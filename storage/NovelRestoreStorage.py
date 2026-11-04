"""一个写入者；运行库原子保存客观与主观资料，个人文件为可恢复投影。"""
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from uuid import uuid4
from urllib.parse import quote

from common.NovelRestoreConfig import NovelRestoreConfig, NovelRestoreNarrativeConfig
from common.NovelRestoreRepository import open_repository
from common.NovelRestoreSchema import (
    ConfirmedAction, EventDraft, Format17GroupDraft, GroupDraft, EventOutcome, NarrativeBatch, NarrativePlan, NarrativeReview, NarrativeSelection,
    ProseFragment, WorldState, MemoryRecord, WorldPerception, RoundSchedule, ReplanSchedule, StatePatch,
    SharedStoryMaterial,
)
from common.Runtime import RunStopped
from common.NovelRestoreReview import require_review
from common.NovelRestoreWorldState import apply_patch, sync_snapshot, validate_references
from common.NovelRestoreStory import story_text
from common.NovelRestoreObjects import effective_catalog
from common.NovelRestoreSubjectiveWorld import MemoryCorruption, NovelRestoreSubjectiveWorld
from storage.NovelRestoreRuntime import CALL_ROLE, NovelRestoreRuntime


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class NarrativeLimitReached(RunStopped):
    """正文实质修订已达上限，保留未完成结果。"""


class NovelRestoreStorage(NovelRestoreRuntime):
    FORMAT = 18

    @classmethod
    def create(cls, config: NovelRestoreConfig):
        config = config.model_copy(update={name: getattr(config, name).expanduser().resolve()
                                          for name in ("repository", "run_directory")})
        config.check_paths()
        # 在创建任何运行文件之前完成输入读取与结构检查。
        with open_repository(config.repository) as repository:
            repository.validate_contract()
            catalog = {kind: list(repository.assets(kind)) for kind in repository.ASSET_KINDS}
            if config.character_action_ratios.keys() - set(catalog["role"]):
                raise ValueError("行动比例覆盖包含未知人物")
            nodes, previous = [], None
            for node in repository.nodes():
                snapshot = repository.snapshot(node["node_id"])
                if snapshot.previous_node_id != previous:
                    raise ValueError("节点快照的前驱与故事顺序不一致")
                validate_references(snapshot, catalog)
                groups = []
                for group in repository.groups(node["node_id"]):
                    events = {event["task_id"]: [p["role_code"] for p in event["participants"]]
                              for event in group.events}
                    if any(set(people) - set(catalog["role"]) for people in events.values()):
                        raise ValueError("事件参与者不在人物目录中")
                    before, after = repository.group_states(group)
                    validate_references(before, catalog)
                    validate_references(after, catalog)
                    groups.append({"group_id": group.group_id, "order": group.order, "events": events, "event_ids": list(events)})
                nodes.append({"node_id": node["node_id"], "order": node["order"], "groups": groups})
                previous = node["node_id"]
            config.validate_branching()
            if config.stop_after_group and not any(
                    group["group_id"] == config.stop_after_group for node in nodes for group in node["groups"]):
                raise ValueError("停止事件组不存在：" + config.stop_after_group)
            run_id = uuid4().hex
            manifest = {"format": cls.FORMAT, "run_id": run_id, "branch_root_id": run_id, "source": repository.source,
                        "config": config.model_dump(mode="json"), "repository_hash": repository.fingerprint(),
                        "catalog": catalog, "characters": repository.assets("role"), "nodes": nodes}
            state = sync_snapshot(None, repository.snapshot(nodes[0]["node_id"]))
        progress = {"stage": "initialized", "node_id": None, "node_order": 0,
                    "group_order": 0, "last_action_id": None}
        directory = config.run_directory
        directory.mkdir(parents=True, exist_ok=False)
        connection = sqlite3.connect(directory / "run.sqlite3")
        try:
            with connection:
                connection.execute("""CREATE TABLE runtime (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), manifest TEXT NOT NULL,
                    version INTEGER NOT NULL, state TEXT NOT NULL, progress TEXT NOT NULL, group_state TEXT)""")
                connection.execute("""CREATE TABLE actions (
                    action_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('prepared','committed')), version INTEGER NOT NULL UNIQUE)""")
                connection.execute("""CREATE INDEX actions_event ON actions(json_extract(payload,'$.position.event_id'),version)
                                      WHERE status='committed'""")
                connection.execute("""CREATE TABLE event_progress (
                    event_id TEXT PRIMARY KEY, node_id TEXT NOT NULL, group_order INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','running','completed','timed_out','failed')),
                    last_action_id TEXT, last_round INTEGER NOT NULL DEFAULT 0, last_turn INTEGER NOT NULL DEFAULT 0,
                    outcome TEXT, outcome_version INTEGER)""")
                connection.execute("CREATE TABLE checkpoints (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                connection.execute("CREATE TABLE jobs (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                connection.execute("CREATE TABLE drafts (group_id TEXT PRIMARY KEY, draft_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, digest TEXT NOT NULL)")
                connection.execute("CREATE TABLE narratives (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                connection.execute("""CREATE TABLE narrative_batches (
                    version_id TEXT NOT NULL, task_id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(version_id,task_id))""")
                connection.execute("""CREATE TABLE calls (
                    id INTEGER PRIMARY KEY, request_key TEXT NOT NULL, occurrence INTEGER NOT NULL,
                    request TEXT NOT NULL, started TEXT NOT NULL, ended TEXT,
                    status INTEGER, response TEXT, error TEXT, usage TEXT, scope TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT '')""")
                connection.execute("CREATE INDEX calls_replay ON calls(scope,request_key,occurrence)")
                connection.executemany("INSERT INTO event_progress(event_id,node_id,group_order,status) VALUES (?,?,?,'pending')",
                                       [(event, node["node_id"], group["order"]) for node in nodes
                                        for group in node["groups"] for event in group["events"]])
                connection.execute("INSERT INTO runtime VALUES (1, ?, 0, ?, ?, NULL)",
                                   (encode(manifest), state.model_dump_json(), encode(progress)))
                connection.executemany("INSERT INTO checkpoints VALUES (?, ?)", [
                    ("self_profiles", encode(manifest["characters"])),
                    ("person_runtime", encode({p: {"can_act": True, "body": "", "effects": []}
                                               for p in catalog["role"]})),
                    ("story_seconds", encode(0)),
                ])
        finally:
            connection.close()
        storage = cls(directory)
        try:
            storage.publish_world()
            storage.publish_characters()
        except Exception:
            storage.close()
            raise
        return storage

    def __init__(self, directory, *, readonly=False, narration_only=False):
        self.directory = Path(directory).expanduser().resolve()
        self.readonly = readonly
        self.connection = None
        self._json_repository = None
        self._lock = None
        try:
            database = self.directory / "run.sqlite3"
            if not database.is_file():
                raise ValueError(f"运行数据库不存在：{database}")
            if not readonly:
                self._lock = (self.directory / ".writer.lock").open("a+b")
                try:
                    fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError("该运行已有写入者") from error
            mode = "ro" if readonly else "rw"
            self.connection = sqlite3.connect(database.as_uri() + f"?mode={mode}", uri=True)
            self.connection.row_factory = sqlite3.Row
            if readonly:
                self.connection.execute("PRAGMA query_only=ON")
                self.connection.execute("BEGIN")
            else:
                self.connection.execute("PRAGMA synchronous=FULL")
            row = self.connection.execute("SELECT manifest FROM runtime WHERE singleton=1").fetchone()
            if row is None:
                raise ValueError("运行初始化尚未完成，请保留现场并使用新的运行目录")
            self.manifest = json.loads(row[0])
            if self.manifest["format"] != self.FORMAT and not (readonly and self.manifest["format"] in range(1, self.FORMAT)):
                raise ValueError("旧运行格式仅支持只读查看；请新建运行，不原地升级")
            saved_config = self.manifest["config"]
            if self.manifest["format"] < 17:
                saved_config = {"max_timelines": 1, **saved_config}
            self.config = NovelRestoreConfig.model_validate(saved_config)
            if not readonly and self.config.run_directory.resolve() != self.directory:
                raise ValueError("运行目录与初始化绑定不一致，不能写入复制的运行")
            self.run_id = self.manifest["run_id"]
            self.memory = NovelRestoreSubjectiveWorld(self.directory / "memories", self.run_id,
                self.manifest["catalog"]["role"], legacy=self.manifest["format"] < 8)
            if not readonly and not narration_only:
                if self.config.repository.is_dir():
                    self._json_repository = open_repository(self.config.repository)
                with self.read_repository() as repository:
                    self._check_repository(repository)
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._json_repository = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def read_repository(self):
        """JSON 在本次写运行中复用启动快照；SQLite 保留独立只读事务。"""
        if self._json_repository is not None:
            return nullcontext(self._json_repository)
        return open_repository(self.config.repository)

    def _check_repository(self, repository):
        if repository.fingerprint() != self.manifest["repository_hash"]:
            raise ValueError("输入 repository 已变化，不能继续本次运行")

    def current(self):
        row = self.connection.execute("SELECT version, state, progress FROM runtime WHERE singleton=1").fetchone()
        return row["version"], WorldState.model_validate_json(row["state"]), json.loads(row["progress"])

    def group_state(self):
        row = self.connection.execute("SELECT group_state FROM runtime WHERE singleton=1").fetchone()
        return WorldState.model_validate_json(row[0]) if row[0] is not None else None

    def _writable(self, *, allow_pending=False):
        if self.readonly:
            raise ValueError("当前运行以只读方式打开")
        if not allow_pending and (self.connection.execute("SELECT 1 FROM actions WHERE status='prepared'").fetchone()
                                  or self.get_checkpoint("pending_wave")):
            raise ValueError("存在待完成的行动交接，须先 recover()")

    def enter_node(self, node_id):
        """前一节点全部事件完成后才同步新节点，重复进入不重置状态。"""
        self._writable()
        version, state, progress = self.current()
        if progress["node_id"] == node_id:
            return version  # 恢复或重复调用不得覆盖该节点已演化的状态。
        next_node = next((row for row in self.manifest["nodes"] if row["order"] == progress["node_order"] + 1), None)
        if next_node is None or next_node["node_id"] != node_id:
            raise ValueError("必须按故事时间进入下一个节点")
        if progress["node_id"] is not None:
            self._require_completed(progress["node_id"])
        with self.read_repository() as repository:
            self._check_repository(repository)
            state = sync_snapshot(state, repository.snapshot(node_id))
        progress.update(stage="node_ready", node_id=node_id, node_order=next_node["order"], group_order=0)
        with self.connection:
            self.connection.execute("UPDATE runtime SET version=?,state=?,progress=?,group_state=NULL WHERE singleton=1",
                                    (version + 1, state.model_dump_json(), encode(progress)))
        self.publish_world()
        return version + 1

    def enter_group(self, order):
        self._writable()
        version, state, progress = self.current()
        node = next((row for row in self.manifest["nodes"] if row["node_id"] == progress["node_id"]), None)
        if node is None:
            raise ValueError("先进入时间节点")
        if type(order) is not int or order not in {group["order"] for group in node["groups"]}:
            raise ValueError("事件组不存在")
        if order == progress["group_order"]:
            return version
        following = [row["order"] for row in node["groups"] if row["order"] > progress["group_order"]]
        if not following or order != following[0]:
            raise ValueError("必须按 order 进入下一事件组")
        if progress["group_order"]:
            self._require_completed(progress["node_id"], progress["group_order"])
        with self.read_repository() as repository:
            self._check_repository(repository)
            group = next(g for g in repository.groups(node["node_id"]) if g.order == order)
            reference, target = repository.group_states(group)
        progress.update(stage="group_ready", group_order=order)
        with self.connection:
            self.connection.execute("UPDATE runtime SET version=?,progress=?,state=?,group_state=? WHERE singleton=1",
                                    (version + 1, encode(progress), state.model_dump_json(), reference.model_dump_json()))
            self._put("checkpoints", "group_target", target.model_dump(mode="json"))
            self._put("checkpoints", "group_clock", {"round": 1, "batch": 1, "stagnant": 0})
            self._put("checkpoints", "last_goal", None)
            self._put("checkpoints", "wave", None)
            self.connection.execute("UPDATE event_progress SET status='running' WHERE node_id=? AND group_order=?",
                                    (progress["node_id"], order))
        self.publish_world()
        return version + 1

    def _require_completed(self, node_id, group_order=None):
        node = next(row for row in self.manifest["nodes"] if row["node_id"] == node_id)
        expected = [event for group in node["groups"] if group_order is None or group["order"] == group_order
                    for event in group["events"]]
        saved = {row["event_id"]: row["status"] for row in self.connection.execute(
            "SELECT event_id,status FROM event_progress WHERE node_id=?", (node_id,))}
        incomplete = [event for event in expected if saved.get(event) != "completed"]
        if not expected or incomplete:
            raise ValueError("全组事件完成后才能继续，未完成事件：" + ", ".join(incomplete))
        for group in node["groups"]:
            if group_order is None or group["order"] == group_order:
                result = self.group_assessment(group["group_id"])
                if not result or not result["achieved"]:
                    raise ValueError("事件组世界状态目标尚未通过验收")

    def group_assessment(self, group_id):
        return self.get_checkpoint("group_assessment/" + group_id)

    def save_round_schedule(self, group, clock, result):
        """原子保存调度及从未知位置补充的模拟初态；不伪造人物移动。"""
        self._writable()
        version, state, progress = self.current()
        if (group.node_id, group.order) != (progress["node_id"], progress["group_order"]) or clock != self.get_checkpoint("group_clock"):
            raise ValueError("行动机会或初始地点的依据已过期")
        payload = result.model_dump(mode="json")
        if clock["stagnant"] >= self.config.stagnation_rounds:
            result = ReplanSchedule.model_validate(payload)
            require_review(self, "schedule", payload)
        else:
            result = RoundSchedule.model_validate(payload)
        states = self.event_states()
        people = {p["role_code"] for event in group.events if states[event["task_id"]]["status"] == "running"
                  for p in event["participants"]}
        changes, entries = [], []
        for person, choice in result.initial_locations.items():
            if person not in people or person not in state.characters:
                raise ValueError("只能为本组活跃事件的参与人物补充初始地点")
            previous = state.characters[person].location
            if (previous.location.status != "unknown" or state.characters[person].life_status == "not_born"
                    or choice.location_id not in self.manifest["catalog"]["location"]):
                raise ValueError("初始地点只允许补充已登记地点与未知位置")
            changes.append({"role_id": person, "set": {"location": {"location": {"status": "known",
                "references": [{"type": "location", "id": choice.location_id}]}, "detail": choice.detail}},
                "basis": choice.basis})
            entries.append({"kind": "simulation_initial_location", "person": person, "node_id": group.node_id,
                "group_id": group.group_id, "round": clock["round"], "location_id": choice.location_id,
                "basis": choice.basis, "detail": choice.detail, "previous": previous.model_dump(mode="json")})
        if changes:
            state = apply_patch(state, StatePatch.model_validate({"characters": changes}))
            validate_references(state, effective_catalog(self))
        schedule = {**result.model_dump(mode="json"), "group_id": group.group_id, "round": clock["round"]}
        with self.connection:
            if changes:
                self.connection.execute("UPDATE runtime SET version=?,state=? WHERE singleton=1", (version + 1, state.model_dump_json()))
                self._put("checkpoints", "location_initializations", self.get_checkpoint("location_initializations", []) + entries)
            self._put("checkpoints", "round_schedule", schedule)
        if changes:
            self.publish_world()
        return schedule

    def record_group_assessment(self, group, version, assessment):
        self._writable()
        current, state, progress = self.current()
        if version != current or (group.node_id, group.order) != (progress["node_id"], progress["group_order"]):
            raise ValueError("事件组验收依据已过期")
        states = self.event_states()
        if any(states[e["task_id"]]["status"] != "completed" for e in group.events):
            raise ValueError("全组事件完成后才能确认组后目标")
        from common.NovelRestoreWorldState import validate_result_evidence
        actions = {a.action_id for event in group.events for a in self.event_actions(event["task_id"])}
        validate_result_evidence(assessment, state.model_dump(mode="json"), actions)
        from common.NovelRestoreReview import require_review, group_delivery
        require_review(self, "group", group_delivery(self, group.group_id, assessment))
        with self.connection:
            self._put("checkpoints", "group_assessment/" + group.group_id,
                      {"version": version, **assessment.model_dump(mode="json")})
            self._put("checkpoints", "last_goal", assessment.model_dump(mode="json"))
            if not assessment.achieved:
                for event in group.events:
                    row = states[event["task_id"]]
                    exhausted = self.get_checkpoint("group_clock", {}).get("round", 1) > self.config.max_rounds
                    if not exhausted:
                        self.connection.execute("UPDATE event_progress SET status='running',outcome=NULL,outcome_version=NULL WHERE event_id=?",
                                                (event["task_id"],))

    def publish_world(self):
        from exporters.NovelRestoreExporter import NovelRestoreExporter
        NovelRestoreExporter(self).publish({"world_state.json": json.dumps(
            self.current()[1].model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"}, "export/world")

    def record_event_outcome(self, outcome: EventOutcome):
        """登记世界的事件判断；超限和失败保留为未完成，不能放行后续组。"""
        self._writable()
        outcome = EventOutcome.model_validate(outcome.model_dump(mode="json"))
        if outcome.run_id != self.run_id:
            raise ValueError("事件结论不属于本次运行")
        payload = encode(outcome.model_dump(mode="json"))
        event = self.connection.execute("SELECT * FROM event_progress WHERE event_id=?", (outcome.event_id,)).fetchone()
        if event is None:
            raise ValueError("事件不存在")
        if event["outcome"] is not None:
            if event["outcome"] != payload:
                raise ValueError("事件结论已保存，不能覆盖")
            return event["outcome_version"]
        version, _, progress = self.current()
        if outcome.expected_version != version:
            raise ValueError("事件结论的状态版本已不匹配")
        if event["status"] != "running" or (event["node_id"], event["group_order"]) != (
                progress["node_id"], progress["group_order"]):
            raise ValueError("只能判断当前组正在模拟的事件")
        for identity in outcome.action_ids:
            row = self.connection.execute("SELECT payload,status FROM actions WHERE action_id=?", (identity,)).fetchone()
            if row is None or row["status"] != "committed" or json.loads(row["payload"])["position"]["event_id"] != outcome.event_id:
                raise ValueError("事件结论只能引用本事件已提交的行动")
        if outcome.status == "completed" and event["last_action_id"] not in outcome.action_ids:
            raise ValueError("完成结论必须包含本事件最新行动")
        if outcome.status == "timed_out" and self.get_checkpoint("group_clock")["round"] <= self.config.max_rounds:
            raise ValueError("事件尚未达到轮数上限")
        with self.connection:
            self.connection.execute("UPDATE event_progress SET status=?,outcome=?,outcome_version=? WHERE event_id=?",
                                    (outcome.status, payload, version + 1, outcome.event_id))
            self.connection.execute("UPDATE runtime SET version=? WHERE singleton=1", (version + 1,))
        return version + 1

    @staticmethod
    def _memory_batches(action):
        batches = defaultdict(list)
        for record in action.memories:
            batches[record.character_id].append(record)
        return batches

    def _validate_action(self, action, *, catalog=None, validate_state=True):
        if catalog is None and action.advance and action.advance.new_objects:
            raise ValueError("新增物品必须通过批次原子交接登记")
        catalog = catalog if catalog is not None else effective_catalog(self)
        version, state, progress = self.current()
        if action.run_id != self.run_id or action.expected_version != version:
            raise ValueError("行动的运行身份或世界状态版本已不匹配")
        position = action.position
        if (position.node_id, position.node_order, position.group_order) != (
                progress["node_id"], progress["node_order"], progress["group_order"]):
            raise ValueError("行动不属于当前节点和事件组")
        node = self.manifest["nodes"][position.node_order - 1]
        group = next(row for row in node["groups"] if row["order"] == position.group_order)
        participants = group["events"].get(position.event_id)
        if participants is None or (action.proposal.character_id is not None and action.proposal.character_id not in participants):
            raise ValueError("行动人物不是当前核心事件的参与者")
        if position.round > self.config.max_rounds:
            raise ValueError("行动超过配置的最大轮数")
        event = self.connection.execute("SELECT * FROM event_progress WHERE event_id=?", (position.event_id,)).fetchone()
        if event is None or event["status"] != "running":
            raise ValueError("事件未开始或已经结束，不能提交新行动")
        if (position.round, position.turn) <= (event["last_round"], event["last_turn"]):
            raise ValueError("同一事件的行动位置必须向前推进")
        wave = self.get_checkpoint("wave", []) or []
        scheduled = next((r for r in wave if r["action_id"] == action.action_id), None)
        clock = self.get_checkpoint("group_clock")
        if scheduled:
            valid = scheduled["position"] == position.model_dump() and scheduled["character_id"] == action.proposal.character_id
        else:
            valid = not wave and (position.round, position.turn) == (clock["round"], clock["batch"])
        if not valid:
            raise ValueError("行动必须使用当前调度位置；每次人物行为计一轮")
        arrangement = scheduled.get("arrangement") if scheduled else None
        if action.proposal.character_id is None:
            if (scheduled and not arrangement) or (not scheduled and participants):
                raise ValueError("人物事件中的环境推进必须来自已保存的安排")
            if action.proposal.mode != "action" or action.proposal.target_ids:
                raise ValueError("环境推进不能冒充人物行为或引用个人感知对象")
            if arrangement and (action.proposal.content, action.environment_basis) != (arrangement["content"], arrangement["basis"]):
                raise ValueError("环境提议与已保存的安排不一致")
        elif arrangement or action.environment_basis is not None:
            raise ValueError("人物自主行为不能附带环境指令")
        before_view = action.views_before.get(action.proposal.character_id)
        if scheduled and before_view != (WorldPerception.model_validate(scheduled["before_view"])
                                                if scheduled.get("before_view") else None):
            raise ValueError("行动前个人感知不能在裁定后被替换")
        targets = {p.perceived_id for p in before_view.personal.objects} if before_view else set()
        if set(action.proposal.target_ids) - targets:
            raise ValueError("行动指向了人物尚未感知的对象")
        self.validate_resolved_targets(action.proposal, before_view, action.resolved_targets, catalog)
        if any(m.actual_kind == "unknown" for m in action.resolved_targets):
            if action.unknown_target_review is None or not action.unknown_target_review.approved:
                raise ValueError("对象仍未知的行动须经过独立检查，确认结果不依赖未知身份或存在")
        self.validate_views(action.observations, catalog)
        self.validate_views(action.views_before)
        if scheduled:
            current_views = self.get_checkpoint("views", {})
            expected = {p: WorldPerception.model_validate(current_views[p]) for p in action.observations if p in current_views}
            if action.views_before != expected:
                raise ValueError("观察者的行动前体验须来自已保存的主观世界")
            if action.memory_review is None or not action.memory_review.approved:
                raise ValueError("批次主观更新必须经过世界检查")
            if {m.character_id for m in action.memories} - set(action.observations):
                raise ValueError("记忆作者须收到本次个人反馈；未形成记忆允许为空")
            if set(action.subjective_inputs) != set(action.observations):
                raise ValueError("须保存每个观察者实际收到的已处理输入")
            if action.proposal.character_id is not None and action.proposal.character_id not in action.observations:
                raise ValueError("行动者须收到本人的行动后体验")
        if action.memory_review is not None and not action.memory_review.approved:
            raise ValueError("人物主观世界未通过检查，不能提交")
        if action.proposal.mode == "think" and set(action.observations) - {action.proposal.character_id} - set(action.advance.observations if action.advance else {}):
            raise ValueError("私有思考不能写入其他人物的体验")
        if action.perceptions:
            raise ValueError("旧感知格式只供历史运行读取，新行动使用个人体验及对应关系")
        for record in action.memories:
            if record.source_shared_id is not None:
                raise ValueError("共享环境体验的记忆不能冒充人物行动记忆")
            if record.source_action_id not in (None, action.action_id):
                raise ValueError("记忆来源行动与实际绑定不一致")
            before = action.views_before.get(record.character_id)
            after = action.observations.get(record.character_id)
            allowed = {o.perceived_id for view in (before, after) if view for o in view.personal.objects}
            if set(record.perceived_ids) - allowed:
                raise ValueError("记忆感知对象没有本人本次体验的来源")
            if scheduled:
                access = action.subjective_inputs[record.character_id]
                encoded = next((e.experience for e in access.experiences if e.source_id == action.action_id
                                and e.source_kind == "action"), None)
                if encoded is None:
                    raise ValueError("未编码的体验不能被总结成该行动的记忆")
                if set(record.perceived_ids) - {o.perceived_id for o in encoded.objects}:
                    raise ValueError("记忆对象须来自处理后可记住的体验")
                if set(record.corrects + record.related_memory_ids) - {m.memory_id for m in access.memories}:
                    raise ValueError("人物只能关联本次实际可回想的旧记忆")
            people = [record.character_id, *record.role_ids, *record.listener_ids]
            if record.speaker_id is not None:
                people.append(record.speaker_id)
            if (set(people) - set(catalog["role"]) or set(record.object_ids) - set(catalog["object"])
                    or set(record.location_ids) - set(catalog["location"])):
                raise ValueError("记忆引用了未知人物、地点或物品")
            if action.proposal.mode == "think" and record.character_id != action.proposal.character_id and record.character_id not in (action.advance.observations if action.advance else {}):
                raise ValueError("私有思考不能写入其他人物的记忆")
        if validate_state:
            validate_references(apply_patch(state, action.state_patch), catalog)
        for character, records in self._memory_batches(action).items():
            self.validate_memories(character, records)

    def commit_action(self, action: ConfirmedAction):
        self._writable(allow_pending=True)
        # 重新验证可变的 Pydantic 实例，避免调用方修改后绕过字段检查。
        action = ConfirmedAction.model_validate(action.model_dump(mode="json"))
        payload = encode(action.model_dump(mode="json"))
        previous = self.connection.execute("SELECT * FROM actions WHERE action_id=?", (action.action_id,)).fetchone()
        if previous is not None:
            if previous["payload"] != payload:
                raise ValueError("行动身份已绑定另一份内容，不能覆盖")
            if previous["status"] == "prepared":
                self.recover()
            return previous["version"]
        self._writable()
        if self.get_checkpoint("wave"):
            raise ValueError("已调度的并发批次须整体提交，不能拆成单独行动")
        self._validate_action(action)
        with self.connection:
            self.connection.execute("INSERT INTO actions VALUES (?,?,'prepared',?)",
                                    (action.action_id, payload, action.expected_version + 1))
        self.recover()
        return action.expected_version + 1

    def recover(self):
        self._writable(allow_pending=True)
        if self.get_checkpoint("pending_wave"):
            return self.recover_wave()
        row = self.connection.execute("SELECT * FROM actions WHERE status='prepared'").fetchone()
        if row is None:
            self.publish_world()
            self.publish_characters()
            return False
        action = ConfirmedAction.model_validate_json(row["payload"])
        version, state, progress = self.current()
        if version != action.expected_version:
            raise ValueError("待交接行动与当前状态版本冲突，保留现场")
        state = apply_patch(state, action.state_patch)
        progress.update(stage="simulating", last_action_id=action.action_id)
        clock = self.advance_clock(1, bool(action.completion and action.completion.achieved))
        with self.connection:
            self._put("checkpoints", "group_clock", clock)
            views = self.get_checkpoint("views", {})
            views.update({p: v.model_dump(mode="json") for p, v in action.observations.items()})
            self._put("checkpoints", "views", views)
            self.connection.execute("UPDATE runtime SET version=?,state=?,progress=? WHERE singleton=1",
                                    (version + 1, state.model_dump_json(), encode(progress)))
            self.connection.execute("UPDATE actions SET status='committed' WHERE action_id=?", (action.action_id,))
            self.connection.execute("UPDATE event_progress SET last_action_id=?,last_round=?,last_turn=? WHERE event_id=?",
                                    (action.action_id, action.position.round, action.position.turn, action.position.event_id))
            if action.completion is not None:
                position = action.position
                exhausted = clock["round"] > self.config.max_rounds
                if action.completion.achieved or exhausted:
                    outcome = EventOutcome(run_id=self.run_id, event_id=position.event_id,
                                           expected_version=version,
                                           status="completed" if action.completion.achieved else "timed_out",
                                           summary=action.completion.summary,
                                           action_ids=[item.action_id for item in self.event_actions(position.event_id)])
                    self.connection.execute("UPDATE event_progress SET status=?,outcome=?,outcome_version=? WHERE event_id=?",
                                            (outcome.status, outcome.model_dump_json(), version + 1, position.event_id))
        self.publish_world()
        self.publish_characters()
        return True

    def _put(self, table, key, payload):
        # table 仅由下列固定调用方提供，不接收外部输入。
        self.connection.execute(f"INSERT INTO {table} VALUES (?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
                                (key, encode(payload)))

    def get_checkpoint(self, key, default=None):
        row = self.connection.execute("SELECT payload FROM checkpoints WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_checkpoint(self, key, payload):
        self._writable()
        with self.connection:
            self._put("checkpoints", key, payload)

    def get_job(self, key):
        row = self.connection.execute("SELECT payload FROM jobs WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_job(self, key, payload):
        self._writable()
        with self.connection:
            self._put("jobs", key, payload)

    def initialize_world(self):
        self._writable()
        if not self.get_checkpoint("world_initialized"):
            self.set_checkpoint("world_initialized", True)
        self.publish_characters()

    def mark_stage(self, stage, error=None):
        self._writable(allow_pending=True)
        if stage == "simulation_complete":
            for node in self.manifest["nodes"]:
                self._require_completed(node["node_id"])
                if any(self.get_draft(group["group_id"]) is None for group in node["groups"]):
                    raise ValueError("所有事件组均须保存正式草稿后才能完成模拟")
        progress = self.current()[2]
        if stage == "group_complete":
            self._require_completed(progress["node_id"], progress["group_order"])
            node = next(n for n in self.manifest["nodes"] if n["node_id"] == progress["node_id"])
            group = next(g for g in node["groups"] if g["order"] == progress["group_order"])
            if group["group_id"] != self.config.stop_after_group or self.get_draft(group["group_id"]) is None:
                raise ValueError("指定停止事件组尚未完成草稿保存")
        progress.update(stage=stage, error=error)
        with self.connection:
            self.connection.execute("UPDATE runtime SET progress=? WHERE singleton=1", (encode(progress),))

    def event_states(self):
        return {row["event_id"]: dict(row) for row in self.connection.execute("SELECT * FROM event_progress")}

    def event_actions(self, event_id=None):
        where, values = ("", ()) if event_id is None else (" AND json_extract(payload,'$.position.event_id')=?", (event_id,))
        return [ConfirmedAction.model_validate_json(row[0]) for row in self.connection.execute(
            "SELECT payload FROM actions WHERE status='committed'" + where + " ORDER BY version", values)]

    def get_action(self, action_id):
        row = self.connection.execute("SELECT payload FROM actions WHERE action_id=? AND status='committed'", (action_id,)).fetchone()
        return ConfirmedAction.model_validate_json(row[0]) if row else None

    def get_draft(self, identity):
        """当前格式按组读取；旧运行的只读 API 保留原来的事件键。"""
        if self.manifest["format"] < 17:
            row = self.connection.execute("SELECT payload FROM drafts WHERE event_id=?", (identity,)).fetchone()
            return EventDraft.model_validate_json(row[0]) if row else None
        row = self.connection.execute("SELECT payload,digest FROM drafts WHERE group_id=?", (identity,)).fetchone()
        if row is None:
            return None
        self.check_draft_digest(row)
        schema = Format17GroupDraft if self.manifest["format"] == 17 else GroupDraft
        return schema.model_validate_json(row["payload"])

    def group_actions(self, node_id, order):
        return [ConfirmedAction.model_validate_json(row[0]) for row in self.connection.execute(
            """SELECT payload FROM actions WHERE status='committed'
               AND json_extract(payload,'$.position.node_id')=? AND json_extract(payload,'$.position.group_order')=?
               ORDER BY version""", (node_id, order))]

    @staticmethod
    def _shared_materials(group_id, actions):
        result = []
        for action in actions:
            advance = action.advance
            if advance and advance.result and advance.shared_id:
                result.append(SharedStoryMaterial(shared_id=advance.shared_id, group_id=group_id, turn=action.position.turn,
                    result=advance.result, related_event_ids=advance.related_event_ids,
                    experiences={p: view.personal for p, view in advance.observations.items()},
                    memories=[m.story_memory() for m in advance.memories],
                    working_states={p: state.story_state() for p, state in advance.working_states.items()}))
        return result

    def group_materials(self, group_id):
        node, group = self._find_group(group_id)
        return self._shared_materials(group_id, self.group_actions(node["node_id"], group["order"]))

    def _find_group(self, group_id):
        for node in self.manifest["nodes"]:
            for group in node["groups"]:
                if group["group_id"] == group_id:
                    return node, group
        raise ValueError("未知事件组：" + group_id)

    def _group_draft(self, node, group):
        self._require_completed(node["node_id"], group["order"])
        group_id = group["group_id"]
        events = group["event_ids"]
        actions = self.group_actions(node["node_id"], group["order"])
        shared = self._shared_materials(group_id, actions)
        materials = [a.story_material() for a in actions]
        # 人物表只解释冻结材料中实际行动或获得体验的人物身份，上游预定参与者不能证明到场。
        people = {}
        event_actions = {event: [] for event in events}
        for material in materials:
            event_actions[material.position.event_id].append(material.action_id)
            people.update(dict.fromkeys(material.perceptions_before))
            if material.proposal:
                people[material.proposal.character_id] = None
        for item in [*materials, *shared]:
            people.update(dict.fromkeys(item.experiences))
            people.update(dict.fromkeys(item.working_states))
            people.update(dict.fromkeys(m.character_id for m in item.memories))
        characters = self.manifest["characters"]
        names = {p: characters[p].get("role_name") or "未命名人物" for p in people}
        results = self.get_checkpoint("core_completed", {})
        return GroupDraft(draft_id="draft/" + quote(group_id, safe=""), run_id=self.run_id, group_id=group_id,
            node_id=node["node_id"], node_order=node["order"], group_order=group["order"], event_ids=events,
            action_ids=[m.action_id for m in materials], participant_ids=list(people), participant_names=names,
            event_action_ids=event_actions,
            batch_elapsed_seconds={a.position.turn: a.advance.elapsed_seconds for a in actions if a.advance},
            achieved_results={e: results[e]["achieved_results"] for e in events},
            materials=materials, shared_materials=shared, shared_material_ids=[m.shared_id for m in shared],
            text=story_text(actions, characters))

    def archive_group(self, group_id):
        """从已提交故事生成唯一正式版本；重复调用只能得到逐项相同的材料。"""
        self._writable()
        draft = self._group_draft(*self._find_group(group_id))
        old = self.get_draft(group_id)
        if old is not None:
            if old != draft:
                raise ValueError("同一事件组的正式草稿不能覆盖")
            return
        with self.connection:
            payload = draft.model_dump_json()
            self.connection.execute("INSERT INTO drafts VALUES (?,?,?,?)",
                                    (group_id, draft.draft_id, payload, sha256(payload.encode()).hexdigest()))

    def reserve(self, key, occurrence, body, limit, *, can_send, scope="simulation", budget_config=None,
                prior_usage=None):
        self._writable()
        # 重演继承费用账本，但请求缓存须隔离；运行身份无需进入人物提示。
        key = sha256(encode([self.run_id, key]).encode()).hexdigest()
        row = self.connection.execute("SELECT * FROM calls WHERE scope=? AND request_key=? AND occurrence=? AND ended IS NOT NULL ORDER BY id DESC LIMIT 1",
                                      (scope, key, occurrence)).fetchone()
        if row is not None:
            return dict(row), True
        if not can_send:
            raise RunStopped("未提供模型密钥，不能发出新请求")
        self.check_budget(budget_config or self.config, scope, prior_usage)
        if self.call_count(scope) >= limit:
            raise RunStopped("累计模型调用预算已用尽")
        with self.connection:
            from model.ModelClient import CALL_SCOPE
            job = body.get("job") or (CALL_SCOPE.get() or ("unknown",))[0]
            role = job.split("/", 1)[0]
            role = "character" if role in ("recall", "memory", "summary") else "narrator" if role == "batch" else "world" if role == "environment" else role
            role = CALL_ROLE.get() or role
            cursor = self.connection.execute("INSERT INTO calls(request_key,occurrence,request,started,scope,role) VALUES (?,?,?,?,?,?)",
                                             (key, occurrence, encode(body), datetime.now(timezone.utc).isoformat(), scope, role))
        return {"id": cursor.lastrowid}, False

    def received(self, call_id, status, raw, usage):
        self._writable()
        with self.connection:
            self.connection.execute("UPDATE calls SET ended=?,status=?,response=?,usage=? WHERE id=?",
                                    (datetime.now(timezone.utc).isoformat(), status, raw, encode(usage), call_id))

    def failed(self, call_id, error):
        self._writable()
        with self.connection:
            self.connection.execute("UPDATE calls SET ended=?,error=? WHERE id=?",
                                    (datetime.now(timezone.utc).isoformat(), error, call_id))

    def narrative_source(self):
        """仅使用库中已确认资料，不读取草稿导出文件或原小说。"""
        if self.connection.execute("SELECT json_extract(progress,'$.stage') FROM runtime WHERE singleton=1").fetchone()[0] != "simulation_complete":
            raise ValueError("全部模拟完成后才能开始叙述")
        if self.manifest["format"] < self.FORMAT:
            raise ValueError("旧运行的草稿未归档当前协议的行动前回想，只能查看历史结果")
        drafts, actions, shared, characters = [], {}, {}, {}
        for node in self.manifest["nodes"]:
            for group in node["groups"]:
                draft = self.get_draft(group["group_id"])
                if (draft is None or draft.run_id != self.run_id or draft.node_id != node["node_id"]
                        or draft.group_id != group["group_id"] or draft.group_order != group["order"]
                        or draft.node_order != node["order"] or draft.event_ids != group["event_ids"]):
                    raise ValueError("叙述来源缺失或组草稿与冻结时间轴不一致")
                drafts.append(draft.model_dump(mode="json", exclude={"materials", "shared_materials"}))
                characters.update({p: {"role_name": name} for p, name in draft.participant_names.items()})
                for event in draft.event_ids:
                    actions[event] = []
                for material in draft.materials:
                    actions[material.position.event_id].append(material.model_dump(mode="json"))
                shared.update({m.shared_id: m.model_dump(mode="json") for m in draft.shared_materials})
        if not drafts:
            raise ValueError("叙述来源必须包含已验收的组草稿")
        return {"run_id": self.run_id, "characters": characters, "drafts": drafts,
                "actions": actions, "shared_materials": shared}

    def create_narrative(self, config, binding, *, selection=None):
        self._writable()
        config = NovelRestoreNarrativeConfig.model_validate(config.model_dump(mode="json"))
        if self.connection.execute("SELECT 1 FROM narratives WHERE key=?", (config.version_id,)).fetchone():
            raise ValueError("叙述版本已存在，请使用 resume 或选择新的版本身份")
        state = {"config": config.model_dump(mode="json"), "binding": binding,
                 "source_hash": sha256(encode(self.narrative_source()).encode()).hexdigest(),
                 "stage": "initialized", "error": None, "epoch": 0, "sequence": 1,
                 "assignment": None, "plan": None, "prose": {}, "links": {},
                 "pending": None, "reviews": [], "issues": [], "prose_meta": {}, "batch_reviews": []}
        if selection is not None:
            state["selection"] = selection
        with self.connection:
            self._put("narratives", config.version_id, state)

    def narrative(self, version_id):
        row = self.connection.execute("SELECT payload FROM narratives WHERE key=?", (version_id,)).fetchone()
        if row is None:
            raise ValueError("叙述版本不存在：" + version_id)
        return json.loads(row[0])

    def update_narrative(self, version_id, **updates):
        self._writable()
        if updates.keys() - {"stage", "error", "assignment", "pending"}:
            raise ValueError("正文与计划须通过专用事务更新")
        state = self.narrative(version_id)
        state.update(updates)
        with self.connection:
            self._put("narratives", version_id, state)

    def check_narrative_source(self, version_id):
        source = self.narrative_source()
        if sha256(encode(source).encode()).hexdigest() != self.narrative(version_id)["source_hash"]:
            raise ValueError("叙述来源草稿或行动已变化，不能继续旧版本")
        return source

    def validate_narrative_plan(self, version_id, plan):
        plan = NarrativePlan.model_validate(plan.model_dump(mode="json"))
        config = self.narrative(version_id)["config"]
        if (plan.run_id, plan.version_id, plan.requirements, plan.revision) != (
                self.run_id, version_id, config["requirements"], 1):
            raise ValueError("叙述计划身份或版本要求不匹配")
        drafts = {row["draft_id"]: row for row in self.check_narrative_source(version_id)["drafts"]}
        if {(fragment.draft_id, fragment.event_id) for fragment in plan.fragments} != {
                (key, event) for key, draft in drafts.items() for event in draft["event_ids"]}:
            raise ValueError("叙述计划必须覆盖全部来源草稿及其事件")
        for fragment in plan.fragments:
            self.validate_fragment(fragment, drafts)
        spec = config["version"]
        if spec["length"] is not None and sum(f.length_budget for f in plan.fragments) != spec["length"]:
            raise ValueError("片段篇幅预算总和须等于版本目标长度")
        by_id = {f.fragment_id: f for f in plan.fragments}
        if any(c.length_budget != sum(by_id[i].length_budget for i in c.fragment_ids) for c in plan.chapters):
            raise ValueError("每章篇幅预算须等于所含片段预算之和")
        if any(not link.acceptance for link in plan.links):
            raise ValueError("关联须明确正文中的成立和回收条件")
        return plan

    def validate_fragment(self, fragment, drafts):
        draft = drafts.get(fragment.draft_id)
        if not draft or fragment.event_id not in draft["event_ids"] or (
                set(fragment.action_ids) - set(draft["event_action_ids"][fragment.event_id])):
            raise ValueError("片段的草稿、事件或行动引用不一致")
        if not fragment.action_ids or not fragment.purpose or not fragment.viewpoint or fragment.length_budget is None:
            raise ValueError("片段须明确具体行动来源、叙述目的、视角和篇幅预算")
        if fragment.viewpoint_role_id is not None and not any(
                fragment.viewpoint_role_id in d["participant_ids"] for d in drafts.values()):
            raise ValueError("片段视角引用未知人物")
        if fragment.knowledge_action_id:
            if not any(fragment.knowledge_action_id in d["action_ids"] for d in drafts.values()):
                raise ValueError("人物知情时点须指向实际已提交行动")
            if fragment.knowledge_action_id not in fragment.action_ids and not fragment.knowledge_basis:
                raise ValueError("使用内容来源之外的知情时点须说明依据")

    def save_narrative_plan(self, version_id, plan):
        self._writable()
        plan = self.validate_narrative_plan(version_id, plan)
        state = self.narrative(version_id)
        payload = plan.model_dump(mode="json")
        if state["plan"] is not None:
            if state["plan"] != payload:
                raise ValueError("已接收的叙述落位不能覆盖；重新落位请另建版本")
            return
        state.update(plan=payload, stage="writing", epoch=state["epoch"] + 1,
                     links={link.link_id: {"link": link.model_dump(mode="json"), "revisions": {}} for link in plan.links})
        with self.connection:
            self._put("narratives", version_id, state)

    @staticmethod
    def link_current(record, prose):
        link = record["link"]
        ends = [link["from_fragment"], link["to_fragment"]]
        return (all(end in prose for end in ends) and
                record["revisions"] == {end: prose[end]["revision"] for end in ends})

    def select_narrative_links(self, version_id, selection, required=()):
        """校验拟修改范围并在副本上应用关系变更；不触碰当前正文。"""
        selection = NarrativeSelection.model_validate(selection.model_dump(mode="json"))
        state = self.narrative(version_id)
        if state["plan"] is None:
            raise ValueError("先保存完整章节片段落位")
        plan = NarrativePlan.model_validate(state["plan"])
        targets = set(selection.fragment_ids)
        if targets - set(plan.reading_order) or set(required) - targets:
            raise ValueError("联合任务包含未知片段或遗漏必需目标")
        links = state["links"]  # narrative() 每次解码新副本。
        for operation in [*selection.upsert_links, *selection.remove_links]:
            old = links.get(operation.link_id)
            if old and {old["link"]["from_fragment"], old["link"]["to_fragment"]} - targets:
                raise ValueError("修改已有关系必须共同读取其原有两端")
        for removal in selection.remove_links:
            if removal.link_id not in links:
                raise ValueError("不能删除不存在的叙述关系")
            del links[removal.link_id]
        for link in selection.upsert_links:
            if not link.acceptance:
                raise ValueError("关联须明确正文中的成立和回收条件")
            if {link.from_fragment, link.to_fragment} - targets:
                raise ValueError("新建或修改关系必须共同处理其两端")
            old = links.get(link.link_id)
            if old is None or old["link"] != link.model_dump(mode="json"):
                links[link.link_id] = {"link": link.model_dump(mode="json"), "revisions": {}}
        NarrativePlan.model_validate({**state["plan"], "links": [row["link"] for row in links.values()]})
        return links

    def preview_narrative_batch(self, version_id, task_id, epoch, selection, batch, review=None):
        state = self.narrative(version_id)
        if state["stage"] in ("complete", "incomplete") or epoch != state["epoch"]:
            raise ValueError("叙述版本已结束或正文依据版本已变化")
        pending = state["pending"]
        if not pending or pending["task_id"] != task_id or pending.get("selection") != selection.model_dump(mode="json"):
            raise ValueError("正文必须绑定已保存的联合任务与选取范围")
        links = self.select_narrative_links(version_id, selection, pending["required"])
        batch = NarrativeBatch.model_validate(batch.model_dump(mode="json"))
        targets = set(selection.fragment_ids)
        if {row.fragment_id for row in batch.texts} != targets:
            raise ValueError("联合交付必须恰好包含本次目标的全部正文")
        internal = {key for key, row in links.items()
                    if {row["link"]["from_fragment"], row["link"]["to_fragment"]} <= targets}
        if set(batch.confirmed_link_ids) != internal:
            raise ValueError("须共同核对本次目标之间的全部有效关联")
        for text in batch.texts:
            if not text.summary.strip():
                raise ValueError("正文须提供供后续检索和全书复核的内容摘要")
            old = state["prose"].get(text.fragment_id)
            if old and old["text"] == text.text:
                continue
            revision = old["revision"] + 1 if old else 1
            if revision > state["config"]["max_revisions"] + 1:
                raise NarrativeLimitReached("片段正文修订上限已用尽：" + text.fragment_id)
            state["prose"][text.fragment_id] = ProseFragment(
                version_id=version_id, plan_revision=state["plan"]["revision"], fragment_id=text.fragment_id,
                revision=revision, task_id=task_id, text=text.text).model_dump(mode="json")
            state.setdefault("prose_meta", {})[text.fragment_id] = {
                "summary": text.summary, "revealed": text.revealed, "change": text.change}
        for identity in internal:
            link = links[identity]["link"]
            links[identity]["revisions"] = {end: state["prose"][end]["revision"]
                                            for end in (link["from_fragment"], link["to_fragment"])}
        if review is not None:
            if not review.approved:
                raise ValueError("联合正文语义检查未通过，不得保存为有效正文")
            for identity in review.preserved_link_ids:
                if identity not in links:
                    raise ValueError("语义检查保留了未知关联")
                ends = {links[identity]["link"]["from_fragment"], links[identity]["link"]["to_fragment"]}
                if not ends.intersection(targets) or not ends <= state["prose"].keys():
                    raise ValueError("只能保留已读取正文的直接关联")
                links[identity]["revisions"] = {end: state["prose"][end]["revision"] for end in ends}
            known = set(NarrativePlan.model_validate(state["plan"]).reading_order)
            if set(review.affected_fragment_ids) - (known - targets):
                raise ValueError("影响传播只能标记本次目标之外的实际片段")
            state.setdefault("batch_reviews", []).append({"task_id": task_id, "result": review.model_dump(mode="json")})
        issues = [i for i in state["issues"] if set(i["fragment_ids"]) - targets]
        if review is not None:
            issues.extend({"fragment_ids": [i], "description": "本次表达、已有线索呈现或读者揭示变化，需要复核"}
                          for i in dict.fromkeys(review.affected_fragment_ids))
        # 未经语义核对的外部关系保留旧确认，自动进入后续联合任务。
        state.update(links=links, epoch=epoch + 1, sequence=state["sequence"] + 1,
                     pending=None, issues=issues, stage="writing", error=None)
        return state

    def commit_narrative_batch(self, version_id, task_id, epoch, selection, batch, review=None):
        self._writable()
        if review is None or not review.approved:
            raise ValueError("保存联合正文前必须通过世界角色的语义核对")
        self.check_narrative_source(version_id)
        payload = encode({"epoch": epoch, "selection": selection.model_dump(mode="json"), "batch": batch.model_dump(mode="json"),
                          "review": review.model_dump(mode="json") if review else None})
        old = self.connection.execute("SELECT payload FROM narrative_batches WHERE version_id=? AND task_id=?",
                                      (version_id, task_id)).fetchone()
        if old is not None:
            if old[0] != payload:
                raise ValueError("联合任务已经绑定另一份结果")
            return
        state = self.preview_narrative_batch(version_id, task_id, epoch, selection, batch, review)
        with self.connection:
            self.connection.execute("INSERT INTO narrative_batches VALUES (?,?,?)", (version_id, task_id, payload))
            self._put("narratives", version_id, state)

    def narrative_status(self, version_id):
        state = self.narrative(version_id)
        order = NarrativePlan.model_validate(state["plan"]).reading_order if state["plan"] else []
        scope = NarrativeScope.scope_name(version_id)
        receipt = self.get_checkpoint("export/narrative/" + version_id, {})
        return {"version_id": version_id, "stage": state["stage"], "error": state["error"],
                "published": state["stage"] == "complete" and receipt.get("complete_epoch") == state["epoch"] and not receipt.get("pending"),
                "plan_ready": state["plan"] is not None, "fragments": len(order), "written": len(state["prose"]),
                "missing": [identity for identity in order if identity not in state["prose"]],
                "pending_links": [key for key, row in state["links"].items() if not self.link_current(row, state["prose"])],
                "issues": state["issues"], "review_count": len(state["reviews"]),
                "calls": self.call_count(scope)}

    def record_narrative_review(self, version_id, epoch, review):
        self._writable()
        review = NarrativeReview.model_validate(review.model_dump(mode="json"))
        state, status = self.narrative(version_id), self.narrative_status(version_id)
        if state["epoch"] != epoch or state["pending"] or not status["plan_ready"] or status["missing"] or status["pending_links"]:
            raise ValueError("内容核对必须针对完整且关联已落实的当前正文")
        if state["stage"] in ("complete", "incomplete") or len(state["reviews"]) > state["config"]["max_review_rounds"]:
            raise ValueError("内容核对已结束或超过配置上限")
        if any(set(issue.fragment_ids) - set(state["prose"]) for issue in review.issues):
            raise ValueError("内容核对的问题引用了未知片段")
        self.check_narrative_source(version_id)
        state["reviews"].append({"epoch": epoch, "result": review.model_dump(mode="json")})
        state.update(issues=[issue.model_dump(mode="json") for issue in review.issues], error=None,
                     stage="complete" if review.approved else "writing")
        with self.connection:
            self._put("narratives", version_id, state)

    def _visible_actions(self, through_version=None):
        version = self.current()[0] if through_version is None else through_version
        return {row[0] for row in self.connection.execute(
            "SELECT action_id FROM actions WHERE status='committed' AND version<=?", (version,))}

    def memories(self, character_id, *, through_version=None):
        self.memory.path(character_id)  # 人物边界必须先验证。
        if self.manifest["format"] < 8:
            return self.memory.read(character_id, committed_actions=self._visible_actions(through_version))
        version = self.current()[0] if through_version is None else through_version
        rows = self.connection.execute("""SELECT actions.version AS visible_version, 0 AS phase,
                   CAST(memory.key AS INTEGER) AS item, memory.value AS payload
            FROM actions, json_each(actions.payload, '$.memories') AS memory
            WHERE actions.status='committed' AND actions.version<=?
              AND json_extract(memory.value, '$.character_id')=?
            UNION ALL
            SELECT json_extract(actions.payload, '$.advance.version_after'), 1,
                   CAST(memory.key AS INTEGER), memory.value
            FROM actions, json_each(actions.payload, '$.advance.memories') AS memory
            WHERE actions.status='committed' AND json_extract(actions.payload, '$.advance.version_after')<=?
              AND json_extract(memory.value, '$.character_id')=?
            ORDER BY visible_version, phase, item""", (version, character_id, version, character_id))
        return [MemoryRecord.model_validate_json(row["payload"]) for row in rows]

    def validate_memories(self, character_id, records):
        existing = {m.memory_id for m in self.memories(character_id)}
        for record in records:
            if (record.run_id, record.character_id) != (self.run_id, character_id):
                raise ValueError("记忆的运行或人物绑定不匹配")
            if record.memory_id in existing or set(record.related_memory_ids + record.corrects) - existing:
                raise ValueError("记忆编号重复，或关联不是本人的已保存记忆")
            existing.add(record.memory_id)

    def search_personal_memories(self, character_id, query):
        """人物自己提出的问题只命中当前运行中本人已提交的主观记忆正文。"""
        limit = min(self.config.memory_candidates, self.config.recent_memories + self.config.related_memories)
        if not query or not query.strip() or not limit:
            return []
        records = self.memories(character_id)
        if any((r.run_id, r.character_id) != (self.run_id, character_id) for r in records):
            raise MemoryCorruption("回想记录须来自本次运行中的本人")
        if len({r.memory_id for r in records}) != len(records):
            raise MemoryCorruption("本人主观历史包含重复的记忆编号")
        return self.memory.search(records, query, limit)

    def retrieve_memories(self, character_id, *, query="", related_ids=(), through_version=None, relevant=None):
        return self.memory.select(self.memories(character_id, through_version=through_version),
                                  query=query, related_ids=related_ids, recent=self.config.recent_memories,
                                  relevant=self.config.related_memories if relevant is None else relevant)

    def status(self):
        row = self.connection.execute("SELECT version,progress FROM runtime WHERE singleton=1").fetchone()
        version, progress = row["version"], json.loads(row["progress"])
        counts, diagnostics = {}, []
        visible = self._visible_actions(version)
        for character in self.manifest["catalog"]["role"]:
            try:
                counts[character] = len(self.memories(character, through_version=version))
            except (MemoryCorruption, OSError) as error:
                counts[character] = None
                diagnostics.append(str(error))
        pending = [row[0] for row in self.connection.execute("SELECT action_id FROM actions WHERE status='prepared'")]
        events = ({row["event_id"]: dict(row) for row in self.connection.execute(
            "SELECT event_id,node_id,group_order,status,last_action_id,last_round,last_turn FROM event_progress")}
                  if self.manifest["format"] >= 2 else None)
        return {"run_id": self.run_id, "repository": str(self.config.repository), "version": version,
                "progress": progress, "committed_actions": len(visible), "pending_actions": pending,
                "memory_counts": counts, "diagnostics": diagnostics, "retrieval_mode": "text_and_entities",
                "events": events, **({"calls": self.connection.execute("SELECT COUNT(*) FROM calls").fetchone()[0],
                                      "unknown_calls": self.connection.execute("SELECT COUNT(*) FROM calls WHERE ended IS NULL").fetchone()[0],
                                      "drafts": self.connection.execute("SELECT COUNT(*) FROM drafts").fetchone()[0],
                                      "usage": self.usage()}
                                     if self.manifest["format"] >= 3 else {}),
                **({"narratives": {row[0]: self.narrative_status(row[0]) for row in self.connection.execute("SELECT key FROM narratives ORDER BY key")}}
                   if self.manifest["format"] >= 17 else {})}


class NarrativeScope:
    """同一账本按叙述版本隔离任务及预算；不复制模型请求实现。"""
    def __init__(self, storage, version_id, *, selection_config=None):
        self.storage = storage
        self.prior_usage = {}
        if selection_config is not None:
            self.scope = "selection/" + quote(version_id, safe="")
            self.config = selection_config
            return
        self.scope = self.scope_name(version_id)
        state = storage.narrative(version_id)
        self.config = NovelRestoreNarrativeConfig.model_validate(state["config"])
        self.prior_usage = state.get("selection", {}).get("usage", {})

    @staticmethod
    def scope_name(version_id):
        return "narrative/" + quote(version_id, safe="")

    def get_job(self, key):
        return self.storage.get_job(self.scope + "/" + key)

    def save_job(self, key, payload):
        self.storage.save_job(self.scope + "/" + key, payload)

    def reserve(self, key, occurrence, body, limit, *, can_send):
        remaining = min(limit, self.config.max_calls) - self.prior_usage.get("calls", 0)
        return self.storage.reserve(key, occurrence, body, remaining, can_send=can_send,
                                    scope=self.scope, budget_config=self.config, prior_usage=self.prior_usage)

    def received(self, call_id, status, raw, usage):
        self.storage.received(call_id, status, raw, usage)

    def failed(self, call_id, error):
        self.storage.failed(call_id, error)
