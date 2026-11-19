"""一个写入者；运行库原子保存客观与主观资料，个人文件为可恢复投影。"""
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from uuid import uuid4
from urllib.parse import quote

from common.NovelRestoreConfig import NovelRestoreConfig
from tools.JsonInput import loads, validate
from tools.RuntimeFiles import check_sqlite_files, regular_opener
from inputs.NovelRestoreRepository import open_repository
from common.NovelRestoreSchema import (
    ConfirmedAction, EventDraft, Format17GroupDraft, Format19GroupDraft, GroupDraft,
    WorldState, MemoryRecord, WorldPerception, RoundSchedule, ReplanSchedule, StatePatch,
    SharedStoryMaterial,
)
from common.Runtime import CALL_ROLE, RunStopped, fingerprint, storage_encode
from execution.NovelRestoreReview import require_review
from simulation.NovelRestoreWorldState import apply_patch, sync_snapshot, validate_references
from simulation.NovelRestoreStory import story_text
from simulation.NovelRestoreObjects import effective_catalog
from simulation.NovelRestoreSubjectiveWorld import MemoryCorruption, NovelRestoreSubjectiveWorld
from storage.NovelRestoreRuntime import NovelRestoreRuntime, _publish_fork
from storage.NovelRestoreNarrativeStorage import NovelRestoreNarrativeStorage


class NovelRestoreStorage(NovelRestoreRuntime, NovelRestoreNarrativeStorage):
    FORMAT = 23

    @classmethod
    def create(cls, config: NovelRestoreConfig):
        if config.run_directory.expanduser().is_symlink():
            raise ValueError("不能通过符号链接创建运行目录")
        config = config.model_copy(update={name: getattr(config, name).expanduser().resolve()
                                          for name in ("repository", "run_directory")})
        config.check_paths()
        # 在创建任何运行文件之前完成输入读取与结构检查。
        with open_repository(config.repository) as repository:
            repository.validate_contract()
            catalog = {kind: list(repository.assets(kind)) for kind in repository.ASSET_KINDS}
            # validate_contract 已逐组回放并检查引用、身份和跨节点连续性。
            # 此处只投影运行目录；逐组重新求 before/after 会反复回放同一历史。
            nodes = []
            for node in repository.nodes():
                groups = []
                for group in repository.groups(node["node_id"]):
                    events = {event["task_id"]: [p["role_code"] for p in event["participants"]]
                              for event in group.events}
                    groups.append({"group_id": group.group_id, "order": group.order, "events": events, "event_ids": list(events)})
                nodes.append({"node_id": node["node_id"], "order": node["order"], "groups": groups})
            config.validate_branching()
            if config.stop_after_group and not any(
                    group["group_id"] == config.stop_after_group for node in nodes for group in node["groups"]):
                raise ValueError("停止事件组不存在：" + config.stop_after_group)
            run_id = uuid4().hex
            manifest = {"format": cls.FORMAT, "run_id": run_id, "branch_root_id": run_id, "source": repository.source,
                        "config": config.model_dump(mode="json"), "repository_hash": repository.fingerprint(),
                        "catalog": catalog, "characters": repository.assets("role"),
                        "locations": repository.assets("location"), "nodes": nodes}
            state = sync_snapshot(None, repository.snapshot(nodes[0]["node_id"]))
        directory = config.run_directory
        if directory.exists():
            raise FileExistsError("运行目录已存在，请保留已有目录：" + str(directory))
        directory.parent.mkdir(parents=True, exist_ok=True)
        # 建库和初始投影成功前不暴露目标目录，初始化故障可用同名目录重试。
        with TemporaryDirectory(prefix=".run-init-", dir=directory.parent) as temporary:
            staged = Path(temporary) / "run"
            staged_config = config.model_copy(update={"run_directory": staged})
            staged_manifest = {**manifest, "config": staged_config.model_dump(mode="json")}
            with cls._create_staged(staged_config, staged_manifest, state) as storage:
                with storage.connection:
                    storage.connection.execute("UPDATE runtime SET manifest=? WHERE singleton=1", (storage_encode(manifest),))
            _publish_fork(staged, directory)
        return cls(directory)

    @classmethod
    def _create_staged(cls, config, manifest, state):
        """仅供 create 在自己的隔离目录内准备初始化文件。"""
        catalog, nodes = manifest["catalog"], manifest["nodes"]
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
                                   (storage_encode(manifest), state.model_dump_json(), storage_encode(progress)))
                connection.executemany("INSERT INTO checkpoints VALUES (?, ?)", [
                    ("self_profiles", storage_encode({})),
                    ("person_runtime", storage_encode({p: {"can_act": True, "body": "", "effects": []}
                                               for p in catalog["role"]})),
                    ("story_seconds", storage_encode(0)),
                ])
        finally:
            connection.close()
        # 临时库尚不接收任务；正式发布后重新打开时统一核对 repository 指纹。
        storage = cls(directory, narration_only=True)
        try:
            storage.publish_world()
            storage.publish_characters()
        except BaseException:
            storage.close()
            raise
        return storage

    def __init__(self, directory, *, readonly=False, narration_only=False):
        self.directory = Path(directory).expanduser().resolve()
        self.readonly = readonly
        self.connection = None
        self._inflight_calls = set()
        self._json_repository = None
        self._lock = None
        try:
            database = self.directory / "run.sqlite3"
            check_sqlite_files(database)
            if not database.is_file():
                raise ValueError(f"运行数据库不存在：{database}")
            if not readonly:
                self._lock = open(self.directory / ".writer.lock", "a+b",
                                  opener=regular_opener)
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
            location = f"{database}:runtime[singleton=1].manifest"
            if row is None:
                raise ValueError(f"{location}: 运行初始化尚未完成，请保留现场并使用新的运行目录")
            self.manifest = loads(row[0], location=location)
            # 只核对所有受支持格式共有的打开条件，不把当前格式的新字段强加给历史运行。
            for key, kind in {"format": int, "config": dict, "run_id": str, "catalog": dict,
                              "characters": dict, "nodes": list}.items():
                if key not in self.manifest or type(self.manifest[key]) is not kind:
                    raise ValueError(f"{location}/{key}: 运行记录缺少字段或字段类型错误，须为 {kind.__name__}")
            if not self.manifest["run_id"].strip():
                raise ValueError(f"{location}/run_id: 运行身份不能为空")
            people = self.manifest["catalog"].get("role")
            if not isinstance(people, list):
                raise ValueError(f"{location}/catalog/role: 人物目录须为身份列表")
            for index, person in enumerate(people):
                if not isinstance(person, str) or not person.strip():
                    raise ValueError(f"{location}/catalog/role/{index}: 人物身份须为非空文本")
            if self.manifest["format"] != self.FORMAT and not (readonly and self.manifest["format"] in range(1, self.FORMAT)):
                raise ValueError(f"{location}/format: 旧运行格式仅支持只读查看；请新建运行，不原地升级")
            saved_config = self.manifest["config"]
            if self.manifest["format"] < 21:
                # Historical manifests remain untouched; obsolete controls
                # are removed only from the read-only configuration view.
                saved_config = {k: v for k, v in saved_config.items()
                                if k not in {"action_ratios", "character_action_ratios", "stagnation_rounds"}}
            if self.manifest["format"] < 17:
                saved_config = {"max_timelines": 1, **saved_config}
            self.config = validate(NovelRestoreConfig, saved_config, location=location + "/config")
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
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._json_repository = None
        connection, self.connection = self.connection, None
        lock, self._lock = self._lock, None
        try:
            if connection is not None:
                connection.close()
        finally:
            if lock is not None:
                lock.close()

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
                                    (version + 1, state.model_dump_json(), storage_encode(progress)))
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
                                    (version + 1, storage_encode(progress), state.model_dump_json(), reference.model_dump_json()))
            self._put("checkpoints", "group_target", target.model_dump(mode="json"))
            from simulation.NovelRestoreGoals import initial_goal_records
            self._put("checkpoints", "group_clock", {"round": 1, "batch": 1})
            self._put("checkpoints", "goal_records", initial_goal_records(group))
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
        from simulation.NovelRestoreGoals import goal_replan_required
        replan = goal_replan_required(self.get_checkpoint("goal_records", {}), self.config.goal_stagnation_batches)
        if replan:
            result = ReplanSchedule.model_validate(payload)
        else:
            result = RoundSchedule.model_validate(payload)
        states = self.event_states()
        active = {e["task_id"] for e in group.events if states[e["task_id"]]["status"] == "running"}
        if any(set(getattr(result, field)) != active for field in ("actors", "opportunities", "scene_decisions")):
            raise ValueError("调度、机会及现场安排必须恰好覆盖本组活跃事件")
        require_review(self, "schedule", payload)
        people = {p["role_code"] for event in group.events if states[event["task_id"]]["status"] == "running"
                  for p in event["participants"]}
        changes, entries = [], []
        for person, choice in result.initial_locations.items():
            if person not in people or person not in state.characters:
                raise ValueError("只能为本组活跃事件的参与人物补充初始地点")
            previous = state.characters[person].position
            if (previous.location_id is not None or state.characters[person].life_status == "not_born"
                    or choice.location_id not in self.manifest["catalog"]["location"]):
                raise ValueError("初始地点只允许补充已登记地点与未知位置")
            changes.append({"role_id": person, "set": {"position": {"location_id": choice.location_id, "position_within_place": choice.detail}},
                "basis": choice.basis})
            entries.append({"kind": "simulation_initial_location", "person": person, "node_id": group.node_id,
                "group_id": group.group_id, "round": clock["round"], "location_id": choice.location_id,
                "basis": choice.basis, "detail": choice.detail, "previous": previous.model_dump(mode="json")})
        if changes:
            state = apply_patch(state, StatePatch.model_validate({"characters": changes}))
            validate_references(state, effective_catalog(self), locations=self.manifest["locations"])
        schedule = {**result.model_dump(mode="json"), "group_id": group.group_id, "round": clock["round"], "replan": replan}
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
        if assessment.achieved and any(states[e["task_id"]]["status"] != "completed" for e in group.events):
            raise ValueError("全组事件完成后才能确认组后目标")
        from simulation.NovelRestoreWorldState import validate_result_evidence
        actions = {a.action_id for event in group.events for a in self.event_actions(event["task_id"])}
        validate_result_evidence(assessment, state.model_dump(mode="json"), actions)
        from execution.NovelRestoreReview import group_delivery
        require_review(self, "group", group_delivery(self, group.group_id, assessment))
        records = self.get_checkpoint("goal_records", {})
        from simulation.NovelRestoreGoals import goals_satisfied
        if assessment.achieved and not goals_satisfied(records):
            raise ValueError("逐项目标尚未全部满足，不能以整体结论绕过未完成项")
        with self.connection:
            self._put("checkpoints", "group_assessment/" + group.group_id,
                      {"version": version, **assessment.model_dump(mode="json")})
            self._put("checkpoints", "last_goal", assessment.model_dump(mode="json"))
            if assessment.achieved:
                # 进入下一组会替换当前账本；通过时保留本组最终记录，
                # 与验收同一事务保存，不进入人物或叙述的故事来源。
                self._put("checkpoints", "goal_records/" + group.group_id, records)
            if not assessment.achieved:
                from simulation.NovelRestoreGoals import invalidate_group_acceptance
                self._put("checkpoints", "goal_records", invalidate_group_acceptance(records, version=version,
                    assessment_key="group_assessment/" + group.group_id,
                    continuity_ok=assessment.continuity_ok and not assessment.contradictions))

    def continue_group(self, group):
        """为剩余状态条件重新开放行动位置；core_completed 与已提交历史不变。"""
        self._writable()
        from simulation.NovelRestoreGoals import goals_satisfied
        progress = self.current()[2]
        if (group.node_id, group.order) != (progress["node_id"], progress["group_order"]):
            raise ValueError("只能继续当前事件组")
        if goals_satisfied(self.get_checkpoint("goal_records", {})):
            raise ValueError("逐项条件齐备后应先正式验收，不能继续安排多余行动")
        with self.connection:
            self.connection.execute("UPDATE event_progress SET status='running',outcome=NULL,outcome_version=NULL "
                "WHERE node_id=? AND group_order=? AND status='completed'", (group.node_id, group.order))

    def publish_world(self):
        from storage.NovelRestoreExporter import NovelRestoreExporter
        NovelRestoreExporter(self).publish({"world_state.json": json.dumps(
            self.current()[1].model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"}, "export/world")

    def _validate_action(self, action, *, catalog=None, validate_state=True):
        """校验单项来源与内容；记忆唯一性由单行动或整批提交统一检查。"""
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
            validate_references(apply_patch(state, action.state_patch), catalog, locations=self.manifest["locations"])

    def recover(self):
        self._writable(allow_pending=True)
        if self.get_checkpoint("pending_wave"):
            return self.recover_wave()
        if self.connection.execute("SELECT 1 FROM actions WHERE status='prepared'").fetchone():
            raise ValueError("待交接行动缺少完整批次记录，不能退回单行动提交；保留现场")
        self.publish_world()
        self.publish_characters()
        return False

    def _put(self, table, key, payload):
        # table 仅由下列固定调用方提供，不接收外部输入。
        self.connection.execute(f"INSERT INTO {table} VALUES (?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
                                (key, storage_encode(payload)))

    def get_checkpoint(self, key, default=None):
        row = self.connection.execute("SELECT payload FROM checkpoints WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_checkpoint(self, key, payload):
        self.set_checkpoints({key: payload})

    def set_checkpoints(self, checkpoints):
        """相关检查点一次交接；只读或尚未恢复的运行不接受新的准备状态。"""
        self._writable()
        with self.connection:
            for key, payload in checkpoints.items():
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
            self.connection.execute("UPDATE runtime SET progress=? WHERE singleton=1", (storage_encode(progress),))

    def event_states(self):
        return {row["event_id"]: dict(row) for row in self.connection.execute("SELECT * FROM event_progress")}

    def event_action_ids(self, event_id):
        """进度引用校验只读已提交身份，不解析正文、记忆和审核附件。"""
        return {row[0] for row in self.connection.execute(
            "SELECT action_id FROM actions WHERE status='committed' AND json_extract(payload,'$.position.event_id')=?",
            (event_id,))}

    def event_actions(self, event_id=None, *, action_ids=None):
        where, values = ("", ()) if event_id is None else (" AND json_extract(payload,'$.position.event_id')=?", (event_id,))
        if action_ids is not None:
            if not action_ids:
                return []
            where += " AND action_id IN (SELECT value FROM json_each(?))"
            values += (json.dumps(sorted(action_ids)),)
        return [self._read_action(row[0]) for row in self.connection.execute(
            "SELECT payload FROM actions WHERE status='committed'" + where + " ORDER BY version", values)]

    def get_action(self, action_id):
        row = self.connection.execute("SELECT payload FROM actions WHERE action_id=? AND status='committed'", (action_id,)).fetchone()
        return self._read_action(row[0]) if row else None

    def _read_action(self, payload):
        if self.manifest["format"] < 21:
            from storage.NovelRestoreLegacyRead import action_record
            return ConfirmedAction.model_validate(action_record(json.loads(payload)))
        return ConfirmedAction.model_validate_json(payload)

    def get_draft(self, identity):
        """当前格式按组读取；旧运行的只读 API 保留原来的事件键。"""
        if self.manifest["format"] < 17:
            row = self.connection.execute("SELECT payload FROM drafts WHERE event_id=?", (identity,)).fetchone()
            from storage.NovelRestoreLegacyRead import event_draft
            return EventDraft.model_validate(event_draft(json.loads(row[0]))) if row else None
        row = self.connection.execute("SELECT payload,digest FROM drafts WHERE group_id=?", (identity,)).fetchone()
        if row is None:
            return None
        self.check_draft_digest(row)
        if self.manifest["format"] == 17:
            schema = Format17GroupDraft
        elif self.manifest["format"] < 20:
            schema = Format19GroupDraft
        else:
            schema = GroupDraft
        return schema.model_validate_json(row["payload"])

    def group_actions(self, node_id, order):
        return [self._read_action(row[0]) for row in self.connection.execute(
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
                prior_usage=None, billable=True):
        self._writable()
        # 重演继承费用账本，但请求缓存须隔离；运行身份无需进入人物提示。
        key = fingerprint([self.run_id, key])
        row = self.connection.execute("SELECT * FROM calls WHERE scope=? AND request_key=? AND occurrence=? ORDER BY id DESC LIMIT 1",
                                      (scope, key, occurrence)).fetchone()
        if row is not None:
            if row["ended"] is None:
                if row["id"] not in getattr(self, "_retry_unknown_calls", set()):
                    raise RunStopped("同一请求已有发送记录但未收到回执，无法确认是否送达；"
                                     "核对预算后可显式 resume --retry-unknown，保留账目与已完成任务")
            elif row["id"] not in getattr(self, "_retry_unsent_calls", set()):
                return dict(row), True
        if not can_send:
            raise RunStopped("未提供模型密钥，不能发出新请求")
        self.check_budget(budget_config or self.config, scope, prior_usage, request=storage_encode(body) if billable else None)
        if limit is not None and self.call_count(scope) >= limit:
            raise RunStopped("累计模型调用预算已用尽")
        with self.connection:
            from common.Runtime import CALL_SCOPE
            job = body.get("job") or (CALL_SCOPE.get() or ("unknown",))[0]
            role = job.split("/", 1)[0]
            role = CALL_ROLE.get() or {"recall": "character", "memory": "character", "summary": "character",
                                       "batch": "narrator", "environment": "world"}.get(role, role)
            cursor = self.connection.execute("INSERT INTO calls(request_key,occurrence,request,started,scope,role) VALUES (?,?,?,?,?,?)",
                                             (key, occurrence, storage_encode(body), datetime.now(timezone.utc).isoformat(), scope, role))
            if row is not None:
                self._put("checkpoints", "request_retry/" + str(cursor.lastrowid),
                          {"previous_call_id": row["id"], "reason": ("explicit_retry_unsent"
                           if row["ended"] is not None else "explicit_retry_unknown")})
                self._retry_unknown_calls.discard(row["id"])
                self._retry_unsent_calls.discard(row["id"])
        self._inflight_calls.add(cursor.lastrowid)
        return {"id": cursor.lastrowid}, False

    def allow_unknown_retries(self):
        """One invocation, retry orphaned or known-unsent calls; preserve the ledger.

        The workflow calls this after checking the execution fingerprint. New
        uncertain deliveries remain protected until another explicit resume.
        """
        self._writable()
        self._retry_unknown_calls = {row[0] for row in self.connection.execute(
            "SELECT id FROM calls WHERE ended IS NULL AND scope='simulation'")} - self._inflight_calls
        # Explicit resume may retry known pre-send failures, but never successful
        # receipts or new failures created during this invocation. Keep the ledger.
        self._retry_unsent_calls = {row[0] for row in self.connection.execute(
            "SELECT id FROM calls WHERE ended IS NOT NULL AND scope='simulation' "
            "AND status IS NULL AND error IN ('ConnectError','ConnectTimeout','PoolTimeout')")}

    def release_call(self, call_id):
        """本写入者不再等待网络；仅真实回执或已知未送达才能结清账目。"""
        self._inflight_calls.discard(call_id)

    def received(self, call_id, status, raw, usage):
        self._writable()
        with self.connection:
            self.connection.execute("UPDATE calls SET ended=?,status=?,response=?,usage=? WHERE id=?",
                                    (datetime.now(timezone.utc).isoformat(), status, raw, storage_encode(usage), call_id))
        self.release_call(call_id)

    def failed(self, call_id, error):
        self._writable()
        with self.connection:
            self.connection.execute("UPDATE calls SET ended=?,error=? WHERE id=?",
                                    (datetime.now(timezone.utc).isoformat(), error, call_id))
        self.release_call(call_id)

    def _visible_actions(self, through_version=None):
        version = self.current()[0] if through_version is None else through_version
        return {row[0] for row in self.connection.execute(
            "SELECT action_id FROM actions WHERE status='committed' AND version<=?", (version,))}

    def memories(self, character_id, *, through_version=None, offset=0, limit=None):
        return self._committed_memories(character_id, through_version, offset=offset, limit=limit)

    def _committed_memories(self, character_id, through_version, *, world_links=None, offset=0, limit=None):
        if type(offset) is not int or offset < 0 or limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("记忆分页偏移和数量须为非负整数")
        self.memory.path(character_id)  # 人物边界必须先验证。
        if self.manifest["format"] < 8:
            records = self.memory.read(character_id, committed_actions=self._visible_actions(through_version))
            return records[offset:None if limit is None else offset + limit]
        version = self.current()[0] if through_version is None else through_version
        # 仅世界按实体筛选时，另读该条记忆所在来源和接收者的客观映射。
        def correspondences(path):
            if world_links is None:
                return ""
            return f""", (SELECT json_extract(view.value, '$.correspondences')
                          FROM json_each(actions.payload, '{path}') AS view
                          WHERE view.key=json_extract(memory.value, '$.character_id')) AS mappings"""
        rows = self.connection.execute(f"""SELECT actions.version AS visible_version, 0 AS phase,
                   CAST(memory.key AS INTEGER) AS item, memory.value AS payload {correspondences('$.observations')}
            FROM actions, json_each(actions.payload, '$.memories') AS memory
            WHERE actions.status='committed' AND actions.version<=?
              AND json_extract(memory.value, '$.character_id')=?
            UNION ALL
            SELECT json_extract(actions.payload, '$.advance.version_after'), 1,
                   CAST(memory.key AS INTEGER), memory.value {correspondences('$.advance.observations')}
            FROM actions, json_each(actions.payload, '$.advance.memories') AS memory
            WHERE actions.status='committed' AND json_extract(actions.payload, '$.advance.version_after')<=?
              AND json_extract(memory.value, '$.character_id')=?
            ORDER BY visible_version, phase, item LIMIT ? OFFSET ?""",
            (version, character_id, version, character_id, -1 if limit is None else limit, offset))
        records = []
        for row in rows:
            memory = MemoryRecord.model_validate_json(row["payload"])
            records.append(memory)
            if world_links is not None and memory.perceived_ids and row["mappings"]:
                seen = set(memory.perceived_ids)
                world_links[memory.memory_id] = {
                    m["actual_id"] for m in json.loads(row["mappings"])
                    if m["perceived_id"] in seen and m["actual_kind"] in ("role", "object", "location")
                    and m.get("actual_id") is not None}
        return records

    def validate_memories(self, character_id, records):
        existing = {m.memory_id for m in self.memories(character_id)}
        for record in records:
            if (record.run_id, record.character_id) != (self.run_id, character_id):
                raise ValueError("记忆的运行或人物绑定不匹配")
            if record.memory_id in existing or set(record.related_memory_ids + record.corrects) - existing:
                raise ValueError("记忆编号重复，或关联不是本人的已保存记忆")
            existing.add(record.memory_id)

    def search_personal_memories(self, character_id, query, *, cues=()):
        """人物自己提出的问题只命中当前运行中本人已提交的主观记忆正文。"""
        limit = min(self.config.memory_candidates, self.config.recent_memories + self.config.related_memories)
        if not query or not query.strip() or not limit:
            return []
        records = self.memories(character_id)
        if any((r.run_id, r.character_id) != (self.run_id, character_id) for r in records):
            raise MemoryCorruption("回想记录须来自本次运行中的本人")
        if len({r.memory_id for r in records}) != len(records):
            raise MemoryCorruption("本人主观历史包含重复的记忆编号")
        return self.memory.search(records, query, limit, cues=cues)

    def retrieve_memories(self, character_id, *, query="", related_ids=(), through_version=None, relevant=None):
        world_links = {} if related_ids else None
        records = self._committed_memories(character_id, through_version, world_links=world_links)
        return self.memory.select(records, world_links=world_links,
                                  query=query, related_ids=related_ids, recent=self.config.recent_memories,
                                  relevant=self.config.related_memories if relevant is None else relevant)

    def status(self):
        row = self.connection.execute("SELECT version,progress FROM runtime WHERE singleton=1").fetchone()
        version, progress = row["version"], json.loads(row["progress"])
        counts, diagnostics = {}, []
        committed = self.connection.execute(
            "SELECT COUNT(*) FROM actions WHERE status='committed' AND version<=?", (version,)).fetchone()[0]
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
        deferred_exports = ([{"key": row["key"], **receipt["error"]}
                             for row in self.connection.execute("SELECT key,payload FROM checkpoints WHERE key='export' OR key LIKE 'export/%' ORDER BY key")
                             if (receipt := json.loads(row["payload"])).get("error")]
                            if self.manifest["format"] >= 3 else [])
        return {"run_id": self.run_id, "repository": str(self.config.repository), "version": version,
                "progress": progress, "committed_actions": committed, "pending_actions": pending,
                "memory_counts": counts, "diagnostics": diagnostics, "deferred_exports": deferred_exports,
                "retrieval_mode": "text_and_entities",
                "events": events, **({"calls": self.connection.execute("SELECT COUNT(*) FROM calls").fetchone()[0],
                                      "pending_group_summary": self.get_checkpoint("pending_group_summary"),
                                      "unknown_calls": self.connection.execute("SELECT COUNT(*) FROM calls WHERE ended IS NULL").fetchone()[0],
                                      "drafts": self.connection.execute("SELECT COUNT(*) FROM drafts").fetchone()[0],
                                      "usage": self.usage()}
                                     if self.manifest["format"] >= 3 else {}),
                **({"narratives": {row[0]: self.narrative_status(row[0]) for row in self.connection.execute("SELECT key FROM narratives ORDER BY key")}}
                   if self.manifest["format"] >= 17 else {})}
