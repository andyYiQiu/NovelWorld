"""同组批次的原子交接、人物工作状态及隔离重演；复用运行库和文件发布器。"""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import ctypes
import json
import math
import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from inputs.NovelWorldState import merge_deltas
from common.Runtime import RunStopped, fingerprint, task_id
from common.NovelRestoreSchema import CharacterState, ConfirmedAction, EventOutcome, RecallExperience, StatePatch, WorldPerception
from simulation.NovelRestoreWorldState import apply_patch, validate_references, validate_result_evidence
from simulation.NovelRestoreObjects import effective_catalog, local_perception_ids, validate_registrations
from tools.JsonInput import loads
from tools.TokenBudget import POLICY as TOKEN_RESERVATION_POLICY, missing_usage_reservation, usage_counts_valid


def _publish_fork(staged, destination):
    """同一文件系统原子发布，竞争者或用户创建的空目录同样不能覆盖。"""
    libc = ctypes.CDLL(None, use_errno=True)
    src, dst = os.fsencode(staged), os.fsencode(destination)
    if sys.platform == "darwin":
        code = libc.renamex_np(src, dst, 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        code = libc.renameat2(-100, src, -100, dst, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise OSError("当前平台不支持新分支目录的原子无覆盖发布")
    if code:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


class NovelRestoreRuntime:
    def character_state(self, identity):
        if identity not in self.manifest["catalog"]["role"]:
            raise ValueError("未知人物")
        value = self.get_checkpoint("character/" + identity, {})
        if self.manifest["format"] < 21:
            from storage.NovelRestoreLegacyRead import character_state
            value = character_state(value)
        return CharacterState.model_validate(value)

    def memory_summaries(self, identity):
        return self.get_checkpoint("summaries/" + identity, [])

    def subjective_world(self, identity):
        """与个人文件同源；只含本人资料，不带世界私有对应关系。"""
        view = self.get_checkpoint("views", {}).get(identity)
        profile = (self.manifest["characters"][identity] if self.manifest["format"] < 10
                   else self.get_checkpoint("self_profiles", {}).get(identity, {}))
        return self.memory.project(identity, profile=profile, view=view,
            working_state=self.character_state(identity), history=self.memories(identity),
            summaries=self.memory_summaries(identity))

    def validate_views(self, views, catalog=None):
        catalog = catalog if catalog is not None else effective_catalog(self)
        if views.keys() - set(catalog["role"]):
            raise ValueError("个人体验引用未知人物")
        for view in views.values():
            view = WorldPerception.model_validate(view.model_dump() if isinstance(view, WorldPerception) else view)
            if not local_perception_ids((o.perceived_id for o in view.personal.objects), catalog):
                raise ValueError("感知对象须使用中性局部编号，不能占用真实资产编号")
            for mapping in view.correspondences:
                if mapping.actual_kind not in ("none", "unknown") and mapping.actual_id not in catalog[mapping.actual_kind]:
                    raise ValueError("主客观对应关系引用未知真实对象")
                if (mapping.actual_kind in ("role", "unknown") and
                        not re.fullmatch(r"p[1-9][0-9]{0,5}", mapping.perceived_id)):
                    raise ValueError("人物或身份未定的感知对象只能用 p1、p2 等中性局部编号")

    def validate_resolved_targets(self, proposal, before_view, resolved, catalog=None):
        mappings = {m.perceived_id: m for m in before_view.correspondences} if before_view else {}
        unknown = {target for target in proposal.target_ids
                   if target in mappings and mappings[target].actual_kind == "unknown"}
        ids = [m.perceived_id for m in resolved]
        if len(ids) != len(set(ids)) or set(ids) != unknown:
            raise ValueError("须恰好解析行动所指向、原先客观对应未知的对象")
        catalog = catalog if catalog is not None else effective_catalog(self)
        for mapping in resolved:
            if mapping.actual_kind == "unknown":
                continue
            if mapping.actual_kind != "none" and mapping.actual_id not in catalog[mapping.actual_kind]:
                raise ValueError("行动对象解析引用未知真实对象")

    def publish_characters(self):
        from storage.NovelRestoreExporter import NovelRestoreExporter
        documents = {self.memory.path(p).relative_to(self.directory).as_posix():
                     json.dumps(self.subjective_world(p), ensure_ascii=False, indent=2) + "\n"
                     for p in self.manifest["catalog"]["role"]}
        if documents:
            NovelRestoreExporter(self).publish(documents, "export/characters")

    def advanced_story_seconds(self, elapsed_seconds):
        """交付、提交和恢复共用同一累计值；单次有限仍可能在相加时溢出。"""
        seconds = self.get_checkpoint("story_seconds", 0) + elapsed_seconds
        if not math.isfinite(seconds):
            raise ValueError("advance.elapsed_seconds 与当前故事时间相加后须为有限数，不能溢出")
        return seconds

    def validate_wave(self, actions, working_states):
        version, state, progress = self.current()
        additions = actions[0].advance.new_objects if actions and actions[0].advance else {}
        catalog = validate_registrations(self, additions,
            [a.state_patch for a in actions] + ([actions[0].advance.state_patch] if actions and actions[0].advance else []),
            [a.action_id for a in actions])
        saved = self.get_checkpoint("wave") or []
        if len(actions) != len(saved) or {a.action_id for a in actions} != {r["action_id"] for r in saved}:
            raise ValueError("统一交接必须覆盖本批全部行动且身份唯一")
        clock = self.get_checkpoint("group_clock")
        if (sorted(a.position.round for a in actions) != list(range(clock["round"], clock["round"] + len(actions)))
                or any(a.position.turn != clock["batch"] for a in actions)):
            raise ValueError("每个行为须恰好占用一个连续轮号，并绑定当前共同批次")
        requests = {row["action_id"]: row for row in saved}
        for action in actions:
            row = requests[action.action_id]
            context = row.get("decision_context", row.get("context", {}))
            if (action.proposal.character_id != row.get("character_id")
                    or action.recollections_before != RecallExperience.from_action_context(context)):
                raise ValueError("行动者和行动前回想须与当时实际提交给该人物的上下文一致")
            if action.working_state_before != CharacterState.story_state_from_context(context):
                raise ValueError("行动前心理须来自该行动者实际作决定的上下文")
            if action.completion is None:
                raise ValueError("批次行动必须包含事件完成判断")
            self._validate_action(action, catalog=catalog, validate_state=False)
            if action.proposal.mode in ("think", "idle") and apply_patch(state, action.state_patch) != state:
                raise ValueError("思考和无行动本身不能直接修改客观世界状态")
        combined = StatePatch.model_validate(merge_deltas([a.state_patch for a in actions]).model_dump())
        advances = [a.advance for a in actions if a.advance is not None]
        if len(advances) != 1 or actions[0].advance is None:
            raise ValueError("每批首条行动必须且只能保存一次世界时间推进")
        advance = advances[0]
        self.advanced_story_seconds(advance.elapsed_seconds)
        resulting_state = apply_patch(apply_patch(state, combined), advance.state_patch)
        validate_references(resulting_state, catalog, locations=self.manifest["locations"])
        group = next(g for n in self.manifest["nodes"] if n["node_id"] == progress["node_id"]
                     for g in n["groups"] if g["order"] == progress["group_order"])
        actual = resulting_state.model_dump(mode="json")
        event_actions = {event_id: {a.action_id for a in self.event_actions(event_id)} for event_id in group["events"]}
        for action in actions:
            event_actions[action.position.event_id].add(action.action_id)
        evidence_actions = set().union(*event_actions.values())
        for action in actions:
            validate_result_evidence(action.completion, actual, evidence_actions,
                                     event_action_ids=event_actions[action.position.event_id])
        expected_shared = task_id("shared", group["group_id"], clock["round"], clock["batch"])
        if (advance.shared_id != expected_shared or advance.version_after != version + len(actions)
                or len(advance.related_event_ids) != len(set(advance.related_event_ids))
                or set(advance.related_event_ids) - set(group["events"])):
            raise ValueError("共享变化的身份、批次版本或事件引用不合法")
        self.validate_views(advance.observations, catalog)
        self.validate_views(advance.final_views, catalog)
        observed = set(advance.observations) | {p for action in actions for p in action.observations}
        if set(advance.final_views) != observed:
            raise ValueError("每个受影响人物须有一个批次结束时的当前视图")
        before_views = self.get_checkpoint("views", {})
        expected_before = {p: WorldPerception.model_validate(before_views[p]) for p in advance.observations if p in before_views}
        if advance.views_before != expected_before or set(advance.subjective_inputs) != set(advance.observations):
            raise ValueError("共享体验须绑定真实批前视图和处理后的个人输入")
        if set(advance.working_states) != set(advance.observations):
            raise ValueError("共享体验须保存每个观察者的当时心理")
        if set(working_states) != set(self.manifest["catalog"]["role"]):
            raise ValueError("每批须保留全部人物状态，未获得体验者沿用原值")
        mental_states = {person: CharacterState.model_validate(mental) for person, mental in working_states.items()}
        for person, mental in mental_states.items():
            if person not in observed and mental != self.character_state(person):
                raise ValueError("未获得本批体验的人物必须沿用原心理，不能按时间公式或世界目标改写")
        for action in actions:
            expected = {p: mental_states[p] for p in action.observations}
            if action.working_states != expected:
                raise ValueError("行动保存的当时心理必须与本批人物更新一致")
        if advance.working_states != {p: mental_states[p] for p in advance.observations}:
            raise ValueError("共享体验保存的当时心理必须与本批人物更新一致")
        # 同一观察者可看到多个事件；相关记忆也须在整个批次内相容。
        batches = defaultdict(list)
        for action in actions:
            for memory in action.memories:
                batches[memory.character_id].append(memory)
        shared_position = actions[0].position.model_copy(update={"event_id": None,
            "round": max(a.position.round for a in actions)})
        for memory in advance.memories:
            if (memory.action_id is not None or memory.source_action_id is not None
                    or memory.source_shared_id != advance.shared_id or memory.position != shared_position
                    or memory.character_id not in advance.observations):
                raise ValueError("共享体验记忆须引用本批共享来源，不能归属于某个事件行动")
            access = advance.subjective_inputs[memory.character_id]
            encoded = next((e.experience for e in access.experiences if e.source_id == advance.shared_id
                            and e.source_kind == "shared"), None)
            if encoded is None or set(memory.perceived_ids) - {o.perceived_id for o in encoded.objects}:
                raise ValueError("共享记忆须来自本人已编码的共享体验")
            if set(memory.corrects + memory.related_memory_ids) - {m.memory_id for m in access.memories}:
                raise ValueError("共享记忆只能关联本次实际可回想的旧记忆")
            people = [memory.character_id, *memory.role_ids, *memory.listener_ids]
            if memory.speaker_id:
                people.append(memory.speaker_id)
            if (set(people) - set(catalog["role"]) or set(memory.object_ids) - set(catalog["object"])
                    or set(memory.location_ids) - set(catalog["location"])):
                raise ValueError("共享记忆引用了未知人物、地点或物品")
            batches[memory.character_id].append(memory)
        for person, memories in batches.items():
            self.validate_memories(person, memories)
        return combined

    def commit_wave(self, actions, working_states, goal, assessment):
        self._writable()
        actions = [ConfirmedAction.model_validate(a.model_dump(mode="json")) for a in actions]
        patch = self.validate_wave(actions, working_states)
        actual = apply_patch(apply_patch(self.current()[1], patch), actions[0].advance.state_patch).model_dump(mode="json")
        group = next(g for n in self.manifest["nodes"] if n["node_id"] == actions[0].position.node_id
                     for g in n["groups"] if g["order"] == actions[0].position.group_order)
        evidence_actions = {a.action_id for a in actions} | {
            a.action_id for event_id in group["events"] for a in self.event_actions(event_id)}
        validate_result_evidence(goal, actual, evidence_actions)
        from execution.NovelRestoreReview import require_review, wave_delivery
        require_review(self, "wave", wave_delivery(actions, working_states, goal, assessment))
        from common.NovelRestoreSchema import WaveAssessment
        from simulation.NovelRestoreGoals import update_goal_records, validate_goal_updates
        assessment = WaveAssessment.model_validate(assessment.model_dump(mode="json"))
        if assessment.goal != goal:
            raise ValueError("逐项目标评估与组验收候选不一致")
        version = self.current()[0]
        validate_goal_updates(self.get_checkpoint("goal_records"), assessment,
                              self.current()[1].model_dump(mode="json"), actual, evidence_actions)
        goal_records = update_goal_records(self.get_checkpoint("goal_records"), assessment,
            self.get_checkpoint("round_schedule", {}), version=version + len(actions),
            window=self.config.goal_progress_window)
        payload = {"actions": [a.model_dump(mode="json") for a in actions], "patch": patch.model_dump(mode="json"),
                   "working_states": working_states, "goal": goal.model_dump(mode="json"),
                   "assessment": assessment.model_dump(mode="json"), "goal_records": goal_records, "version": version}
        with self.connection:
            self._put("checkpoints", "pending_wave", payload)
            for index, action in enumerate(actions, 1):
                self.connection.execute("INSERT INTO actions VALUES (?,?,'prepared',?)",
                                        (action.action_id, action.model_dump_json(), version + index))
        self.recover()

    def recover_wave(self):
        """客观状态、个人体验、记忆和心理一起提交，再刷新可重建的文件。"""
        from simulation.NovelRestoreVisibility import visibility_receipts, working_state_approval
        payload = self.get_checkpoint("pending_wave")
        if not payload:
            return False
        actions = [ConfirmedAction.model_validate(a) for a in payload["actions"]]
        version, state, progress = self.current()
        if version != payload["version"]:
            raise ValueError("待交接批次的世界版本发生冲突")
        from common.NovelRestoreSchema import WaveAssessment
        from execution.NovelRestoreReview import require_review, wave_delivery
        from simulation.NovelRestoreGoals import update_goal_records
        assessment = WaveAssessment.model_validate(payload["assessment"])
        require_review(self, "wave", wave_delivery(actions, payload["working_states"], assessment.goal, assessment))
        expected_goals = update_goal_records(self.get_checkpoint("goal_records"), assessment,
            self.get_checkpoint("round_schedule", {}), version=version + len(actions),
            window=self.config.goal_progress_window)
        if payload["goal_records"] != expected_goals or payload["goal"] != assessment.goal.model_dump(mode="json"):
            raise ValueError("待交接目标记录与本批已审反馈不一致")
        # 完整行动和心理已由上述内容回执绑定；这里只重算独立保存的派生补丁，
        # 不再次读取所有人物记忆、重复执行整批业务校验。
        patch = StatePatch.model_validate(merge_deltas([a.state_patch for a in actions]).model_dump())
        if patch.model_dump(mode="json") != payload["patch"]:
            raise ValueError("待交接状态变化与已审行动不一致")
        state = apply_patch(state, patch)
        advance = actions[0].advance
        story_seconds = self.advanced_story_seconds(advance.elapsed_seconds)
        state = apply_patch(state, advance.state_patch)
        runtimes = {**self.get_checkpoint("person_runtime", {}),
                    **{p: c.state.model_dump() for p, c in advance.runtime_changes.items()}}
        clock = self.advance_clock(len(actions))
        progress.update(stage="simulating", error=None, last_action_id=actions[-1].action_id)
        end_version = version + len(actions)
        views = self.get_checkpoint("views", {})
        core = self.get_checkpoint("core_completed", {})
        with self.connection:
            # 两份记录均由本次准备事务写入；恢复不能提交缺失或被替换的行动。
            prepared = [(row["action_id"], fingerprint(loads(row["payload"], location="待交接行动记录")), row["status"], row["version"])
                        for row in self.connection.execute(
                "SELECT action_id,payload,status,version FROM actions WHERE status='prepared' ORDER BY version")]
            expected = [(action.action_id, fingerprint(action.model_dump(mode="json")), "prepared", version + index)
                        for index, action in enumerate(actions, 1)]
            if prepared != expected:
                raise ValueError("待交接行动记录与完整批次不一致，保留现场，不能提交部分或未审核内容")
            objects = self.get_checkpoint("runtime_objects", {})
            objects.update({key: {**item.model_dump(mode="json"), "origin": "simulation",
                                 "registered_at_version": end_version}
                            for key, item in advance.new_objects.items()})
            self._put("checkpoints", "runtime_objects", objects)
            for action in actions:
                position = action.position
                self.connection.execute("UPDATE actions SET status='committed' WHERE action_id=?", (action.action_id,))
                self.connection.execute("UPDATE event_progress SET last_action_id=?,last_round=?,last_turn=? WHERE event_id=?",
                    (action.action_id, position.round, position.turn, position.event_id))
                if action.completion.achieved:
                    core.setdefault(position.event_id, action.completion.model_dump(mode="json"))
                if position.event_id in core:
                    outcome = EventOutcome(run_id=self.run_id, event_id=position.event_id, expected_version=version,
                        status="completed", summary=core[position.event_id]["summary"],
                        action_ids=[a.action_id for a in self.event_actions(position.event_id)])
                    self.connection.execute("UPDATE event_progress SET status='completed',outcome=?,outcome_version=? WHERE event_id=?",
                        (outcome.model_dump_json(), end_version, position.event_id))
            self.connection.execute("UPDATE runtime SET version=?,state=?,progress=? WHERE singleton=1",
                (end_version, state.model_dump_json(), json.dumps(progress, ensure_ascii=False)))
            for person, mental in payload["working_states"].items():
                observed = person in advance.observations or any(person in a.observations for a in actions)
                origin = None
                if not observed:
                    before = self.character_state(person).model_dump()
                    if mental == before:
                        origin = working_state_approval(self, person, before)
                    else:
                        raise ValueError("未获得本批体验的人物必须沿用原心理，不能按时间公式或世界目标改写")
                self._put("checkpoints", "character/" + person, mental)
                if observed:
                    # 仅批准本批已独立核对的新心理，不搬运上一版本的任意可见性收据。
                    for key, receipt in visibility_receipts(self, person, {"working_state": mental},
                                                            actions[0].action_id + "/memory_review").items():
                        self._put("checkpoints", key, receipt)
                    if mental.get("needs"):
                        self._put("checkpoints", "subjective_initialized/" + person, True)
                elif origin is not None:
                    # 未收到新体验者只继承已核准的原值。
                    for key, receipt in visibility_receipts(self, person, {"working_state": mental},
                                                            origin["source"], state_origin=origin).items():
                        self._put("checkpoints", key, receipt)
            self._put("checkpoints", "person_runtime", runtimes)
            self._put("checkpoints", "story_seconds", story_seconds)
            for action in actions:
                for person, access in action.subjective_inputs.items():
                    self._put("checkpoints", "access/" + person, access.model_dump())
            for person, access in advance.subjective_inputs.items():
                self._put("checkpoints", "access/" + person, access.model_dump())
            views.update({p: v.model_dump(mode="json") for p, v in advance.final_views.items()})
            # 只把本批已审的最终个人体验绑定到提交后的版本；不搬运旧感知。
            for person, view in advance.final_views.items():
                for key, receipt in visibility_receipts(self, person,
                        {"perception": view.personal.model_dump(mode="json")},
                        advance.shared_id + "/approved-final-view").items():
                    self._put("checkpoints", key, receipt)
            for key, value in (("views", views), ("core_completed", core), ("group_clock", clock),
                               ("goal_records", payload["goal_records"]),
                               ("last_goal", payload["goal"]), ("wave", None), ("pending_wave", None)):
                self._put("checkpoints", key, value)
            # 接续来自同一已审 assessment，与行动一起提交。中断恢复不重新选人，
            # 也不把先后回应放进共享前态的同一批。
            from simulation.NovelRestoreScene import continuation_schedule
            previous = self.get_checkpoint("round_schedule") or {}
            group = next(g for n in self.manifest["nodes"] if n["node_id"] == actions[0].position.node_id
                         for g in n["groups"] if g["order"] == actions[0].position.group_order)
            active = {e for e in group["events"] if self.event_states()[e]["status"] == "running"}
            following = continuation_schedule(assessment, actions, previous, active,
                payload["goal_records"], clock, end_version, self.config)
            if following is not None:
                self._put("checkpoints", "round_schedule", following)
        self.publish_world()
        self.publish_characters()
        return True

    def advance_clock(self, count):
        """与行动同一事务保存；一项行为计一轮，恢复不能重复累计。"""
        clock = self.get_checkpoint("group_clock")
        return {"round": clock["round"] + count, "batch": clock["batch"] + 1}

    def stop_group(self, reason):
        self._writable()
        progress = self.current()[2]
        progress.update(stage="incomplete", error=reason)
        with self.connection:
            self.connection.execute("UPDATE event_progress SET status='timed_out' WHERE node_id=? AND group_order=? AND status='running'",
                                    (progress["node_id"], progress["group_order"]))
            self.connection.execute("UPDATE runtime SET progress=? WHERE singleton=1",
                                    (json.dumps(progress, ensure_ascii=False),))

    def save_group_start(self, group):
        self._save_group_start(group.group_id)

    def _save_group_start(self, group_id):
        key = "group_start/" + group_id
        if self.get_checkpoint(key):
            return
        # 同一个只写者持有锁；副本用于新尝试，当前运行不会回滚。
        tables = ("runtime", "event_progress", "checkpoints")
        data = {table: [dict(row) for row in self.connection.execute("SELECT * FROM " + table)] for table in tables}
        # 不将更早组的完整检查点副本再嵌套保存，避免随组数指数增长。
        data["checkpoints"] = [row for row in data["checkpoints"] if not row["key"].startswith(("group_start/", "export", "execution_binding", "selection/", "branch_"))]
        # 行动带世界版本，账本和草稿则各自追加。保存其精确上界，之后可从历史节点派生。
        data["last_call_id"] = self.connection.execute("SELECT coalesce(max(id),0) FROM calls").fetchone()[0]
        data["pending_call_ids"] = [row[0] for row in self.connection.execute("SELECT id FROM calls WHERE ended IS NULL")]
        data["last_draft_rowid"] = self.connection.execute("SELECT coalesce(max(rowid),0) FROM drafts").fetchone()[0]
        self.set_checkpoint(key, data)

    @classmethod
    def retry_group(cls, directory, destination, group_id, *, preserve_progress=False):
        """隔离新尝试；可从组入口重演，或保留当前组已提交的行动继续。"""
        with cls(directory) as source:
            source._writable()
            snapshot = source.get_checkpoint("group_start/" + group_id)
            if not snapshot:
                raise ValueError("指定事件组没有保存可重演的组前检查点")
            if preserve_progress:
                _, _, progress = source.current()
                group = next(g for n in source.manifest["nodes"] for g in n["groups"] if g["group_id"] == group_id)
                node = next(n for n in source.manifest["nodes"] if group in n["groups"])
                if ((progress["node_id"], progress["group_order"]) != (node["node_id"], group["order"])
                        or progress["stage"] not in ("failed", "paused", "simulating")):
                    raise ValueError("只可保留当前未结束事件组的已提交进度")
                snapshot = {table: [dict(row) for row in source.connection.execute("SELECT * FROM " + table)]
                            for table in ("runtime", "event_progress", "checkpoints")}
                transient = {"wave", "pending_wave", "round_schedule", "round_context", "round_perception", "round_access"}
                snapshot["checkpoints"] = [row for row in snapshot["checkpoints"]
                    if row["key"] not in transient
                    and not row["key"].startswith(("export", "execution_binding", "access/", "visibility/", "semantic_", "review_", "memory_coverage/", "personal_recall/", "selection/", "branch_"))]
                progress = {k: v for k, v in progress.items() if k != "error"}
                snapshot["runtime"][0]["progress"] = json.dumps({**progress, "stage": "simulating"})
            destination = Path(destination).expanduser()
            if destination.is_symlink():
                raise ValueError("不能通过符号链接写重试目标")
            destination = destination.resolve()
            if destination.exists():
                raise ValueError("重试目标已存在，请保留已有目录：" + str(destination))
            config = source.config.model_copy(update={"run_directory": destination,
                                                       "seed": source.config.seed + (not preserve_progress),
                                                       "branch_plan": None, "max_timelines": 1})
            config.check_paths()
            # 前缀、账目和投影全部准备成功才发布目标，失败后可重试同一目录名。
            destination.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".group-retry-", dir=destination.parent) as temp:
                staged = Path(temp) / "run"
                with cls.create(config.model_copy(update={"run_directory": staged})) as child:
                    manifest = {**child.manifest, "parent_run_id": source.run_id, "retry_group_id": group_id,
                                "config": config.model_dump(mode="json")}
                    if preserve_progress:
                        manifest["resume_after_version"] = snapshot["runtime"][0]["version"]
                    source._copy_prefix(child, snapshot, manifest, preserve_progress=preserve_progress)
                _publish_fork(staged, destination)
            with cls(destination, readonly=True) as child:
                return child.status()

    @classmethod
    def fork_node(cls, directory, destination, node_id, *, seed, max_timelines=1, opportunity=None, max_calls=None, max_tokens=None):
        """从节点首组的已保存入口派生；父库可由编排者持锁，本方法只读同一 SQLite 快照。"""
        with cls(directory, readonly=True) as source:
            config_data = {**source.config.model_dump(), "run_directory": Path(destination), "seed": seed,
                           "branch_plan": None, "max_timelines": max_timelines, "stop_after_group": None}
            if max_calls is not None:
                config_data["max_calls"] = max_calls
            if max_tokens is not None:
                config_data["max_tokens"] = max_tokens
            config = type(source.config).model_validate(config_data)
            for name in ("max_calls", "max_tokens"):
                parent_limit = getattr(source.config, name)
                if parent_limit is not None and getattr(config, name) > parent_limit:
                    raise ValueError("子分支预算不能超出父运行上限")
            return source._fork_from_checkpoint(node_id, config, opportunity=opportunity)

    @classmethod
    def fork_experiment(cls, directory, config, *, node_id):
        """采用独立配置开一棵新实验树，其共同前缀来自来源运行的已保存节点入口。"""
        with cls(directory, readonly=True) as source:
            config = type(source.config).model_validate(config)
            if config.repository.expanduser().resolve() != source.config.repository:
                raise ValueError("新实验必须使用来源运行的同一 repository")
            destination = config.run_directory.expanduser().resolve()
            if destination.is_relative_to(source.directory) or source.directory.is_relative_to(destination):
                raise ValueError("新实验目录与来源运行必须相互独立，不能嵌套")
            return source._fork_from_checkpoint(node_id, config, new_tree=True)

    def _fork_from_checkpoint(self, node_id, config, *, new_tree=False, opportunity=None):
        destination = config.run_directory.expanduser()
        if destination.is_symlink():
            raise ValueError("不能通过符号链接写分支目标")
        destination = destination.resolve()
        config = config.model_copy(update={"run_directory": destination, "repository": self.config.repository})
        if self.manifest["format"] != self.FORMAT or self.config.run_directory.resolve() != self.directory:
            raise ValueError("只能从原目录中当前格式的运行创建分支")
        nodes = {n["node_id"]: n for n in self.manifest["nodes"]}
        if node_id not in nodes:
            raise ValueError("分叉时间节点不存在：" + node_id)
        node = nodes[node_id]
        birth = self.manifest.get("fork_node_id") or self.manifest.get("source_node_id")
        if birth is not None and node["order"] < nodes[birth]["order"]:
            raise ValueError("不能从当前分支出生前的节点派生")
        first = node["groups"][0]["group_id"]
        snapshot = self.get_checkpoint("group_start/" + first)
        if not snapshot or not {"last_call_id", "last_draft_rowid"} <= snapshot.keys():
            raise ValueError("缺少带前缀费用边界的节点首组检查点：" + node_id)
        runtime = snapshot["runtime"][0]
        progress = json.loads(runtime["progress"])
        if (progress["node_id"], progress["group_order"], progress["stage"]) != (node_id, 1, "group_ready"):
            raise ValueError("检查点不是指定节点的首组入口")
        config.validate_branching()
        if new_tree:
            lineage = {"parent_run_id": self.run_id, "source_directory": str(self.directory), "source_node_id": node_id,
                       "source_group_id": first, "source_version": runtime["version"], "source_call_id": snapshot["last_call_id"]}
        else:
            lineage = {"branch_root_id": self.manifest.get("branch_root_id", self.run_id),
                       "parent_run_id": self.run_id, "fork_node_id": node_id, "fork_group_id": first,
                       "fork_version": runtime["version"], "fork_call_id": snapshot["last_call_id"]}

        if opportunity is not None:
            split = self.get_checkpoint("branch_split")
            if (not split or split["node_id"] != node_id or opportunity not in split["alternatives"]
                    or opportunity["name"] != destination.name):
                raise ValueError("探索方向与世界在当前分叉点提交的决策不一致")
            lineage["branch_opportunity"] = opportunity

        def existing_child():
            with type(self)(destination, readonly=True) as existing:
                if (existing.config != config or existing.manifest["repository_hash"] != self.manifest["repository_hash"]
                        or (new_tree and existing.manifest.get("branch_root_id") != existing.run_id)
                        or any(existing.manifest.get(k) != v for k, v in lineage.items())):
                    raise ValueError("目标已存在且不属于同一分叉或配置：" + str(destination))
                return existing.status()

        if destination.exists():
            return existing_child()
        # 在暂存和建父目录前检查目标不能写入输入 repository。
        if config.repository.is_dir() and destination.is_relative_to(config.repository):
            raise ValueError("分支目录不能放在输入 repository 内")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".node-fork-", dir=destination.parent) as temp:
            staged = Path(temp) / "run"
            with type(self).create(config.model_copy(update={"run_directory": staged})) as child:
                if child.manifest["repository_hash"] != self.manifest["repository_hash"]:
                    raise ValueError("输入 repository 已变化，不能派生分支")
                manifest = {**child.manifest, **lineage, "config": config.model_dump(mode="json")}
                self._copy_prefix(child, snapshot, manifest, node_fork=True)
                child._save_group_start(first)
            try:
                _publish_fork(staged, destination)
            except FileExistsError:
                return existing_child()
        return existing_child()

    @staticmethod
    def check_draft_digest(row):
        if sha256(row["payload"].encode()).hexdigest() != row["digest"]:
            raise ValueError("正式组草稿在验收冻结后已变化，拒绝读取或派生")

    def _copy_prefix(self, child, snapshot, manifest, *, preserve_progress=False, node_fork=False):
        pending_at_entry = set(snapshot.get("pending_call_ids", ())) if node_fork else set()
        if node_fork:
            calls = self.connection.execute("SELECT * FROM calls WHERE scope='simulation' AND id<=?", (snapshot["last_call_id"],))
            # 当前人物心理和已提交体验取自快照。父运行的审核签名、按轮感知和访问投影不能给子运行授权。
            snapshot["checkpoints"] = [row for row in snapshot["checkpoints"]
                                       if not row["key"].startswith(("visibility/", "semantic_", "review_", "memory_coverage/", "personal_recall/", "access/", "selection/", "branch_"))
                                       and row["key"] not in {"round_schedule", "round_context", "round_perception", "round_access"}]
            draft_rows = self.connection.execute("SELECT * FROM drafts WHERE rowid<=?", (snapshot["last_draft_rowid"],))
        else:
            # 隔离重演仍承担失败尝试的全部账单。
            calls = self.connection.execute("SELECT * FROM calls WHERE scope NOT LIKE 'selection/%'")
            draft_rows = self.connection.execute("SELECT * FROM drafts")
        prefix_version = snapshot["runtime"][0]["version"]
        snapshot["actions"] = [dict(row) for row in self.connection.execute(
            "SELECT * FROM actions WHERE status='committed' AND version<=? ORDER BY version", (prefix_version,))]
        completed = {row["event_id"] for row in snapshot["event_progress"] if row["status"] == "completed"}
        completed_groups = {g["group_id"] for n in manifest["nodes"] for g in n["groups"] if completed.issuperset(g["events"])}
        snapshot["drafts"] = [dict(row) for row in draft_rows if row["group_id"] in completed_groups]
        if preserve_progress:
            # 个人当前感知还原为已提交批次的反馈，未提交轮次重新感知并审核。
            views = {}
            for row in snapshot["actions"]:
                action = json.loads(row["payload"])
                views.update((action.get("advance") or {}).get("final_views", {}))
            snapshot["checkpoints"] = [row for row in snapshot["checkpoints"] if row["key"] != "views"]
            snapshot["checkpoints"].append({"key": "views", "payload": json.dumps(views, ensure_ascii=False)})
        exports = [dict(row) for row in child.connection.execute("SELECT * FROM checkpoints WHERE key LIKE 'export/%'")]
        with child.connection:
            for table in ("actions", "event_progress", "checkpoints", "drafts"):
                child.connection.execute("DELETE FROM " + table)
                for row in snapshot[table]:
                    if table == "event_progress" and row["outcome"]:
                        outcome = json.loads(row["outcome"])
                        outcome["run_id"] = child.run_id
                        row = {**row, "outcome": json.dumps(outcome, ensure_ascii=False)}
                    if table == "drafts":
                        self.check_draft_digest(row)
                    if table in ("actions", "drafts"):
                        raw = json.loads(row["payload"])
                        raw["run_id"] = child.run_id
                        memories = raw.get("memories", []) + (raw.get("advance") or {}).get("memories", [])
                        for memory in memories:
                            memory["run_id"] = child.run_id
                        row = {**row, "payload": json.dumps(raw, ensure_ascii=False)}
                    if table == "drafts":
                        row["digest"] = sha256(row["payload"].encode()).hexdigest()
                    columns = list(row)
                    child.connection.execute("INSERT INTO " + table + " (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")", tuple(row.values()))
            child.connection.executemany("INSERT INTO checkpoints(key,payload) VALUES (?,?)",
                [(r["key"], r["payload"]) for r in exports])
            runtime = snapshot["runtime"][0]
            child.connection.execute("UPDATE runtime SET manifest=?,version=?,state=?,progress=?,group_state=? WHERE singleton=1",
                (json.dumps(manifest, ensure_ascii=False), runtime["version"], runtime["state"], runtime["progress"], runtime["group_state"]))
            for row in calls:
                values = dict(row)
                if row["id"] in pending_at_entry:
                    # 父运行可能在入口之后才收到该旧请求的回执；只继承当时已知账目。
                    values.update(ended=None, status=None, response=None, error=None, usage=None)
                columns = list(values)
                child.connection.execute("INSERT INTO calls (" + ",".join(columns) + ") VALUES ("
                                         + ",".join("?" for _ in columns) + ")", tuple(values.values()))
        if node_fork:
            inherited = child.usage("simulation")
            manifest.update(inherited_calls=inherited["calls"], inherited_tokens=inherited["tokens"])
            with child.connection:
                child.connection.execute("UPDATE runtime SET manifest=? WHERE singleton=1",
                                         (json.dumps(manifest, ensure_ascii=False),))
        child.manifest = manifest
        child.publish_world()
        child.publish_characters()

    def call_count(self, scope):
        return self.connection.execute("SELECT COUNT(*) FROM calls WHERE scope=?", (scope,)).fetchone()[0]

    def usage(self, scope=None):
        where, values = (" WHERE scope=?", (scope,)) if scope else ("", ())
        role_column = "role"
        if self.manifest["format"] != self.FORMAT and not any(
                r["name"] == "role" for r in self.connection.execute("PRAGMA table_info(calls)")):
            role_column = "'legacy' AS role"
        # 长运行的账本含大量请求与原始响应；预算统计只读取计量列。
        rows = self.connection.execute(f"SELECT id,started,ended,status,usage,{role_column} FROM calls" + where, values)
        count, total, unreported, roles = 0, 0, 0, {}
        reserved, unreserved, reservations, inflight = 0, 0, [], 0
        for row in rows:
            count += 1
            # 历史接口回执可能含非法 JSON/非对象/字符串 Token 等。它们不能使
            # 状态查询崩溃，也不能被当作零费用帮助后续请求绕过 Token 预算。
            invalid = False
            try:
                usage = json.loads(row["usage"] or "{}")
            except (TypeError, ValueError, RecursionError):
                usage, invalid = {}, True
            if not isinstance(usage, dict):
                usage, invalid = {}, True
            fields = ("total_tokens", "prompt_tokens", "completion_tokens")
            invalid = invalid or not usage_counts_valid(usage)
            tokens = None if invalid else usage.get("total_tokens")
            if not invalid and tokens is None and all(usage.get(key) is not None for key in fields[1:]):
                tokens = usage["prompt_tokens"] + usage["completion_tokens"]
            # 本写入者仍可收到并发请求的用量；重开或继承的未结请求已不在其掌管中。
            unresolved = row["ended"] is None and row["id"] not in self._inflight_calls
            pending = row["ended"] is None and row["id"] in self._inflight_calls
            # HTTP 错误也可能附带已计量分项；不能因缺少 total 而抹掉已报告费用。
            if tokens is None and (invalid or unresolved or pending or row["status"] == 200
                                   or any(key in usage for key in fields)):
                if not pending:
                    unreported += 1
                # Only unresolved rows load the large request. Raw receipts and
                # usage remain untouched; reopening/retrying recomputes the same allowance.
                request = self.connection.execute("SELECT request FROM calls WHERE id=?", (row["id"],)).fetchone()[0]
                allowance = missing_usage_reservation(request, usage)
                if allowance is None:
                    unreserved += 1
                elif pending:
                    inflight += allowance
                else:
                    reserved += allowance
                if not pending:
                    reservations.append({"call_id": row["id"], "tokens": allowance})
            tokens = tokens or 0
            total += tokens
            role = row["role"]
            item = roles.setdefault(role, {"calls": 0, "tokens": 0, "seconds": 0.0})
            item["calls"] += 1
            item["tokens"] += tokens
            if row["ended"]:
                item["seconds"] += max(0, (datetime.fromisoformat(row["ended"]) - datetime.fromisoformat(row["started"])).total_seconds())
        return {"calls": count, "tokens": total, "unreported_usage_calls": unreported, "roles": roles,
                "reserved_tokens": reserved, "inflight_tokens": inflight, "budget_tokens": total + reserved + inflight,
                "unreserved_usage_calls": unreserved, "token_reservation_policy": TOKEN_RESERVATION_POLICY,
                "token_reservations": reservations}

    def check_budget(self, config, scope, prior_usage=None, *, request=None):
        prior_usage = prior_usage or {}
        if config.max_tokens is not None:
            usage = self.usage(scope)
            # Old selection summaries without reservation data must not turn
            # unknown usage into zero when passed into a narrative version.
            prior_unreserved = prior_usage.get("unreserved_usage_calls", prior_usage.get("unreported_usage_calls", 0))
            if usage["unreserved_usage_calls"] or prior_unreserved:
                raise RunStopped("存在未返回 Token 用量且无法预留预算的请求，须核对原始账本")
            if usage["budget_tokens"] + prior_usage.get("budget_tokens", prior_usage.get("tokens", 0)) >= config.max_tokens:
                reason = "（含未返回 Token 用量的保守预留）" if (usage["reserved_tokens"] or usage["inflight_tokens"]
                    or prior_usage.get("reserved_tokens") or prior_usage.get("inflight_tokens")) else ""
                raise RunStopped("累计 Token 预算已用尽" + reason)
            if request is not None:
                allowance = missing_usage_reservation(request, {})
                used = usage["budget_tokens"] + prior_usage.get("budget_tokens", prior_usage.get("tokens", 0))
                if allowance is None or used + allowance > config.max_tokens:
                    raise RunStopped("剩余 Token 预算不足以容纳本次请求的保守预留；未发送请求")
        if config.max_elapsed_seconds is not None:
            start = prior_usage.get("started") or self.connection.execute(
                "SELECT MIN(started) FROM calls WHERE scope=?", (scope,)).fetchone()[0]
            if not start:
                return
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(start)).total_seconds()
            if elapsed >= config.max_elapsed_seconds:
                raise RunStopped("累计运行时间上限已用尽")
