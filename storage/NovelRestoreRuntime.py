"""同组批次的原子交接、人物工作状态及隔离重演；复用运行库和文件发布器。"""
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path

from contract.NovelWorldState import merge_deltas
from common.Runtime import RunStopped
from common.NovelRestoreSchema import CharacterState, ConfirmedAction, EventOutcome, StatePatch, WorldPerception
from common.NovelRestoreWorldState import apply_patch, validate_references, validate_result_evidence
from common.NovelRestoreObjects import effective_catalog, validate_registrations

CALL_ROLE = ContextVar("novel_restore_call_role", default=None)


class NovelRestoreRuntime:
    def character_state(self, identity):
        if identity not in self.manifest["catalog"]["role"]:
            raise ValueError("未知人物")
        return CharacterState.model_validate(self.get_checkpoint("character/" + identity, {}))

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

    def subjective_worlds(self):
        return {p: self.subjective_world(p) for p in self.manifest["catalog"]["role"]}

    def validate_views(self, views, catalog=None):
        catalog = catalog if catalog is not None else effective_catalog(self)
        if views.keys() - set(catalog["role"]):
            raise ValueError("个人体验引用未知人物")
        for view in views.values():
            view = WorldPerception.model_validate(view.model_dump() if isinstance(view, WorldPerception) else view)
            for mapping in view.correspondences:
                if mapping.actual_kind not in ("none", "unknown") and mapping.actual_id not in catalog[mapping.actual_kind]:
                    raise ValueError("主客观对应关系引用未知真实对象")

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
        from exporters.NovelRestoreExporter import NovelRestoreExporter
        documents = {self.memory.path(p).relative_to(self.directory).as_posix():
                     json.dumps(self.subjective_world(p), ensure_ascii=False, indent=2) + "\n"
                     for p in self.manifest["catalog"]["role"]}
        if documents:
            NovelRestoreExporter(self).publish(documents, "export/characters")

    def validate_wave(self, actions, working_states):
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
        for action in actions:
            if action.completion is None:
                raise ValueError("批次行动必须包含事件完成判断")
            self._validate_action(action, catalog=catalog, validate_state=False)
            if action.proposal.mode in ("think", "idle") and apply_patch(self.current()[1], action.state_patch) != self.current()[1]:
                raise ValueError("思考和无行动本身不能直接修改客观世界状态")
        combined = StatePatch.model_validate(merge_deltas([a.state_patch for a in actions]).model_dump())
        advances = [a.advance for a in actions if a.advance is not None]
        if len(advances) != 1 or actions[0].advance is None:
            raise ValueError("每批首条行动必须且只能保存一次世界时间推进")
        advance = advances[0]
        validate_references(apply_patch(apply_patch(self.current()[1], combined), advance.state_patch), catalog)
        from agents.NovelRestoreAgent import task_id
        progress = self.current()[2]
        group = next(g for n in self.manifest["nodes"] if n["node_id"] == progress["node_id"]
                     for g in n["groups"] if g["order"] == progress["group_order"])
        actual = apply_patch(apply_patch(self.current()[1], combined), advance.state_patch).model_dump(mode="json")
        event_actions = {event_id: {a.action_id for a in self.event_actions(event_id)} for event_id in group["events"]}
        for action in actions:
            event_actions[action.position.event_id].add(action.action_id)
        evidence_actions = set().union(*event_actions.values())
        for action in actions:
            validate_result_evidence(action.completion, actual, evidence_actions,
                                     event_action_ids=event_actions[action.position.event_id])
        expected_shared = task_id("shared", group["group_id"], clock["round"], clock["batch"])
        if (advance.shared_id != expected_shared or advance.version_after != self.current()[0] + len(actions)
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
            raise ValueError("每批须推进全部人物的需求，包含本批未行动的人物")
        for state in working_states.values():
            CharacterState.model_validate(state)
        for action in actions:
            expected = {p: CharacterState.model_validate(working_states[p]) for p in action.observations}
            if action.working_states != expected:
                raise ValueError("行动保存的当时心理必须与本批人物更新一致")
        if advance.working_states != {p: CharacterState.model_validate(working_states[p]) for p in advance.observations}:
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

    def commit_wave(self, actions, working_states, goal, made_progress):
        self._writable()
        actions = [ConfirmedAction.model_validate(a.model_dump(mode="json")) for a in actions]
        patch = self.validate_wave(actions, working_states)
        actual = apply_patch(apply_patch(self.current()[1], patch), actions[0].advance.state_patch).model_dump(mode="json")
        group = next(g for n in self.manifest["nodes"] if n["node_id"] == actions[0].position.node_id
                     for g in n["groups"] if g["order"] == actions[0].position.group_order)
        evidence_actions = {a.action_id for a in actions} | {
            a.action_id for event_id in group["events"] for a in self.event_actions(event_id)}
        validate_result_evidence(goal, actual, evidence_actions)
        from common.NovelRestoreReview import require_review, wave_delivery
        require_review(self, "wave", wave_delivery(actions, working_states, goal, made_progress))
        version = self.current()[0]
        payload = {"actions": [a.model_dump(mode="json") for a in actions], "patch": patch.model_dump(mode="json"),
                   "working_states": working_states, "goal": goal.model_dump(mode="json"),
                   "made_progress": made_progress, "version": version}
        with self.connection:
            self._put("checkpoints", "pending_wave", payload)
            for index, action in enumerate(actions, 1):
                self.connection.execute("INSERT INTO actions VALUES (?,?,'prepared',?)",
                                        (action.action_id, action.model_dump_json(), version + index))
        self.recover()

    def recover_wave(self):
        """客观状态、个人体验、记忆和心理一起提交，再刷新可重建的文件。"""
        from common.NovelRestoreVisibility import visibility_receipts
        payload = self.get_checkpoint("pending_wave")
        if not payload:
            return False
        actions = [ConfirmedAction.model_validate(a) for a in payload["actions"]]
        version, state, progress = self.current()
        if version != payload["version"]:
            raise ValueError("待交接批次的世界版本发生冲突")
        state = apply_patch(state, StatePatch.model_validate(payload["patch"]))
        advance = actions[0].advance
        state = apply_patch(state, advance.state_patch)
        runtimes = {**self.get_checkpoint("person_runtime", {}),
                    **{p: c.state.model_dump() for p, c in advance.runtime_changes.items()}}
        clock = self.advance_clock(len(actions), payload["made_progress"])
        progress.update(stage="simulating", last_action_id=actions[-1].action_id)
        end_version = version + len(actions)
        views = self.get_checkpoint("views", {})
        core = self.get_checkpoint("core_completed", {})
        with self.connection:
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
                    core[position.event_id] = action.completion.model_dump(mode="json")
                if position.event_id in core:
                    outcome = EventOutcome(run_id=self.run_id, event_id=position.event_id, expected_version=version,
                        status="completed", summary=core[position.event_id]["summary"],
                        action_ids=[a.action_id for a in self.event_actions(position.event_id)])
                    self.connection.execute("UPDATE event_progress SET status='completed',outcome=?,outcome_version=? WHERE event_id=?",
                        (outcome.model_dump_json(), end_version, position.event_id))
            self.connection.execute("UPDATE runtime SET version=?,state=?,progress=? WHERE singleton=1",
                (end_version, state.model_dump_json(), json.dumps(progress, ensure_ascii=False)))
            for person, mental in payload["working_states"].items():
                self._put("checkpoints", "character/" + person, mental)
                observed = person in advance.observations or any(person in a.observations for a in actions)
                if observed:
                    # 仅批准本批已独立核对的新心理，不搬运上一版本的任意可见性收据。
                    for key, receipt in visibility_receipts(self, person, {"working_state": mental},
                                                            actions[0].action_id + "/memory_review").items():
                        self._put("checkpoints", key, receipt)
                    if mental.get("needs"):
                        self._put("checkpoints", "subjective_initialized/" + person, True)
            self._put("checkpoints", "person_runtime", runtimes)
            self._put("checkpoints", "story_seconds", self.get_checkpoint("story_seconds", 0) + advance.elapsed_seconds)
            for action in actions:
                for person, access in action.subjective_inputs.items():
                    self._put("checkpoints", "access/" + person, access.model_dump())
            for person, access in advance.subjective_inputs.items():
                self._put("checkpoints", "access/" + person, access.model_dump())
            views.update({p: v.model_dump(mode="json") for p, v in advance.final_views.items()})
            for key, value in (("views", views), ("core_completed", core), ("group_clock", clock),
                               ("last_goal", payload["goal"]), ("wave", None), ("pending_wave", None)):
                self._put("checkpoints", key, value)
        self.publish_world()
        self.publish_characters()
        return True

    def advance_clock(self, count, made_progress):
        """与行动同一事务保存；一项行为计一轮，恢复不能重复累计。"""
        clock = self.get_checkpoint("group_clock")
        return {"round": clock["round"] + count, "batch": clock["batch"] + 1,
                "stagnant": 0 if made_progress else clock["stagnant"] + count}

    def stop_group(self, reason):
        self._writable()
        progress = self.current()[2]
        with self.connection:
            self.connection.execute("UPDATE event_progress SET status='timed_out' WHERE node_id=? AND group_order=? AND status='running'",
                                    (progress["node_id"], progress["group_order"]))
        self.mark_stage("incomplete", reason)

    def save_group_start(self, group):
        key = "group_start/" + group.group_id
        if self.get_checkpoint(key):
            return
        # 同一个只写者持有锁；副本用于新尝试，当前运行不会回滚。
        tables = ("runtime", "event_progress", "checkpoints")
        data = {table: [dict(row) for row in self.connection.execute("SELECT * FROM " + table)] for table in tables}
        # 不将更早组的完整检查点副本再嵌套保存，避免随组数指数增长。
        data["checkpoints"] = [row for row in data["checkpoints"] if not row["key"].startswith(("group_start/", "export", "execution_binding"))]
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
                    and not row["key"].startswith(("export", "execution_binding", "access/", "visibility/", "semantic_"))]
                progress = {k: v for k, v in progress.items() if k != "error"}
                snapshot["runtime"][0]["progress"] = json.dumps({**progress, "stage": "simulating"})
            config = source.config.model_copy(update={"run_directory": Path(destination).expanduser().resolve(),
                                                       "seed": source.config.seed + (not preserve_progress)})
            prefix_version = snapshot["runtime"][0]["version"]
            snapshot["actions"] = [dict(row) for row in source.connection.execute(
                "SELECT * FROM actions WHERE status='committed' AND version<=? ORDER BY version", (prefix_version,))]
            completed = {row["event_id"] for row in snapshot["event_progress"] if row["status"] == "completed"}
            snapshot["drafts"] = [dict(row) for row in source.connection.execute("SELECT * FROM drafts") if row["event_id"] in completed]
            if preserve_progress:
                # 个人当前感知还原为已提交批次的反馈，未提交轮次重新感知并审核。
                views = {}
                for row in snapshot["actions"]:
                    action = json.loads(row["payload"])
                    views.update((action.get("advance") or {}).get("final_views", {}))
                snapshot["checkpoints"] = [row for row in snapshot["checkpoints"] if row["key"] != "views"]
                snapshot["checkpoints"].append({"key": "views", "payload": json.dumps(views, ensure_ascii=False)})
            child = cls.create(config)
            try:
                exports = [dict(row) for row in child.connection.execute("SELECT * FROM checkpoints WHERE key LIKE 'export/%'")]
                manifest = {**child.manifest, "parent_run_id": source.run_id, "retry_group_id": group_id}
                if preserve_progress:
                    manifest["resume_after_version"] = prefix_version
                with child.connection:
                    for table in ("actions", "event_progress", "checkpoints", "drafts"):
                        child.connection.execute("DELETE FROM " + table)
                        for row in snapshot[table]:
                            if table == "event_progress" and row["outcome"]:
                                outcome = json.loads(row["outcome"])
                                outcome["run_id"] = child.run_id
                                row = {**row, "outcome": json.dumps(outcome, ensure_ascii=False)}
                            if table in ("actions", "drafts"):
                                raw = json.loads(row["payload"])
                                raw["run_id"] = child.run_id
                                memories = raw.get("memories", []) + (raw.get("advance") or {}).get("memories", [])
                                for memory in memories:
                                    memory["run_id"] = child.run_id
                                row = {**row, "payload": json.dumps(raw, ensure_ascii=False)}
                            columns = list(row)
                            child.connection.execute("INSERT INTO " + table + " (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")", tuple(row.values()))
                    child.connection.executemany("INSERT INTO checkpoints(key,payload) VALUES (?,?)",
                        [(r["key"], r["payload"]) for r in exports])
                    runtime = snapshot["runtime"][0]
                    child.connection.execute("UPDATE runtime SET manifest=?,version=?,state=?,progress=?,group_state=? WHERE singleton=1",
                        (json.dumps(manifest, ensure_ascii=False), runtime["version"], runtime["state"], runtime["progress"], runtime["group_state"]))
                    for row in source.connection.execute("SELECT * FROM calls"):
                        columns = list(row.keys())
                        child.connection.execute("INSERT INTO calls (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")", tuple(row))
                child.manifest = manifest
                child.publish_world()
                child.publish_characters()
                return child.status()
            finally:
                child.close()

    def usage(self, scope=None):
        where, values = (" WHERE scope=?", (scope,)) if scope else ("", ())
        rows = self.connection.execute("SELECT * FROM calls" + where, values).fetchall()
        total, unreported, roles = 0, 0, {}
        for row in rows:
            usage = json.loads(row["usage"] or "{}")
            tokens = usage.get("total_tokens")
            if tokens is None and usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
                tokens = usage["prompt_tokens"] + usage["completion_tokens"]
            if tokens is None and row["ended"] and row["status"] == 200:
                unreported += 1
            tokens = tokens or 0
            total += tokens
            role = row["role"] if "role" in row.keys() else "legacy"
            item = roles.setdefault(role, {"calls": 0, "tokens": 0, "seconds": 0.0})
            item["calls"] += 1
            item["tokens"] += tokens
            if row["ended"]:
                item["seconds"] += max(0, (datetime.fromisoformat(row["ended"]) - datetime.fromisoformat(row["started"])).total_seconds())
        return {"calls": len(rows), "tokens": total, "unreported_usage_calls": unreported, "roles": roles}

    def check_budget(self, config, scope):
        usage = self.usage(scope)
        if config.max_tokens is not None and usage["unreported_usage_calls"]:
            raise RunStopped("存在未返回 Token 用量的请求，无法继续保证累计 Token 预算")
        if config.max_tokens is not None and usage["tokens"] >= config.max_tokens:
            raise RunStopped("累计 Token 预算已用尽")
        start = self.connection.execute("SELECT MIN(started) FROM calls WHERE scope=?", (scope,)).fetchone()[0]
        if config.max_elapsed_seconds is not None and start:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(start)).total_seconds()
            if elapsed >= config.max_elapsed_seconds:
                raise RunStopped("累计运行时间上限已用尽")
