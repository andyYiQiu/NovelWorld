"""世界角色负责语义，程序绑定身份、引用与提交位置。"""
import asyncio
from pathlib import Path

from common.NovelRestoreConfig import BranchDecision

from agents.NovelRestoreAgent import task_id
from agents.NovelRestoreCharacterAgent import NovelRestoreCharacterAgent
from common.Runtime import JobFailed, RunStopped
from common.NovelRestoreSchema import (
    ConfirmedAction, MemoryRecord, RecallExperience,
    GroupAssessment, MemoryApproval, WaveResolution,
    StoryPosition, RoundSchedule, ReplanSchedule, StatePatch,
    PerceptionBatch, WorldPerception, WaveAssessment, WorldAdvance,
    UnknownTargetReview,
)
from contract.NovelWorldState import merge_deltas
from common.NovelRestoreWorldState import apply_patch, resolve_pointer, validate_references, validate_result_evidence
from common.NovelRestoreObjects import object_assets, object_inventory, validate_registrations, effective_catalog
from common.NovelRestoreExperience import prepare_experience, advance_needs
from common.NovelRestoreVisibility import visible_task, private_perceptions, CharacterFeedback, record_visible
from common.NovelRestoreContext import (
    REFERENCE_STATE_POLICY, action_history, interaction_progress, pending_observations, compact_next_reference,
)


from common.NovelRestoreReview import reviewed_task, seal_review, wave_delivery, group_delivery


class NovelRestoreWorldAgent:
    def __init__(self, storage, repository, executor):
        self.storage, self.repository, self.executor = storage, repository, executor
        self.prompt = (Path(__file__).resolve().parents[1] / "prompts/NovelRestoreWorldPrompt.md").read_text()
        self.world = {"assets": {kind: repository.assets(kind) for kind in repository.ASSET_KINDS},
                      "worlds": repository.worlds(), "nodes": repository.nodes(),
                      "state_before": {n["node_id"]: repository.snapshot(n["node_id"]).model_dump(mode="json")
                                       for n in repository.nodes()},
                      "groups": [group.model_dump(mode="json") for node in repository.nodes()
                                 for group in repository.groups(node["node_id"])]}

    def continuation(self, group):
        groups = self.world["groups"]
        index = next(i for i, row in enumerate(groups) if row["group_id"] == group.group_id)
        if index + 1 == len(groups):
            return None
        from common.NovelRestoreSchema import EventGroup
        return self.repository.group_states(EventGroup.model_validate(groups[index + 1]))[0].model_dump(mode="json")

    async def initialize(self):
        self.storage.initialize_world()

    async def assess_group(self, group):
        version, state, _ = self.storage.current()
        target = self.storage.get_checkpoint("group_target")
        actions = {e["task_id"]: [a.model_dump(mode="json") for a in self.storage.event_actions(e["task_id"])]
                   for e in group.events}
        current_state = state.model_dump(mode="json")
        action_ids = {a["action_id"] for records in actions.values() for a in records}
        result, review_job = await reviewed_task(self.executor, task_id("world", "assess_group", group.group_id, version),
            self.prompt, {"operation": "assess_group", "group": group.model_dump(mode="json"),
                "reference_state_policy": REFERENCE_STATE_POLICY,
                "reference_before": self.storage.group_state().model_dump(mode="json"),
                "target_state": target,
                "next_reference": compact_next_reference(target, self.continuation(group)),
                "current_state": current_state,
                "events": {e["task_id"]: self.storage.event_states()[e["task_id"]] for e in group.events},
                "actions": action_history(actions)}, GroupAssessment,
            lambda judgment: validate_result_evidence(judgment, current_state, action_ids),
            lambda value: ["/" + key for key in ("achieved", "summary", "achieved_results", "remaining", "continuity_ok", "contradictions")])
        seal_review(self.storage, "group", group_delivery(self.storage, group.group_id, result), review_job)
        self.storage.record_group_assessment(group, version, result)
        return result.achieved

    def runtime_context(self, group):
        """感知与裁定共用真实局面；目标另由规划及验收任务按需添加。"""
        people = {p["role_code"] for e in group.events for p in e["participants"]}
        views = self.storage.get_checkpoint("views", {})
        state = self.storage.current()[1].model_dump(mode="json")
        locations = {i for e in group.events for i in e.get("location_ids", [])}
        for person in people:
            locations.update(r["id"] for r in state["characters"][person]["location"]["location"]["references"]
                             if r["type"] == "location")
        for person, character in state["characters"].items():
            if any(r["type"] == "location" and r["id"] in locations
                   for r in character["location"]["location"]["references"]):
                people.add(person)
        for person in tuple(people):
            for mapping in views.get(person, {}).get("correspondences", []):
                if mapping.get("actual_kind") == "role":
                    people.add(mapping["actual_id"])
        actions = {e["task_id"]: [a.model_dump(mode="json", exclude={"completion"})
                                   for a in self.storage.event_actions(e["task_id"])] for e in group.events}
        subjective = {}
        for person in sorted(people):
            working = self.storage.character_state(person).model_dump(mode="json")
            perception = views.get(person, {}).get("personal", {}).get("content", "")
            query = " ".join([perception, *working["goals"], *working["plan"]])
            memories = self.storage.retrieve_memories(person, query=query, relevant=self.storage.config.memory_candidates)
            subjective[person] = {"character_id": person, "profile": self.storage.manifest["characters"][person],
                "current": {"perception": views.get(person, {}).get("personal"), "working_state": working},
                "history": [m.model_dump(mode="json") for m in memories],
                "summaries": self.storage.memory_summaries(person)}
            # 候选记忆的客观来源必须可直接核对；其他历史可使用世界查询。
            for memory in memories:
                if memory.action_id:
                    action = self.storage.get_action(memory.action_id)
                elif memory.source_shared_id:
                    row = self.storage.connection.execute("""SELECT payload FROM actions WHERE status='committed'
                        AND json_extract(payload,'$.advance.shared_id')=?""", (memory.source_shared_id,)).fetchone()
                    action = ConfirmedAction.model_validate_json(row[0]) if row else None
                else:
                    action = None
                if action:
                    records = actions.setdefault(action.position.event_id, [])
                    if not any(r["action_id"] == action.action_id for r in records):
                        records.append(action.model_dump(mode="json", exclude={"completion"}))
        role_assets = {p: self.world["assets"]["role"][p] for p in sorted(people)}
        object_ids = {i for e in group.events for i in e.get("object_ids", [])}
        for identity, item in state["objects"].items():
            references = (item["owner"]["references"] + item["holder"]["references"]
                          + item["location"]["location"]["references"])
            if any((r["type"] == "role" and r["id"] in people) or
                   (r["type"] == "location" and r["id"] in locations) for r in references):
                object_ids.add(identity)
        for person in people:
            for mapping in views.get(person, {}).get("correspondences", []):
                if mapping.get("actual_kind") == "object":
                    object_ids.add(mapping["actual_id"])
        assets = {**self.world["assets"], "role": role_assets,
                  "object": {i: row for i, row in object_assets(self.storage).items() if i in object_ids}}
        return {"environment": {"assets": assets, "worlds": self.world["worlds"]},
                "current_state": state,
                "subjective_worlds": subjective,
                "views": views,
                "actions": action_history(actions), "person_runtime": self.storage.get_checkpoint("person_runtime", {}),
                "reference_state_policy": REFERENCE_STATE_POLICY,
                "initial_locations": self.storage.get_checkpoint("location_initializations", []),
                "story_seconds": self.storage.get_checkpoint("story_seconds", 0),
                "retrieval_scope": "人物历史为候选，行动含本组及候选记忆来源。旧 subjective_inputs 保留 experiences/processing；其中档案、心理及更早记忆副本，以及历史审核、完成结论和运行元数据按 action_id 查询完整行动。实际发生、状态变化、个人体验和未知对象依据保留；旧批准不能证明当前事实。未展示不等于不存在。"}

    def scene_evidence(self, anchor, runtime):
        """读取环境提议所指的现场事实。它能否支持本次变化，由独立审读结合处境判断。"""
        if anchor and anchor.startswith("action/"):
            action = self.storage.get_action(anchor)
            if action is None:
                raise ValueError(f"scene_anchor 没有对应的已提交行动：{anchor}")
            return {"result": action.result, "shared_result": action.advance.result if action.advance else None}
        if anchor == "/story_seconds":
            return runtime["story_seconds"]
        domains = ("characters", "locations", "objects", "relations", "public_state", "constraints")
        if not anchor or not anchor.startswith(tuple(f"/current_state/{name}/" for name in domains)):
            raise ValueError("环境尝试缺少现场依据：scene_anchor 须引用已提交的 action_id 或具体的 /current_state 路径；"
                             "定时变化可引用 /story_seconds。若眼下没有足够条件，可把行动机会交给有理由行动的人物。")
        value = resolve_pointer(runtime, anchor)
        if value is None or isinstance(value, (list, dict)) or (isinstance(value, str) and value.strip() in ("", "unknown")):
            raise ValueError(f"scene_anchor 须指向一项已知的具体事实：{anchor}")
        return value

    async def consider_branch(self, group):
        key = task_id("branch_decision", group.node_id)
        saved = self.storage.get_checkpoint(key)
        if saved:
            return bool(saved["alternatives"])
        # 决策只展开已成立的现场与最近经过。每个候选须给出可回指此输入的依据。
        _, state, _ = self.storage.current()
        evidence = {f"state/{section}/{identity}": value for section, items in
                    state.model_dump(mode="json").items() if isinstance(items, dict)
                    for identity, value in items.items()}
        rows = self.storage.connection.execute("""SELECT payload FROM actions WHERE status='committed'
                                                  ORDER BY version DESC LIMIT 12""")
        for row in rows:
            action = ConfirmedAction.model_validate_json(row[0])
            evidence[action.action_id] = action.result
        context = {"operation": "consider_branch", "node_id": group.node_id,
                   "capacity": self.storage.config.max_timelines, "evidence": evidence,
                   "story_seconds": self.storage.get_checkpoint("story_seconds", 0)}

        def validate(decision):
            choices = decision.alternatives
            if len(choices) == 1 or len(choices) > context["capacity"]:
                raise ValueError("继续时 alternatives 留空；分叉至少两个走法且不超过剩余容量")
            intents = {choice.opportunity.strip() for choice in choices}
            if len(intents) != len(choices) or "" in intents:
                raise ValueError("每条线须有不同的具体探索机会")
            if any(set(choice.source_ids) - evidence.keys() for choice in choices):
                raise ValueError("分支机会必须引用本节点实际现场的依据")

        decision = await self.executor.run_task(task_id("world", "consider_branch", group.node_id), "world",
            self.prompt, context, BranchDecision, validate)
        saved = {"node_id": group.node_id, **decision.model_dump(mode="json")}
        # 目录标识也被用作择线编号，不允许其中包含尚未发生的故事设想。
        for index, choice in enumerate(saved["alternatives"], 1):
            choice["name"] = str(index)
        with self.storage.connection:
            self.storage._put("checkpoints", key, saved)
            if decision.alternatives:
                self.storage._put("checkpoints", "branch_split", saved)
                # 节点决策在首次行动前完成。其账目也属于子线共同承担的前缀。
                start_key = "group_start/" + group.group_id
                entry = self.storage.get_checkpoint(start_key)
                entry["last_call_id"] = self.storage.connection.execute("SELECT coalesce(max(id),0) FROM calls").fetchone()[0]
                self.storage._put("checkpoints", start_key, entry)
        return bool(decision.alternatives)

    async def schedule(self, group, clock):
        states = self.storage.event_states()
        current_people = self.storage.current()[1].characters
        target = self.storage.get_checkpoint("group_target")
        eligible = {e["task_id"]: [p["role_code"] for p in e["participants"]] for e in group.events
                    if states[e["task_id"]]["status"] == "running"}
        participants = {person for people in eligible.values() for person in people}
        last_goal = self.storage.get_checkpoint("last_goal")
        completed = self.storage.get_checkpoint("core_completed", {})
        runtime = self.runtime_context(group)
        interactions = interaction_progress(runtime["actions"], eligible, group=(group.node_id, group.order))
        replan = clock["stagnant"] >= self.storage.config.stagnation_rounds
        def validate(result):
            if set(result.actors) != set(eligible):
                raise ValueError("actors 必须以当前活跃事件编号为键，值只能是该事件的参与人物编号或 null；"
                                 f"当前事件及可选人物：{eligible}；不要以人物编号为键，也不要填写行动内容")
            people = [p for p in result.actors.values() if p is not None]
            if len(people) != len(set(people)):
                raise ValueError("同一人物不能在同批两个事件中分别行动")
            if set(result.initial_locations) - participants:
                raise ValueError("初始地点只能为本组活跃事件的参与人物选择")
            for person, choice in result.initial_locations.items():
                if current_people[person].location.location.status != "unknown":
                    raise ValueError("仅当前位置未知的人物可以补充初始地点")
                if current_people[person].life_status == "not_born":
                    raise ValueError("尚未出生的人物不能补充当前位置")
                if choice.location_id not in self.storage.manifest["catalog"]["location"]:
                    raise ValueError("补充的初始地点必须属于 repository 地点目录")
            for event, people in eligible.items():
                if event in result.arrangements:
                    self.scene_evidence(result.arrangements[event].scene_anchor, runtime)
                if result.actors[event] is not None and result.actors[event] not in people:
                    raise ValueError("下一行动者须属于该事件；环境推进使用 null 并提交发生依据")
                if result.actors[event] and current_people[result.actors[event]].life_status in ("dead", "not_born"):
                    raise ValueError("客观状态中已死亡或尚未出生的人物不能主动行动")
                if result.actors[event] and not self.storage.get_checkpoint("person_runtime", {}).get(result.actors[event], {}).get("can_act", True):
                    raise ValueError("当前无法自主行动的人物不能被安排主动行为")
        job = task_id("world", "schedule", group.group_id, clock["round"])
        context = {"operation": "schedule", "eligible": eligible, "clock": clock, "replan": replan,
                "group": group.model_dump(mode="json"), "target_state": target,
                "next_reference": compact_next_reference(target, self.continuation(group)),
                "last_goal": ({k: last_goal[k] for k in ("achieved", "continuity_ok")} if last_goal else None),
                "assessment_feedback": ({k: last_goal.get(k, []) for k in ("remaining", "contradictions")} if last_goal else None),
                "core_completed": {event: {"achieved": item["achieved"]} for event, item in completed.items()},
                "interaction_progress": interactions, **runtime}
        if group.node_id == self.storage.manifest.get("fork_node_id") and self.storage.manifest.get("branch_opportunity"):
            context["branch_opportunity"] = self.storage.manifest["branch_opportunity"]
        if not replan:
            return await self.executor.run_task(job, "world", self.prompt, context, RoundSchedule, validate)

        previous = self.storage.get_checkpoint("round_schedule") or {}
        prior_focus = {}
        if previous.get("group_id") == group.group_id and previous.get("scene_decisions"):
            context["previous_scene_decisions"] = previous["scene_decisions"]
            prior_focus = {e: {"actor": previous["actors"][e], "pacing": d["pacing"],
                               "after_action_id": d["after_action_id"],
                               "anchor": (previous["arrangements"].get(e) or {}).get("scene_anchor") or
                                         d.get("focus_anchor") or d["after_action_id"]}
                           for e, d in previous["scene_decisions"].items() if e in eligible}

        def validate_replan(result):
            validate(result)
            if set(result.scene_decisions) != set(eligible):
                raise ValueError("scene_decisions 须恰好覆盖本轮所有活跃事件")
            for event, decision in result.scene_decisions.items():
                recent = interactions[event]["recent_actions"]
                last_id = recent[-1]["action_id"] if recent else None
                if decision.after_action_id != last_id:
                    raise ValueError(f"{event} 的重规划须从该事件最后一次已提交行动 {last_id} 起比较")
                if event in result.arrangements:
                    if decision.focus_anchor not in (None, result.arrangements[event].scene_anchor):
                        raise ValueError(f"{event} 的环境机会须沿用该环境安排的现场依据")
                else:
                    self.scene_evidence(decision.focus_anchor or last_id, runtime)

        def scene_for_review(candidate):
            # 审读者只核对当前现场能否支持这一步。调度者的目标、缺口与查档结果不传入。
            return {"eligible": eligible, "interaction_progress": interactions,
                    "previous_scheduled_focus": prior_focus,
                    "pending_observations": pending_observations(runtime["actions"], interactions),
                    "current_state": runtime["current_state"], "views": runtime["views"],
                    "person_runtime": runtime["person_runtime"], "story_seconds": runtime["story_seconds"],
                    "working_states": {p: s["current"]["working_state"] for p, s in runtime["subjective_worlds"].items()},
                    "initial_location_assets": {p: self.world["assets"]["location"][choice.location_id]
                                                for p, choice in candidate.initial_locations.items()},
                    "scene_focus": self.scene_focus(candidate.model_dump(mode="json"), runtime, eligible),
                    "scene_evidence": {event: self.scene_evidence(attempt.scene_anchor, runtime)
                                       for event, attempt in candidate.arrangements.items()}}

        def review_paths(candidate):
            # 逐事件、逐人覆盖，避免审了第一个事件就放行同批其他安排。
            return [f"/{field}/" + key.replace("~", "~0").replace("/", "~1")
                    for field in ("scene_decisions", "actors", "arrangements", "initial_locations")
                    for key in getattr(candidate, field)]

        result, review_job = await reviewed_task(self.executor, job, self.prompt, context, ReplanSchedule,
            validate_replan, review_paths, review_evidence=scene_for_review)
        seal_review(self.storage, "schedule", result.model_dump(mode="json"), review_job)
        return result

    def scene_focus(self, plan, runtime, event_ids):
        """仅投影本批事件的取景方式和可回读的现场事实。"""
        focus = {}
        for event in event_ids:
            decision = plan.get("scene_decisions", {}).get(event)
            if decision is None:
                continue
            attempt = plan.get("arrangements", {}).get(event)
            anchor = attempt["scene_anchor"] if attempt else decision.get("focus_anchor") or decision["after_action_id"]
            if anchor:
                focus[event] = {"pacing": decision["pacing"], "anchor": anchor,
                                "fact": self.scene_evidence(anchor, runtime)}
        return focus

    def _current_schedule(self, group, round_number):
        plan = self.storage.get_checkpoint("round_schedule") or {}
        if plan.get("group_id") == group.group_id and plan.get("round") == round_number:
            return plan
        return {}

    async def perceive(self, group, people, clock):
        runtime = self.runtime_context(group)
        plan = self._current_schedule(group, clock["round"])
        focus = self.scene_focus(plan, runtime, [event for event, actor in plan.get("actors", {}).items() if actor in people])
        def validate(result):
            if set(result.views) != set(people):
                raise ValueError("行动前感知须恰好覆盖本批选中的人物")
            self.storage.validate_views(result.views)
        result = await visible_task(self.executor, task_id("world", "perceive", group.group_id, clock["round"]),
            self.prompt, {"operation": "perceive", "selected": people,
                **runtime, **({"scene_focus": focus} if focus else {})}, PerceptionBatch, validate,
            lambda value: {p: {"perception": v.personal.model_dump()} for p, v in value.views.items()},
            fixed=private_perceptions, private_review=private_perceptions)
        return result.views

    async def resolve_wave(self, group, wave, proposals):
        version, state, _ = self.storage.current()
        runtime = self.runtime_context(group)
        before_views = runtime["views"]
        wave_id = task_id(group.group_id, wave[0]["position"]["round"], wave[0]["position"]["turn"])
        shared_id = task_id("shared", group.group_id, wave[0]["position"]["round"], wave[0]["position"]["turn"])
        rows = {row["action_id"]: (row, proposal) for row, proposal in zip(wave, proposals)}
        def final_views(value):
            views = {}
            for resolved in value.actions:
                for person, view in resolved.observations.items():
                    views.setdefault(person, []).append(view)
            people = set(views) | set(value.advance.observations)
            if set(value.final_views) - people:
                raise ValueError("当前视图只能属于本批有个人体验的人物")
            final = {}
            for person in people:
                parts = [*views.get(person, [])]
                if person in value.advance.observations:
                    parts.append(value.advance.observations[person])
                if person in value.final_views:
                    final[person] = value.final_views[person]
                elif all(view == parts[0] for view in parts):
                    final[person] = parts[0]
                else:
                    raise ValueError(f"{person} 本批有多份不同体验，须提供 final_views 中的统一当前视图；原体验均须保留")
            return final
        def validate(result):
            if len(result.actions) != len(rows) or {r.action_id for r in result.actions} != set(rows):
                raise ValueError("统一判断须恰好覆盖本批每一项提议")
            catalog = validate_registrations(self.storage, result.advance.new_objects,
                [a.state_patch for a in result.actions] + [result.advance.state_patch], rows)
            self.storage.validate_views(result.advance.observations, catalog)
            self.storage.validate_views(result.final_views, catalog)
            if result.advance.observations and not result.advance.result:
                raise ValueError("自然经过向人物反馈变化时须保存实际发生的故事内容")
            if (len(result.advance.related_event_ids) != len(set(result.advance.related_event_ids)) or
                    set(result.advance.related_event_ids) - {e["task_id"] for e in group.events}):
                raise ValueError("共享材料的 related_event_ids 必须唯一且属于本事件组")
            if result.advance.related_event_ids and not result.advance.result:
                raise ValueError("advance.result 为空，表示没有共享故事材料；advance.related_event_ids 必须为 []")
            if set(result.advance.runtime_changes) - set(self.storage.manifest["catalog"]["role"]):
                raise ValueError("身体/认知变化引用未知人物")
            for change in result.advance.runtime_changes.values():
                if set(change.source_action_ids) - set(rows):
                    raise ValueError("本批身体/认知变化须引用本批真实行动或明确时间经过依据")
            for trace in result.advance.processing:
                if trace.stage != "execution" or not trace.source_ids or set(trace.source_ids) - set(rows):
                    raise ValueError("执行偏差须有行动前已生效的执行作用与本批行动依据")
                for source in trace.source_ids:
                    person = rows[source][1].character_id
                    own = {e["effect_id"]: e for e in runtime["person_runtime"].get(person, {}).get("effects", [])}
                    if set(trace.effect_ids) - own.keys() or any(own[e]["stage"] != "execution" for e in trace.effect_ids):
                        raise ValueError("不能借用其他人的认知作用解释本人的执行偏差")
            for resolved in result.actions:
                _, proposal = rows[resolved.action_id]
                self.storage.validate_views(resolved.observations, catalog)
                before = before_views.get(proposal.character_id)
                self.storage.validate_resolved_targets(proposal, WorldPerception.model_validate(before) if before else None,
                                                       resolved.resolved_targets, catalog)
                if proposal.character_id is not None and proposal.character_id not in resolved.observations:
                    raise ValueError("须向行动者反馈本人能体验到的结果")
                if proposal.mode == "think" and set(resolved.observations) - {proposal.character_id}:
                    raise ValueError("私有思考不得通过个人体验泄漏")
                if proposal.mode in ("think", "idle") and apply_patch(state, resolved.state_patch) != state:
                    raise ValueError("思考及无行动不能直接改变外部状态")
            final_views(result)
            combined = StatePatch.model_validate(merge_deltas([r.state_patch for r in result.actions]).model_dump())
            validate_references(apply_patch(apply_patch(state, combined), result.advance.state_patch), catalog)
        def project(value):
            final = final_views(value)
            experiences = {p: [] for p in final}
            for action in value.actions:
                for person, view in action.observations.items():
                    experiences[person].append({"source_id": action.action_id, "source_kind": "action",
                                                "observation": view.personal.model_dump()})
            for person, view in value.advance.observations.items():
                experiences[person].append({"source_id": shared_id, "source_kind": "shared",
                                            "observation": view.personal.model_dump()})
            return {p: {"perception": final[p].personal.model_dump(), "observations": experiences[p]} for p in final}
        # 从实存状态回读供裁定及独立审核共同核对；环境安排自己的说明不充当证据。
        anchors = {row["action_id"]: self.scene_evidence(row["arrangement"]["scene_anchor"], runtime)
                   for row in wave if (row.get("arrangement") or {}).get("scene_anchor")}
        plan = self._current_schedule(group, wave[0]["position"]["round"])
        event_ids = [row["position"]["event_id"] for row in wave]
        focus = self.scene_focus(plan, runtime, event_ids)
        decisions = plan.get("scene_decisions", {})
        boundaries = {event: decisions[event]["stop_before"] for event in event_ids if event in decisions}
        def review_resolution(value):
            evidence = private_perceptions(value)
            if boundaries:
                evidence["approved_boundaries"] = boundaries
            return evidence

        result = await visible_task(self.executor, task_id("world", "wave", wave_id), self.prompt,
            {"operation": "resolve_wave", **runtime,
             **({"scene_evidence": anchors} if anchors else {}),
             **({"scene_focus": focus} if focus else {}),
             "state_consistency_feedback": (self.storage.get_checkpoint("last_goal") or {}).get("contradictions", []),
             "object_inventory": object_inventory(self.storage),
             "intents": [{**row, "proposal": p.model_dump(mode="json")} for row, p in zip(wave, proposals)],
             "current_version": version}, WaveResolution, validate, project,
            fixed=private_perceptions, private_review=review_resolution,
            observer_evidence=lambda value: {p: self.storage.subjective_world(p) for p in final_views(value)
                                             if p not in runtime["subjective_worlds"]})
        result.actions.sort(key=lambda action: list(rows).index(action.action_id))
        unresolved = [r for r in result.actions if any(m.actual_kind == "unknown" for m in r.resolved_targets)]
        unknown_review = None
        if unresolved:
            unknown_review = await self.executor.run_task(task_id("world", "review_unknown_targets", wave_id),
                "world", self.prompt, {"operation": "review_unknown_targets", **runtime,
                    "intents": [{**row, "proposal": proposal.model_dump(mode="json")} for row, proposal in rows.values()],
                    "actions": [r.model_dump(mode="json") for r in result.actions],
                    "unresolved_action_ids": [r.action_id for r in unresolved],
                    "advance": result.advance.model_dump(mode="json")}, UnknownTargetReview)
            if not unknown_review.approved:
                raise RunStopped("未知对象使实际结果无法确定：" + "; ".join(unknown_review.issues))
        final = final_views(result)
        runtimes = {**runtime["person_runtime"], **{p: c.state.model_dump() for p, c in result.advance.runtime_changes.items()}}
        states = advance_needs({p: self.storage.character_state(p).model_dump()
                                for p in self.storage.manifest["catalog"]["role"]}, result.advance.elapsed_seconds)
        observations = {}
        for resolved in result.actions:
            row, proposal = rows[resolved.action_id]
            for person, view in resolved.observations.items():
                before = before_views.get(person)
                observations.setdefault(person, []).append({"source_id": resolved.action_id, "source_kind": "action",
                    "position": row["position"], "observation": view.personal.model_dump(),
                    "before_perception": WorldPerception.model_validate(before).personal.model_dump() if before else None,
                    "own_proposal": proposal.model_dump(mode="json") if person == proposal.character_id else None})
        for person, view in result.advance.observations.items():
            before = before_views.get(person)
            observations.setdefault(person, []).append({"source_id": shared_id, "source_kind": "shared",
                "observation": view.personal.model_dump(),
                "before_perception": WorldPerception.model_validate(before).personal.model_dump() if before else None})
        people, feedback = sorted(observations), {}
        for person in people:
            if person not in runtime["subjective_worlds"]:
                runtime["subjective_worlds"][person] = self.storage.subjective_world(person)
        access = await prepare_experience(self, wave_id + "/after", people, observations=observations,
                                          runtimes=runtimes, states=states,
                                          elapsed_seconds=result.advance.elapsed_seconds,
                                          acted=[p.character_id for p in proposals if p.character_id]) if people else {}
        for revision in range(self.storage.config.semantic_repairs + 1):
            summaries = await asyncio.gather(*(NovelRestoreCharacterAgent(person, self.storage, self.executor).summarize_wave(
                wave_id, access[person], revision, feedback.get(person, []),
                catalog=effective_catalog(self.storage, result.advance.new_objects)) for person in people), return_exceptions=True)
            for summary in summaries:
                if isinstance(summary, BaseException):
                    raise summary
            approval = await self.executor.run_task(task_id("world", "review_memory", wave_id, revision), "world", self.prompt,
                {"operation": "review_memory", "observations": observations,
                 "state_before": state.model_dump(mode="json"), "resolution": result.model_dump(mode="json"),
                 "subjective_worlds": runtime["subjective_worlds"],
                 "views_before": before_views,
                 "previous_actions": runtime["actions"], "person_runtime": runtimes,
                 "subjective_inputs": {p: v.model_dump() for p, v in access.items()},
                 "summaries": {p: s.model_dump(mode="json") for p, s in zip(people, summaries)}}, MemoryApproval)
            if approval.approved:
                break
            feedback = {p: approval.character_issues.get(p, ["请核对本人收到的体验与记忆引用，不借用未获知的信息。"])
                        for p in people}
            if revision < self.storage.config.semantic_repairs:
                def check_feedback(value):
                    if set(value.messages) != set(people):
                        raise ValueError("返修提示须恰好覆盖本批人物")
                safe = await visible_task(self.executor, task_id("world", "feedback", wave_id, revision), self.prompt,
                    {"operation": "repair_character_feedback", "review": approval.model_dump(),
                     "subjective_inputs": {p: v.model_dump() for p, v in access.items()}},
                    CharacterFeedback, check_feedback,
                    lambda value: {p: {"feedback": messages} for p, messages in value.messages.items()},
                    initial=CharacterFeedback(messages=feedback))
                feedback = safe.messages
                # 有提取失真时，审核者知道人物不该知道的原始身份。人物返修仅收固定提示。
                for person in people:
                    effects = [*runtime["person_runtime"].get(person, {}).get("effects", []),
                               *runtimes.get(person, {}).get("effects", [])]
                    if any(effect["stage"] == "retrieval" for effect in effects):
                        feedback[person] = ["请仅依据本次获得的体验、当时的回想和可见编号核对自己的记录。"]
                        record_visible(self.storage, person, {"feedback": feedback[person]}, "recall/private-repair")
        else:
            raise JobFailed("人物主观世界未通过检查：" + "; ".join(approval.issues))
        actions = []
        mental = {**states, **{p: s.working_state.model_dump() for p, s in zip(people, summaries)}}
        for resolved in result.actions:
            row, proposal = rows[resolved.action_id]
            contributions = [m for summary in summaries for m in summary.memories if m.source_action_id == resolved.action_id]
            actions.append(ConfirmedAction(action_id=resolved.action_id, run_id=self.storage.run_id,
                expected_version=version, position=StoryPosition.model_validate(row["position"]), proposal=proposal,
                recollections_before=RecallExperience.from_action_context(row["context"]),
                result=resolved.result, state_patch=resolved.state_patch, resolved_targets=resolved.resolved_targets,
                unknown_target_review=unknown_review if resolved in unresolved else None,
                views_before={p: before_views[p] for p in resolved.observations if p in before_views},
                observations=resolved.observations, memory_review=approval,
                environment_basis=row["arrangement"]["basis"] if row.get("arrangement") else None,
                working_states={p: mental[p] for p in resolved.observations},
                subjective_inputs={p: access[p] for p in resolved.observations},
                memories=[MemoryRecord(**m.model_dump(mode="json"), memory_id=task_id(resolved.action_id, i),
                    run_id=self.storage.run_id, action_id=resolved.action_id, position=row["position"])
                    for i, m in enumerate(contributions, 1)]))
        shared_memories = [m for summary in summaries for m in summary.memories if m.source_shared_id == shared_id]
        shared_position = {**wave[0]["position"], "event_id": None,
                           "round": max(row["position"]["round"] for row in wave)}
        actions[0].advance = WorldAdvance(**result.advance.model_dump(mode="json"), shared_id=shared_id,
            version_after=version + len(actions),
            views_before={p: before_views[p] for p in result.advance.observations if p in before_views},
            subjective_inputs={p: access[p] for p in result.advance.observations},
            working_states={p: mental[p] for p in result.advance.observations}, final_views=final,
            memories=[MemoryRecord(**m.model_dump(mode="json"), memory_id=task_id(shared_id, i),
                run_id=self.storage.run_id, action_id=None, position=shared_position)
                for i, m in enumerate(shared_memories, 1)])
        # 真实结果已固定，目标检查只能交付完成判断，不能改行动、感知或人物心理。
        combined = StatePatch.model_validate(merge_deltas([a.state_patch for a in actions]).model_dump())
        current_state = apply_patch(apply_patch(state, combined), result.advance.state_patch).model_dump(mode="json")
        event_actions = {event["task_id"]: {a.action_id for a in self.storage.event_actions(event["task_id"])}
                         for event in group.events}
        for action in actions:
            event_actions[action.position.event_id].add(action.action_id)
        action_ids = set().union(*event_actions.values())
        def check_assessment(value):
            if set(value.events) != {a.position.event_id for a in actions}:
                raise ValueError("完成检查须覆盖本批全部事件")
            for event_id, judgment in value.events.items():
                validate_result_evidence(judgment, current_state, action_ids, event_action_ids=event_actions[event_id])
            validate_result_evidence(value.goal, current_state, action_ids)
        target = self.storage.get_checkpoint("group_target")
        assessment, review_job = await reviewed_task(self.executor, task_id("world", "assess_wave", wave_id), self.prompt,
            {"operation": "assess_wave", "group": group.model_dump(mode="json"),
             "reference_state_policy": REFERENCE_STATE_POLICY,
             "target_state": target, "next_reference": compact_next_reference(target, self.continuation(group)),
             "state_before": state.model_dump(mode="json"),
             "current_state": current_state,
             "core_completed": {event: {"achieved": item["achieved"]}
                                for event, item in self.storage.get_checkpoint("core_completed", {}).items()},
             "previous_actions": runtime["actions"],
             "actions": [a.model_dump(mode="json") for a in actions], "working_states": mental},
            WaveAssessment, check_assessment,
            lambda value: ["/events/" + e.replace("~", "~0").replace("/", "~1") for e in value.events] + ["/goal", "/made_progress"])
        for action in actions:
            action.completion = assessment.events[action.position.event_id]
        self.storage.validate_wave(actions, mental)
        seal_review(self.storage, "wave", wave_delivery(actions, mental, assessment.goal, assessment.made_progress), review_job)
        return actions, mental, assessment.goal, assessment.made_progress

    async def summarize_group(self, group):
        for person in self.storage.manifest["catalog"]["role"]:
            saved = self.storage.memory_summaries(person)
            if any(s["group_id"] == group.group_id and s["kind"] == "event" for s in saved):
                continue
            records = [m.model_dump(mode="json") for m in self.storage.memories(person)
                       if (m.position.node_id, m.position.group_order) == (group.node_id, group.order)]
            if not records:
                continue
            actor = NovelRestoreCharacterAgent(person, self.storage, self.executor)
            summary = await self.checked_summary(actor, group.group_id, records, [])
            saved.append({"group_id": group.group_id, "kind": "event", **summary.model_dump(mode="json")})
            events = [s for s in saved if s["kind"] == "event"]
            if len(events) % self.storage.config.summary_interval == 0:
                stages = [s for s in saved if s["kind"] == "stage"]
                inputs = stages[-1:] + events[-self.storage.config.summary_interval:]
                stage = await self.checked_summary(actor, group.group_id + "/stage", [], inputs)
                saved.append({"group_id": group.group_id, "kind": "stage", **stage.model_dump(mode="json")})
            self.storage.set_checkpoint("summaries/" + person, saved)
        self.storage.publish_characters()

    async def checked_summary(self, actor, identity, records, summaries):
        feedback = []
        for revision in range(self.storage.config.semantic_repairs + 1):
            result = await actor.summarize_history(identity, records, summaries, revision, feedback)
            check = await self.executor.run_task(task_id("world", "review_summary", identity, actor.character_id, revision),
                "world", self.prompt, {"operation": "review_summary", "character_id": actor.character_id,
                    "records": records, "summaries": summaries, "summary": result.model_dump(mode="json")}, MemoryApproval)
            if check.approved:
                return result
            feedback = check.issues
        raise JobFailed("人物摘要未通过来源核对")
