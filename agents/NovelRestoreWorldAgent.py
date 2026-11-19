"""世界角色负责语义，程序绑定身份、引用与提交位置。"""
from execution.NovelRestoreTaskPrompt import load_prompt

from common.NovelRestoreConfig import BranchDecision

from common.Runtime import CandidateError, JobFailed, RunStopped, fingerprint, task_id
from agents.NovelRestoreCharacterAgent import NovelRestoreCharacterAgent
from common.NovelRestoreSchema import (
    ConfirmedAction,
    GroupAssessment, MemoryApproval, WaveResolution,
    RoundSchedule, ReplanSchedule, SceneDecision, StatePatch,
    PerceptionBatch, WorldPerception, WaveAssessment,
    UnknownTargetReview,
)
from inputs.NovelWorldState import merge_deltas
from simulation.NovelRestoreWorldState import resolved_views, apply_patch, scene_fact, validate_references, validate_result_evidence
from simulation.NovelRestoreObjects import object_inventory, validate_registrations
from simulation.NovelRestoreInitialProfile import initialize_profiles
from simulation.NovelRestoreVisibility import visible_task, private_perceptions
from simulation.NovelRestoreContext import REFERENCE_STATE_POLICY, runtime_context, progress_actions, action_history, shared_history, interaction_progress, pending_observations, compact_next_reference
from simulation.NovelRestoreGoals import initial_goal_records, goals_satisfied, goal_replan_required, validate_goal_updates, reusable_goal_updates
from simulation.NovelRestoreScene import validate_continuations


from execution.NovelRestoreReview import reviewed_task, seal_review, group_delivery


class NovelRestoreWorldAgent:
    def __init__(self, storage, repository, supervisor):
        self.storage, self.repository, self.supervisor = storage, repository, supervisor
        self.prompt = load_prompt()
        nodes = repository.nodes()
        self.world = {"assets": {kind: repository.assets(kind) for kind in repository.ASSET_KINDS},
                      "worlds": repository.worlds(),
                      "opening_snapshot": repository.snapshot(nodes[0]["node_id"]).model_dump(mode="json"),
                      "groups": [group for node in nodes
                                 for group in repository.groups(node["node_id"])]}

    def continuation(self, group):
        groups = self.world["groups"]
        index = next(i for i, row in enumerate(groups) if row.group_id == group.group_id)
        if index + 1 == len(groups):
            return None
        return self.repository.group_states(groups[index + 1])[0].model_dump(mode="json")

    async def initialize(self):
        await initialize_profiles(self)
        self.storage.initialize_world()

    async def assess_group(self, group):
        version, state, _ = self.storage.current()
        events = self.storage.event_states()
        target = self.storage.get_checkpoint("group_target")
        current_state = state.model_dump(mode="json")
        action_ids = set().union(*(self.storage.event_action_ids(e["task_id"]) for e in group.events))
        goals = self.storage.get_checkpoint("goal_records") or initial_goal_records(group)
        # 逐项结论已经独立审读。组末默认只展开它们真正引用的已提交经过，疑点仍可补查。
        cited = {identity for row in goals.values() for field in ("latest", "completion_evidence")
                 for identity in (row.get(field) or {}).get("action_ids", [])}
        evidence_actions = {e["task_id"]: [a.model_dump(mode="json")
            for a in self.storage.event_actions(e["task_id"], action_ids=cited)] for e in group.events}
        def validate(judgment):
            validate_result_evidence(judgment, current_state, action_ids)
            if judgment.achieved and not goals_satisfied(goals):
                raise ValueError("正式组完成须有全部逐项目标的已审核满足记录，不得用总判断覆盖尚缺条件")
        result, review_job = await reviewed_task(self.supervisor, task_id("world", "assess_group", group.group_id, version),
            self.prompt, {"operation": "assess_group", "group": group.model_dump(mode="json"),
                "reference_state_policy": REFERENCE_STATE_POLICY,
                "reference_before": self.storage.group_state().model_dump(mode="json"),
                "target_state": target,
                "next_reference": compact_next_reference(target, self.continuation(group)),
                "current_state": current_state,
                "goal_records": goals,
                "events": {e["task_id"]: events[e["task_id"]] for e in group.events},
                "actions": action_history(evidence_actions), "shared_advances": shared_history(evidence_actions),
                "evidence_policy": "goal_records 为已审核逐项证据；核对当前有效性、同时满足与承接。actions 仅展开被引经过；疑点可用 world_lookup 补查。"}, GroupAssessment,
            validate,
            lambda value: ["/" + key for key in ("achieved", "summary", "achieved_results", "remaining", "continuity_ok", "contradictions")])
        seal_review(self.storage, "group", group_delivery(self.storage, group.group_id, result), review_job)
        self.storage.record_group_assessment(group, version, result)
        return result.achieved

    def scene_evidence(self, anchor, runtime):
        """读取环境提议所指的现场事实。它能否支持本次变化，由独立审读结合处境判断。"""
        if anchor and anchor.startswith("action/"):
            action = self.storage.get_action(anchor)
            if action is None:
                raise ValueError(f"scene_anchor 没有对应的已提交行动：{anchor}")
            return {"result": action.result, "shared_result": action.advance.result if action.advance else None}
        if anchor == "/story_seconds":
            return runtime["story_seconds"]
        domains = ("characters", "locations", "objects", "relations")
        if not anchor or not anchor.startswith(tuple(f"/current_state/{name}/" for name in domains)):
            raise ValueError("环境尝试缺少现场依据：scene_anchor 须引用已提交的 action_id 或具体的 /current_state 路径；"
                             "定时变化可引用 /story_seconds。若眼下没有足够条件，可把行动机会交给有理由行动的人物。")
        return scene_fact(runtime["current_state"], anchor.removeprefix("/current_state"))

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

        decision = await self.supervisor.run_task(task_id("world", "consider_branch", group.node_id), "world",
            self.prompt, context, BranchDecision, validate)
        saved = {"node_id": group.node_id, **decision.model_dump(mode="json")}
        # 目录标识也被用作择线编号，不允许其中包含尚未发生的故事设想。
        for index, choice in enumerate(saved["alternatives"], 1):
            choice["name"] = str(index)
        checkpoints = {key: saved}
        if decision.alternatives:
            checkpoints["branch_split"] = saved
            # 节点决策在首次行动前完成。其账目也属于子线共同承担的前缀。
            start_key = "group_start/" + group.group_id
            entry = self.storage.get_checkpoint(start_key)
            entry["last_call_id"] = self.storage.connection.execute("SELECT coalesce(max(id),0) FROM calls").fetchone()[0]
            checkpoints[start_key] = entry
        self.storage.set_checkpoints(checkpoints)
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
        runtime = runtime_context(self, group)
        interactions = interaction_progress(runtime["actions"], eligible, group=(group.node_id, group.order),
                                            shared_advances=runtime["shared_advances"])
        goals = self.storage.get_checkpoint("goal_records") or initial_goal_records(group)
        replan = goal_replan_required(goals, self.storage.config.goal_stagnation_batches)
        def validate(result):
            issues = []
            def issue(location, message, kind="candidate_error"):
                issues.append({"loc": ("result", *location), "type": kind, "msg": message})

            if set(result.actors) != set(eligible):
                raise ValueError("actors 必须以当前活跃事件编号为键，值只能是该事件的参与人物编号或 null；"
                                 f"当前事件及可选人物：{eligible}；不要以人物编号为键，也不要填写行动内容")
            if set(result.opportunities) != set(eligible):
                raise ValueError("每个选中机会须说明目标缺口、本人动机、可能回应和预期贡献")
            for index, event in enumerate(sorted(eligible)):
                opportunity = result.opportunities[event]
                position = ("decisions", index, "opportunity")
                if len(opportunity.goal_ids) != len(set(opportunity.goal_ids)) or set(opportunity.goal_ids) - set(goals):
                    issue((*position, "goal_ids"), "机会只能关联本组已有的稳定目标，不得新增或重复目标")
                if set(opportunity.perceivers) - set(current_people):
                    issue((*position, "perceivers"), "机会接收者须为本线实际人物")
                try:
                    self.scene_evidence(opportunity.scene_anchor, runtime)
                except ValueError as error:
                    issue((*position, "scene_anchor"), str(error))
                if event in result.arrangements and opportunity.scene_anchor != result.arrangements[event].scene_anchor:
                    raise ValueError("环境机会与环境尝试必须使用相同现场依据")
            if not result.scene_decisions:
                result.scene_decisions = {event: SceneDecision(pacing="stay",
                    after_action_id=(interactions[event]["recent_actions"][-1]["action_id"]
                                     if interactions[event]["recent_actions"] else None),
                    focus_anchor=opportunity.scene_anchor, opportunity="机会依据与预期贡献见 opportunities/" + event,
                    stop_before=opportunity.stop_before) for event, opportunity in result.opportunities.items()}
            people = [p for p in result.actors.values() if p is not None]
            if len(people) != len(set(people)):
                raise ValueError("同一人物不能在同批两个事件中分别行动")
            for person, choice in result.initial_locations.items():
                position = ("initial_locations", person)
                if person not in participants:
                    issue(position, f"{person} 不是本组活跃事件参与者；撤回该初始地点安排（null）", "candidate_record")
                elif current_people[person].position.location_id is not None:
                    issue(position, f"{person} 仅当前位置未知时才可补充初始地点；实际位置为 "
                          f"{current_people[person].position.model_dump(mode='json')}。具体房间未明不等于地点未知；"
                          "撤回该初始地点安排（null），不能用初始化代替移动或补造所在房间", "candidate_record")
                elif current_people[person].life_status == "not_born":
                    issue(position, f"{person} 尚未出生，撤回该初始地点安排（null）", "candidate_record")
                elif choice.location_id not in self.storage.manifest["catalog"]["location"]:
                    issue((*position, "location_id"), f"{person} 的初始地点必须属于 repository 地点目录")
            for index, (event, people) in enumerate(sorted(eligible.items())):
                if result.actors[event] is not None and result.actors[event] not in people:
                    issue(("decisions", index, "actor"), "下一行动者须属于该事件；环境推进使用 null 并提交发生依据")
                elif result.actors[event] and current_people[result.actors[event]].life_status in ("dead", "not_born"):
                    issue(("decisions", index, "actor"), "客观状态中已死亡或尚未出生的人物不能主动行动")
                elif result.actors[event] and not self.storage.get_checkpoint("person_runtime", {}).get(result.actors[event], {}).get("can_act", True):
                    issue(("decisions", index, "actor"), "当前无法自主行动的人物不能被安排主动行为")
            if issues:
                raise CandidateError("\n".join(row["msg"] for row in issues), issues)
        job = task_id("world", "schedule", group.group_id, clock["round"])
        context = {"operation": "schedule", "eligible": eligible, "clock": clock, "replan": replan,
                "group": group.model_dump(mode="json"), "target_state": target,
                "next_reference": compact_next_reference(target, self.continuation(group)),
                "last_goal": ({k: last_goal[k] for k in ("achieved", "continuity_ok")} if last_goal else None),
                "assessment_feedback": ({k: last_goal.get(k, []) for k in ("remaining", "contradictions")} if last_goal else None),
                "core_completed": {event: {"achieved": item["achieved"]} for event, item in completed.items()},
                "goal_records": goals,
                "interaction_progress": interactions, **runtime}
        if group.node_id == self.storage.manifest.get("fork_node_id") and self.storage.manifest.get("branch_opportunity"):
            context["branch_opportunity"] = self.storage.manifest["branch_opportunity"]
        previous = self.storage.get_checkpoint("round_schedule") or {}
        if previous.get("group_id") == group.group_id:
            context["previous_opportunities"] = previous.get("opportunities", {})
        prior_focus = {}
        if previous.get("group_id") == group.group_id and previous.get("scene_decisions"):
            context["previous_scene_decisions"] = {
                event: {key: decision[key] for key in ("pacing", "after_action_id", "focus_anchor", "stop_before")}
                for event, decision in previous["scene_decisions"].items()}
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
                opportunity = result.opportunities[event]
                if decision.stop_before != opportunity.stop_before or decision.focus_anchor != opportunity.scene_anchor:
                    raise ValueError("实际机会停点与机会规划的现场依据、停点须一致")
                if event in result.arrangements:
                    if decision.focus_anchor not in (None, result.arrangements[event].scene_anchor):
                        raise ValueError(f"{event} 的环境机会须沿用该环境安排的现场依据")
                else:
                    self.scene_evidence(decision.focus_anchor or last_id, runtime)

        def scene_for_review(candidate):
            # Planning review sees goals in a separate section; actual causal
            # adjudication below still receives only approved scene facts.
            return {"operation": "schedule", "eligible": eligible, "interaction_progress": interactions,
                    "goal_records": goals, "target_state": target,
                    "previous_opportunities": context.get("previous_opportunities", {}),
                    "planning_policy": "目标仅用来审核机会贡献，不能证明现场事实、人物知情或实际成功。",
                    "previous_scheduled_focus": prior_focus,
                    "pending_observations": pending_observations(runtime["actions"], interactions,
                                                                 shared_advances=runtime["shared_advances"]),
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
                    for field in ("opportunities", "scene_decisions", "actors", "arrangements", "initial_locations")
                    for key in getattr(candidate, field)]

        result, review_job = await reviewed_task(self.supervisor, job, self.prompt, context, ReplanSchedule if replan else RoundSchedule,
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
        runtime = runtime_context(self, group, selected=people)
        plan = self._current_schedule(group, clock["round"])
        focus = self.scene_focus(plan, runtime, [event for event, actor in plan.get("actors", {}).items() if actor in people])
        def validate(result):
            if set(result.views) != set(people):
                raise ValueError("行动前感知须恰好覆盖本批选中的人物")
            self.storage.validate_views(result.views)
        result = await visible_task(self.supervisor, task_id("world", "perceive", group.group_id, clock["round"]),
            self.prompt, {"operation": "perceive", "selected": people,
                **runtime, **({"scene_focus": focus} if focus else {})}, PerceptionBatch, validate,
            lambda value: {p: {"perception": v.personal.model_dump()} for p, v in value.views.items()},
            fixed=private_perceptions, private_review=private_perceptions)
        return result.views

    async def adjudicate_wave(self, group, wave, proposals, runtime):
        version, state, _ = self.storage.current()
        before_views = runtime["views"]
        wave_id = task_id(group.group_id, wave[0]["position"]["round"], wave[0]["position"]["turn"])
        shared_id = task_id("shared", group.group_id, wave[0]["position"]["round"], wave[0]["position"]["turn"])
        rows = {row["action_id"]: (row, proposal) for row, proposal in zip(wave, proposals)}
        def validate(result):
            if len(result.actions) != len(rows) or {r.action_id for r in result.actions} != set(rows):
                raise ValueError("统一判断须恰好覆盖本批每一项提议")
            self.storage.advanced_story_seconds(result.advance.elapsed_seconds)
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
            resolved_views(result)
            combined = StatePatch.model_validate(merge_deltas([r.state_patch for r in result.actions]).model_dump())
            validate_references(apply_patch(apply_patch(state, combined), result.advance.state_patch), catalog,
                                locations=self.world["assets"]["location"])
        def project(value):
            final = resolved_views(value)
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

        approved_profiles = self.storage.get_checkpoint("self_profiles", {})
        def observer_boundaries(value):
            """只向知情审核补齐本人的确切已批准起点边界。"""
            evidence = {}
            for person in resolved_views(value):
                if person not in approved_profiles:
                    raise RunStopped("人物缺少已批准的起点档案，不能审核个人体验：" + person)
                evidence[person] = ({"profile": approved_profiles[person]}
                                    if person in runtime["subjective_worlds"]
                                    else self.storage.subjective_world(person))
            return evidence

        unknown_review, unknown_reviews = None, {}
        async def review_unknown_targets(value, revision):
            nonlocal unknown_review
            unknown_review = None
            unresolved = [r.action_id for r in value.actions
                          if any(m.actual_kind == "unknown" for m in r.resolved_targets)]
            if not unresolved:
                return []
            context = {"operation": "review_unknown_targets", **runtime,
                "intents": [{**row, "proposal": proposal.model_dump(mode="json")} for row, proposal in rows.values()],
                "actions": [r.model_dump(mode="json") for r in value.actions],
                "unresolved_action_ids": unresolved, "advance": value.advance.model_dump(mode="json")}
            base = task_id("world", "review_unknown_targets", wave_id)
            job = unknown_reviews.setdefault(fingerprint(context), base if not revision else task_id(base, revision))
            unknown_review = await self.supervisor.run_task(job, "world", self.prompt, context, UnknownTargetReview)
            return ["未知对象使实际结果无法确定：" + issue for issue in unknown_review.issues]

        result = await visible_task(self.supervisor, task_id("world", "wave", wave_id), self.prompt,
            {"operation": "resolve_wave", **runtime,
             **({"scene_evidence": anchors} if anchors else {}),
             **({"scene_focus": focus} if focus else {}),
             **({"approved_boundaries": boundaries} if boundaries else {}),
             "object_inventory": object_inventory(self.storage),
             "intents": [{**row, "proposal": p.model_dump(mode="json")} for row, p in zip(wave, proposals)],
             "current_version": version}, WaveResolution, validate, project,
            fixed=private_perceptions, private_review=review_resolution, world_review=review_unknown_targets,
            observer_evidence=observer_boundaries)
        order = {identity: index for index, identity in enumerate(rows)}
        result.actions.sort(key=lambda action: order[action.action_id])
        return result, unknown_review

    async def assess_wave(self, group, wave_id, state, actions, mental, runtime):
        # 真实结果已固定，目标检查只能交付完成判断，不能改行动、感知或人物心理。
        combined = StatePatch.model_validate(merge_deltas([a.state_patch for a in actions]).model_dump())
        current_state = apply_patch(apply_patch(state, combined), actions[0].advance.state_patch).model_dump(mode="json")
        event_actions = {event["task_id"]: self.storage.event_action_ids(event["task_id"])
                         for event in group.events}
        for action in actions:
            event_actions[action.position.event_id].add(action.action_id)
        action_ids = set().union(*event_actions.values())
        goals = self.storage.get_checkpoint("goal_records") or initial_goal_records(group)
        def check_assessment(value):
            if set(value.events) != {a.position.event_id for a in actions}:
                raise ValueError("完成检查须覆盖本批全部事件")
            for event_id, judgment in value.events.items():
                validate_result_evidence(judgment, current_state, action_ids, event_action_ids=event_actions[event_id])
            validate_result_evidence(value.goal, current_state, action_ids)
            validate_goal_updates(goals, value, state.model_dump(mode="json"), current_state, action_ids)
            eligible = {e["task_id"]: [p["role_code"] for p in e["participants"]] for e in group.events}
            runtimes = {**runtime["person_runtime"], **{p: c.state.model_dump()
                        for p, c in actions[0].advance.runtime_changes.items()}}
            validate_continuations(value, actions, eligible, current_state, runtimes)
        target = self.storage.get_checkpoint("group_target")
        assessment, review_job = await reviewed_task(self.supervisor, task_id("world", "assess_wave", wave_id), self.prompt,
            {"operation": "assess_wave", "group": group.model_dump(mode="json"),
             "reference_state_policy": REFERENCE_STATE_POLICY,
             "target_state": target, "next_reference": compact_next_reference(target, self.continuation(group)),
             "state_before": state.model_dump(mode="json"),
             "current_state": current_state,
             "goal_records": goals,
             "reusable_goal_ids": sorted(reusable_goal_updates(goals, state.model_dump(mode="json"), current_state)),
             "previous_opportunities": (self.storage.get_checkpoint("round_schedule") or {}).get("opportunities", {}),
             "core_completed": {event: {"achieved": item["achieved"]}
                                for event, item in self.storage.get_checkpoint("core_completed", {}).items()},
             "previous_actions": runtime["actions"], "previous_shared_advances": runtime["shared_advances"],
             "retrieval_scope": runtime["retrieval_scope"],
             "actions": progress_actions(actions), "working_states": mental,
             "person_runtime_after": {**runtime["person_runtime"], **{p: c.state.model_dump()
                                      for p, c in actions[0].advance.runtime_changes.items()}}},
            WaveAssessment, check_assessment,
            lambda value: ["/events/" + e.replace("~", "~0").replace("/", "~1") for e in value.events]
                          + ["/goal"] + ["/goal_updates/" + g.replace("~", "~0").replace("/", "~1") for g in value.goal_updates]
                          + ["/scene_continuations/" + e.replace("~", "~0").replace("/", "~1")
                             for e in value.scene_continuations])
        return assessment, review_job

    async def summarize_group(self, group):
        """显式按需整理一个已冻结组；辅助任务故障不改变模拟完成状态。"""
        if self.storage.get_draft(group.group_id) is None:
            raise ValueError("只能整理已冻结组的本人历史")
        async def summarize(person):
            saved = self.storage.memory_summaries(person)
            if any(s["group_id"] == group.group_id and s["kind"] == "event" for s in saved):
                return None
            records = [m.model_dump(mode="json") for m in self.storage.memories(person)
                       if (m.position.node_id, m.position.group_order) == (group.node_id, group.order)]
            if not records:
                return None
            actor = NovelRestoreCharacterAgent(person, self.storage, self.supervisor)
            summary = await self.checked_summary(actor, group.group_id, records, [])
            saved.append({"group_id": group.group_id, "kind": "event", **summary.model_dump(mode="json")})
            return person, saved
        failed = {}
        for person in self.storage.manifest["catalog"]["role"]:
            key = task_id("auxiliary_summary", group.group_id, person)
            try:
                result = await summarize(person)
            except (JobFailed, RunStopped) as error:
                self.storage.set_checkpoint(key, {"status": "failed", "error": str(error)})
                failed[person] = str(error)
                continue
            if result is not None:
                self.storage.set_checkpoint("summaries/" + person, result[1])
            self.storage.set_checkpoint(key, {"status": "complete"})
        self.storage.publish_characters()
        return failed

    async def checked_summary(self, actor, identity, records, summaries):
        feedback = []
        for revision in range(self.supervisor.semantic_attempts()):
            result = await actor.summarize_history(identity, records, summaries, revision, feedback,
                                                   recovery=self.supervisor.semantic_recovery(revision))
            check = await self.supervisor.run_task(task_id("world", "review_summary", identity, actor.character_id, revision),
                "world", self.prompt, {"operation": "review_summary", "character_id": actor.character_id,
                    "records": records, "summaries": summaries, "summary": result.model_dump(mode="json")}, MemoryApproval)
            if check.approved:
                return result
            feedback = check.issues
        raise JobFailed("人物摘要未通过来源核对")
