"""按故事顺序模拟；同组人物共享批前状态，整批结果原子提交。"""
from common.Runtime import JobFailed, fingerprint, gather_batch, task_id
from agents.NovelRestoreCharacterAgent import NovelRestoreCharacterAgent
from agents.NovelRestoreWorldAgent import NovelRestoreWorldAgent
from common.NovelRestoreSchema import (
    ActionProposal, CharacterState, ConfirmedAction, MemoryApproval, MemoryRecord,
    RecallExperience, StoryPosition, WorldAdvance, WorldPerception,
)
from simulation.NovelRestoreExperience import prepare_experience, personal_lookup
from simulation.NovelRestoreLookup import WorldLookup
from simulation.NovelRestoreGoals import goal_exhausted, goals_satisfied
from storage.NovelRestoreExporter import NovelRestoreExporter
from storage.NovelRestoreLog import NovelRestoreLog
from simulation.NovelRestoreContext import runtime_context
from simulation.NovelRestoreWorldState import resolved_views
from simulation.NovelRestoreObjects import effective_catalog
from simulation.NovelRestoreVisibility import visible_task, CharacterFeedback, record_visible
from execution.NovelRestoreReview import seal_review, wave_delivery


class NovelRestoreSimulation:
    def __init__(self, storage, repository, supervisor):
        self.storage, self.repository, self.supervisor = storage, repository, supervisor
        supervisor.world_lookup = WorldLookup(storage, repository)
        self.world = NovelRestoreWorldAgent(storage, repository, supervisor)
        self.characters = {identity: NovelRestoreCharacterAgent(identity, storage, supervisor)
                           for identity in storage.manifest["catalog"]["role"]}

    async def run(self, *, recovered=False):
        """recovered 只跳过工作流刚完成的恢复；所有后续校验仍照常执行。"""
        if not recovered:
            self.storage.recover()
        NovelRestoreExporter(self.storage).export_story_progress()
        NovelRestoreLog.report("simulation", step="初始化人物起点")
        await self.world.initialize()
        for node in self.storage.manifest["nodes"]:
            if node["order"] < self.storage.current()[2]["node_order"]:
                continue
            self.storage.enter_node(node["node_id"])
            for group in self.repository.groups(node["node_id"]):
                if group.order < self.storage.current()[2]["group_order"]:
                    continue
                self.storage.enter_group(group.order)
                self.storage.save_group_start(group)
                NovelRestoreLog.report("simulation", step="进入事件组", node=node["node_id"], group=group.group_id)
                if (group.order == 1 and self.storage.config.max_timelines > 1
                        and node["node_id"] != self.storage.manifest.get("fork_node_id")
                        and await self.world.consider_branch(group)):
                    self.storage.mark_stage("branch_ready")
                    return True
                if not await self.simulate_group(group):
                    if self.storage.current()[2]["stage"] != "incomplete":
                        self.storage.mark_stage("incomplete", "当前事件组未满足全部事件及世界目标")
                    return False
                at_stop = group.group_id == self.storage.config.stop_after_group
                if at_stop:
                    self.storage.mark_stage("group_complete")
                NovelRestoreExporter(self.storage).export_simulation()
                NovelRestoreLog.report("simulation", step="事件组验收通过，正式草稿已保存", group=group.group_id)
                if at_stop:
                    return True
        self.storage.mark_stage("simulation_complete")
        return True

    async def next_wave(self, group):
        clock = self.storage.get_checkpoint("group_clock")
        schedule = self.storage.get_checkpoint("round_schedule")
        if (schedule and "scene_continuation_version" in schedule
                and schedule["scene_continuation_version"] != self.storage.current()[0]):
            schedule = None  # 其他变化已使当前互动的接续依据失效。
        if not schedule or (schedule["group_id"], schedule["round"]) != (group.group_id, clock["round"]):
            result = await self.world.schedule(group, clock)
            schedule = self.storage.save_round_schedule(group, clock, result)
        elif schedule.get("initial_locations"):
            self.storage.publish_world()  # 恢复数据库已提交、文件尚未刷新的位置补充。
        NovelRestoreLog.report("simulation",
            step="场景直接接续，复用已审核体验" if "scene_continuation_version" in schedule else "调度与感知已就绪",
            group=group.group_id, round=clock["round"])
        states = self.storage.event_states()
        wave = []
        remaining = self.storage.config.max_rounds - clock["round"] + 1
        for event in group.events:
            identity = event["task_id"]
            if states[identity]["status"] != "running":
                continue
            if len(wave) == remaining:
                break
            person = schedule["actors"][identity]
            position = StoryPosition(node_id=group.node_id, node_order=group.node_order, group_order=group.order,
                                     event_id=identity, round=clock["round"] + len(wave), turn=clock["batch"])
            wave.append({"action_id": task_id("action", identity, position.round, position.turn),
                         "character_id": person, "position": position.model_dump(),
                         "arrangement": schedule["arrangements"].get(identity)})
        ready = self.storage.get_checkpoint("round_context")
        key = [group.group_id, clock["round"]]
        if ready != key:
            people = sorted({r["character_id"] for r in wave if r["character_id"] is not None})
            if self.storage.get_checkpoint("round_perception") != key:
                if "scene_continuation_version" in schedule:
                    # 上批获准的最终体验已经是当前输入；没有额外世界变化，
                    # 不请求模型换措辞再生成相同现场。新场景仍走正常感知。
                    saved_views = self.storage.get_checkpoint("views", {})
                    views = {p: WorldPerception.model_validate(saved_views[p]) for p in people}
                else:
                    views = await self.world.perceive(group, people, clock) if people else {}
                current = self.storage.get_checkpoint("views", {})
                current.update({p: v.model_dump(mode="json") for p, v in views.items()})
                self.storage.set_checkpoints({"views": current, "round_perception": key})
            newcomers = [p for p in people if not self.storage.get_checkpoint("subjective_initialized/" + p)]
            runtimes = self.storage.get_checkpoint("person_runtime", {})
            async def initialize(person):
                actor = self.characters[person]
                perception = WorldPerception.model_validate(self.storage.get_checkpoint("views")[person]).personal
                profile = actor.profile
                if any(effect["stage"] == "retrieval" for effect in runtimes.get(person, {}).get("effects", [])):
                    access = await prepare_experience(self.world, task_id("initial_access", *key, person),
                                                      [person])
                    profile = access[person].profile
                return await actor.initialize_subjective(group.group_id, perception, profile=profile)

            initialized = await gather_batch([initialize(person) for person in newcomers])
            # 所有人完成前不改变工作状态；失败时已交付任务可回放。
            checkpoints = {}
            for person, mental in zip(newcomers, initialized):
                checkpoints["character/" + person] = mental.model_dump()
                checkpoints["subjective_initialized/" + person] = True
            self.storage.set_checkpoints(checkpoints)
            if self.storage.get_checkpoint("round_access") != key:
                access = await prepare_experience(self.world, group.group_id + "/" + str(clock["round"]) + "/before", people)
                checkpoints = {"access/" + person: value.model_dump() for person, value in access.items()}
                checkpoints["round_access"] = key
                self.storage.set_checkpoints(checkpoints)
            self.storage.set_checkpoint("round_context", key)
            self.storage.publish_characters()
        for row in wave:
            person = row["character_id"]
            row["context"] = self.characters[person].context(StoryPosition.model_validate(row["position"])) if person else {}
            row["before_view"] = self.storage.get_checkpoint("views", {}).get(person)
        return wave

    async def simulate_group(self, group):
        events = {event["task_id"]: event for event in group.events}
        while True:
            wave = self.storage.get_checkpoint("wave")
            if not wave:
                states = self.storage.event_states()
                if any(states[e]["status"] in ("failed", "timed_out") for e in events):
                    return False
                completed = all(states[e]["status"] == "completed" for e in events)
                goals = self.storage.get_checkpoint("goal_records", {})
                if completed and goals_satisfied(goals):
                    assessment = self.storage.group_assessment(group.group_id)
                    achieved = bool(assessment and assessment["version"] == self.storage.current()[0]
                                    and assessment["achieved"])
                    if not achieved:
                        achieved = await self.world.assess_group(group)
                    if achieved:
                        self.storage.archive_group(group.group_id)
                        return True
                clock = self.storage.get_checkpoint("group_clock")
                if clock["round"] == 1 and self.storage.config.max_rounds < len(events):
                    self.storage.stop_group("轮数上限不足以让本组全部事件同时开始")
                    return False
                stop_reason = None
                if clock["round"] > self.storage.config.max_rounds:
                    stop_reason = "组模拟轮数已用尽"
                elif goal_exhausted(self.storage.get_checkpoint("goal_records", {}), self.storage.config):
                    stop_reason = "有限探索后仍无已知可行推进路径；保留未完成线，不认定目标已证明不可达"
                if stop_reason:
                    # The limit bounds exploration, not the truth of a goal.
                    # Reuse this version's approved terminal assessment, or
                    # obtain it before recording the unfinished branch.
                    assessment = self.storage.group_assessment(group.group_id)
                    if not assessment or assessment["version"] != self.storage.current()[0]:
                        await self.world.assess_group(group)
                    self.storage.stop_group(stop_reason)
                    return False
                if completed:
                    # 事件已发生但状态/承接仍缺条件，不为重新调度额外请求整组验收。
                    self.storage.continue_group(group)
                wave = await self.next_wave(group)
                if not wave:
                    continue
                self.storage.set_checkpoint("wave", wave)
            if any((r["position"]["node_id"], r["position"]["group_order"]) != (group.node_id, group.order) for r in wave):
                raise ValueError("保存的行动批次不属于当前事件组")

            async def propose(row):
                if row["character_id"] is None:
                    return ActionProposal(character_id=None, mode="action", content=row["arrangement"]["content"])
                actor = self.characters[row["character_id"]]
                proposal = await actor.propose_action(row["context"], recall=personal_lookup(self.world, row))
                # 获准的实际回想进入本次行动前材料；恢复完成决策时从同一 checkpoint 还原。
                delivered = actor.context(StoryPosition.model_validate(row["position"]))
                if delivered != row["context"]:
                    row["decision_context"] = delivered
                return proposal

            proposals = await gather_batch([propose(row) for row in wave])
            # 初次请求保持不变，实际交付另存；重放已完成的决策不会改变任务指纹。
            self.storage.set_checkpoint("wave", wave)
            wave = [dict(row) for row in wave]
            for row in wave:
                row["context"] = row.pop("decision_context", row["context"])
            actions, mental, goal, assessment = await self.resolve_wave(group, wave, proposals)
            self.storage.commit_wave(actions, mental, goal, assessment)
            NovelRestoreExporter(self.storage).export_simulation()
            NovelRestoreLog.report("simulation", step="本轮原子提交，已执行文件导出", group=group.group_id,
                                   round=max(a.position.round for a in actions), actions=len(actions))

    async def resolve_wave(self, group, wave, proposals):
        """同批前态 → 世界裁定 → 主观更新 → 进展审核 → 完整待提交交付。"""
        if not wave or len(wave) != len(proposals) or len({r["action_id"] for r in wave}) != len(wave):
            raise ValueError("行动批次须非空、身份唯一，并与人物提议逐项对应")
        version, state, _ = self.storage.current()
        runtime = runtime_context(self.world, group)
        before_views = runtime["views"]
        wave_id = task_id(group.group_id, wave[0]["position"]["round"], wave[0]["position"]["turn"])
        shared_id = task_id("shared", group.group_id, wave[0]["position"]["round"], wave[0]["position"]["turn"])
        rows = {row["action_id"]: (row, proposal) for row, proposal in zip(wave, proposals)}
        result, unknown_review = await self.world.adjudicate_wave(group, wave, proposals, runtime)
        mental, access, summaries, approval = await self.update_characters(
            wave_id, shared_id, proposals, result, rows, runtime)
        unresolved = [r for r in result.actions if any(m.actual_kind == "unknown" for m in r.resolved_targets)]
        actions = []
        for resolved in result.actions:
            row, proposal = rows[resolved.action_id]
            contributions = [m for summary in summaries for m in summary.memories if m.source_action_id == resolved.action_id]
            actions.append(ConfirmedAction(action_id=resolved.action_id, run_id=self.storage.run_id,
                expected_version=version, position=StoryPosition.model_validate(row["position"]), proposal=proposal,
                recollections_before=RecallExperience.from_action_context(row["context"]),
                working_state_before=CharacterState.story_state_from_context(row["context"]),
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
            working_states={p: mental[p] for p in result.advance.observations}, final_views=resolved_views(result),
            memories=[MemoryRecord(**m.model_dump(mode="json"), memory_id=task_id(shared_id, i),
                run_id=self.storage.run_id, action_id=None, position=shared_position)
                for i, m in enumerate(shared_memories, 1)])
        assessment, review_job = await self.world.assess_wave(group, wave_id, state, actions, mental, runtime)
        for action in actions:
            action.completion = assessment.events[action.position.event_id]
        self.storage.validate_wave(actions, mental)
        seal_review(self.storage, "wave", wave_delivery(actions, mental, assessment.goal, assessment), review_job)
        return actions, mental, assessment.goal, assessment

    async def update_characters(self, wave_id, shared_id, proposals, result, rows, runtime):
        """交付获准体验，返修相应人物的主观更新；不修改已裁定的客观结果。"""
        version, state, _ = self.storage.current()
        before_views = runtime["views"]
        runtimes = {**runtime["person_runtime"], **{p: c.state.model_dump() for p, c in result.advance.runtime_changes.items()}}
        states = {p: self.storage.character_state(p).model_dump()
                  for p in self.storage.manifest["catalog"]["role"]}
        for proposal in proposals:
            if proposal.character_id is not None:
                states[proposal.character_id]["intention"] = proposal.purpose
                record_visible(self.storage, proposal.character_id,
                    {"working_state": states[proposal.character_id]}, "selected-personal-plan/" + wave_id)
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
        catalog = effective_catalog(self.storage, result.advance.new_objects)
        access = await prepare_experience(self.world, wave_id + "/after", people, observations=observations,
                                          runtimes=runtimes, states=states, catalog=catalog,
                                          acted=[p.character_id for p in proposals if p.character_id]) if people else {}
        summary_by_person, repair_people, coverage = {}, people, {}
        attempts = self.supervisor.semantic_attempts()
        for revision in range(attempts):
            if not people:
                summaries, approval = [], MemoryApproval(approved=True, notes=["本批无个人体验，无新增主观更新。"])
                break
            repaired = await gather_batch([self.characters[person].summarize_wave(
                wave_id, access[person], revision, feedback.get(person, []),
                catalog=catalog, recovery=self.supervisor.semantic_recovery(revision)) for person in repair_people])
            summary_by_person.update(zip(repair_people, repaired))
            summaries = [summary_by_person[p] for p in people]
            # 审核新记忆是否承接已放行体验；保留本批客观结果以区分误解与真实事实，
            # 不重复发送全部旧行动。完整候选、本人回想与认知条件仍直接提供。
            review_context = {"operation": "review_memory",
                 "observations": {p: observations[p] for p in repair_people},
                 "state_before": state.model_dump(mode="json"), "resolution": result.model_dump(mode="json"),
                 "subjective_worlds": {p: runtime["subjective_worlds"][p] for p in repair_people},
                 "character_assets": {p: value for p, value in runtime["environment"]["assets"]["role"].items() if p in repair_people},
                 "views_before": {p: value for p, value in before_views.items() if p in repair_people},
                 "person_runtime": {p: value for p, value in runtimes.items() if p in repair_people},
                 "retrieval_scope": runtime["retrieval_scope"],
                 "subjective_inputs": {p: access[p].model_dump() for p in repair_people},
                 "summaries": {p: summary_by_person[p].model_dump(mode="json") for p in repair_people}}
            # 已通过的其他人物候选及全部依据均未改变；此次仅审核返修人物。
            review_job = task_id("world", "review_memory", wave_id, revision)
            approval = await self.supervisor.run_task(review_job,
                "world", self.world.prompt, review_context, MemoryApproval)
            located = [p for p in repair_people if approval.character_issues.get(p)]
            unlocated = (set(approval.character_issues) - set(repair_people)
                         or set(approval.issues) - {issue for items in approval.character_issues.values() for issue in items})
            failed = set() if approval.approved else set(repair_people if unlocated or not located else located)
            for person in repair_people:
                if person in failed:
                    coverage.pop(person, None)
                else:
                    coverage[person] = {"review_job": review_job,
                        "candidate": fingerprint(summary_by_person[person].model_dump(mode="json"))}
            if approval.approved:
                break
            # 定位明确时只重写并重审相关人物；其他人的候选和依赖未变。
            # 无法定位的整批问题保留全员返修，不把未点名当作通过。
            repair_people = sorted(failed)
            located_issues = {issue for items in approval.character_issues.values() for issue in items}
            general_issues = [issue for issue in approval.issues if issue not in located_issues]
            feedback = {p: list(dict.fromkeys([*approval.character_issues.get(p, []), *general_issues]))
                        or ["请核对本人收到的体验与记忆引用，不借用未获知的信息。"]
                        for p in repair_people}
            if revision < attempts - 1:
                def check_feedback(value):
                    if set(value.messages) != set(repair_people):
                        raise ValueError("返修提示须恰好覆盖本次需要返修的人物")
                safe = await visible_task(self.supervisor, task_id("world", "feedback", wave_id, revision), self.world.prompt,
                    {"operation": "repair_character_feedback", "review": approval.model_dump(),
                     "subjective_inputs": {p: access[p].model_dump() for p in repair_people}},
                    CharacterFeedback, check_feedback,
                    lambda value: {p: {"feedback": messages} for p, messages in value.messages.items()},
                    initial=CharacterFeedback(messages=feedback))
                feedback = safe.messages
                # safe 已按本人的实际输入独立审查；包括有认知作用的情况。
                # 不再用无定位的固定句覆盖已获准的具体返修意见。
        else:
            raise JobFailed("人物主观世界未通过检查：" + "; ".join(approval.issues))
        if set(coverage) != set(people):
            raise ValueError("主观更新缺少覆盖全部人物的独立审核")
        self.storage.set_checkpoint("memory_coverage/" + wave_id, {
            "version": version, "people": coverage})
        mental = {**states, **{p: s.working_state.model_dump() for p, s in zip(people, summaries)}}
        return mental, access, summaries, approval

