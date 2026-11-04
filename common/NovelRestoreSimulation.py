"""按故事顺序模拟；同组人物共享批前状态，整批结果原子提交。"""
import asyncio

from agents.NovelRestoreAgent import task_id
from agents.NovelRestoreCharacterAgent import NovelRestoreCharacterAgent
from agents.NovelRestoreWorldAgent import NovelRestoreWorldAgent
from common.NovelRestoreSchema import StoryPosition, ActionProposal, WorldPerception
from common.NovelRestoreExperience import prepare_experience
from common.NovelRestoreLookup import WorldLookup
from exporters.NovelRestoreExporter import NovelRestoreExporter


class NovelRestoreSimulation:
    def __init__(self, storage, repository, executor):
        self.storage, self.repository = storage, repository
        executor.world_lookup = WorldLookup(storage, repository)
        self.world = NovelRestoreWorldAgent(storage, repository, executor)
        self.characters = {identity: NovelRestoreCharacterAgent(identity, storage, executor)
                           for identity in storage.manifest["catalog"]["role"]}

    async def run(self):
        self.storage.recover()
        NovelRestoreExporter(self.storage).export_story_progress()
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
                if (group.order == 1 and self.storage.config.max_timelines > 1
                        and node["node_id"] != self.storage.manifest.get("fork_node_id")
                        and await self.world.consider_branch(group)):
                    self.storage.mark_stage("branch_ready")
                    return True
                if not await self.simulate_group(group):
                    self.storage.mark_stage("incomplete", "当前事件组未满足全部事件及世界目标")
                    return False
                NovelRestoreExporter(self.storage).export_story_progress()
                if group.group_id == self.storage.config.stop_after_group:
                    self.storage.mark_stage("group_complete")
                    return True
        self.storage.mark_stage("simulation_complete")
        return True

    async def next_wave(self, group):
        clock = self.storage.get_checkpoint("group_clock")
        schedule = self.storage.get_checkpoint("round_schedule")
        if not schedule or (schedule["group_id"], schedule["round"]) != (group.group_id, clock["round"]):
            result = await self.world.schedule(group, clock)
            schedule = self.storage.save_round_schedule(group, clock, result)
        elif schedule.get("initial_locations"):
            self.storage.publish_world()  # 恢复数据库已提交、文件尚未刷新的位置补充。
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
                views = await self.world.perceive(group, people, clock) if people else {}
                with self.storage.connection:
                    current = self.storage.get_checkpoint("views", {})
                    current.update({p: v.model_dump(mode="json") for p, v in views.items()})
                    self.storage._put("checkpoints", "views", current)
                    self.storage._put("checkpoints", "round_perception", key)
            for person in people:
                if not self.storage.get_checkpoint("subjective_initialized/" + person):
                    perception = WorldPerception.model_validate(self.storage.get_checkpoint("views")[person]).personal
                    mental = await self.characters[person].initialize_subjective(group.group_id, perception)
                    with self.storage.connection:
                        self.storage._put("checkpoints", "character/" + person, mental.model_dump())
                        self.storage._put("checkpoints", "subjective_initialized/" + person, True)
            if self.storage.get_checkpoint("round_access") != key:
                access = await prepare_experience(self.world, group.group_id + "/" + str(clock["round"]) + "/before", people)
                with self.storage.connection:
                    for person, value in access.items():
                        self.storage._put("checkpoints", "access/" + person, value.model_dump())
                    self.storage._put("checkpoints", "round_access", key)
            with self.storage.connection:
                self.storage._put("checkpoints", "round_context", key)
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
                if all(states[e]["status"] == "completed" for e in events):
                    assessment = self.storage.group_assessment(group.group_id)
                    achieved = bool(assessment and assessment["achieved"])
                    if not achieved:
                        achieved = await self.world.assess_group(group)
                    if achieved:
                        self.storage.archive_group(group.group_id)
                        await self.world.summarize_group(group)
                        return True
                    if not any(self.storage.event_states()[e]["status"] == "running" for e in events):
                        return False
                clock = self.storage.get_checkpoint("group_clock")
                if clock["round"] == 1 and self.storage.config.max_rounds < len(events):
                    self.storage.stop_group("轮数上限不足以让本组全部事件同时开始")
                    return False
                if clock["round"] > self.storage.config.max_rounds:
                    self.storage.stop_group("组模拟轮数已用尽")
                    return False
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
                return await actor.propose_action(row["context"])

            proposals = await asyncio.gather(*(propose(row) for row in wave), return_exceptions=True)
            for proposal in proposals:
                if isinstance(proposal, BaseException):
                    raise proposal
            actions, mental, goal, progress = await self.world.resolve_wave(group, wave, proposals)
            self.storage.commit_wave(actions, mental, goal, progress)
            NovelRestoreExporter(self.storage).export_story_progress()
