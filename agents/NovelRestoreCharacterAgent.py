"""一名人物绑定一份档案和动态记忆，不持有全知世界上下文。"""
from hashlib import sha256
from pathlib import Path
import random

from pydantic import Field, field_validator

from agents.NovelRestoreAgent import task_id
from common.Runtime import JobFailed
from common.NovelRestoreVisibility import is_visible, record_visible
from common.NovelRestoreObjects import effective_catalog
from common.NovelRestoreSchema import (
    CharacterDecision, CharacterMemories, MemorySummary, CharacterState, PersonalPerception, Record, SubjectiveInput,
)


class RecallQuery(Record):
    query: str | None = Field(max_length=240)

    @field_validator("query")
    @classmethod
    def meaningful(cls, value):
        return (value.strip() or None) if value is not None else None


def personal_recalls(memories):
    """短编号只在本次交付内有效，不编码说话者、历史行动或来源身份。"""
    return [{"memory_id": f"r{i}", "content": m.content, "source": m.source}
            for i, m in enumerate(memories, 1)]


class NovelRestoreCharacterAgent:
    def __init__(self, character_id, storage, executor):
        self.character_id, self.storage, self.executor = character_id, storage, executor
        self.prompt = (Path(__file__).resolve().parents[1] / "prompts/NovelRestoreCharacterPrompt.md").read_text()

    @property
    def profile(self):
        return self.storage.get_checkpoint("self_profiles", {}).get(self.character_id, {})

    @staticmethod
    def memory_view(records):
        """保留个人内容和检索引用，不把全局事件位置、轮次或运行身份交给人物。"""
        hidden = {"run_id", "action_id", "source_action_id", "position"}
        return [{k: v for k, v in (r.model_dump(mode="json") if hasattr(r, "model_dump") else r).items()
                 if k not in hidden} for r in records]

    async def recall_query(self, identity, access, perception):
        context = {"operation": "recall_query", "character_id": self.character_id, "profile": access.profile,
                   "perception": perception, "working_state": access.working_state.model_dump(), "observations": []}
        result = await self.executor.run_task(task_id("recall_query", identity, self.character_id), self.role,
                                              self.prompt, context, RecallQuery)
        return result.query

    async def initialize_subjective(self, identity, perception):
        def validate(state):
            if not state.needs:
                raise ValueError("初始化须形成本人的需求，不用剧情目标代替")
        return await self.executor.run_task(task_id("subjective_init", identity, self.character_id), self.role,
            self.prompt, {"operation": "initialize_subjective", "character_id": self.character_id,
                          "profile": self.profile,
                          "perception": perception.model_dump()}, CharacterState, validate)

    def context(self, position):
        ratios = self.storage.config.character_action_ratios.get(self.character_id, self.storage.config.action_ratios)
        randomizer = random.Random(task_id(self.storage.config.seed, self.character_id,
                                          position.event_id, position.round, position.turn))
        modes = ["action", "talk", "think", "idle"]
        mode = randomizer.choices(modes, weights=[ratios[key] for key in modes])[0]
        access = SubjectiveInput.model_validate(self.storage.get_checkpoint("access/" + self.character_id,
            {"working_state": self.storage.character_state(self.character_id).model_dump()}))
        working = access.working_state.model_dump()
        view = self.storage.get_checkpoint("views", {}).get(self.character_id, {}).get("personal")
        perception = PersonalPerception.model_validate(view)
        return {"character_id": self.character_id, "profile": access.profile if access.profile is not None else self.profile,
                "position": position.model_dump(),
                "perception": perception.model_dump(), "mode": mode, "working_state": working,
                "memories": personal_recalls(access.memories)}

    @property
    def role(self):
        return "character_" + sha256(self.character_id.encode()).hexdigest()[:12]

    async def summarize_wave(self, wave_id, access, revision=0, feedback=(), *, catalog=None):
        observations = [{"source_id": e.source_id, "source_kind": e.source_kind,
                         "observation": e.experience.model_dump()}
                        for e in access.experiences if e.experience is not None]
        if not observations:
            return CharacterMemories(working_state=access.working_state)
        if not access.working_state.needs:
            access.working_state = await self.initialize_subjective(
                wave_id + "/observed", PersonalPerception.model_validate(observations[0]["observation"]))
        if not is_visible(self.storage, self.character_id, "observations", observations):
            raise JobFailed("个人体验尚未通过知情边界检查")
        sources = {f"o{i}": row["source_id"] for i, row in enumerate(observations, 1)}
        observations = [{**row, "source_id": alias} for alias, row in zip(sources, observations)]
        record_visible(self.storage, self.character_id, {"observations": observations}, "approved/observation-alias")
        allowed = {row["source_id"]: row["source_kind"] for row in observations}
        references = {f"r{i}": m.memory_id for i, m in enumerate(access.memories, 1)}
        catalog = catalog if catalog is not None else effective_catalog(self.storage)
        objects = {row["source_id"]: {obj["perceived_id"] for obj in row["observation"]["objects"]}
                   for row in observations}
        def validate(result):
            if not result.working_state.needs or set(access.working_state.needs) - set(result.working_state.needs):
                raise ValueError("更新须保留已有需求；已满足可将紧迫程度降为零")
            for memory in result.memories:
                source = memory.source_action_id or memory.source_shared_id
                if (memory.character_id != self.character_id or not source or
                        bool(memory.source_action_id) == bool(memory.source_shared_id) or
                        allowed.get(source) != ("action" if memory.source_action_id else "shared")):
                    raise ValueError("记忆须属于本人并准确引用行动或共享体验的来源")
                if set(memory.perceived_ids) - objects[source]:
                    raise ValueError("记忆感知对象必须来自本人的这次体验")
                if set(memory.corrects + memory.related_memory_ids) - references.keys():
                    raise ValueError("记忆关联与纠正只能引用本人已存在的记忆")
                people = memory.role_ids + memory.listener_ids + ([memory.speaker_id] if memory.speaker_id else [])
                if (set(people) - set(catalog["role"]) or set(memory.object_ids) - set(catalog["object"])
                        or set(memory.location_ids) - set(catalog["location"])):
                    raise ValueError("记忆引用了未知人物、物品或地点")
        result = await self.executor.run_task(task_id("memory", wave_id, self.character_id, revision), self.role,
            self.prompt, {"operation": "summarize_memory", "character_id": self.character_id, "profile": access.profile,
                "observations": observations, "feedback": list(feedback),
                "working_state": access.working_state.model_dump(),
                "memories": personal_recalls(access.memories)}, CharacterMemories, validate)

        # 人物只引用本次短编号，持久化时再绑定真正来源，回放使用相同的映射。
        for m in result.memories:
            if m.source_action_id:
                m.source_action_id = sources[m.source_action_id]
            if m.source_shared_id:
                m.source_shared_id = sources[m.source_shared_id]
            m.corrects = [references[i] for i in m.corrects]
            m.related_memory_ids = [references[i] for i in m.related_memory_ids]
        return result

    async def summarize_history(self, identity, records, summaries=(), revision=0, feedback=()):
        allowed = {r["memory_id"] for r in records} | {i for s in summaries for i in s["memory_ids"]}
        def validate(result):
            if len(result.memory_ids) != len(set(result.memory_ids)) or set(result.memory_ids) - allowed:
                raise ValueError("摘要只能引用本人已提供的原始记忆")
        return await self.executor.run_task(task_id("summary", identity, self.character_id, revision), "world",
            "按人物当时的认识归档既有主观历史，保留误解和不确定，不补事实。输出带来源的摘要；这不是人物当前回想。",
            {"operation": "summarize_history", "character_id": self.character_id,
                "records": self.memory_view(records),
                "summaries": [{k: s[k] for k in ("text", "memory_ids")} for s in summaries],
                "feedback": list(feedback)}, MemorySummary, validate)

    async def propose_action(self, context):
        position = context["position"]
        def validate(decision):
            needs = context["working_state"]["needs"]
            # 只比较本轮抽中的行为类型；完整原始响应仍在任务和请求账本中。
            candidates = [candidate for candidate in decision.candidates if candidate.proposal.mode == context["mode"]]
            if not candidates:
                raise ValueError("候选行动必须属于本人且采用本次抽取的方式")
            for candidate in candidates:
                proposal = candidate.proposal
                if proposal.character_id != self.character_id:
                    raise ValueError("候选行动必须属于本人且采用本次抽取的方式")
                if set(proposal.target_ids) - {o["perceived_id"] for o in context["perception"]["objects"]}:
                    raise ValueError("行动只能引用当前个人感知中的对象")
                if set(candidate.need_changes) != set(needs):
                    raise ValueError("每项候选须预测全部本人需求的变化")
            decision.candidates = candidates
        decision = await self.executor.run_task(
            task_id("character", position["event_id"], position["round"], position["turn"]), self.role,
            self.prompt, {k: v for k, v in context.items() if k != "position"}, CharacterDecision, validate)
        # urgency 越高越迫切；只比较本人预想的满足程度，完全不使用剧情完成度。
        needs = context["working_state"]["needs"]
        def utility(candidate):
            return sum(need["urgency"] ** 2 - min(10, max(0,
                need["urgency"] + candidate.need_changes[name])) ** 2 for name, need in needs.items())
        return max(decision.candidates, key=utility).proposal
