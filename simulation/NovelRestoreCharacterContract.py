"""人物只交付选择与主观变化；程序绑定身份、来源及未变化状态。"""
from copy import deepcopy
from typing import Annotated, Literal

from pydantic import Field, create_model

from tools.ToolArguments import payload_data
from common.Runtime import digest
from common.NovelRestoreSchema import (
    CharacterDecision, CharacterMemories, CharacterState, Identity, MemorySource, Need, ActionMode, Record,
)


class NeedUpdate(Need):
    existing_need: Identity | None = Field(default=None, description="更新已有需求时选择其现有编号；新需求留空，编号由程序分配。")


class MentalUpdate(Record):
    needs: list[NeedUpdate] = Field(default_factory=list, description="一次交付全部有实质变化的需求及新需求；未列出的已有需求沿用。")
    emotion: str | None = None
    beliefs: list[Identity] | None = None
    goals: list[Identity] | None = None
    intention: str | None = None


class MentalInitialization(MentalUpdate):
    needs: list[NeedUpdate] = Field(min_length=1,
        description="初始化须交付至少一项当下有依据的本人需求；更新任务才可为空并沿用已有需求。")


class MemoryContent(Record):
    observation_id: Identity | None = Field(default=None, description="选择本次 observations 中的来源；只有一份体验时可省略。")
    kind: Literal["event", "chat", "thought"]
    content: Identity
    source: MemorySource
    importance: Annotated[int, Field(ge=1, le=5, strict=True)] = 3
    perceived_ids: list[Identity] = Field(default_factory=list,
        description="只引用 observation_id 所选体验的 observation.objects[].perceived_id，不能跨体验借用。")
    related_memory_ids: list[Identity] = Field(default_factory=list,
        description="只引用本次 memories 中本人已获得的 memory_id；不是世界行动编号。")
    corrects: list[Identity] = Field(default_factory=list,
        description="本条认识纠正的旧记忆，只取本次 memories 中的 memory_id；没有则为空。")
    occurred_at: str | None = None


class MemoryDelivery(Record):
    memories: list[MemoryContent] = Field(default_factory=list)
    working_state: MentalUpdate = Field(default_factory=MentalUpdate)


def _mental(value, original, job):
    update = MentalUpdate.model_validate(payload_data(value))
    result = deepcopy(original or {"needs": {}, "emotion": "", "beliefs": [], "goals": []})
    needs, used = result.setdefault("needs", {}), set()
    for index, need in enumerate(update.needs):
        identity = need.existing_need
        values = need.model_dump(exclude={"existing_need"})
        if identity is not None:
            if identity not in needs or identity in used:
                raise ValueError("需求更新必须唯一绑定本人已有需求")
            used.add(identity)
        else:
            # Needs are shown back to their owner. A reversible task path would
            # expose world-only person/event/round metadata through the key.
            identity = "need_" + digest((job, index))
            if identity in needs:
                raise ValueError("新需求身份与已有记录冲突")
        needs[identity] = values
    for name in ("emotion", "beliefs", "goals", "intention"):
        value = getattr(update, name)
        if value is not None:
            result[name] = value
    return CharacterState.model_validate(result)

def mental_contract(schema, context, owner, job):
    base = MentalInitialization if context.get("operation") == "initialize_subjective" else MentalUpdate
    payload_type = base
    return payload_type, lambda value: _mental(payload_type.model_validate(payload_data(value)), context.get("working_state"), job)

def decision_contract(schema, context, owner, job):
    targets = {o["perceived_id"] for o in context["perception"]["objects"]}
    config = owner.config
    candidate = create_model("PersonalActionChoice", __base__=Record,
        mode=(ActionMode, Field(description="本段主要行为性质；动作及伴随言语可连续发生，私有思考须在正文中区分。")),
        purpose=(Identity, Field(pattern=r"\S", description="本人此次行为的目的。")),
        stop_before=(Identity, Field(pattern=r"\S", description="预期需要本人重新选择或等待他人回应的位置。")),
        content=(Identity, Field(description="一次具体连贯的本人尝试及完整言语；不预定成功或替别人决定。")),
        target_ids=(list[Identity], Field(default_factory=list, max_length=len(targets),
            description="只选 perception.objects 中已有的 perceived_id，且不得重复；没有对象则为 []。不是实际人物或资产编号。")),
        expected_outcome=(Identity, Field(description="按本人认识预计会发生什么，尚未实际发生。")),
        tradeoffs=(Identity, Field(description="本人预期的需要满足、代价、风险、不确定条件及替代办法；不作数值评分。")))
    payload_type = create_model("CharacterDecisionInput", __base__=Record,
        candidates=(list[candidate], Field(min_length=1, max_length=config.plan_candidates)),
        selected_index=(int, Field(ge=0, le=config.plan_candidates - 1, strict=True,
            description="本人选定的 candidates 下标，从0开始，必须指向本次实际交付的候选；程序不评分或改选。")),
        selection_reason=(Identity, Field(description="本人的选择理由及愿意承担的代价，不受世界剧情目标控制。")))

    def assemble(payload):
        value = payload_type.model_validate(payload_data(payload))
        result = []
        for row in value.candidates:
            if set(row.target_ids) - targets or len(row.target_ids) != len(set(row.target_ids)):
                raise ValueError("行动对象必须唯一来自本人当前感知")
            result.append({"proposal": {"character_id": "self", "mode": row.mode,
                "purpose": row.purpose, "stop_before": row.stop_before,
                "content": row.content, "target_ids": row.target_ids},
                "expected_outcome": row.expected_outcome, "tradeoffs": row.tradeoffs})
        return schema(candidates=result, selected_index=value.selected_index, selection_reason=value.selection_reason)
    return payload_type, assemble

def memory_contract(schema, context, owner, job):
    observations = {row["source_id"]: row for row in context["observations"]}
    recalled = {row["memory_id"] for row in context.get("memories", [])}
    mental_type = MentalUpdate
    payload_type = create_model("PersonalMemoryDelivery", __base__=MemoryDelivery,
        working_state=(mental_type, Field(default_factory=mental_type)))

    def assemble(payload):
        value = payload_type.model_validate(payload_data(payload))
        memories = []
        for memory in value.memories:
            source = memory.observation_id
            if source is None and len(observations) == 1:
                source = next(iter(observations))
            if source not in observations:
                raise ValueError("记忆必须绑定本次本人实际获得的一份体验")
            observation = observations[source]
            if observation["source_kind"] not in ("action", "shared"):
                raise ValueError("记忆来源类别无效")
            objects = {row["perceived_id"] for row in observation["observation"]["objects"]}
            if set(memory.perceived_ids) - objects:
                raise ValueError("记忆对象必须属于对应的这次体验")
            if set(memory.related_memory_ids + memory.corrects) - recalled:
                raise ValueError("关联或纠正只能引用本人本次已获得的回想")
            source_field = "source_action_id" if observation["source_kind"] == "action" else "source_shared_id"
            memories.append({**memory.model_dump(exclude={"observation_id"}), "character_id": "self",
                             source_field: source})
        return schema(memories=memories, working_state=_mental(value.working_state,
            context["working_state"], job + "/working_state"))
    return payload_type, assemble

def bind_character_result(result, person):
    """人物只用 self 表示本人，在世界侧验证并恢复归属。"""
    if isinstance(result, CharacterDecision):
        owned = [candidate.proposal for candidate in result.candidates]
    elif isinstance(result, CharacterMemories):
        owned = result.memories
    else:
        return
    for row in owned:
        if row.character_id != "self":
            raise ValueError("本人的 character_id 须填写 self")
        row.character_id = person
    if isinstance(result, CharacterMemories):
        for memory in result.memories:
            if (memory.speaker_id not in (None, "self")
                    or any(ref != "self" for ref in memory.role_ids + memory.listener_ids)
                    or memory.object_ids or memory.location_ids):
                raise ValueError("记忆只能用 self 指代本人，感知对象须使用本次体验中的局部编号")
            if memory.speaker_id == "self":
                memory.speaker_id = person
            memory.role_ids = [person for _ in memory.role_ids]
            memory.listener_ids = [person for _ in memory.listener_ids]

