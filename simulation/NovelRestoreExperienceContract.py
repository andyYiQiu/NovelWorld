"""认知处理只交付有据变化，未受影响的完整输入由程序沿用。"""
from copy import deepcopy
from pydantic import AliasChoices, BaseModel, Field, create_model, model_validator

from common.NovelRestoreSchema import (
    CharacterState, Identity, MemorySource, PersonalPerception, ProcessingContent, Record,
)
from common.Runtime import CandidateError


class RetrievalProcessing(ProcessingContent):
    source_ids: list[Identity] = Field(min_length=1,
        description="只引用本次 hits 中实际受到提取作用影响的 memory_id。")


class ExperienceProcessing(ProcessingContent):
    source_ids: list[Identity] = Field(min_length=1,
        description="encoding 引用本人物 observations 的 source_id；retrieval 仅引用 profile 或 working_state。")


class ExperienceChange(Record):
    source_id: Identity = Field(validation_alias=AliasChoices("source_id", "action_id"))
    experience: PersonalPerception | None = Field(description="该来源的实际编码结果；显式 null 表示未形成体验，须有编码作用依据。")

    @model_validator(mode="before")
    @classmethod
    def bind_legacy_kind(cls, value):
        # The source category is immutable metadata, always rebound from input.
        return {k: v for k, v in value.items() if k != "source_kind"} if isinstance(value, dict) else value


class ProfileText(Record):
    facts: list[Identity]


class ExperienceChanges(Record):
    working_state: CharacterState | None = Field(default=None, description="仅交付受提取作用影响的心理；省略或 null 沿用原值，不是人物总结新体验。")
    profile: ProfileText | None = Field(default=None, description="仅交付受提取作用影响的本人认识文字；省略或 null 沿用。")
    experiences: list[ExperienceChange] = Field(default_factory=list,
        description="只列改变或抑制的当次体验；未列来源由程序完整沿用，不抄写未受影响内容。")
    processing: list[ExperienceProcessing] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def allow_legacy_empty_memories(cls, value):
        # This stage never retrieves history. Only the old empty list is equivalent.
        if isinstance(value, dict) and value.get("memories") == []:
            return {k: v for k, v in value.items() if k != "memories"}
        return value


class ExperienceDelivery(Record):
    people: dict[Identity, ExperienceChanges] = Field(default_factory=dict,
        description="只列有认知处理变化的人物；未列人物及未列内容由程序沿用，全部结果仍独立审核。")


class RecallChange(Record):
    memory_id: Identity
    content: Identity | None = Field(default=None, description="受提取作用影响后的回想；省略沿用命中原文。")
    source: MemorySource | None = Field(default=None, description="受提取作用影响后的来源认识；省略沿用。")
    omitted: bool = Field(default=False, strict=True, description="true 表示本次无法取回，须有提取作用依据；不可同时交付 content/source。")

    @model_validator(mode="after")
    def unambiguous_omission(self):
        if self.omitted and (self.content is not None or self.source is not None):
            raise ValueError("抑制回想不能同时交付其内容或来源；只选择 omitted=true")
        return self


class RecallDelivery(Record):
    changes: list[RecallChange] = Field(default_factory=list,
        description="只列本次 hits 中实际改变或抑制的记忆，未列项完整沿用。")
    processing: list[RetrievalProcessing] = Field(default_factory=list)


def _invalid(location, message):
    raise CandidateError(message, [{"loc": location, "msg": message, "type": "candidate_error"}])


def experience_contract(schema, context):
    from simulation.NovelRestoreExperience import ExperienceBatch, RecallResult

    if schema is ExperienceBatch:
        def assemble(payload):
            value = ExperienceDelivery.model_validate(payload.model_dump() if isinstance(payload, BaseModel) else payload)
            sources = context["people"]
            for person in value.people.keys() - sources.keys():
                _invalid(("result", "people", person), f"人物 {person} 不在本次 people 中")
            people = {}
            for person, source in sources.items():
                change = value.people.get(person, ExperienceChanges())
                experiences = {row["source_id"]: {"source_id": row["source_id"], "source_kind": row["source_kind"],
                    "experience": deepcopy(row["observation"])} for row in source["observations"]}
                seen = set()
                for index, item in enumerate(change.experiences):
                    if item.source_id not in experiences or item.source_id in seen:
                        _invalid(("result", "people", person, "experiences", index, "source_id"),
                                 f"人物 {person} 的体验来源须唯一选自本次 observations：{list(experiences)}")
                    seen.add(item.source_id)
                    experiences[item.source_id]["experience"] = item.experience.model_dump() if item.experience else None
                people[person] = {"working_state": change.working_state.model_dump() if change.working_state is not None
                                 else deepcopy(source["working_state"]),
                                 "profile": change.profile.model_dump() if change.profile is not None else deepcopy(source["profile"]),
                                 "experiences": list(experiences.values()), "processing": []}
                effects = {effect["effect_id"]: effect["stage"] for effect in source.get("runtime", {}).get("effects", [])}
                for index, trace in enumerate(change.processing):
                    stages = {effects.get(identity) for identity in trace.effect_ids}
                    if len(stages) != 1 or not stages <= {"encoding", "retrieval"}:
                        _invalid(("result", "people", person, "processing", index, "effect_ids"),
                                 "认知处理须引用此人当前生效且阶段相同的编码或提取作用")
                    people[person]["processing"].append({**trace.model_dump(), "stage": next(iter(stages))})
            return schema(people=people)
        return ExperienceDelivery, assemble

    if schema is RecallResult:
        hits = {row["memory_id"]: row for row in context.get("hits", [])}

        def legacy_delivery(cls, value):
            # Old memories was a complete list. Missing hits meant suppression;
            # preserve that meaning explicitly, then run the unchanged safeguards.
            if isinstance(value, dict) and "changes" not in value and isinstance(value.get("memories"), list):
                seen = {row.get("memory_id") for row in value["memories"] if isinstance(row, dict)}
                return {**{k: v for k, v in value.items() if k != "memories"},
                        "changes": [*value["memories"], *({"memory_id": identity, "omitted": True}
                                                        for identity in hits if identity not in seen)]}
            return value

        payload_type = create_model("RecallResultInput", __base__=RecallDelivery,
            __validators__={"legacy_delivery": model_validator(mode="before")(legacy_delivery)})

        def assemble(payload):
            if "hits" not in context:
                raise ValueError("回想装配缺少本次 hits，不能猜测原始记忆")
            value = payload_type.model_validate(payload.model_dump() if isinstance(payload, BaseModel) else payload)
            memories, seen = deepcopy(hits), set()
            for index, change in enumerate(value.changes):
                if change.memory_id not in hits or change.memory_id in seen:
                    _invalid(("result", "changes", index, "memory_id"), "回想更新须唯一引用本次 hits 的 memory_id")
                seen.add(change.memory_id)
                if change.omitted:
                    del memories[change.memory_id]
                else:
                    memories[change.memory_id].update(change.model_dump(exclude={"memory_id", "omitted"}, exclude_none=True))
            return schema(memories=list(memories.values()),
                          processing=[{**trace.model_dump(), "stage": "retrieval"} for trace in value.processing])
        return payload_type, assemble
    return None
