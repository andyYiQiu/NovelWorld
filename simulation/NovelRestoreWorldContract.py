"""世界任务的语义交付；感知对应、行动归属和新对象身份由程序绑定。"""
from typing import Literal
from pydantic import BaseModel, Field, create_model, model_validator

from common.Runtime import digest, encode
from tools.ToolArguments import payload_data
from common.NovelRestoreSchema import (
    Identity, Record, Correspondence, WaveAdvance, WorldPerception, WorldResolution,
    CognitiveEffect, PersonRuntime, RuntimeChange, SimulatedObject, ReplanSchedule,
    EnvironmentAttempt, SceneDecision, OpportunityPlan, InitialLocationChoice, ProcessingContent,
)
from inputs.NovelWorldStateSchema import ObjectDelta, ObjectStatePatch, CharacterStatePatch, ActionTerm


class ExecutionProcessing(ProcessingContent):
    source_ids: list[Identity] = Field(min_length=1, description="只引用本批 intents 中实际受到执行作用影响的 action_id。")


class PerceivedContent(Record):
    description: Identity
    actual_kind: Literal["role", "object", "location", "none", "unknown"]
    actual_id: Identity | None = None
    basis: Identity

    @model_validator(mode="before")
    @classmethod
    def program_owned_label(cls, value):
        # This payload has no references to local labels: _perception allocates
        # every label and both mappings together. A legacy label is not input.
        return {k: v for k, v in value.items() if k != "perceived_id"} if isinstance(value, dict) else value


class PerceptionContent(Record):
    content: Identity
    objects: list[PerceivedContent] = Field(default_factory=list,
        description="逐项描述所见及世界侧对应依据；局部编号及两侧对应表由程序生成。")

    @model_validator(mode="before")
    @classmethod
    def paired_delivery(cls, value):
        # A complete old-style pair can be projected without guessing matching
        # text or identities. Incomplete/ambiguous pairs still fail validation.
        if isinstance(value, dict) and set(value) <= {"personal", "correspondences"} and "personal" in value:
            old = WorldPerception.model_validate(value)
            mappings = {m.perceived_id: m for m in old.correspondences}
            return {"content": old.personal.content, "objects": [
                {"description": obj.description, **mappings[obj.perceived_id].model_dump(exclude={"perceived_id"})}
                for obj in old.personal.objects]}
        return value


def _perception(value, reserved):
    labels = (f"p{i}" for i in range(1, 1_000_000) if f"p{i}" not in reserved)
    personal, mappings = [], []
    for obj in value.objects:
        label = next(labels, None)
        if label is None:
            raise ValueError("本次感知的中性局部编号空间已用尽")
        personal.append({"perceived_id": label, "description": obj.description})
        mappings.append({"perceived_id": label, **obj.model_dump(exclude={"description"})})
    return WorldPerception(personal={"content": value.content, "objects": personal}, correspondences=mappings)


class ActionResolution(WorldResolution):
    observations: dict[Identity, PerceptionContent] = Field(description="选择实际获得体验的人物，交付各自体验内容。")
    resolved_targets: list[Correspondence] = Field(default_factory=list,
        description="恰好覆盖本次 proposal.target_ids 中在行动前视图对应为 unknown 的对象；保留其旧 perceived_id，不列已知目标或无关对象。")


class EffectInput(CognitiveEffect):
    effect_id: Identity = Field(description="保留本人已有作用时引用其现有编号；新作用仅给本批本人内唯一的局部别名，正式身份由程序分配。不能借用他人的已有作用。")


class RuntimeInput(PersonRuntime):
    effects: list[EffectInput] = Field(default_factory=list)


class RuntimeChangeInput(RuntimeChange):
    state: RuntimeInput


class AdvanceInput(WaveAdvance):
    observations: dict[Identity, PerceptionContent] = Field(default_factory=dict)
    runtime_changes: dict[Identity, RuntimeChangeInput] = Field(default_factory=dict)
    new_objects: dict[Identity, SimulatedObject] = Field(default_factory=dict,
        description="本批登记的新普通道具，以本批唯一局部别名为键；不得使用已有目录身份。物品补丁及客观对象引用用同一别名，正式SIM_OBJ身份由程序统一绑定；正文不受替换。仍须先比较已有物品并给实际来源。")
    processing: list[ExecutionProcessing] = Field(default_factory=list)


def _opaque_id(prefix, *parts):
    return prefix + digest(parts)


def _bind_object_references(value, bindings):
    """Visit validated objects; rewrite only explicitly typed object references.

    This is deterministic identity binding, not schema-to-question generation.
    Text, untyped JSON payloads and unrelated identifier fields are never changed.
    """
    if isinstance(value, ObjectDelta):
        value.object_id = bindings.get(value.object_id, value.object_id)
    elif isinstance(value, ObjectStatePatch) and value.container_id:
        value.container_id = bindings.get(value.container_id, value.container_id)
    elif isinstance(value, ActionTerm):
        value.related_entity_ids = [bindings.get(i,i) for i in value.related_entity_ids]
    elif isinstance(value, Correspondence) and value.actual_kind == "object":
        value.actual_id = bindings.get(value.actual_id, value.actual_id)
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _bind_object_references(getattr(value, name), bindings)
    elif isinstance(value, dict):
        for child in value.values():
            _bind_object_references(child, bindings)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _bind_object_references(child, bindings)


def _reserved_asset_ids(context, owner):
    """Include the full catalog even when the delivered scene is narrower."""
    catalog = getattr(owner, "manifest", {}).get("catalog", {})
    reserved = {identity for values in catalog.values() for identity in values}
    if hasattr(owner, "get_checkpoint"):
        reserved.update(owner.get_checkpoint("runtime_objects", {}))
    reserved.update(row["object_id"] for row in context.get("object_inventory", []))
    for assets in context.get("environment", {}).get("assets", {}).values():
        reserved.update(assets)
    reserved.update(context.get("current_state", {}).get("objects", {}))
    return reserved


def _bind_wave_identities(candidate, context, owner, job):
    reserved = _reserved_asset_ids(context, owner)
    aliases = candidate.advance.new_objects
    if set(aliases) & reserved:
        raise ValueError("新物品局部别名不能占用已有目录身份")
    bindings = {alias: _opaque_id("SIM_OBJ_", job, alias) for alias in aliases}
    if set(bindings.values()) & (reserved | set(aliases)):
        raise ValueError("新物品正式身份与已有身份或本批别名冲突")
    if bindings:
        _bind_object_references(candidate, bindings)
    candidate.advance.new_objects = {bindings[alias]: item for alias, item in aliases.items()}
    runtimes = context.get("person_runtime", {})
    existing = {person: {effect["effect_id"] for effect in runtime.get("effects", [])}
                for person, runtime in runtimes.items()}
    all_existing = set().union(*existing.values()) if existing else set()
    generated = set()
    for person, change in candidate.advance.runtime_changes.items():
        own = existing.get(person, set())
        for effect in change.state.effects:
            if effect.effect_id in own:
                continue
            if effect.effect_id in all_existing:
                raise ValueError("不能把其他人的已有认知作用绑定给本人")
            identity = _opaque_id("effect_", job, person, effect.effect_id)
            if identity in all_existing or identity in generated:
                raise ValueError("新认知作用身份发生冲突")
            generated.add(identity)
            effect.effect_id = identity
    # advance.processing describes execution under the shared PRE-state. New
    # effects cannot explain an earlier action; its old-ID validation stays in
    # WorldAgent instead of silently rebinding a new alias into that evidence.
    return candidate


def schedule_contract(schema, context, owner, job):
    eligible = context["eligible"]
    current = context.get("current_state", {}).get("characters", {})
    runtime = context.get("person_runtime", {})
    available = {event: [p for p in people
                        if current.get(p, {}).get("life_status") not in {"dead", "not_born"}
                        and runtime.get(p, {}).get("can_act", True)]
                 for event, people in eligible.items()}
    people = sorted({person for choices in available.values() for person in choices})
    person_type = Literal[tuple(people)] | None if people else type(None)
    fields = {
        "actor": (person_type, Field(description="选择本事件可行动人物；null 表示环境尝试。")),
        "opportunity": (OpportunityPlan, ...),
    }
    # The anchor is already chosen once in opportunity. Do not ask the model
    # to copy it into a second field and compare the copies.
    environment_type = create_model("EnvironmentContent", __base__=Record,
        content=(Identity, ...), basis=(Identity, ...))
    fields["environment"] = (environment_type | None, None)
    if schema is ReplanSchedule:
        fields["pacing"] = (Literal["stay", "move_on"], ...)
        fields["scene_reason"] = (Identity, Field(pattern=r"\S"))
    item_type = create_model("ScheduledOpportunity", __base__=Record, **fields)
    events = sorted(eligible)
    payload_type = create_model(schema.__name__ + "Input", __base__=Record,
        decisions=(list[item_type], Field(min_length=len(events), max_length=len(events),
            description="按此顺序一次交付各事件完整机会；事件归属由程序绑定：" + encode(events))),
        initial_locations=(dict[Identity, InitialLocationChoice | None], Field(default_factory=dict,
            description="仅为本组当前位置 unknown 且已出生的人物提出初始地点；已知宅院而房间不明不属于未知地点。"
                        "null 表示撤回该项安排；其余内容保留，实际状态不因此改变。")))

    def assemble(payload):
        value = payload_type.model_validate(payload_data(payload))
        actors, arrangements, opportunities, scenes = {}, {}, {}, {}
        for event, row in zip(events, value.decisions):
            if row.actor is not None and row.actor not in available[event]:
                raise ValueError(f"事件 {event} 只能选择可行动人物 {available[event]} 或环境尝试")
            if (row.actor is None) != (row.environment is not None):
                raise ValueError(f"事件 {event} 只有环境推进须交付 environment 内容与依据")
            anchor = row.opportunity.scene_anchor
            # Unambiguous alternate root, never a guessed identity or a
            # substring extracted from an explanation. Outer validation
            # still verifies this exact fact against the actual world.
            if anchor.startswith(tuple("/" + domain + "/" for domain in
                    ("characters", "locations", "objects", "relations"))):
                row.opportunity.scene_anchor = "/current_state" + anchor
            actors[event], opportunities[event] = row.actor, row.opportunity
            if row.environment is not None:
                arrangements[event] = EnvironmentAttempt(**row.environment.model_dump(),
                                                          scene_anchor=row.opportunity.scene_anchor)
            if schema is ReplanSchedule:
                recent = context.get("interaction_progress", {}).get(event, {}).get("recent_actions", [])
                scenes[event] = SceneDecision(pacing=row.pacing, opportunity=row.scene_reason,
                    after_action_id=recent[-1]["action_id"] if recent else None,
                    focus_anchor=row.opportunity.scene_anchor, stop_before=row.opportunity.stop_before)
        return schema(actors=actors, arrangements=arrangements, opportunities=opportunities,
                      scene_decisions=scenes, initial_locations={p: choice for p, choice in value.initial_locations.items()
                                                                 if choice is not None})
    return payload_type, assemble

def perception_contract(schema, context, owner, job):
    people = sorted(context["selected"])
    payload_type = create_model("PerceptionBatchInput", __base__=Record,
        views=(list[PerceptionContent], Field(min_length=len(people), max_length=len(people),
            description="按此顺序分别形成完整感知，接收者由程序绑定：" + encode(people))))
    def assemble(payload):
        delivered = payload_type.model_validate(payload_data(payload))
        reserved = _reserved_asset_ids(context, owner)
        return schema(views={person: _perception(view, reserved) for person, view in zip(people, delivered.views)})
    return payload_type, assemble

def resolution_contract(schema, context, owner, job):
    identities = [row["action_id"] for row in context["intents"]]
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("本批行动身份必须非空且唯一")
    payload_type = create_model("WaveResolutionInput", __base__=Record,
        actions=(list[ActionResolution], Field(min_length=len(identities), max_length=len(identities),
            description="按 intents 的现有顺序分别交付实际结果；行动身份由程序绑定，全部共享同一前态。")),
        advance=(AdvanceInput, Field(
            description="整批唯一的共享经过与时间推进；与 actions 同级，不能放入 actions 的单项中。")),
        final_views=(dict[Identity, PerceptionContent], Field(default_factory=dict,
            description="同一人物在本批有多份不同体验时，必须提供统一的批末视图；各原始体验仍完整保留。只可包含本批获得体验的人物。")))

    def assemble(payload):
        value = payload_type.model_validate(payload_data(payload))
        reserved = _reserved_asset_ids(context, owner)
        views = lambda items: {person: _perception(view, reserved) for person, view in items.items()}
        candidate = schema(actions=[{**row.model_dump(exclude={"observations"}), "action_id": identity,
                                    "observations": views(row.observations)}
                               for identity, row in zip(identities, value.actions)],
                      advance={**value.advance.model_dump(exclude={"observations", "processing"}),
                               "processing": [{**trace.model_dump(), "stage": "execution"} for trace in value.advance.processing],
                               "observations": views(value.advance.observations)}, final_views=views(value.final_views))
        return _bind_wave_identities(candidate, context, owner, job)
    return payload_type, assemble

