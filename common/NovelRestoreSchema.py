"""还原系统的数据契约；结构验证不替代世界智能体的语义判断。"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, field_validator, model_serializer, model_validator

from contract.NovelWorldStateSchema import AssetState, StateDelta
from contract.NovelWorldState import public_state


Identity = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]
ActionMode = Literal["action", "talk", "think", "idle"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class WorldState(AssetState):
    """与公开节点快照同构；共用上游字段类型，不保存原文证据。"""
    node_id: Identity | None = None
    previous_node_id: Identity | None = None

    @field_validator('*', mode='before')
    @classmethod
    def empty_collections(cls, value, info):
        if info.field_name in {'node_id', 'previous_node_id'}:
            return value
        return super().empty_collections(value, info)

    @model_validator(mode="before")
    @classmethod
    def snapshot_envelope(cls, value):
        return value

    @model_serializer(mode="wrap")
    def without_evidence(self, handler):
        return public_state(handler(self), include_evidence=False)


class StatePatch(StateDelta):
    @model_serializer(mode="wrap")
    def without_evidence(self, handler):
        return public_state(handler(self), include_evidence=False)


class NodeSnapshot(WorldState):
    node_id: Identity
    previous_node_id: Identity | None


class EventGroup(Record):
    group_id: Identity
    node_id: Identity
    node_order: PositiveInt
    order: PositiveInt
    events: list[dict[str, JsonValue]] = Field(min_length=1)
    state_delta: StatePatch


class StoryPosition(Record):
    node_id: Identity
    node_order: PositiveInt
    group_order: PositiveInt
    event_id: Identity | None
    round: PositiveInt = Field(description="事件组内一次人物或环境行为的计数编号；同批编号不表示因果先后")
    turn: PositiveInt = Field(description="事件组内共同裁定的批次号；同批行为共享前态")

    @property
    def sort_key(self):
        # 用于同一人物的记忆排序，不把提交耗时解释为组内事件先后。
        return self.node_order, self.group_order, self.round, self.turn


class MemoryContribution(Record):
    """角色提交内容，运行及行动身份由程序绑定。"""
    character_id: Identity
    kind: Literal["event", "chat", "thought"]
    content: Identity
    source: Literal["experience", "observation", "reported", "inference", "imagination", "plan"]
    role_ids: list[Identity] = Field(default_factory=list)
    object_ids: list[Identity] = Field(default_factory=list)
    location_ids: list[Identity] = Field(default_factory=list)
    speaker_id: Identity | None = None
    listener_ids: list[Identity] = Field(default_factory=list)
    related_memory_ids: list[Identity] = Field(default_factory=list)
    corrects: list[Identity] = Field(default_factory=list)
    importance: Annotated[int, Field(ge=1, le=5, strict=True)] = 3
    occurred_at: str | None = None
    source_action_id: Identity | None = None
    source_shared_id: Identity | None = None
    perceived_ids: list[Identity] = Field(default_factory=list, description="本次个人体验中的对象编号；不要求对应真实实体")


class MemoryRecord(MemoryContribution):
    memory_id: Identity
    run_id: Identity
    action_id: Identity | None
    position: StoryPosition


class PerceivedObject(Record):
    perceived_id: Identity
    description: Identity


class PersonalPerception(Record):
    content: Identity
    objects: list[PerceivedObject] = Field(default_factory=list)


class Correspondence(Record):
    perceived_id: Identity
    actual_kind: Literal["role", "object", "location", "none", "unknown"]
    actual_id: Identity | None = None
    basis: Identity = Field(description="世界侧依据：当前条件如何产生此个人体验；不得交给人物")

    @model_validator(mode="after")
    def bound_target(self):
        if (self.actual_kind in ("none", "unknown")) != (self.actual_id is None):
            raise ValueError("真实对象须有编号；确定无对应或尚未确定时编号须为 null")
        return self


class WorldPerception(Record):
    """个人可读体验与世界私有对应关系；保存时绑定人物、行动及版本。"""
    personal: PersonalPerception
    correspondences: list[Correspondence] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_mapping(self):
        objects = [p.perceived_id for p in self.personal.objects]
        mapped = [p.perceived_id for p in self.correspondences]
        if len(set(objects)) != len(objects) or len(set(mapped)) != len(mapped) or set(objects) != set(mapped):
            raise ValueError("个人感知对象必须唯一且恰好各有一条世界对应关系；"
                             f"personal.objects 的 perceived_id={objects}，correspondences 的 perceived_id={mapped}；"
                             f"personal.objects 缺少 {sorted(set(mapped) - set(objects))}，"
                             f"correspondences 缺少 {sorted(set(objects) - set(mapped))}。"
                             "两处须按原体验对应，去除重复编号；不得新增经历或虚构真实对象。")
        return self


class PerceptionBatch(Record):
    views: dict[Identity, WorldPerception]


class StateFactEvidence(Record):
    path: Identity = Field(description="当前实际状态 current_state 的 JSON Pointer，例如 /characters/人物编号/life_status")
    value: JsonValue = Field(description="该路径上的实际值；不得填写目标或未来参考状态的值")


class ResultEvidence(Record):
    result_index: Annotated[int, Field(ge=0, strict=True)] = Field(description="对应 achieved_results 的从 0 开始的下标")
    action_ids: list[Identity] = Field(default_factory=list, description="本组已发生或本批已裁定的行动编号")
    state_facts: list[StateFactEvidence] = Field(default_factory=list, description="当前实际状态中的路径及实际值")

    @model_validator(mode="after")
    def has_source(self):
        if not self.action_ids and not self.state_facts:
            raise ValueError("每项已达成结果须引用实际行动或当前状态")
        return self


class CompletionJudgment(Record):
    achieved: bool = Field(strict=True)
    summary: Identity
    achieved_results: list[Identity] = Field(default_factory=list)
    result_evidence: list[ResultEvidence] = Field(default_factory=list,
        description="逐项为 achieved_results 提供实际行动或当前状态依据；不得引用目标快照")
    remaining: list[Identity] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_result(self):
        if self.achieved and (not self.achieved_results or self.remaining):
            raise ValueError("达标必须列出实际结果且没有未满足要求")
        if not self.achieved and not self.remaining:
            raise ValueError("未达标必须说明尚未满足的要求")
        return self


class GroupAssessment(CompletionJudgment):
    continuity_ok: bool = Field(strict=True)
    contradictions: list[Identity] = Field(default_factory=list)

    @model_validator(mode="after")
    def safe_continuation(self):
        if self.achieved and (not self.continuity_ok or self.contradictions):
            raise ValueError("后续承接或关键事实仍有矛盾，不能确认组完成")
        return self


class Need(Record):
    urgency: Annotated[float, Field(ge=0, le=10)]
    description: Identity
    drift_per_hour: Annotated[float, Field(ge=-10, le=10)] = 0


class CognitiveEffect(Record):
    effect_id: Identity
    stage: Literal["perception", "execution", "encoding", "retrieval"]
    mechanism: Identity
    cause: Identity
    recovery: Identity


class PersonRuntime(Record):
    """世界掌握的能力与认知条件，不能作为人物自我介绍。"""
    can_act: bool = True
    body: str = ""
    effects: list[CognitiveEffect] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_effects(self):
        if len({e.effect_id for e in self.effects}) != len(self.effects):
            raise ValueError("认知作用身份重复")
        return self


class RuntimeChange(Record):
    state: PersonRuntime
    basis: Identity
    source_action_ids: list[Identity] = Field(default_factory=list)


class ProcessingTrace(Record):
    stage: Literal["perception", "execution", "encoding", "retrieval"]
    effect_ids: list[Identity] = Field(min_length=1)
    source_ids: list[Identity] = Field(default_factory=list)
    explanation: Identity


class CharacterState(Record):
    needs: dict[Identity, Need] = Field(default_factory=dict)
    emotion: str = ""
    beliefs: list[Identity] = Field(default_factory=list)
    goals: list[Identity] = Field(default_factory=list)
    plan: list[Identity] = Field(default_factory=list)


class MemorySummary(Record):
    text: Identity
    memory_ids: list[Identity] = Field(min_length=1)


class CharacterSummary(MemorySummary):
    group_id: Identity
    kind: Literal["event", "stage"]


class SubjectiveCurrent(Record):
    perception: PersonalPerception | None
    working_state: CharacterState


class SubjectiveWorld(Record):
    """每人一份的文件契约；私有认知作用与主客观对应不在这个结构中。"""
    character_id: Identity
    profile: dict[str, JsonValue]
    current: SubjectiveCurrent
    history: list[MemoryRecord]
    summaries: list[CharacterSummary]


class EnvironmentAttempt(Record):
    content: Identity
    basis: Identity = Field(description="当前真实条件支持的发生依据，不填写剧情目标或人物行动指令")
    scene_anchor: Identity | None = Field(default=None, description="新调度须给一条现场依据：已提交的 action_id，或 /current_state 下具体事实的 JSON Pointer，定时变化可引用 /story_seconds。")


class InitialLocationChoice(Record):
    location_id: Identity
    basis: Identity = Field(description="从已有资料选择地点的理由；属于下游补充而非原始已知事实")
    detail: str | None = None


class RoundSchedule(Record):
    actors: dict[Identity, Identity | None]
    arrangements: dict[Identity, EnvironmentAttempt] = Field(default_factory=dict)
    initial_locations: dict[Identity, InitialLocationChoice] = Field(default_factory=dict,
        description="仅为本组活跃事件中当前位置未知的参与人物补充初始地点；不是移动或已经发生的动作")

    @model_validator(mode="after")
    def explicit_environment(self):
        if set(self.arrangements) != {event for event, person in self.actors.items() if person is None}:
            raise ValueError("每个环境推进须有明确尝试；同一事件本批不能同时指派人物和环境")
        return self


class CharacterMemories(Record):
    memories: list[MemoryContribution] = Field(default_factory=list)
    working_state: CharacterState


class MemoryApproval(Record):
    approved: bool = Field(strict=True)
    issues: list[Identity] = Field(default_factory=list)
    character_issues: dict[Identity, list[Identity]] = Field(default_factory=dict)
    notes: list[Identity] = Field(default_factory=list, description="仅世界可见的审核依据或已接纳主观疑点；不作为未决错误")

    @model_validator(mode="after")
    def consistent(self):
        if self.approved == bool(self.issues):
            raise ValueError("记忆检查通过不得有遗留问题；拒绝须说明问题")
        if self.approved and any(self.character_issues.values()):
            raise ValueError("人物仍有未解决问题，不能确认记忆通过")
        return self


class UnknownTargetReview(Record):
    approved: bool = Field(strict=True)
    issues: list[Identity] = Field(default_factory=list)
    basis: Identity = Field(description="依据实际局面解释结果是否依赖未知对象的身份或存在")

    @model_validator(mode="after")
    def consistent(self):
        if self.approved == bool(self.issues):
            raise ValueError("未知对象检查通过不得遗留问题；拒绝须说明问题")
        return self


class WorldResolution(Record):
    result: Identity
    state_patch: StatePatch = Field(default_factory=StatePatch)
    observations: dict[Identity, WorldPerception]


class ResolvedIntent(WorldResolution):
    action_id: Identity
    resolved_targets: list[Correspondence] = Field(default_factory=list,
        description="对本次行动指向、此前客观对应尚未知的感知对象所作的有依据解析")


class SimulatedObject(Record):
    object_name: Identity
    description: Identity
    aliases: list[Identity] = Field(default_factory=list)
    basis: Identity = Field(description="此普通道具实际存在或产生的依据；登记不等于此刻制造出来")
    considered_object_ids: list[Identity] = Field(default_factory=list, description="已比较的上游及已登记下游物品编号")
    reuse_reason: Identity = Field(description="先比较上游物品的用途、实际位置和可取得性；说明为何没有适用的现成物品，不能只说需要新增")
    source_action_ids: list[Identity] = Field(min_length=1, description="证明此物实际存在的已提交行动或本批实际发生编号，不能只引用个人想法")


class WaveAdvance(Record):
    """同批自然经过一次：不把时间流逝的后果冒充思考/等待造成。"""
    elapsed_seconds: Annotated[float, Field(gt=0)]
    state_patch: StatePatch = Field(default_factory=StatePatch)
    runtime_changes: dict[Identity, RuntimeChange] = Field(default_factory=dict)
    new_objects: dict[Identity, SimulatedObject] = Field(default_factory=dict,
        description="本批登记的普通道具；编号以 SIM_OBJ_ 开头且未使用。先复用上游及已有下游物品；对应状态须在本批物品补丁保存。不是额外的环境发生。")
    processing: list[ProcessingTrace] = Field(default_factory=list)
    basis: Identity
    result: str = ""
    observations: dict[Identity, WorldPerception] = Field(default_factory=dict)
    related_event_ids: list[Identity] = Field(default_factory=list,
        description="自然变化与哪些本组事件有关；不将变化默认归入第一个事件")


class WaveResolution(Record):
    actions: list[ResolvedIntent] = Field(min_length=1)
    advance: WaveAdvance
    final_views: dict[Identity, WorldPerception] = Field(default_factory=dict,
        description="同一人物本批有多份不同体验时的统一当前视图；各项体验仍分别保留")


class WaveAssessment(Record):
    events: dict[Identity, CompletionJudgment]
    goal: GroupAssessment
    made_progress: bool


class ActionProposal(Record):
    character_id: Identity | None
    mode: ActionMode
    content: Identity
    target_ids: list[Identity] = Field(default_factory=list, description="当前个人感知中的对象编号；仅提出尝试，不宣布成功")


class ActionCandidate(Record):
    proposal: ActionProposal
    expected_outcome: Identity
    need_changes: dict[Identity, Annotated[float, Field(ge=-10, le=10)]] = Field(default_factory=dict)


class CharacterDecision(Record):
    candidates: list[ActionCandidate] = Field(min_length=1, max_length=3)


class RecalledMemory(Record):
    memory_id: Identity
    content: Identity
    source: Identity


class EncodedExperience(Record):
    source_id: Identity = Field(validation_alias=AliasChoices("source_id", "action_id"))
    source_kind: Literal["action", "shared"] = "action"
    experience: PersonalPerception | None


class SubjectiveInput(Record):
    """人物本次真正可用的材料；世界私有 processing 不传给人物。"""
    working_state: CharacterState
    profile: dict[str, JsonValue] | None = None
    memories: list[RecalledMemory] = Field(default_factory=list)
    experiences: list[EncodedExperience] = Field(default_factory=list)
    processing: list[ProcessingTrace] = Field(default_factory=list)


class WorldAdvance(WaveAdvance):
    """一次共同时间经过及其独立来源的主观记录；物理上随本批保存。"""
    shared_id: Identity | None = None
    version_after: Annotated[int, Field(ge=0, strict=True)] | None = None
    views_before: dict[Identity, WorldPerception] = Field(default_factory=dict)
    subjective_inputs: dict[Identity, SubjectiveInput] = Field(default_factory=dict)
    working_states: dict[Identity, CharacterState] = Field(default_factory=dict)
    memories: list[MemoryRecord] = Field(default_factory=list)
    final_views: dict[Identity, WorldPerception] = Field(default_factory=dict)


class StoryMaterial(Record):
    """模拟已产生的故事内容；不含幕后计划、状态补丁或审核记录。"""
    action_id: Identity
    position: StoryPosition
    proposal: ActionProposal | None
    result: Identity
    perceptions_before: dict[Identity, PersonalPerception]
    experiences: dict[Identity, PersonalPerception]
    memories: list[MemoryContribution]
    working_states: dict[Identity, CharacterState]
    background_result: str = ""


class SharedStoryMaterial(Record):
    shared_id: Identity
    group_id: Identity
    result: Identity
    related_event_ids: list[Identity] = Field(default_factory=list)
    experiences: dict[Identity, PersonalPerception] = Field(default_factory=dict)
    memories: list[MemoryContribution] = Field(default_factory=list)
    working_states: dict[Identity, CharacterState] = Field(default_factory=dict)


class ConfirmedAction(Record):
    action_id: Identity
    run_id: Identity
    expected_version: Annotated[int, Field(ge=0, strict=True)]
    position: StoryPosition
    proposal: ActionProposal
    result: Identity
    state_patch: StatePatch = Field(default_factory=StatePatch)
    memories: list[MemoryRecord] = Field(default_factory=list)
    completion: CompletionJudgment | None = None
    # 仅解析格式1—7的历史行动；新写入必须使用下方主客观分离记录。
    perceptions: dict[Identity, Identity] = Field(default_factory=dict)
    views_before: dict[Identity, WorldPerception] = Field(default_factory=dict)
    observations: dict[Identity, WorldPerception] = Field(default_factory=dict)
    resolved_targets: list[Correspondence] = Field(default_factory=list)
    unknown_target_review: UnknownTargetReview | None = None
    memory_review: MemoryApproval | None = None
    environment_basis: Identity | None = None
    # 一批只在首条行动保存一次；SQLite 事务同时应用，草稿不复制这些幕后字段。
    advance: WorldAdvance | None = None
    subjective_inputs: dict[Identity, SubjectiveInput] = Field(default_factory=dict)
    working_states: dict[Identity, CharacterState] = Field(default_factory=dict)

    def story_material(self):
        return StoryMaterial(action_id=self.action_id, position=self.position,
            proposal=self.proposal if self.proposal.character_id is not None else None,
            result=self.result, perceptions_before={p: v.personal for p, v in self.views_before.items()},
            experiences={p: v.personal for p, v in self.observations.items()},
            memories=[MemoryContribution.model_validate(m.model_dump(include=set(MemoryContribution.model_fields)))
                      for m in self.memories], working_states=self.working_states)

    @model_validator(mode="after")
    def matching_memories(self):
        keys = [(row.character_id, row.memory_id) for row in self.memories]
        if len(set(keys)) != len(keys):
            raise ValueError("同一人物的记忆身份重复")
        for row in self.memories:
            if (row.run_id, row.action_id, row.position) != (self.run_id, self.action_id, self.position):
                raise ValueError("记忆必须绑定产生它的已确认行动与故事位置")
        return self


class EventOutcome(Record):
    """世界角色交付的事件结论；程序只核对其引用与保存位置。"""
    run_id: Identity
    event_id: Identity
    expected_version: Annotated[int, Field(ge=0, strict=True)]
    status: Literal["completed", "timed_out", "failed"]
    summary: Identity
    action_ids: list[Identity] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_action_references(self):
        if len(set(self.action_ids)) != len(self.action_ids):
            raise ValueError("事件结论的行动引用不能重复")
        if self.status == "completed" and not self.action_ids:
            raise ValueError("完成结论必须引用实际已提交的行动")
        return self


class EventDraft(Record):
    draft_id: Identity
    run_id: Identity
    event_id: Identity
    node_id: Identity
    node_order: PositiveInt
    group_order: PositiveInt
    participant_ids: list[Identity]
    action_ids: list[Identity] = Field(min_length=1)
    achieved_results: list[str]
    text: Identity
    materials: list[StoryMaterial] = Field(default_factory=list)
    shared_material_ids: list[Identity] = Field(default_factory=list)


class NarrativeFragment(Record):
    fragment_id: Identity
    draft_id: Identity
    event_id: Identity
    # 按实际行动与语义选取定位；允许同源内容在不同阅读位置再次呈现。
    selected_content: Identity
    action_ids: list[Identity] = Field(default_factory=list)
    viewpoint: str | None = None
    purpose: str = ""
    reveal: list[str] = Field(default_factory=list)
    withhold: list[str] = Field(default_factory=list)
    length_budget: PositiveInt | None = None
    knowledge_basis: list[Identity] = Field(default_factory=list)
    viewpoint_role_id: Identity | None = None
    knowledge_action_id: Identity | None = None


class ChapterPlan(Record):
    chapter_id: Identity
    title: str
    purpose: str = ""
    fragment_ids: list[Identity] = Field(min_length=1)
    length_budget: PositiveInt | None = None
    setting: str = ""


class NarrativeLink(Record):
    link_id: Identity
    kind: Literal["foreshadow", "suspense", "echo", "retelling"]
    from_fragment: Identity
    to_fragment: Identity
    content: Identity
    acceptance: list[Identity] = Field(default_factory=list)


class NarrativeLayout(Record):
    fragments: list[NarrativeFragment] = Field(min_length=1)
    chapters: list[ChapterPlan] = Field(min_length=1)
    links: list[NarrativeLink] = Field(default_factory=list)

    @property
    def reading_order(self):
        return [identity for chapter in self.chapters for identity in chapter.fragment_ids]

    @model_validator(mode="after")
    def complete_placement(self):
        identities = [row.fragment_id for row in self.fragments]
        chapters = [row.chapter_id for row in self.chapters]
        placed = self.reading_order
        if len(set(identities)) != len(identities) or len(set(chapters)) != len(chapters):
            raise ValueError("片段和章节身份必须各自唯一")
        if len(placed) != len(set(placed)) or set(placed) != set(identities):
            raise ValueError("所有片段必须恰好落位一次；重复呈现应使用不同片段身份")
        if len({row.link_id for row in self.links}) != len(self.links):
            raise ValueError("叙述关联身份重复")
        for link in self.links:
            if link.from_fragment not in placed or link.to_fragment not in placed:
                raise ValueError("叙述关联引用了未落位片段")
            if link.from_fragment == link.to_fragment:
                raise ValueError("叙述关联需要两个不同片段")
            if link.kind == "foreshadow" and placed.index(link.from_fragment) >= placed.index(link.to_fragment):
                raise ValueError("伏笔必须位于其回收片段之前")
        return self


class ChapterSketch(Record):
    chapter_id: Identity
    title: Identity
    purpose: Identity
    draft_ids: list[Identity] = Field(min_length=1)
    length_budget: PositiveInt
    setting: str = ""


class NarrativeOutline(Record):
    chapters: list[ChapterSketch] = Field(min_length=1)


class ChapterLayout(Record):
    fragments: list[NarrativeFragment] = Field(min_length=1)


class NarrativeConnections(Record):
    links: list[NarrativeLink] = Field(default_factory=list)


class NarrativePlan(NarrativeLayout):
    run_id: Identity
    version_id: Identity
    revision: PositiveInt = 1
    requirements: str


class ProseFragment(Record):
    version_id: Identity
    plan_revision: PositiveInt
    fragment_id: Identity
    revision: PositiveInt
    task_id: Identity
    text: Identity = Field(pattern=r"\S")


class NarrativeAssignment(Record):
    constraints: list[Identity] = Field(min_length=1)
    focus: Identity


class NarrativeIssue(Record):
    fragment_ids: list[Identity] = Field(min_length=1)
    description: Identity


class NarrativeReview(Record):
    approved: bool = Field(strict=True)
    summary: Identity
    issues: list[NarrativeIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_verdict(self):
        if self.approved == bool(self.issues):
            raise ValueError("内容核对通过时不能仍有问题；未通过须指出具体问题及片段")
        return self


class LinkRemoval(Record):
    link_id: Identity
    reason: Identity


class NarrativeSelection(Record):
    fragment_ids: list[Identity] = Field(min_length=1)
    upsert_links: list[NarrativeLink] = Field(default_factory=list)
    remove_links: list[LinkRemoval] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_targets(self):
        upserts = [link.link_id for link in self.upsert_links]
        removals = [link.link_id for link in self.remove_links]
        if (len(set(self.fragment_ids)) != len(self.fragment_ids) or len(set(upserts)) != len(upserts)
                or len(set(removals)) != len(removals) or set(upserts) & set(removals)):
            raise ValueError("联合目标和关系操作不得重复或冲突")
        return self


class FragmentText(Record):
    fragment_id: Identity
    text: Identity = Field(pattern=r"\S")
    summary: str = ""
    revealed: list[Identity] = Field(default_factory=list)
    change: Literal["initial", "wording", "meaning"] = Field(default="meaning",
        description="首写、仅措辞、已有内容的呈现或揭示变化；均不得增添或改写故事事实")


class NarrativeBatchReview(NarrativeReview):
    preserved_link_ids: list[Identity] = Field(default_factory=list)
    affected_fragment_ids: list[Identity] = Field(default_factory=list)


class NarrativeBatch(Record):
    texts: list[FragmentText] = Field(min_length=1)
    confirmed_link_ids: list[Identity] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_items(self):
        if len({row.fragment_id for row in self.texts}) != len(self.texts):
            raise ValueError("每个联合目标只交付一份正文")
        if len(set(self.confirmed_link_ids)) != len(self.confirmed_link_ids):
            raise ValueError("关联确认重复")
        return self
