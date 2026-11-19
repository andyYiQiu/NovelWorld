"""还原系统的数据契约；结构验证不替代世界智能体的语义判断。"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, field_validator, model_serializer, model_validator

from inputs.NovelWorldStateSchema import AssetState, StateDelta, normalize_snapshot
from inputs.NovelWorldState import public_state


Identity = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]
ActionMode = Literal["action", "talk", "think", "idle"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SemanticPayload(Record):
    """Explicit opt-in for content-only tasks with no persistent record assembly.

    Storage records must use a task contract instead. This marker never changes
    validation or adds model calls.
    """


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
        # AssetState 会剥离这两个节点字段，而运行中的 WorldState 需要保留它们。
        return normalize_snapshot(value)

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


MemorySource = Literal["experience", "observation", "reported", "inference", "imagination", "plan"]


class MemoryContent(Record):
    """人物亲历、观察或推断的内容及来源。"""
    character_id: Identity
    kind: Literal["event", "chat", "thought"]
    content: Identity
    source: MemorySource
    role_ids: list[Identity] = Field(default_factory=list)
    object_ids: list[Identity] = Field(default_factory=list)
    location_ids: list[Identity] = Field(default_factory=list)
    speaker_id: Identity | None = None
    listener_ids: list[Identity] = Field(default_factory=list)
    related_memory_ids: list[Identity] = Field(default_factory=list)
    corrects: list[Identity] = Field(default_factory=list)
    occurred_at: str | None = None
    source_action_id: Identity | None = None
    source_shared_id: Identity | None = None
    perceived_ids: list[Identity] = Field(default_factory=list, description="只填写对应的当次个人体验里出现的 perceived_id")


class MemoryContribution(MemoryContent):
    importance: Annotated[int, Field(ge=1, le=5, strict=True)] = 3


class StoryMemory(MemoryContent):
    memory_id: Identity


class MemoryRecord(MemoryContribution):
    memory_id: Identity
    run_id: Identity
    action_id: Identity | None
    position: StoryPosition

    def story_memory(self):
        return StoryMemory.model_validate(self.model_dump(include=set(StoryMemory.model_fields)))


class PerceivedObject(Record):
    perceived_id: Identity = Field(description="本份个人视图内唯一的局部编号，与 correspondences 一一对应；不填真实资产编号。人物或身份未定对象使用 p1 至 p999999 的中性编号，不带前导0")
    description: Identity


class PersonalPerception(Record):
    content: Identity
    objects: list[PerceivedObject] = Field(default_factory=list)


class Correspondence(Record):
    perceived_id: Identity = Field(description="与 personal.objects 对应的局部编号；解析已有行动目标时原样沿用目标编号，不创建新编号")
    actual_kind: Literal["role", "object", "location", "none", "unknown"]
    actual_id: Identity | None = Field(default=None,
        description="actual_kind 为 role/object/location 时须填对应目录的真实编号；none/unknown 时必须为 null。只供世界使用")
    basis: Identity = Field(description="世界侧依据：当前条件如何产生此个人体验；不得交给人物")

    @model_validator(mode="after")
    def bound_target(self):
        if (self.actual_kind in ("none", "unknown")) != (self.actual_id is None):
            raise ValueError("真实对象须有编号；确定无对应或尚未确定时编号须为 null")
        return self


class WorldPerception(Record):
    """个人可读体验与世界私有对应关系；保存时绑定人物、行动及版本。"""
    personal: PersonalPerception
    correspondences: list[Correspondence] = Field(default_factory=list,
        description="与 personal.objects 按 perceived_id 恰好一一对应；两边均不得重复、漏项或多项，不靠新增经历补齐映射")

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
    views: dict[Identity, WorldPerception] = Field(description="以本批选中人物的真实编号为键，须恰好覆盖这些人物；每份内容只包含该人的实际感知")


class StateFactEvidence(Record):
    path: Identity = Field(pattern=r"^/(?:[^~]|~[01])*$",
        description="相对于 current_state 根的 JSON Pointer，如 /characters/R1/life_status；不带 /current_state 前缀。字段名内 / 写为 ~1、~ 写为 ~0；数组用从0开始的下标")
    value: JsonValue = Field(description="该路径上的实际值；不得填写目标或未来参考状态的值")


class ResultEvidence(Record):
    result_index: Annotated[int, Field(ge=0, strict=True)] = Field(description="对应 achieved_results 的从 0 开始的下标")
    action_ids: list[Identity] = Field(default_factory=list, description="本组已发生或本批已裁定的行动编号，不得重复；与 state_facts 至少一项非空")
    state_facts: list[StateFactEvidence] = Field(default_factory=list, description="当前实际状态中的路径及实际值；与 action_ids 至少一项非空")

    @model_validator(mode="after")
    def has_source(self):
        if not self.action_ids and not self.state_facts:
            raise ValueError("每项已达成结果须引用实际行动或当前状态")
        return self


class CompletionJudgment(Record):
    achieved: bool = Field(strict=True, description="true 须有 achieved_results 且 remaining 为空；false 须用 remaining 明确尚未满足的要求")
    summary: Identity
    achieved_results: list[Identity] = Field(default_factory=list)
    result_evidence: list[ResultEvidence] = Field(default_factory=list,
        description="每项 achieved_results 恰好一条，result_index 为 0..n-1 且不重复；不得引用目标快照。确认事件完成须引用本事件实际行动")
    remaining: list[Identity] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_result(self):
        if self.achieved and (not self.achieved_results or self.remaining):
            raise ValueError("达标必须列出实际结果且没有未满足要求")
        if not self.achieved and not self.remaining:
            raise ValueError("未达标必须说明尚未满足的要求")
        return self


class GroupAssessment(CompletionJudgment):
    continuity_ok: bool = Field(strict=True, description="确认 achieved=true 时必须为 true")
    contradictions: list[Identity] = Field(default_factory=list, description="关键事实或后续承接矛盾；非空时不能确认 achieved=true")

    @model_validator(mode="after")
    def safe_continuation(self):
        if self.achieved and (not self.continuity_ok or self.contradictions):
            raise ValueError("后续承接或关键事实仍有矛盾，不能确认组完成")
        return self


class Need(Record):
    urgency: Identity = Field(description="本人依据当前体验表达的紧迫感，不是数值评分或统一刻度。")
    description: Identity


class PlanStep(Record):
    description: Identity
    status: Literal["pending", "completed"] = "pending"


class ShortPlan(Record):
    purpose: Identity
    steps: list[PlanStep] = Field(min_length=1)
    next_step: Annotated[int, Field(ge=0, strict=True)] = 0
    status: Literal["active", "completed", "blocked", "abandoned"] = "active"
    feedback: str = Field(default="", description="本人依据实际感知解释继续、完成、阻塞、调整或放弃；不等于客观成功。")

    @model_validator(mode="after")
    def valid_step(self):
        if self.next_step >= len(self.steps):
            raise ValueError("下一步须引用本计划实际存在的步骤")
        if self.status == "active" and self.steps[self.next_step].status == "completed":
            raise ValueError("仍在执行的下一步不能已经标记完成；重试应说明新理由并列为待办")
        return self


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
    effects: list[CognitiveEffect] = Field(default_factory=list, description="此人的认知作用；effect_id 在本人列表中不得重复，不能引用他人的作用")

    @model_validator(mode="after")
    def unique_effects(self):
        if len({e.effect_id for e in self.effects}) != len(self.effects):
            raise ValueError("认知作用身份重复")
        return self


class RuntimeChange(Record):
    state: PersonRuntime
    basis: Identity
    source_action_ids: list[Identity] = Field(default_factory=list, description="本批真实行动编号；若由自然时间经过产生，留空并在 basis 说明实际依据")


class ProcessingContent(Record):
    effect_ids: list[Identity] = Field(min_length=1, description="引用此人已生效且 stage 一致的认知作用；不能用本批刚生成的作用倒推此前处理")
    source_ids: list[Identity] = Field(default_factory=list, description="本次处理来源编号；执行偏差须引用本批人物行动，其他阶段引用当前任务提供的体验或记忆来源")
    explanation: Identity

    @model_validator(mode="before")
    @classmethod
    def program_stage(cls, value):
        # Content delivery never controls this metadata. Persistent traces still
        # validate their own declared stage, including when historical data is read.
        if "stage" not in cls.model_fields and isinstance(value, dict):
            return {key: item for key, item in value.items() if key != "stage"}
        return value


class ProcessingTrace(ProcessingContent):
    stage: Literal["perception", "execution", "encoding", "retrieval"]


class FeltNeed(Record):
    description: Identity
    urgency: str = ""


class StoryState(Record):
    needs: dict[Identity, FeltNeed] = Field(default_factory=dict)
    emotion: str = ""
    beliefs: list[Identity] = Field(default_factory=list)
    goals: list[Identity] = Field(default_factory=list)
    intention: str = Field(default="", description="本人当前目的及打算；依实际体验改变，不是待自动执行的步骤表。")
    plan: list[Identity] = Field(default_factory=list, description="旧运行的历史打算，只读取证。")
    short_plan: ShortPlan | None = None


class CharacterState(StoryState):
    needs: dict[Identity, Need] = Field(default_factory=dict)

    def story_state(self):
        return StoryState(**self.model_dump(exclude={"needs"}),
                          needs={key: FeltNeed(description=need.description, urgency=need.urgency) for key, need in self.needs.items()})

    @classmethod
    def story_state_from_context(cls, context):
        """只取行动者作决定时获准使用的心理，不从结果或后来的记忆推断。"""
        value = context.get("working_state")
        return cls.model_validate(value).story_state() if value is not None else None


class MemorySummary(SemanticPayload):
    text: Identity
    memory_ids: list[Identity] = Field(min_length=1, description="本人已保存、用于本次总结的记忆编号；不得引用他人记忆或行动编号")


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
    location_id: Identity = Field(description="已有 repository 地点目录编号；不得创建地点编号或用地点名称代替")
    basis: Identity = Field(description="从已有资料选择地点的理由；属于下游补充而非原始已知事实")
    detail: str | None = None


class OpportunityPlan(Record):
    goal_ids: list[Identity] = Field(min_length=1, description="本次机会针对 goal_records 中哪些稳定目标；不新建验收目标。")
    gap: Identity
    scene_anchor: Identity = Field(description="一条已提交 action_id 或 /current_state 下具体状态属性的 JSON Pointer。"
        "已知位置、引用等复合属性可整体引用，未知部分仍保留未知；不引用整个实体、目标或预测，不拼接多条路径或解释。")
    perceivers: list[Identity] = Field(default_factory=list, description="按现场接触条件实际可能获得机会的人物；不代表已经知情。")
    motive: Identity = Field(description="与此人已经有据的需要、信念、短计划或承诺如何相容；环境机会说明无需人物动机的依据。")
    possible_responses: list[Identity] = Field(min_length=1, description="几种合理回应及各自可能改善或损害什么；均为预测。")
    alternatives: list[Identity] = Field(default_factory=list, description="比较过的其他合法机会及取舍；不得伪造依据。")
    expected_contribution: Identity
    risk: Identity
    stop_before: Identity
    change_from_previous: Identity = Field(description="上次安排和实际反馈之后，本次具体改变什么；首次说明这是首次机会。")
    path_status: Literal["viable", "exploring", "no_known_path"] = "viable"


class RoundSchedule(Record):
    actors: dict[Identity, Identity | None] = Field(description="键为当前活跃事件编号，值为该事件的一位可行动参与人物编号；环境推进填 null。不得填写行动正文，同一人物本批不能重复指派")
    arrangements: dict[Identity, EnvironmentAttempt] = Field(default_factory=dict,
        description="键须恰好为 actors 中值为 null 的事件编号；人物行动事件不能同时填写环境尝试")
    initial_locations: dict[Identity, InitialLocationChoice] = Field(default_factory=dict,
        description="仅为本组活跃事件中当前位置未知的参与人物补充初始地点；不是移动或已经发生的动作")
    opportunities: dict[Identity, OpportunityPlan] = Field(description="恰好覆盖 actors 的事件键；世界机会预测仅留后台，不能作为人物输入或实际因果证据。")
    scene_decisions: dict[Identity, "SceneDecision"] = Field(default_factory=dict)

    @model_validator(mode="after")
    def explicit_environment(self):
        if set(self.arrangements) != {event for event, person in self.actors.items() if person is None}:
            raise ValueError("每个环境推进须有明确尝试；同一事件本批不能同时指派人物和环境")
        return self


class SceneDecision(Record):
    pacing: Literal["stay", "move_on"] = Field(description="保留尚有意义的选择或观察，或收束已演出的琐事，转到下一处有依据的机会。")
    after_action_id: Identity | None = Field(description="本事件 recent_actions 的最后一条实际行动编号；本事件尚无行动时填 null。定位比较起点，不作为新结果的因果证明。")
    focus_anchor: Identity | None = Field(default=None, description="人物机会聚焦的一项已提交行动或具体现场事实，默认取 after_action_id；环境机会沿用 arrangement.scene_anchor。程序只转交回读的事实，不向人物传递调度理由。")
    opportunity: Identity = Field(pattern=r"\S", description="与刚才的实际经过相比，本次具体给谁什么选择或观察，或哪个已有过程可自然发展。说明继续停留或收束的理由。")
    stop_before: Identity = Field(pattern=r"\S", description="本次机会止于何处；应留给谁回应，或哪项后果仍需实际裁定。不能预写当事人的选择或事件结局。")


class ReplanSchedule(RoundSchedule):
    scene_decisions: dict[Identity, SceneDecision] = Field(description="键须恰好覆盖当前全部活跃事件编号；每项比较该事件最后一次已提交行动")


class CharacterMemories(Record):
    memories: list[MemoryContribution] = Field(default_factory=list)
    working_state: CharacterState


class NarrativePlanReview(Record):
    approved: bool = Field(strict=True)
    issues: list[Identity] = Field(default_factory=list)
    repair_scope: Literal["all", "chapters", "links"] = Field(default="all",
        description="仅所有问题都局限于指定章节时选 chapters；仅关联问题选 links；大纲、篇幅分配或无法定位时选 all。")
    chapter_ids: list[Identity] = Field(default_factory=list,
        description="chapters 返修须列出全部受影响章节；其余范围留空。不能通过缩小范围漏掉问题。")

    @model_validator(mode="after")
    def consistent(self):
        if self.approved == bool(self.issues):
            raise ValueError("通过不得有未决问题；拒绝须说明具体问题")
        if len(self.chapter_ids) != len(set(self.chapter_ids)):
            raise ValueError("返修章节不得重复")
        if not self.approved and (self.repair_scope == "chapters") != bool(self.chapter_ids):
            raise ValueError("仅章节返修须列出非空章节编号，其他范围不能夹带章节")
        return self


class MemoryApproval(Record):
    approved: bool = Field(strict=True, description="通过时 issues 及各人的 character_issues 都须为空；拒绝时 issues 至少一项")
    issues: list[Identity] = Field(default_factory=list, description="需要修复的具体问题；approved=false 时必填，不把已接纳疑点列作错误")
    character_issues: dict[Identity, list[Identity]] = Field(default_factory=dict,
        description="以受影响人物编号定位返修问题；通过时各列表必须为空")
    notes: list[Identity] = Field(default_factory=list, description="仅世界可见的审核依据或已接纳主观疑点；不作为未决错误")

    @model_validator(mode="after")
    def consistent(self):
        if self.approved == bool(self.issues):
            raise ValueError("记忆检查通过不得有遗留问题；拒绝须说明问题")
        if self.approved and any(self.character_issues.values()):
            raise ValueError("人物仍有未解决问题，不能确认记忆通过")
        return self


class UnknownTargetReview(Record):
    approved: bool = Field(strict=True, description="通过时 issues 为空，拒绝时 issues 至少一项")
    issues: list[Identity] = Field(default_factory=list, description="需要修复的具体问题；approved=false 时必填")
    basis: Identity = Field(description="依据实际局面解释结果是否依赖未知对象的身份或存在")

    @model_validator(mode="after")
    def consistent(self):
        if self.approved == bool(self.issues):
            raise ValueError("未知对象检查通过不得遗留问题；拒绝须说明问题")
        return self


class WorldResolution(Record):
    result: Identity
    state_patch: StatePatch = Field(default_factory=StatePatch)
    observations: dict[Identity, WorldPerception] = Field(description="以实际获得本次体验的人物编号为键，须包含行动者本人；私有思考不能向其他人反馈")


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
    processing: list[ProcessingTrace] = Field(default_factory=list,
        description="本批执行偏差依据：stage 只能为 execution，source_ids 非空且引用本批行动，effect_ids 引用对应行动者在前态已生效的 execution 作用")
    basis: Identity
    result: str = Field(default="", description="共享经过的实际故事内容；observations 非空时必须填写。为空表示没有共享故事材料，此时 related_event_ids 须为空")
    observations: dict[Identity, WorldPerception] = Field(default_factory=dict,
        description="以实际感知共享经过的人物编号为键；有反馈时 result 必须保存共享经过")
    related_event_ids: list[Identity] = Field(default_factory=list,
        description="自然变化相关的本组事件编号，不得重复；不将变化默认归入第一个事件。result 为空时必须为空")


class WaveResolution(Record):
    actions: list[ResolvedIntent] = Field(min_length=1)
    advance: WaveAdvance
    final_views: dict[Identity, WorldPerception] = Field(default_factory=dict,
        description="以本批有体验的人物编号为键；同一人物有多份不同体验时必须给统一当前视图，各项原体验分别保留")


class GoalUpdate(Record):
    status: Literal["satisfied", "unsatisfied", "unknown"]
    change: Literal["improved", "regressed", "unchanged"]
    explanation: Identity
    action_ids: list[Identity] = Field(default_factory=list)
    state_facts: list[StateFactEvidence] = Field(default_factory=list)
    before_facts: list[StateFactEvidence] = Field(default_factory=list, description="批前实际状态；变化必须有实际行动或前后证据，不能引用目标快照。")
    exploration_gain: list[Identity] = Field(default_factory=list, description="实际解决的未知或证伪的路径；与条件改善分列，不能抵消退步或清零停滞。")
    blocker: Identity
    failure_kind: Literal["none", "not_perceived", "not_believed", "conditions_missing", "needs_conflict", "execution_failed", "unknown"]
    intermediate_conditions: list[Identity] = Field(default_factory=list, description="可调整的规划假设，不是新硬目标。")
    alternative_paths: list[Identity] = Field(default_factory=list)
    conflicts: list[Identity] = Field(default_factory=list, description="与其他稳定目标的已知冲突或待核实矛盾。")
    next_change: Identity


class SceneContinuation(Record):
    actor: Identity
    source_action_id: Identity = Field(description="本批该事件中，此人实际获得体验的行动；不能仅凭在场推定知情。")
    reason: Identity = Field(description="当前实质问题仍未解决，为什么此人可以依据实际体验和本人动机直接接续；不预写其回答。")
    stop_before: Identity = Field(description="下一段尝试的必要选择或回应停点；不限制普通动作数量或要求逐句确认。")


class WaveAssessment(Record):
    events: dict[Identity, CompletionJudgment] = Field(description="以本批事件编号为键，须恰好覆盖本批全部事件")
    goal: GroupAssessment
    goal_updates: dict[Identity, GoalUpdate] = Field(description="恰好覆盖 goal_records 全部稳定目标，同一候选后态逐项判断；未获机会与无效尝试由程序分别累计。")
    scene_continuations: dict[Identity, SceneContinuation] = Field(default_factory=dict,
        description="可在当前互动中直接接续的事件及人物；有新阻碍、需换场景、问题已有结果或无法确认时不填。不是人物行动或成功承诺。")


class ActionProposal(Record):
    character_id: Identity | None
    mode: ActionMode
    content: Identity
    target_ids: list[Identity] = Field(default_factory=list, description="当前个人感知中的对象编号；仅提出尝试，不宣布成功")
    short_plan: ShortPlan | None = None
    selection_reason: str = ""
    purpose: str = ""
    stop_before: str = Field(default="", description="本人预期需要重新决定或等待他人回应的位置。")


class ActionCandidate(Record):
    proposal: ActionProposal
    expected_outcome: Identity
    tradeoffs: Identity = Field(description="本人预计的需要满足、时间资源代价、风险、未知条件与替代办法；不做数值评分。")


class CharacterDecision(Record):
    candidates: list[ActionCandidate] = Field(min_length=1)
    selected_index: Annotated[int, Field(ge=0, strict=True)]
    selection_reason: Identity

    @model_validator(mode="after")
    def selected_candidate_exists(self):
        if self.selected_index >= len(self.candidates):
            raise ValueError("人物选定项须属于本次候选，不由程序改选")
        return self


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


class RecallExperience(Record):
    """人物当时体验到的回想。世界侧的命中编号、映射、诊断不进入这个结构。"""
    content: Identity
    source: MemorySource

    @classmethod
    def from_action_context(cls, context):
        return [cls(content=m["content"], source=m["source"]) for m in context.get("memories", [])]


class Format17StoryMaterial(Record):
    """模拟已产生的故事内容；不含幕后计划、状态补丁或审核记录。"""
    action_id: Identity
    position: StoryPosition
    proposal: ActionProposal | None
    result: Identity
    perceptions_before: dict[Identity, PersonalPerception]
    experiences: dict[Identity, PersonalPerception]
    memories: list[StoryMemory]
    working_states: dict[Identity, StoryState]


class Format19StoryMaterial(Format17StoryMaterial):
    recollections_before: list[RecallExperience]


class StoryMaterial(Format19StoryMaterial):
    working_state_before: StoryState | None


class SharedStoryMaterial(Record):
    turn: PositiveInt
    shared_id: Identity
    group_id: Identity
    result: Identity
    related_event_ids: list[Identity] = Field(default_factory=list)
    experiences: dict[Identity, PersonalPerception] = Field(default_factory=dict)
    memories: list[StoryMemory] = Field(default_factory=list)
    working_states: dict[Identity, StoryState] = Field(default_factory=dict)


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

    recollections_before: list[RecallExperience] = Field(default_factory=list)
    working_state_before: StoryState | None = None

    def story_material(self):
        return StoryMaterial(action_id=self.action_id, position=self.position, recollections_before=self.recollections_before,
            working_state_before=self.working_state_before,
            proposal=self.proposal if self.proposal.character_id is not None else None,
            result=self.result, perceptions_before={p: v.personal for p, v in self.views_before.items()},
            experiences={p: v.personal for p, v in self.observations.items()},
            memories=[m.story_memory() for m in self.memories],
            working_states={p: state.story_state() for p, state in self.working_states.items()})

    @model_validator(mode="after")
    def matching_memories(self):
        if self.proposal.character_id is None and (self.recollections_before or self.working_state_before is not None):
            raise ValueError("仅人物行动可以携带行动前回想或心理")
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


class LegacyStoryMaterial(Format17StoryMaterial):
    memories: list[MemoryContribution]
    working_states: dict[Identity, CharacterState]
    background_result: str = ""


class EventDraft(Record):
    """格式 16 及以前的历史事件草稿，只读。"""
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
    materials: list[LegacyStoryMaterial] = Field(default_factory=list)
    shared_material_ids: list[Identity] = Field(default_factory=list)


class Format17GroupDraft(Record):
    """一个通过验收的事件组。一批内的行动同时发生，公共经历只归档一次。"""
    draft_id: Identity
    run_id: Identity
    group_id: Identity
    event_ids: list[Identity] = Field(min_length=1)
    node_id: Identity
    node_order: PositiveInt
    group_order: PositiveInt
    participant_ids: list[Identity]
    participant_names: dict[Identity, str]
    action_ids: list[Identity] = Field(min_length=1)
    event_action_ids: dict[Identity, list[Identity]]
    batch_elapsed_seconds: dict[int, float]
    achieved_results: dict[Identity, list[str]]
    materials: list[Format17StoryMaterial] = Field(min_length=1)
    shared_materials: list[SharedStoryMaterial]
    shared_material_ids: list[Identity]
    text: Identity


class Format19GroupDraft(Format17GroupDraft):
    materials: list[Format19StoryMaterial] = Field(min_length=1)


class GroupDraft(Format19GroupDraft):
    materials: list[StoryMaterial] = Field(min_length=1)


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
    from_fragment: Identity = Field(description="当前叙述计划中的片段编号，与 to_fragment 不同；foreshadow 的此端须在阅读顺序中更早")
    to_fragment: Identity = Field(description="当前叙述计划中的另一片段编号，与 from_fragment 不同")
    content: Identity
    acceptance: list[Identity] = Field(default_factory=list, description="新交付叙述关联须给至少一条具体可核对的验收条件；不能只填写关系名")


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


class NarrativeAssignment(SemanticPayload):
    constraints: list[Identity] = Field(min_length=1)
    focus: Identity


class NarrativeIssue(Record):
    fragment_ids: list[Identity] = Field(min_length=1, description="问题涉及的当前实际片段编号；批次审核只定位本批目标，后续影响使用 affected_fragment_ids")
    description: Identity


class NarrativeReview(Record):
    approved: bool = Field(strict=True, description="通过时 issues 必须为空；拒绝时至少一项具体问题并定位实际片段")
    summary: Identity
    issues: list[NarrativeIssue] = Field(default_factory=list, description="approved=false 时至少一项，approved=true 时必须为空")

    @model_validator(mode="after")
    def consistent_verdict(self):
        if self.approved == bool(self.issues):
            raise ValueError("内容核对通过时不能仍有问题；未通过须指出具体问题及片段")
        return self


class LinkRemoval(Record):
    link_id: Identity
    reason: Identity


class NarrativeSelection(Record):
    fragment_ids: list[Identity] = Field(min_length=1, description="本批联合加工的既有片段编号，不得重复")
    upsert_links: list[NarrativeLink] = Field(default_factory=list,
        description="新增或更新的叙述关联；link_id 不得重复，也不得同时出现在 remove_links")
    remove_links: list[LinkRemoval] = Field(default_factory=list,
        description="删除的已有叙述关联及理由；link_id 不得重复，也不得同时出现在 upsert_links")

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
    summary: str = Field(default="", description="本次正文的非空摘要；新写作交付必须填写，不能只给正文")
    revealed: list[Identity] = Field(default_factory=list)
    change: Literal["initial", "wording", "meaning"] = Field(default="meaning",
        description="首写、仅措辞、已有内容的呈现或揭示变化；均不得增添或改写故事事实")


class NarrativeBatchReview(NarrativeReview):
    preserved_link_ids: list[Identity] = Field(default_factory=list,
        description="本批目标的直接关联中已核对保留的 link_id；两端都须有已有正文或本批候选正文，只有计划但未写正文的关联不能确认；不能顺带确认链上其他关系")
    affected_fragment_ids: list[Identity] = Field(default_factory=list,
        description="当前计划中需后续复核的其他片段编号；不能填本批目标或不存在的片段")

    def validate_context(self, *, links, targets, written, reading_order):
        """Tool delivery and persistence share the same contextual inputs."""
        targets, written = set(targets), set(written)
        if any(set(issue.fragment_ids) - targets for issue in self.issues):
            raise ValueError("本批拒收问题须定位本批加工片段；后续影响使用 affected_fragment_ids")
        for identity in self.preserved_link_ids:
            if identity not in links:
                raise ValueError("语义检查保留了未知关联：" + identity)
            link = links[identity]["link"]
            ends = {link["from_fragment"], link["to_fragment"]}
            if not ends.intersection(targets):
                raise ValueError("只能保留本批目标的直接关联，不能顺带确认链上其他关系")
            if missing := ends - written:
                raise ValueError(f"关联 {identity} 的片段 {sorted(missing)} 尚无正文；只能保留已读取正文的直接关联")
        if set(self.affected_fragment_ids) - (set(reading_order) - targets):
            raise ValueError("影响传播引用未知片段或当前目标")


class NarrativeBatch(Record):
    texts: list[FragmentText] = Field(min_length=1, description="本批每个联合目标片段恰好一份正文，不得重复或遗漏")
    confirmed_link_ids: list[Identity] = Field(default_factory=list, description="本批已核对的实际叙述关联编号，不得重复")

    @model_validator(mode="after")
    def unique_items(self):
        if len({row.fragment_id for row in self.texts}) != len(self.texts):
            raise ValueError("每个联合目标只交付一份正文")
        if len(set(self.confirmed_link_ids)) != len(self.confirmed_link_ids):
            raise ValueError("关联确认重复")
        return self
