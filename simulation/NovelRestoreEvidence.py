"""模型选择证据路径；程序从原任务实际快照绑定值，不使用发送投影或目标状态。"""
from copy import deepcopy
from typing import Annotated, Literal, Tuple

from pydantic import BaseModel, Field, WrapValidator, create_model, model_validator

from common.Runtime import CandidateError
from common.NovelRestoreSchema import (
    CompletionJudgment, GoalUpdate, GroupAssessment, Identity, Record, ResultEvidence,
    StateFactEvidence, WaveAssessment, SceneContinuation,
)
from simulation.NovelRestoreWorldState import resolve_pointer


class StateFactInput(Record):
    path: Identity = Field(pattern=r"^/(?:[^~]|~[01])*$",
        description="相对于实际快照根的 JSON Pointer，如 /characters/R1/life_status；不带状态字段前缀。字段名内 / 写为 ~1、~ 写为 ~0；数组下标从0开始。只交付 path，原值由程序读取；state_facts 对应 current_state，before_facts 对应 state_before。")

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_value(cls, value):
        # Historical deliveries included value. It is program-owned and never
        # trusted, even when it happens to match; no other extra key is ignored.
        return {k: v for k, v in value.items() if k != "value"} if isinstance(value, dict) else value


class ResultEvidenceInput(ResultEvidence):
    state_facts: list[StateFactInput] = Field(default_factory=list,
        description="当前实际快照的证据路径；程序绑定原值，不填写 value。")


class CompletionInput(CompletionJudgment):
    result_evidence: list[ResultEvidenceInput] = Field(default_factory=list,
        description=CompletionJudgment.model_fields["result_evidence"].description)


class GroupAssessmentInput(GroupAssessment):
    result_evidence: list[ResultEvidenceInput] = Field(default_factory=list,
        description=CompletionJudgment.model_fields["result_evidence"].description)


class CompletionContent(Record):
    """批内语义内容；完成标志只从对应逐项目标装配。"""
    summary: Identity
    achieved_results: list[Identity] = Field(default_factory=list)
    result_evidence: list[ResultEvidenceInput] = Field(default_factory=list)
    remaining: list[Identity] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_flags(cls, value):
        # 兼容旧回执的程序字段，不信任或要求模型重新修正这些值。
        return {k: v for k, v in value.items() if k not in {"achieved", "continuity_ok"}} if isinstance(value, dict) else value


class GroupContent(CompletionContent):
    contradictions: list[Identity] = Field(default_factory=list)


class GoalUpdateInput(GoalUpdate):
    state_facts: list[StateFactInput] = Field(default_factory=list)
    before_facts: list[StateFactInput] = Field(default_factory=list,
        description="批前实际 state_before 的证据路径；变化必须有实际行动或前后证据，不引用目标或未来状态。")


class UnchangedGoalInput(Record):
    unchanged: Literal[True]
    reason: Identity = Field(description="本轮为何不改变该项目标的判断、阻碍与下一步建议；不得把新的贡献写成沿用。")


def _validate_feedback(value, handler):
    # Select the declared form before validation, so an error points to the
    # real field rather than two competing Union branch labels.
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, dict):
        return handler(value)
    return (UnchangedGoalInput if "unchanged" in value else GoalUpdateInput).model_validate(value)


GoalFeedbackInput = Annotated[GoalUpdateInput | UnchangedGoalInput, WrapValidator(_validate_feedback)]


def _bind_fact(fact, snapshot, location, source, issues):
    try:
        fact["value"] = deepcopy(resolve_pointer(snapshot, fact["path"]))
    except ValueError as error:
        message = f"证据 {'.'.join(map(str, location))} 须读取 {source} 实际快照：{error}"
        issues.append({"loc": location, "msg": message, "type": "candidate_error"})


def _bind_facts(facts, snapshot, location, source, issues):
    for index, fact in enumerate(facts):
        _bind_fact(fact, snapshot, (*location, index, "path"), source, issues)


def _bind_result(result, current, location, issues):
    _bind_facts(result["state_facts"], current, (*location, "state_facts"), "current_state（批后）", issues)


def _bind_judgment(judgment, current, location, issues):
    for index, result in enumerate(judgment["result_evidence"]):
        _bind_result(result, current, (*location, "result_evidence", index), issues)


def evidence_contract(schema, context):
    """返回明确的任务输入与装配器；不递归猜测同名字段的业务含义。"""
    inputs = {StateFactEvidence: StateFactInput, ResultEvidence: ResultEvidenceInput,
              CompletionJudgment: CompletionInput, GroupAssessment: GroupAssessmentInput,
              GoalUpdate: GoalUpdateInput}
    goals = []
    reusable = {}
    if schema is WaveAssessment:
        from simulation.NovelRestoreGoals import reusable_goal_updates
        reusable = reusable_goal_updates(context["goal_records"], context.get("state_before"), context.get("current_state"))
        # encode() sorts mapping keys on the wire. Use the same order for IDs.
        goals = sorted(context["goal_records"])
        if not goals:
            raise ValueError("逐项目标评估必须有本组的稳定目标")
        events = sorted({action["position"]["event_id"] for action in context["actions"]})
        definitions = {identity: row.get("definition", {}) for identity, row in context["goal_records"].items()}
        event_goals = {definition["event_id"]: identity for identity, definition in definitions.items()
                       if definition.get("kind") == "event"}
        if set(events) - event_goals.keys():
            raise ValueError("本批事件缺少对应逐项目标定义，不能计算完成标志")
        # Encode reuse eligibility at each fixed list position.  A homogeneous
        # union lets the model choose ``unchanged`` for an ineligible goal and
        # postpones a deterministic constraint until after delivery.
        goal_update_types = tuple(
            GoalFeedbackInput if identity in reusable else GoalUpdateInput
            for identity in goals
        )
        goal_updates_type = Tuple[goal_update_types]
        payload_type = create_model("WaveAssessmentInput", __base__=Record,
            scene_continuations=(dict[Identity, SceneContinuation], Field(default_factory=dict,
                description="当前互动内可直接接续的人物，按事件编号索引；无合法接续时为空。")),
            events=(list[CompletionContent], Field(min_length=len(events), max_length=len(events),
                description="按本批 actions 中 position.event_id 去重后字典排序逐项评估；事件归属由程序绑定。")),
            goal=(GroupContent, ...),
            goal_updates=(goal_updates_type,
                Field(
                description="按 goal_records 键的字典排序逐项判断；不填写目标编号。程序绑定证据原值。"
                "仅 reusable_goal_ids 内且本轮语义确实未变的项目可交付 unchanged=true 与 reason；"
                "程序沿用经核对的依据，不计新收益，独立审核仍逐项进行；其他项目交付完整更新。")))
    else:
        payload_type = inputs.get(schema)
        if payload_type is None:
            return None

    def assemble(payload):
        data = payload.model_dump() if isinstance(payload, BaseModel) else payload
        data = payload_type.model_validate(data).model_dump()
        before, current = context.get("state_before"), context.get("current_state")
        issues = []
        if schema is WaveAssessment:
            for index, judgment in enumerate(data["events"]):
                _bind_judgment(judgment, current, ("result", "events", index), issues)
            data["events"] = dict(zip(events, data["events"]))
            _bind_judgment(data["goal"], current, ("result", "goal"), issues)
            data["goal_updates"] = list(data["goal_updates"])
            for index, update in enumerate(data["goal_updates"]):
                location = ("result", "goal_updates", index)
                if update.get("unchanged"):
                    identity = goals[index]
                    if identity not in reusable:
                        raise ValueError(f"目标 {identity} 的原依据已变化或不足，不能沿用；须完整评估")
                    update = deepcopy(reusable[identity])
                    update["explanation"] = data["goal_updates"][index]["reason"]
                    data["goal_updates"][index] = update
                _bind_facts(update["before_facts"], before, (*location, "before_facts"), "state_before（批前）", issues)
                _bind_facts(update["state_facts"], current, (*location, "state_facts"), "current_state（批后）", issues)
            data["goal_updates"] = dict(zip(goals, data["goal_updates"]))
            updates = data["goal_updates"]
            for event, judgment in data["events"].items():
                judgment["achieved"] = updates[event_goals[event]]["status"] == "satisfied"
            data["goal"]["achieved"] = all(update["status"] == "satisfied" for update in updates.values())
            data["goal"]["continuity_ok"] = all(updates[identity]["status"] == "satisfied"
                for identity, definition in definitions.items() if definition.get("kind") == "continuity")
        elif schema is GoalUpdate:
            _bind_facts(data["before_facts"], before, ("result", "before_facts"), "state_before（批前）", issues)
            _bind_facts(data["state_facts"], current, ("result", "state_facts"), "current_state（批后）", issues)
        elif schema in (CompletionJudgment, GroupAssessment):
            _bind_judgment(data, current, ("result",), issues)
        elif schema is ResultEvidence:
            _bind_result(data, current, ("result",), issues)
        else:
            _bind_fact(data, current, ("result", "path"), "current_state（批后）", issues)
        if issues:
            raise CandidateError("\n".join(issue["msg"] for issue in issues), issues)
        return schema.model_validate(data)
    return payload_type, assemble
