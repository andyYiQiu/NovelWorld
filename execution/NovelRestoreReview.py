"""按用途复用语义审核规则；验收候选独立审读、有限修复及版本绑定回执。"""
from typing import Literal

from pydantic import Field, create_model

from common.NovelRestoreSchema import (Identity, Record, MemoryApproval, UnknownTargetReview,
    NarrativeIssue, NarrativeReview, NarrativeBatchReview, NarrativePlanReview)
from common.Runtime import encode
from simulation.NovelRestoreWorldState import resolve_pointer


class SemanticCheck(Record):
    path: Identity = Field(description="required_paths 中的一项；完整审读该位置的全部判断和文字")
    verdict: Literal["supported", "contradicted", "insufficient"]
    reason: Identity = Field(description="说明事实如何支持或反驳结论；缺证时说明缺什么，不把目标当实际")
    evidence_paths: list[Identity] = Field(min_length=1, description="从 evidence 对象自身起算的真实 JSON Pointer，不带 /evidence 前缀。须逐层按实际字段和数组下标引用，不能把单条行动的 result、state_patch 当作根字段。允许引用完整行动或状态容器，不必猜测叶子字段；包括反证或显示缺证的容器。")


class SemanticReview(Record):
    checks: list[SemanticCheck] = Field(min_length=1)

    @property
    def approved(self):
        return all(check.verdict == "supported" for check in self.checks)


def validate_review(review, paths, evidence):
    covered = [check.path for check in review.checks]
    if len(covered) != len(set(covered)) or set(covered) != set(paths):
        raise ValueError("语义审核须恰好覆盖全部 required_paths，不能漏审或重复")
    errors = []
    for check in review.checks:
        for path in check.evidence_paths:
            try:
                resolve_pointer(evidence, path)
            except ValueError as error:
                errors.append(str(error))
    if errors:
        raise ValueError("\n".join(dict.fromkeys(errors)))


def review_job_for(supervisor, job, context, schema, prompt, role):
    """复用确切候选及完整依赖的审读；依赖不明时保留完整材料作保守失效。"""
    from common.Runtime import fingerprint
    from execution.NovelRestoreTaskPrompt import task_instructions
    from simulation.NovelRestoreLookup import lookup_for
    storage = getattr(supervisor.storage, "storage", supervisor.storage)
    lookup, allow_timeline = lookup_for(supervisor, role, context)
    dependency = fingerprint({"run": storage.run_id, "role": role, "context": context,
                              "lookup_scope": lookup.stamp() if lookup else None,
                              "allow_timeline": allow_timeline,
                              "schema": schema.model_json_schema(),
                              "instructions": task_instructions(prompt, context)})
    key = "review_dependencies/" + dependency
    previous = storage.get_checkpoint(key)
    saved = storage.get_job(previous) if previous else None
    selected = previous if saved and saved.get("status") == "complete" else job
    storage.set_checkpoint(key, selected)
    storage.set_checkpoint("review_coverage/" + job, {
        "run_id": storage.run_id, "version": storage.current()[0],
        "dependency": dependency, "review_job": selected, "reused": selected != job,
        "reason": "候选与完整依赖相同" if selected != job else "首次或依赖已变化，独立审读"})
    return selected


async def reviewed_task(supervisor, job, prompt, context, schema, validate, paths, *, review_evidence=None):
    """不处理人物可见返修；这些意见仅返还世界验收任务。"""
    from common.Runtime import task_id, fingerprint
    from common.Runtime import JobFailed
    owner = getattr(supervisor.storage, "storage", supervisor.storage)
    def state_stamp():
        version, state, progress = owner.current()
        return fingerprint({"version": version, "state": state.model_dump(mode="json"), "progress": progress})
    starting_state = state_stamp()
    feedback, lookups, reviews = [], [], {}
    for revision in range(supervisor.semantic_attempts()):
        generation = job if not revision else task_id(job, "semantic_repair", revision)
        request = context if not revision else {**context, "rejected_candidate": candidate_data,
                                                "semantic_feedback": feedback, "prior_lookup_evidence": lookups}
        candidate = await supervisor.run_task(generation, "world", prompt, request, schema, validate,
                                              recovery=supervisor.semantic_recovery(revision))
        candidate_data = candidate.model_dump(mode="json")
        delivery = supervisor.storage.get_job(generation)
        if delivery:
            for found in delivery.get("lookups", []):
                if found not in lookups:
                    lookups.append(found)
        evidence = review_evidence(candidate) if review_evidence is not None else dict(context)
        if lookups:
            from simulation.NovelRestoreLookup import current_lookup_evidence
            sources = current_lookup_evidence(lookups) if context["operation"] == "schedule" else list(lookups)
            if sources:
                evidence["lookup_evidence"] = sources
        required = paths(candidate)
        for path in required:
            resolve_pointer(candidate_data, path)
        review_context = {"operation": "review_semantics", "reviewed_operation": context["operation"],
                          "candidate": candidate_data, "required_paths": required, "evidence": evidence}
        # 仅复用同一任务内完全相同的候选与依据；拒收也可重放，不能因此获得批准。
        # 恢复时从持久任务重建此索引，run_task 仍核对提示、协议和完整输入身份。
        review_job = review_job_for(supervisor,
            reviews.setdefault(fingerprint(review_context), task_id("semantic", job, revision)),
            review_context, SemanticReview, "你负责独立语义审读，只检查，不改写候选。", "world_semantic")
        review = await supervisor.run_task(review_job, "world_semantic", "你负责独立语义审读，只检查，不改写候选。",
            review_context,
            SemanticReview, lambda value: validate_review(value, required, evidence))
        if state_stamp() != starting_state:
            raise JobFailed("语义审核期间世界状态已改变，不能沿用旧依据")
        if review.approved:
            owner.set_checkpoint("semantic_candidate/" + review_job,
                {"version": owner.current()[0], "candidate": fingerprint(candidate_data),
                 "review_identity": supervisor.storage.get_job(review_job)["identity"]})
            return candidate, review_job
        feedback = [check.model_dump(mode="json") for check in review.checks if check.verdict != "supported"]
    raise JobFailed("验收语义审核未通过：" + "; ".join(row["path"] + ": " + row["reason"] for row in feedback))


def _receipt_key(storage, purpose, payload):
    from common.Runtime import fingerprint
    return "semantic_acceptance/" + fingerprint({"run": storage.run_id, "version": storage.current()[0],
                                                "purpose": purpose, "payload": payload})


def seal_review(storage, purpose, payload, job):
    """批准只绑定确切交付；原候选、依据和审核保留在已有 jobs，不复制事实库。"""
    saved = storage.get_job(job)
    if not saved or saved["status"] != "complete" or not SemanticReview.model_validate(saved["result"]).approved:
        raise ValueError("没有完成且通过的独立语义审核")
    from common.Runtime import fingerprint
    reviewed = storage.get_checkpoint("semantic_candidate/" + job)
    if purpose == "wave":
        candidate = payload["assessment"]
        events = {a["position"]["event_id"]: a["completion"] for a in payload["actions"]}
        if candidate["events"] != events or candidate["goal"] != payload["goal"]:
            raise ValueError("行动与组结论必须对应同一份已审逐项目标评估")
    elif purpose == "group":
        candidate = payload["assessment"]
    elif purpose == "schedule":
        candidate = payload
    else:
        raise ValueError("未知语义回执用途")
    if reviewed != {"version": storage.current()[0], "candidate": fingerprint(candidate), "review_identity": saved["identity"]}:
        raise ValueError("待提交结论不是该语义审核实际通过的候选")
    storage.set_checkpoint(_receipt_key(storage, purpose, payload), {"job": job, "identity": saved["identity"]})


def require_review(storage, purpose, payload):
    receipt = storage.get_checkpoint(_receipt_key(storage, purpose, payload))
    saved = storage.get_job(receipt["job"]) if receipt else None
    if (not saved or saved["status"] != "complete" or saved["identity"] != receipt["identity"]
            or not SemanticReview.model_validate(saved["result"]).approved):
        raise ValueError("交付缺少当前版本和内容对应的独立语义审核：" + purpose)


def wave_delivery(actions, working_states, goal, assessment):
    return {"actions": [a.model_dump(mode="json") for a in actions], "working_states": working_states,
            "goal": goal.model_dump(mode="json"), "assessment": assessment.model_dump(mode="json")}


def group_delivery(storage, group_id, assessment):
    return {"group_id": group_id, "current_state": storage.current()[1].model_dump(mode="json"),
            "goal_records": storage.get_checkpoint("goal_records"),
            "assessment": assessment.model_dump(mode="json")}


def review_contract(schema, context):
    if schema is MemoryApproval:
        people = sorted(context["summaries"]) if context.get("operation") == "review_memory" else [context["character_id"]]
        payload_type = create_model("MemoryReviewInput", __base__=Record,
            checks=(list[list[Identity]], Field(min_length=len(people), max_length=len(people),
                description="按此顺序完整审读各人的认识与记忆；每人交付具体问题列表，无问题显式给 []：" + encode(people))),
            issues=(list[Identity], Field(description="无法归于一人的共同问题；无问题显式给 []。")),
            notes=(list[Identity], Field(default_factory=list)))

        def assemble(payload):
            value = payload_type.model_validate(payload)
            # Common issues must reach every affected person's repair, even when
            # other findings are individually located.
            located = {person: list(dict.fromkeys(value.issues + issues))
                       for person, issues in zip(people, value.checks)}
            issues = list(dict.fromkeys(value.issues + [issue for items in value.checks for issue in items]))
            return schema(approved=not issues, issues=issues, character_issues=located, notes=value.notes)
        return payload_type, assemble

    if schema not in (UnknownTargetReview, NarrativeReview, NarrativeBatchReview, NarrativePlanReview):
        return None
    fields = {"issues": (list[NarrativeIssue] if schema in (NarrativeReview, NarrativeBatchReview)
                         else list[Identity], Field(description="本次完整审读后仍需修正的问题；无问题显式给 []。"))}
    if schema is UnknownTargetReview:
        fields["basis"] = (Identity, ...)
    elif schema is NarrativePlanReview:
        fields.update(repair_scope=(Literal["all", "chapters", "links"], "all"),
                      chapter_ids=(list[Identity], Field(default_factory=list)))
    else:
        fields["summary"] = (Identity, ...)
        if schema is NarrativeBatchReview:
            fields.update(preserved_link_ids=(list[Identity], Field(default_factory=list)),
                          affected_fragment_ids=(list[Identity], Field(default_factory=list)))
    payload_type = create_model(schema.__name__ + "Input", __base__=Record, **fields)

    def assemble(payload):
        value = payload_type.model_validate(payload)
        return schema(approved=not value.issues, **value.model_dump())
    return payload_type, assemble

class SemanticJudgment(Record):
    verdict: Literal["supported", "contradicted", "insufficient"]
    reason: Identity = Field(description="说明实存依据如何支持、反驳本项判断，或缺少什么；不是仅宣布通过。")

def semantic_contract(schema, context, owner, job):
    from tools.ToolArguments import payload_data
    required = list(context["required_paths"])
    evidence = context["evidence"]
    if not required or not evidence:
        raise ValueError("独立审核必须有实际候选与依据")
    refs = ["/" + str(key).replace("~", "~0").replace("/", "~1") for key in evidence]
    payload_type = create_model("SemanticReviewInput", __base__=Record,
        checks=(list[SemanticJudgment], Field(min_length=len(required), max_length=len(required),
            description="按 required_paths 的现有顺序逐项审读，不遗漏、不代写路径。")))

    def assemble(payload):
        value = payload_type.model_validate(payload_data(payload))
        return schema(checks=[{**row.model_dump(), "path": path, "evidence_paths": list(refs)}
                              for path, row in zip(required, value.checks)])
    return payload_type, assemble
