"""世界交付先独立审读，人物请求只接受已放行的确切内容。"""
from hashlib import sha256

from pydantic import Field, model_validator

from common.Runtime import fingerprint, task_id
from common.Runtime import JobFailed, RunStopped
from simulation.NovelRestoreObjects import effective_catalog, local_perception_ids
from common.NovelRestoreSchema import CharacterState, Identity, PersonalPerception, Record
from execution.NovelRestoreTaskPrompt import load_prompt


class VisibilityIssue(Record):
    character_id: Identity
    path: Identity
    reason: Identity


class VisibilityReview(Record):
    approved: bool = Field(strict=True)
    covered_characters: list[Identity] = Field(description="恰好为 candidates 的人物键，不含只在 evidence 中出现的人物；candidates={} 时必须为 []")
    issues: list[VisibilityIssue] = Field(default_factory=list)
    world_issues: list[Identity] = Field(default_factory=list, description="仅 resolve_wave 审核使用：世界裁定的实际结果、状态变化、认知作用、自主行为或观察接收者选择错误；仅世界可见")

    @model_validator(mode="after")
    def consistent(self):
        if self.approved == bool(self.issues or self.world_issues):
            raise ValueError("通过不得遗留知情边界问题；拒收须指出人物、位置和依据")
        return self


class PersonalVisibilityReview(VisibilityReview):
    world_issues: list[Identity] = Field(default_factory=list, max_length=0,
        description="本任务只审核个人交付，此项必须为空；问题须在 issues 中给出接收者、位置和依据。")


class EnvironmentVisibilityReview(VisibilityReview):
    environment_checks: dict[Identity, Identity] = Field(default_factory=dict,
        description="通过时按 environment_action_ids 逐项给出现场依据、实际后果和反应边界；拒收可省略。仅世界侧留档。")


class CharacterFeedback(Record):
    messages: dict[Identity, list[Identity]]


class PersonalPerceptionRepair(Record):
    views: dict[Identity, PersonalPerception] = Field(default_factory=dict)
    actions: dict[Identity, dict[Identity, PersonalPerception]] = Field(default_factory=dict)
    advance: dict[Identity, PersonalPerception] = Field(default_factory=dict)
    final_views: dict[Identity, PersonalPerception] = Field(default_factory=dict)


def apply_personal_repair(candidate, repair):
    """只接受个人文字替换，私有对应、实际结果与状态沿用已裁定内容。"""
    data = candidate.model_dump(mode="json")
    def replace(views, changes):
        if set(views) != set(changes):
            raise ValueError("个人反馈返修须覆盖原有接收者，不能新增或删除")
        for person, value in changes.items():
            views[person]["personal"] = value.model_dump(mode="json")
    replace(data.get("views", {}), repair.views)
    actions = {a["action_id"]: a["observations"] for a in data.get("actions", [])}
    if set(actions) != set(repair.actions):
        raise ValueError("个人反馈返修须覆盖原有行动，不能新增或删除")
    for identity, views in actions.items():
        replace(views, repair.actions[identity])
    replace(data.get("advance", {}).get("observations", {}), repair.advance)
    replace(data.get("final_views", {}), repair.final_views)
    return type(candidate).model_validate(data)


def private_perceptions(value):
    """修可见措辞时，实际结果、状态补丁和对象对应均保持原值。"""
    data = value.model_dump(mode="json")
    views = list(data.get("views", {}).values())
    for action in data.get("actions", []):
        views.extend(action["observations"].values())
    views.extend(data.get("advance", {}).get("observations", {}).values())
    views.extend(data.get("final_views", {}).values())
    for view in views:
        view.pop("personal")
    return data


def _receipt_key(storage, person, field, value):
    if field == "working_state":
        value = CharacterState.model_validate(value).model_dump(mode="json")
    return "visibility/" + fingerprint({"run": storage.run_id, "version": storage.current()[0],
                                       "person": person, "field": field, "value": value})


def _working_state_key(person, value):
    # 先按统一状态模型归一默认字段，避免等价记录失去既有来源。
    state = CharacterState.model_validate(value).model_dump(mode="json")
    return "working_state_approval/" + fingerprint({"person": person, "state": state})


def working_state_approval(storage, person, value):
    """只证明本人当前已存的确切心理；批准来源随共同历史快照继承。"""
    if CharacterState.model_validate(value) != storage.character_state(person):
        return None
    return storage.get_checkpoint(_working_state_key(person, value))


def visibility_receipts(storage, person, fields, source, *, state_origin=None):
    """生成当前版本的确切资料收据；调用方可与已审核的业务提交处于同一事务。"""
    receipts = {_receipt_key(storage, person, field, value): {"source": source} for field, value in fields.items()}
    if "working_state" in fields:
        key = _working_state_key(person, fields["working_state"])
        # 只在真正放行状态时立据；普通 character checkpoint 写入不会获得证明。
        # 保留最初的批准来源，刷新版本收据不能将原来源改成一次机械复制。
        origin = ({**state_origin, "derived_from": state_origin["content_fingerprint"]} if state_origin is not None
                  else {"source": source, "run_id": storage.run_id, "version": storage.current()[0]})
        receipts[key] = storage.get_checkpoint(key) or {
            **origin, "content_fingerprint": key.removeprefix("working_state_approval/")}
    return receipts


def record_visible(storage, person, fields, source):
    """只保存内容指纹和批准来源，不另复制一份世界或人物资料。"""
    storage.set_checkpoints(visibility_receipts(storage, person, fields, source))


def is_visible(storage, person, field, value):
    return bool(storage.get_checkpoint(_receipt_key(storage, person, field, value)))


def valid_personal_profile(value):
    return value == {} or (isinstance(value, dict) and set(value) == {"facts"}
                           and isinstance(value["facts"], list)
                           and all(isinstance(fact, str) and fact.strip() for fact in value["facts"]))


def is_initial_profile(storage, person, value):
    """初始自知资料按人和确切内容绑定，可与共同历史一并继承。"""
    profiles = storage.get_checkpoint("self_profiles", {})
    return (valid_personal_profile(value) and person in profiles and profiles[person] == value
            and storage.get_checkpoint("initial_profile_approvals", {}).get(person) == fingerprint([person, value]))


def require_visible(storage, role, context):
    """最终发送边界的确定性检查；漏接检查、换人、改内容或换版本均不能放行。"""
    person = context.get("character_id")
    if person not in storage.manifest["catalog"]["role"] or role != "character_" + sha256(person.encode()).hexdigest()[:12]:
        raise JobFailed("人物输入缺少正确的接收者身份")
    operation = context.get("operation")
    fields = {
        None: {"profile", "perception", "working_state", "memories"},
        "initialize_subjective": {"profile", "perception"},
        "summarize_memory": {"profile", "observations", "feedback", "working_state", "memories"},
    }.get(operation)
    metadata = {"character_id"} if operation is None else {"character_id", "operation"}
    if fields is None or set(context) != fields | metadata:
        raise JobFailed("人物输入含未授权字段或缺少约定字段")
    if operation is None:
        access = storage.get_checkpoint("access/" + person)
        if not access or access.get("profile") is None or context["profile"] != access["profile"]:
            raise JobFailed("人物行动必须使用本批已准备的本人可知档案")
    for field, key, prefix in (("memories", "memory_id", "r"), ("observations", "source_id", "o")):
        if any(row.get(key) != f"{prefix}{i}" for i, row in enumerate(context.get(field, []), 1)):
            raise JobFailed("人物输入只接受本次交付的局部引用：" + field)
    views = [context.get("perception") or {}, *(o["observation"] for o in context.get("observations", []))]
    aliases = [o["perceived_id"] for view in views for o in view.get("objects", [])]
    if aliases and not local_perception_ids(aliases, effective_catalog(storage)):
        raise JobFailed("人物输入的局部感知编号不能占用真实资产编号")
    for field in fields:
        value = context[field]
        if field == "profile":
            if is_initial_profile(storage, person, value) or (
                    valid_personal_profile(value) and is_visible(storage, person, field, value)):
                continue
            raise JobFailed("人物档案缺少本人确切内容的知情批准")
        if value not in (None, {}, [], "") and not is_visible(storage, person, field, value):
            raise JobFailed("人物输入尚未通过知情边界检查：" + field)


async def visible_task(supervisor, job, prompt, context, schema, validate, project, *, fixed=None, initial=None,
                       private_review=None, observer_evidence=None, world_review=None):
    """候选→独立审读→世界有界返修；只返回通过的交付。

    world_review 仅使用裁定可读的事实依据；它的具体问题可原样返还裁定。
    private_review 的具体意见只返还世界生产者；人物只接收独立审查通过的个人交付。
    """
    storage = supervisor.storage
    original, feedback, world_feedback, lookup_evidence = None, [], [], []
    reviews = {}
    world_resolution = context.get("operation") == "resolve_wave"
    if world_review is not None and not world_resolution:
        raise ValueError("额外世界事实审核仅用于未提交的 resolve_wave 候选")
    review_schema = VisibilityReview if world_resolution else PersonalVisibilityReview
    review_prompt = prompt if "<!-- tasks:" in prompt else load_prompt()
    environment_ids = set()
    if world_resolution:
        environment_ids = {row["action_id"] for row in context.get("intents", [])
                           if row.get("proposal", {}).get("character_id", "") is None}
        if environment_ids:
            review_schema = EnvironmentVisibilityReview
    attempts = supervisor.semantic_attempts()
    for revision in range(attempts):
        generation = job if not revision else task_id(job, "visibility_repair", revision)
        request = context if not revision else {**context, "unsafe_delivery": candidate.model_dump(mode="json"),
                                                "visibility_feedback": feedback}
        if world_feedback:
            request["world_feedback"] = world_feedback
            request["repair_world_resolution"] = True
        def check(value):
            validate(value)
            if original is not None and fixed is not None and fixed(value) != original:
                raise ValueError("知情边界修复只能修改个人交付，不得改写已经裁定的实际结果或私有依据")
        if revision and fixed is private_perceptions and not world_feedback:
            request["repair_personal_only"] = True
            from simulation.NovelRestoreVisibilityContract import repair_layout
            request["repair_targets"] = repair_layout(request)[1]
            def check_repair(value):
                check(apply_personal_repair(candidate, value))
            repair = await supervisor.run_task(generation, "world", prompt, request, PersonalPerceptionRepair, check_repair,
                                              recovery=supervisor.semantic_recovery(revision))
            candidate = apply_personal_repair(candidate, repair)
        else:
            candidate = initial if initial is not None and revision == 0 else await supervisor.run_task(
                generation, "world", prompt, request, schema, check,
                recovery=supervisor.semantic_recovery(revision))
        check(candidate)
        if original is None and fixed is not None:
            original = fixed(candidate)
        people = project(candidate)
        if not people and private_review is None and world_review is None:
            return candidate
        def check_review(review):
            if context.get("operation") == "repair_character_feedback":
                for issue in review.issues:
                    root = issue.path.lstrip("/").split("/", 1)[0].split(".", 1)[0].split("[", 1)[0]
                    if root in {"experiences", "working_state", "memories", "observations", "perception", "profile"}:
                        raise ValueError("本次候选只有 feedback 返修提示；只能定位 feedback 中的泄密或无依据断言。"
                                         "experiences、working_state 等是待后续人物返修的问题，不是本次候选。")
            if review.world_issues and not world_resolution:
                raise ValueError("本任务不是世界裁定，world_issues 必须为空；请在 issues 中按人物、位置和依据说明问题")
            if (len(review.covered_characters) != len(people) or set(review.covered_characters) != set(people)
                    or any(issue.character_id not in people for issue in review.issues)):
                raise ValueError(f"知情边界检查必须覆盖全部接收者：covered_characters 必须为 {sorted(people)}，不能重复或加入仅供世界取证的人物；"
                                 f"实际收到 {review.covered_characters}。issues 只能属于这些接收者")
            if environment_ids:
                checks = review.environment_checks
                if (set(checks) - environment_ids or any(not reason.strip() for reason in checks.values())
                        or review.approved and set(checks) != environment_ids):
                    raise ValueError(f"environment_checks 只能核对 {sorted(environment_ids)}，通过时须逐项说明现场依据、后果及反应边界")
        review_context = {"operation": "review_visibility", "candidates": people, "evidence": context}
        if environment_ids:
            review_context["environment_action_ids"] = sorted(environment_ids)
        delivery_job = storage.get_job(generation) if initial is None or revision else None
        if delivery_job and delivery_job.get("lookups"):
            for found in delivery_job["lookups"]:
                if found not in lookup_evidence:
                    lookup_evidence.append(found)
        if lookup_evidence:
            review_context["lookup_evidence"] = lookup_evidence
        if private_review is not None:
            review_context["private_evidence"] = private_review(candidate)
        if observer_evidence is not None:
            review_context["observer_evidence"] = observer_evidence(candidate)
        # 同一候选、私有依据与个人来源完全未变时重放原审核；任何差异仍独立审读。
        review_key = fingerprint({"context": review_context, "version": storage.current()[0]})
        review_job = reviews.setdefault(review_key, task_id("visibility", job, revision))
        from execution.NovelRestoreReview import review_job_for
        review_job = review_job_for(supervisor, review_job, review_context, review_schema, review_prompt, "world_visibility")
        bounded = "approved_boundaries" in review_context.get("private_evidence", {})
        review = await supervisor.run_task(review_job, "world_visibility", review_prompt,
            review_context, review_schema, check_review)
        # 返修任务始终由世界承担，可读取本线私有依据；保留定位和具体原因。
        # 这些意见不直接交给人物，合并后的个人交付仍须独立通过知情审核。
        feedback = [issue.model_dump() for issue in review.issues]
        world_issues = review.world_issues
        if review.approved and world_review is not None:
            # 未知对象等补充事实检查与原候选共用有限返修预算；不能在审核后直接硬停，
            # 也不能在补充审核尚未通过时向人物放行该候选。
            world_issues = await world_review(candidate, revision)
        if world_issues:
            if revision == attempts - 1:
                raise RunStopped("世界裁定未通过事实与人物自主性检查：" + "; ".join(world_issues))
            # 候选尚未提交；事实本身有问题时需重新裁定，不能只改个人措辞。
            original = None
            # The resolver is the world role and may read the schedule's
            # approved stop plus world-side audit findings.  Keep a concise
            # repair direction, but never replace the exact failure with it:
            # doing so makes repeated re-adjudication unable to fix the
            # objective action, elapsed time or state patch that was rejected.
            guidance = ["请按本批提议与现场依据重审实际动作、经过时间及共享后果，保留仍待人物自主作出的选择。"] if bounded else []
            world_feedback = list(dict.fromkeys([*guidance, *world_issues]))
            continue
        world_feedback = []
        if review.approved:
            for person, fields in people.items():
                record_visible(storage, person, fields, review_job)
            return candidate
    raise JobFailed(job + "：人物交付未通过知情边界检查：" + "; ".join(
        f'{issue["character_id"]} {issue["path"]}：{issue["reason"]}' for issue in feedback))
