"""完整任务使用 AutoGen AssistantAgent / FunctionTool；适配本项目的回执和权限。

来源：AutoGen 0.7.5 _assistant_agent.py、tools/_function_tool.py。
工具执行和字段校验使用 SDK；不拆字段问答，不解析模型正文。
"""
from copy import deepcopy

from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import ToolCallRequestEvent, ToolCallExecutionEvent
from autogen_core import FunctionCall
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import (AssistantMessage, FunctionExecutionResult,
                                FunctionExecutionResultMessage, UserMessage)
from autogen_core.tools import FunctionTool
from pydantic import TypeAdapter, ValidationError
from jsonpointer import JsonPointer

from common.Runtime import CALL_ROLE, CALL_SCOPE, CandidateError, JobFailed, RunStopped, check_batch, fingerprint, encode
from execution.NovelRestoreTaskPrompt import task_instructions
from tools.ToolArguments import normalize_arguments
from tools.JsonPatch import amendment_plan, amendment_type, apply_amendments
from tools.ToolSchema import compact_schema
from simulation.NovelRestoreContext import TASK_VIEW_POLICY, world_task_context, amendment_view, personal_repair_view


PROTOCOL = "autogen-supervised-tasks-v20-method142"


class CheckedFunctionTool(FunctionTool):
    """使 SDK 的实际校验与 additionalProperties=false 声明一致。

    AutoGen 0.7.5 的外层参数模型默认忽略额外键，可能静默丢掉放错层级的内容。
    用 Pydantic 的原生 extra 策略校验，工具执行/错误回传仍由 AutoGen 完成。
    """
    def __init__(self, *args, on_invalid=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_invalid = on_invalid

    @property
    def schema(self):
        # FunctionTool still owns validation/execution. $refs are relative to
        # the complete parameters document, including its original $defs.
        parameters = compact_schema(self.args_type().model_json_schema())
        parameters["additionalProperties"] = False
        return {"name": self.name, "description": self.description,
                "parameters": parameters, "strict": False}

    async def run_json(self, args, cancellation_token, call_id=None):
        try:
            args = normalize_arguments(args, self.args_type(), on_invalid=self.on_invalid)
        except ValidationError as error:
            issues = error.errors(include_url=False, include_input=False, include_context=False)
            raise ValueError("工具参数校验失败：" + encode(issues, compact=True)) from None
        return await super().run_json(args, cancellation_token, call_id)


def task_system(prompt, context):
    return task_instructions(prompt, context) + (
        "\n通过 submit_result 工具一次提交本次完整结果。工具定义由程序提供，按声明参数交付；"
        "不要把 JSON 写在普通消息里，不要逐字段请求确认。格式归一、固定身份和记录装配由程序处理；"
        "若反馈缺失或错误的语义内容，使用指定工具补充，保留未受影响的内容。"
        "需要补查且有 world_lookup 工具时，先查询再提交。"
        "同一次回复可以并行调用多个只读 world_lookup；submit_result 必须单独调用一次，不能与查询混用。"
        "提交成功后任务即完成，不需额外总结。"
        "工具参数中的故事正文应充分保留实际过程、对白与个人体验，不把结构化交付当作简写要求。")


async def run_tool_task(supervisor, job, role, prompt, context, schema, validate, *, recovery=False, personal_lookup=None):
    from execution.NovelRestoreTaskContract import prepare_contract
    from simulation.NovelRestoreCharacterContract import bind_character_result
    from model.ModelClient import ModelDeliveryError, ModelTransportError, RequestContextTooLarge
    from simulation.NovelRestoreVisibility import require_visible, record_visible
    from simulation.NovelRestoreLookup import lookup_for, WorldQuery, CurrentWorldQuery

    owner = getattr(supervisor.storage, "storage", supervisor.storage)
    character = role.startswith("character_")
    if personal_lookup and not character:
        raise ValueError("本人回想工具只属于人物行为任务")
    person = context.get("character_id") if character else None
    if character:
        require_visible(owner, role, context)
        context = {**context, "character_id": "self"}
    else:
        context = {**context, "run_id": owner.run_id}
    system = task_system(prompt, context)
    lookup, allow_timeline = lookup_for(supervisor, role, context)
    wire_context = (world_task_context(personal_repair_view(context), allow_scope=lookup is not None)
                    if role.startswith("world") else context)
    if wire_context != context:
        system += TASK_VIEW_POLICY
    payload_type, assemble = prepare_contract(schema, context, owner, job)
    query_type = WorldQuery if allow_timeline else CurrentWorldQuery
    identity = fingerprint({"protocol": PROTOCOL, "role": role, "system": system, "context": context,
                            "wire_context": wire_context,
                            "contract": payload_type.model_json_schema(),
                            "lookup": query_type.model_json_schema() if lookup else None,
                            "format_repairs": supervisor.storage.config.format_repairs,
                            "supervisor_turns": supervisor.storage.config.supervisor_turns,
                            "supervisor_prompt": supervisor.recovery_instructions(),
                            "recovery": recovery, "personal_lookup": bool(personal_lookup),
                            "personal_sources": getattr(personal_lookup, "dependency", None),
                            "max_personal_lookups": owner.config.max_personal_lookups if personal_lookup else None,
                            "max_world_lookups": supervisor.storage.config.max_world_lookups if lookup else None})
    saved = supervisor.storage.get_job(job) or {
        "identity": identity, "status": "pending", "protocol": PROTOCOL,
        "turn": 0, "repairs": 0, "lookup_repairs": 0, "protocol_repairs": 0,
        "lookups": [], "agent_state": None, "phase": "supervisor" if recovery else "worker", "supervisor_turns": 0,
    }
    if saved["identity"] != identity:
        raise RunStopped("任务来源、工具契约或提示改变，不能复用旧回执：" + job)
    stamp = lookup.stamp() if lookup else None
    if lookup and saved.get("lookup_stamp", stamp) != stamp:
        raise RunStopped("世界资料已变化，不能复用或继续旧任务")
    saved["lookup_stamp"] = stamp
    if saved["status"] == "pending" and supervisor.exhausted(saved):
        saved.update(status="failed", error=saved.get("error") or "任务修复次数已用尽")
        supervisor.storage.save_job(job, saved)
    if saved["status"] == "failed":
        raise JobFailed(job + ": " + saved["error"])
    if saved["status"] == "complete":
        result = schema.model_validate(saved["result"])
        validate(result)
        if character and context.get("operation") == "initialize_subjective":
            record_visible(owner, person, {"working_state": result.model_dump()}, job)
        return result

    accepted, fatal, calls_in_turn, lookup_limit_errors = None, None, [], 0
    failure_kind, amendment_failure, repaired_arguments = None, None, None

    def check_tool():
        names = [call.name for call in calls_in_turn]
        if names == ["amend_result"] and saved.get("candidate_amendment"):
            return
        if personal_lookup and names == ["personal_recall"]:
            return
        if names != ["submit_result"] and not (lookup and names and set(names) == {"world_lookup"}):
            raise ValueError("只读 world_lookup 可以并行；submit_result 必须单独调用一次，不可与查询或另一交付混用")

    def local_amendment(args, issues):
        nonlocal amendment_failure
        allowed = amendment_plan(args, issues, tools[0].args_type())
        if allowed:
            amendment_failure = {"candidate": deepcopy(args), "errors": issues,
                                 "allowed_paths": dict(sorted(allowed.items())),
                                 "record_paths": [JsonPointer.from_parts([str(p) for p in issue["loc"]]).path
                                                  for issue in issues if issue["type"] == "candidate_record"]}

    async def submit_result(result):
        nonlocal accepted, fatal, failure_kind
        check_tool()
        try:
            candidate = assemble(result)
            if character:
                bind_character_result(candidate, person)
            validate(candidate)
        except RunStopped as error:
            fatal = error
            raise
        except CandidateError as error:
            failure_kind = "business"
            local_amendment({"result": result.model_dump(mode="json")
                            if hasattr(result, "model_dump") else result}, error.issues)
            raise
        except ValueError:
            failure_kind = "business"
            raise
        accepted = candidate
        return "完整候选已通过程序校验；独立语义审核和世界提交由外层流程执行。"

    submit_result.__annotations__ = {"result": payload_type, "return": str}
    def invalid_structure(args, issues):
        nonlocal failure_kind
        failure_kind = "structure"
        local_amendment(args, issues)

    tools = [CheckedFunctionTool(submit_result,
        "一次提交本任务完整候选。正文与必要判断一起交付；程序校验结构与业务范围。", on_invalid=invalid_structure)]

    if personal_lookup:
        from agents.NovelRestoreCharacterAgent import RecallQuery
        async def personal_recall(query):
            nonlocal fatal, lookup_limit_errors
            check_tool()
            if len(saved.get("personal_queries", [])) >= owner.config.max_personal_lookups:
                lookup_limit_errors += 1
                raise ValueError("本次回想额度已用尽，请依当前获准资料提交行为")
            # SDK 执行工具仍占人物任务槽；认知加工可能调用世界模型，先释放槽避免并发为1时互锁。
            supervisor.semaphore.release()
            try:
                result = await personal_lookup(query.model_dump())
            except (RunStopped, JobFailed) as error:
                fatal = error
                raise
            finally:
                await supervisor.semaphore.acquire()
            saved.setdefault("personal_queries", []).append(query.model_dump())
            saved["lookups"].append(result)
            return encode(result, compact=True)
        personal_recall.__annotations__ = {"query": RecallQuery, "return": str}
        tools.append(CheckedFunctionTool(personal_recall,
            "可选：按本人问题查已提交的个人经历；只返回实际可得回想。无须回想可直接提交行为，不用满额度。"))

    def invalid_patch(args, issues):
        nonlocal failure_kind
        failure_kind = "structure"
        # Typed SDK validation can reject one value before execution. Merge
        # the delivered values into a copy so valid sibling repairs survive;
        # the full original contract determines the next unresolved fields.
        values = args.get("values") if isinstance(args, dict) else None
        if isinstance(values, list) and len(values) == len(saved["candidate_amendment"]["allowed_paths"]):
            try:
                merge_amendment(values)
            except (ValidationError, ValueError):
                pass

    def merge_amendment(values):
        nonlocal repaired_arguments
        plan = saved["candidate_amendment"]
        # The code owns paths and operations; the model supplies only meanings.
        edits = [{"op": op, "path": path, "value": value}
                 for (path, op), value in zip(plan["allowed_paths"].items(), values)]
        repaired_arguments = apply_amendments(plan["candidate"], edits, plan["allowed_paths"], tools[0].args_type(),
            record_paths=plan.get("record_paths", ()), on_invalid=local_amendment)

    async def amend_result(values) -> str:
        nonlocal failure_kind
        check_tool()
        failure_kind = "structure"
        try:
            merge_amendment(TypeAdapter(amend_result.__annotations__["values"]).dump_python(values, mode="json"))
        except ValidationError as error:
            issues = error.errors(include_url=False, include_input=False, include_context=False)
            raise ValueError("待补内容尚未通过校验：" + encode(issues, compact=True)) from None
        return await submit_result(repaired_arguments["result"])

    if lookup:
        async def world_lookup(query) -> str:
            nonlocal fatal, lookup_limit_errors
            check_tool()
            if lookup.stamp() != stamp:
                fatal = RunStopped("世界查询资料已经变化")
                raise fatal
            if len(saved["lookups"]) >= supervisor.storage.config.max_world_lookups:
                lookup_limit_errors += 1
                raise ValueError("世界查询额度已用尽；已读资料保留，请依据现有资料调用 submit_result 完整交付")
            result = lookup.query(query, stamp=stamp, allow_timeline=allow_timeline, candidate_context=wire_context)
            if lookup.stamp() != stamp:
                fatal = RunStopped("世界查询期间资料已经变化")
                raise fatal
            saved["lookups"].append(deepcopy(result))
            return encode(result, compact=True)

        world_lookup.__annotations__ = {"query": query_type, "return": str}
        tools.append(CheckedFunctionTool(world_lookup,
            "只读本线世界资料。history仅返回客观经过、提议与变化，shared_advances保存共享经过；"
            "个人体验、记忆及当时认识用kind=action查已提交行动。当前批次候选尚未提交，直接审读本任务材料。"
            "严格遵守本任务来源权限。"))

    model_context = UnboundedChatCompletionContext()
    active_tools = tools[:1] if saved.get("lookup_closed") else tools
    agent = supervisor.task_agent(role, system, active_tools, model_context, saved)
    if saved["agent_state"] is not None:
        await agent.load_state(saved["agent_state"])
    if not await model_context.get_messages():
        await model_context.add_message(UserMessage(source="user", content=encode({"context": wire_context}, compact=True)))
    saved["agent_state"] = await agent.save_state()
    supervisor.storage.save_job(job, saved)

    async def compact_context(*, force=False):
        nonlocal model_context, agent
        compact = await supervisor.retained_context(model_context, saved,
            recovery=saved["phase"] == "supervisor")
        before = encode([m.model_dump() for m in await model_context.get_messages()])
        after = encode([m.model_dump() for m in await compact.get_messages()])
        if not force and len(after) >= len(before):
            return False
        model_context = compact
        agent = supervisor.task_agent(role, system, active_tools, model_context, saved)
        saved["agent_state"] = await agent.save_state()
        saved["context_compactions"] = saved.get("context_compactions", 0) + 1
        return True

    try:
        while True:
            check_batch()
            if (lookup or personal_lookup) and not saved.get("lookup_closed") and (
                    (len(saved["lookups"]) >= supervisor.storage.config.max_world_lookups if lookup else
                     len(saved.get("personal_queries", [])) >= owner.config.max_personal_lookups)
                    or saved["phase"] == "supervisor" and saved["supervisor_turns"] >= supervisor.storage.config.supervisor_turns - 1):
                saved["lookup_closed"] = True
                await model_context.add_message(UserMessage(source="user",
                    content="本任务查询已关闭；已读资料全部保留，请依据现有资料调用 submit_result 完整交付，不再查询。"))
                active_tools = tools[:1]
                agent = supervisor.task_agent(role, system, active_tools, model_context, saved)
                saved["agent_state"] = await agent.save_state()
                supervisor.storage.save_job(job, saved)
            accepted, fatal, calls_in_turn, lookup_limit_errors = None, None, [], 0
            failure_kind, amendment_failure, repaired_arguments = None, None, None
            task_agent = agent
            if saved.get("candidate_amendment"):
                # Semantic edits retain the original sources and one candidate;
                # previous rejected drafts are not sent again.
                repair_context = UnboundedChatCompletionContext()
                plan = saved["candidate_amendment"]
                amend_result.__annotations__["values"] = amendment_type(tools[0].args_type(), plan["allowed_paths"])
                amend_tool = CheckedFunctionTool(amend_result,
                    "按 revision_items 顺序一次提交修正值，类型由本工具声明。不要另写补丁位置或操作符；"
                    "证据字段可交付原任务要求的依据引用。程序合并后执行完整校验及独立审核。", on_invalid=invalid_patch)
                repair_sources, repair_candidate = amendment_view(context, plan["candidate"], plan["allowed_paths"],
                    world=role.startswith("world"), allow_scope=lookup is not None and not saved.get("lookup_closed"))
                await repair_context.add_message(UserMessage(source="user", content=encode({
                    "context": repair_sources, "candidate": repair_candidate, "errors": plan["errors"],
                    "revision_items": list(plan["allowed_paths"]),
                    **({"last_error": plan["last_error"]} if plan.get("last_error") else {}),
                    "prior_lookup_evidence": supervisor.lookup_evidence(saved),
                    "lookup_closed": bool(saved.get("lookup_closed"))},
                    compact=True)))
                task_agent = supervisor.task_agent(role, system +
                    "\n本次进入局部返修：依据原任务资料判断错误字段，通过 amend_result 一次提交全部修正。"
                    "不再调用 submit_result；仅判断 revision_items 对应的内容，位置与操作由程序确定。"
                    "需要资料且查询未关闭时可使用 world_lookup。", [amend_tool, *active_tools[1:]], repair_context, saved)
            failures, finished = [], False
            async with supervisor.semaphore:
                check_batch()
                scope = CALL_SCOPE.set((job, PROTOCOL, saved["turn"]))
                role_token = CALL_ROLE.set("supervisor" if saved["phase"] == "supervisor" else
                                           "character" if character else role)
                try:
                    try:
                        async for event in task_agent.run_stream(task=None, output_task_messages=False):
                            if isinstance(event, ToolCallRequestEvent):
                                calls_in_turn = event.content
                            elif isinstance(event, ToolCallExecutionEvent):
                                failures.extend(r.content for r in event.content if r.is_error)
                            elif isinstance(event, TaskResult):
                                finished = True
                    except RequestContextTooLarge:
                        if saved.get("candidate_amendment"):
                            raise
                        if not await compact_context():
                            raise
                        supervisor.storage.save_job(job, saved)
                        continue  # 未发送 HTTP，不消耗请求或修复轮次。
                    except ModelDeliveryError as error:
                        # HTTP 回执已保存，但接口未完整交付消息。未知送达不进入本分支。
                        finished = True
                        failure_kind = ("lookup" if error.calls and all(call.name in {"world_lookup", "personal_recall"} for call in error.calls)
                                        else "syntax")
                        failures.append(str(error))
                        if error.calls:
                            # Pi / Pydantic AI: retain the rejected call and return
                            # an error on that call, rather than retrying blind.
                            # Arguments occur once, not again inside the error.
                            calls_in_turn = error.calls
                            await model_context.add_message(AssistantMessage(source=role, content=error.calls,
                                                                             thought=error.thought))
                            await model_context.add_message(FunctionExecutionResultMessage(content=[
                                FunctionExecutionResult(call_id=call.id, name=call.name, is_error=True,
                                    content="上次接口未交付可用的工具调用：" + error.errors.get(call.id,
                                        "同批其他调用的参数无效，本次调用未执行；请连同失败调用重新提交。")
                                    + "。请依据原任务重新交付完整语义结果，保留未受影响的内容。")
                                for call in error.calls]))
                        else:
                            await model_context.add_message(UserMessage(source="user",
                                content="上次接口未交付可用的工具调用：" + str(error)[:1000]
                                        + "。请根据此错误重新完整交付本次任务。"))
                finally:
                    CALL_SCOPE.reset(scope)
                    CALL_ROLE.reset(role_token)
            if fatal:
                saved.update(turn=saved["turn"] + 1, error=str(fatal), agent_state=await agent.save_state())
                supervisor.storage.save_job(job, saved)
                raise fatal
            if not finished:
                raise JobFailed("SDK 没有交付完整任务结果")
            if saved["phase"] == "supervisor":
                saved["supervisor_turns"] += 1
            if lookup and lookup.stamp() != stamp:
                raise RunStopped("回答期间世界资料已变化，不能接收旧交付")
            if accepted is not None and not failures:
                saved.pop("candidate_amendment", None)
                saved.update(status="complete", error=None, result=accepted.model_dump(mode="json"),
                             agent_state=await agent.save_state())
                supervisor.storage.save_job(job, saved)
                if character and context.get("operation") == "initialize_subjective":
                    record_visible(owner, person, {"working_state": accepted.model_dump()}, job)
                return accepted
            names = [call.name for call in calls_in_turn]
            if not failures and names and set(names) in ({"world_lookup"}, {"personal_recall"}):
                # Query receipts are already persisted in saved.lookups. Drop
                # the model's exploratory reasoning and native tool transcript
                # before the follow-up so the same evidence is sent once.
                await compact_context(force=True)
                decision = "stop" if supervisor.exhausted(saved) else "retry"
            elif failures or not calls_in_turn:
                # AutoGen reports FunctionTool errors in the native conversation.
                # A failed read or routing mistake must not spend result repairs.
                saved["failure_kind"] = failure_kind or (
                    "lookup" if names and set(names) in ({"world_lookup"}, {"personal_recall"}) else "protocol")
                saved["error"] = "；".join(failures) if failures else "未调用 submit_result 或 world_lookup 工具"
                if amendment_failure:
                    saved["candidate_amendment"] = amendment_failure
                elif saved.get("candidate_amendment"):
                    if failure_kind == "business":
                        saved.pop("candidate_amendment")
                        call_id = calls_in_turn[0].id
                        await model_context.add_message(AssistantMessage(source=role, content=[FunctionCall(
                            id=call_id, name="submit_result", arguments=encode(repaired_arguments, compact=True))]))
                        await model_context.add_message(FunctionExecutionResultMessage(content=[
                            FunctionExecutionResult(call_id=call_id, name="submit_result", is_error=True,
                                content="候选已通过结构校验，请依据原任务资料修复业务问题：" + saved["error"])]))
                        # Local repair may have queried evidence outside the
                        # original conversation. Restore it even if it adds size.
                        await compact_context(force=True)
                    else:
                        saved["candidate_amendment"]["last_error"] = saved["error"]
                if not calls_in_turn and not failures:
                    await model_context.add_message(UserMessage(
                        source="user", content="请调用已提供的工具交付完整结果；普通正文不会被解析成状态。"))
                decision = ("retry" if lookup_limit_errors and len(failures) == lookup_limit_errors
                            and not supervisor.exhausted(saved) else
                            supervisor.recovery_decision(saved, names, bool(lookup or personal_lookup)))
            else:
                decision = "stop" if supervisor.exhausted(saved) else "retry"
                if decision == "stop":
                    saved["error"] = "总 agent 的恢复调用上限已用尽，尚未交付可接受结果"
            if decision == "supervisor":
                agent, model_context = await supervisor.start_recovery(role, system, active_tools, model_context, saved)
            elif decision == "retry" and failures and [call.name for call in calls_in_turn] == ["submit_result"]:
                # Repair needs the latest complete candidate and every source,
                # not all superseded candidates. Reuse the same SDK transform
                # before another paid request, not only after hitting a limit.
                await compact_context()
            exhausted = decision == "stop"
            saved.update(turn=saved["turn"] + 1, agent_state=await agent.save_state())
            if exhausted:
                saved.update(status="failed", error=saved.get("error") or "总 agent 的恢复调用上限已用尽")
            supervisor.storage.save_job(job, saved)
            if exhausted:
                raise JobFailed(saved["error"])
    except (ModelTransportError, RunStopped) as error:
        saved.update(failure_kind="transport" if isinstance(error, ModelTransportError) else "protected",
                     error=str(error), agent_state=await agent.save_state())
        supervisor.recovery_decision(saved, [], bool(lookup))
        supervisor.storage.save_job(job, saved)
        raise
    except JobFailed as error:
        if saved["status"] != "failed" or saved.get("error") != str(error):
            saved.update(status="failed", error=str(error))
            supervisor.storage.save_job(job, saved)
        raise JobFailed(job + ": " + str(error)) from error
