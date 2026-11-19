"""工程总 agent：任务分发、故障升级与独立模型恢复，不裁定故事事实。

复用 AutoGen 0.7.5 AssistantAgent 的工具执行/状态；升级顺序参考
LangGraph PregelRunner 的 error-handler-after-retries。业务权限是本项目适配。
"""
import asyncio
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, FunctionExecutionResultMessage, UserMessage

from common.Runtime import RunStopped, encode
from storage.NovelRestoreLog import NovelRestoreLog


# 分类由失败发生的位置提供；不从错误文案猜测，也不允许模型改变策略。
RECOVERY_POLICIES = {
    "format": ("repairs", "本地归一后仍未形成完整交付，由原任务补充内容"),
    "syntax": ("repairs", "完整内容无法可靠读取，由原任务重新交付语义结果"),
    "structure": ("repairs", "保留已归一的候选，仅补充仍缺失或有误的内容"),
    "business": ("repairs", "业务校验未通过，保留来源和候选并由原校验重新验收"),
    "lookup": ("lookup_repairs", "只修复查询或根据已有资料收束，不消耗结果返修额度"),
    "protocol": ("protocol_repairs", "纠正工具调用方式，不扩大原任务权限"),
    "transport": (None, "传输失败由接口层处理；总 agent 不重发送达情况未知的请求"),
    "budget": (None, "预算或执行保护已触发；总 agent 不绕过限制继续请求"),
    "protected": (None, "来源、权限、预算或送达保护触发；保存现场，不启动模型恢复"),
}


class NovelRestoreSupervisorAgent:
    def __init__(self, storage, *, client=None, responder=None):
        if (client is None) == (responder is None):
            raise ValueError("必须明确提供一个执行后端")
        self.storage, self.client, self.responder = storage, client, responder
        self.semaphore = asyncio.Semaphore(storage.config.concurrency)

    async def run_task(self, job, role, prompt, context, schema, validate=lambda result: None, *, recovery=False,
                       personal_lookup=None):
        """三个领域角色共用此入口；正常路径不增加总 agent 模型请求。"""
        NovelRestoreLog.report("task_started", job=job, role=role)
        if self.client is not None:
            from execution.NovelRestoreToolTask import run_tool_task
            result = await run_tool_task(self, job, role, prompt, context, schema, validate, recovery=recovery,
                                       personal_lookup=personal_lookup)
        else:
            from execution.NovelRestoreScriptedBackend import run_record_task
            result = await run_record_task(self, job, role, prompt, context, schema, validate, personal_lookup=personal_lookup)
        NovelRestoreLog.report("task_completed", job=job, role=role)
        return result

    def semantic_attempts(self):
        """正常语义返修后至多一次总管候选修复，仍须由原独立审核者验收。"""
        return self.storage.config.semantic_repairs + 1 + int(
            self.client is not None and self.storage.config.supervisor_turns > 0)

    def semantic_recovery(self, revision):
        return revision > self.storage.config.semantic_repairs

    def exhausted(self, saved):
        if saved.get("phase", "worker") == "supervisor":
            return saved.get("supervisor_turns", 0) >= self.storage.config.supervisor_turns
        return any(saved.get(key, 0) > self.storage.config.format_repairs
                   for key in ("repairs", "lookup_repairs", "protocol_repairs"))

    def recovery_decision(self, saved, names, has_lookup):
        """明确分类、有限返修、一次升级；保护异常仍应由调用方原样抛出。"""
        fallback = ("format" if names == ["submit_result"] else
                    "lookup" if has_lookup and names and set(names) in ({"world_lookup"}, {"personal_recall"}) else "protocol")
        kind = saved.get("failure_kind", fallback)
        counter, reason = RECOVERY_POLICIES.get(kind, (None, "未知故障分类，保留现场并停止自动恢复"))
        if counter is None:
            decision = "stop"
        elif saved.get("phase", "worker") == "supervisor":
            decision = "stop" if self.exhausted(saved) else "retry"
            reason += "；总 agent 调用额度已用尽" if decision == "stop" else "；在剩余接管额度内修复"
        else:
            saved[counter] = saved.get(counter, 0) + 1
            decision = ("retry" if not self.exhausted(saved) else
                        "supervisor" if self.storage.config.supervisor_turns else "stop")
            reason += {"retry": "；使用剩余原任务返修额度", "supervisor": "；原任务额度已尽，升级一次",
                       "stop": "；原任务额度已尽且未启用模型接管"}[decision]
        saved["recovery_route"] = {"failure_kind": kind, "action": decision, "reason": reason}
        return decision

    def task_agent(self, role, system, tools, model_context, saved):
        recovery = saved.get("phase", "worker") == "supervisor"
        if recovery:
            system += "\n\n" + self.recovery_instructions()
        return AssistantAgent("supervisor" if recovery else role, self.client, system_message=system,
                              tools=tools, model_context=model_context, max_tool_iterations=1,
                              reflect_on_tool_use=False, model_client_stream=False)

    @staticmethod
    def recovery_instructions():
        return (Path(__file__).resolve().parents[1] /
                "prompts/NovelRestoreSupervisorPrompt.md").read_text(encoding="utf-8")

    @staticmethod
    def lookup_evidence(saved):
        """请求中只发送一份相同查询依据，全部查询记录及额度仍原样保留。"""
        evidence = []
        for found in saved["lookups"]:
            if found not in evidence:
                evidence.append(found)
        return evidence

    async def retained_context(self, previous_context, saved, *, recovery=False):
        """借鉴 Pi transformContext，使用 SDK 上下文保留来源与最近失败交付。

        不摘要或截断故事资料；全部回执仍在账本。查询证据只发送一份，
        最后候选连同原生工具结果成对保留，不把所有失败正文再次拼入上下文。
        """
        messages = await previous_context.get_messages()
        context = UnboundedChatCompletionContext()
        if not messages:
            return context
        await context.add_message(messages[0])
        # 后续的普通闲聊不能挤掉待修的完整交付及其工具错误。
        last_call = next((i for i in range(len(messages) - 1, 0, -1)
                          if isinstance(messages[i], AssistantMessage)
                          and isinstance(messages[i].content, list)
                          and any(call.name == "submit_result" for call in messages[i].content)), None)
        if last_call is None:
            last_call = next((i for i in range(len(messages) - 1, 0, -1)
                          if isinstance(messages[i], AssistantMessage) and not (
                              isinstance(messages[i].content, list) and messages[i].content
                              and all(call.name == "world_lookup" for call in messages[i].content))), None)
        retained = [] if last_call is None else [messages[last_call]]
        if last_call is not None and last_call + 1 < len(messages) and isinstance(
                messages[last_call + 1], FunctionExecutionResultMessage):
            retained.append(messages[last_call + 1])
        note = {"prior_lookup_evidence": self.lookup_evidence(saved),
                "instruction": "原任务资料完整保留；依据已读资料和最后候选交付，不重复查询已有依据。"}
        # 原生工具结果已包含同一错误时不再复制；后续新路由错误仍须保留。
        if saved.get("error") and not any(saved["error"] in result.content
                for message in retained if isinstance(message, FunctionExecutionResultMessage)
                for result in message.content if result.is_error):
            note["last_error"] = saved["error"]
        if saved.get("lookup_closed"):
            note["instruction"] += "查询已关闭，请调用 submit_result 完整交付。"
        if recovery:
            # 升级原因完整留在任务账本；线上只保留升级身份与最新有效反馈。
            note["engineering_recovery"] = {key: value for key, value in saved.get("recovery", {}).items()
                                             if key != "cause"}
        await context.add_message(UserMessage(source="user", content=encode(note, compact=True)))
        for message in retained:
            await context.add_message(message)
        return context

    async def start_recovery(self, role, system, tools, previous_context, saved):
        route = saved.get("recovery_route", {})
        kind = saved.get("failure_kind", route.get("failure_kind", "format"))
        if route.get("action") == "stop" or RECOVERY_POLICIES.get(kind, (None,))[0] is None:
            raise RunStopped("此类故障不允许升级模型接管：" + str(kind))
        saved.update(phase="supervisor", supervisor_turns=0,
                     recovery={"role": role, "started_after_turn": saved["turn"], "cause": saved["error"],
                               "failure_kind": kind, "route_reason": route.get("reason", RECOVERY_POLICIES[kind][1])})
        context = await self.retained_context(previous_context, saved, recovery=True)
        return self.task_agent(role, system, tools, context, saved), context
