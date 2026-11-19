"""官方 OpenAI SDK 传输与 AutoGen 工具消息适配；账本仍由本项目负责。

消息转换直接复用 microsoft/autogen 的 python-v0.7.5 _openai_client；
HTTP 使用 openai/openai-python v1.109.1 的 with_raw_response，先记账后解析。
固定依赖版本是这些上游内部转换函数的兼容边界，不在这里实现 agent 循环。
"""
import asyncio
import math
import os
import re
from urllib.parse import urlsplit

from autogen_core import FunctionCall
from autogen_core.models import (AssistantMessage, ChatCompletionClient, CreateResult,
                                FunctionExecutionResultMessage, RequestUsage, SystemMessage, UserMessage)
from autogen_ext.models.openai._openai_client import convert_tool_choice, convert_tools, to_oai_type
import httpx
from httpx import AsyncClient as HTTPClient
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from openai._models import construct_type
from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall

from common.Runtime import CALL_SCOPE, RunStopped, check_batch, digest, encode, require
from tools.JsonInput import loads
from tools.TokenBudget import usage_counts_valid


class ModelResponseError(Exception):
    """接口未交付可用响应；不同于业务内容未通过检查。"""


class ModelTransportError(ModelResponseError):
    """接口连接或暂时不可用，已耗尽网络层重试。"""


class RequestContextTooLarge(RunStopped):
    """完整请求尚未发送；允许调用方在保持必要资料完整时收束会话历史。"""


class ModelDeliveryError(ModelResponseError):
    """已收到 HTTP 200，但无完整可用交付；保留回执交由任务级策略处理。"""

    def __init__(self, message, *, calls=(), errors=None, thought=None):
        super().__init__(message)
        self.calls = list(calls)
        self.errors = errors or {}
        self.thought = thought


def receipt_usage(payload):
    """保留回执的费用字段；矛盾计数由账本记作未知，有效分项只提高预留。"""
    usage = payload.get('usage') if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return {}
    try:
        encode(usage).encode('utf-8')
    except (ValueError, RecursionError):
        return {}
    return usage


def request_metrics(body):
    """Wire sizes, not estimated billable tokens; usage comes from the provider."""
    messages = body.get('messages', [])
    first_user = next((i for i, message in enumerate(messages) if message.get('role') == 'user'), None)
    return {'request_chars': len(encode(body)), 'tool_chars': len(encode(body.get('tools', []))),
            'message_chars': {role: sum(len(encode(m)) for m in messages if m.get('role') == role)
                              for role in ('system', 'user', 'assistant', 'tool')},
            'initial_context_chars': len(encode(messages[first_user])) if first_user is not None else 0,
            'continuation_chars': sum(len(encode(m)) for m in messages[first_user + 1:]) if first_user is not None else 0,
            'reasoning_chars': sum(len(m.get('reasoning_content') or '') for m in messages)}


def transport_messages(messages, model):
    """Keep provider reasoning in its native field, never turn it into story text.

    AutoGen supplies thought from CreateResult. In this client it comes solely
    from the provider's reasoning_content, so retain it for tool continuations
    as required by DeepSeek, while using the SDK for all other message fields.
    """
    converted = []
    for message in messages:
        thought = message.thought if isinstance(message, AssistantMessage) else None
        clean = message.model_copy(update={'thought': None}) if thought is not None else message
        parts = to_oai_type(clean, model=model, include_name_in_message=False)
        for part in parts:
            if thought is not None and part['role'] == 'assistant':
                part = {**part, 'reasoning_content': thought}
            converted.append(part)
    return converted


class ModelClient(ChatCompletionClient):
    def __init__(self, config, storage, log, max_calls, max_request_chars=0):
        self.config, self.storage, self.log = config, storage, log
        self.limit = min((n for n in (max_calls, config['max_calls']) if n is not None), default=None)
        self.max_request_chars = max_request_chars
        self.stopped = None
        self.usage = RequestUsage(0, 0)
        self._sdk = None

    @staticmethod
    def read_config(path):
        with open(path, encoding='utf-8') as stream:
            config = loads(stream.read(), location=str(path), expected=dict)
        def check(condition, message):
            require(condition, f'{path}: {message}')
        def finite(value):
            if type(value) not in (int, float):
                return False
            try:
                return math.isfinite(value)
            except OverflowError:
                return False
        allowed = {'api_url', 'model', 'api_key_env', 'timeout_seconds', 'max_retries', 'max_calls', 'request_options'}
        check(not set(config) - allowed, '不支持的模型配置字段：' + str(sorted(set(config) - allowed)))
        check(all(isinstance(config.get(k), str) and config[k].strip()
                    for k in ('api_url', 'model', 'api_key_env')), '缺少模型、接口或密钥环境变量名')
        try:
            parsed = urlsplit(config['api_url'])
            port = parsed.port  # urlsplit 本身不会检查端口的格式或范围。
            httpx.URL(config['api_url'])  # 使用实际发送器的地址规则，尚不构造请求或读取密钥。
        except (ValueError, httpx.InvalidURL) as error:
            raise ValueError(f'{path}: api_url 地址、主机或端口非法') from error
        check(parsed.scheme in ('http', 'https') and parsed.hostname and not parsed.username
                and not parsed.password and not parsed.query and not parsed.fragment
                and (port is None or port > 0) and not any(char.isspace() for char in config['api_url'])
                and parsed.path.endswith('/chat/completions'), 'api_url 须为完整 Chat Completions 接口')
        check(bool(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', config['api_key_env'])), '密钥环境变量名非法')
        for key in ('max_calls', 'max_retries'):
            if key == 'max_calls' and key in config and config[key] is None:
                continue  # 显式 null 解除请求次数预算，未配置或非法值仍拒绝。
            check(type(config.get(key)) is int and config[key] >= (1 if key == 'max_calls' else 0), key + ' 非法')
        timeout = config.get('timeout_seconds')
        check(finite(timeout) and timeout > 0, 'timeout_seconds 须为可用于请求的有限正数')
        options = config.get('request_options', {})
        check(isinstance(options, dict) and not set(options) - {'temperature', 'max_tokens', 'top_p', 'seed'},
                'request_options 仅允许 temperature/max_tokens/top_p/seed')
        for key, value in options.items():
            check(finite(value), 'request_options.' + key + ' 须为可用于请求的有限数值')
            if key in ('seed', 'max_tokens'):
                check(type(value) is int and (key == 'seed' or value > 0), key + ' 非法')
            if key == 'temperature':
                check(0 <= value <= 2, 'temperature 须在 0 到 2 之间')
            if key == 'top_p':
                check(0 < value <= 1, 'top_p 须在 0 到 1 之间')
        return config

    @property
    def model_info(self):
        return {'vision': False, 'function_calling': True, 'json_output': False,
                'family': 'unknown', 'structured_output': False}

    @property
    def capabilities(self):
        return self.model_info

    async def create(self, messages, *, tools=(), tool_choice='auto', json_output=None,
                     extra_create_args=None, cancellation_token=None):
        require(not json_output and not extra_create_args, '使用原生工具交付，不接受隐式模型参数或额外 JSON 模式')
        scope = CALL_SCOPE.get()
        require(scope is not None, '模型请求必须属于持久化智能体任务')
        for message in messages:
            require(isinstance(message, (SystemMessage, UserMessage, AssistantMessage, FunctionExecutionResultMessage)),
                    '模型通道只接受 AutoGen 原生文本及工具消息')
            require(not isinstance(message, UserMessage) or isinstance(message.content, str),
                    '当前模型配置不支持多模态输入')
        converted = transport_messages(messages, self.config['model'])
        from tools.ContextReferences import compact_messages
        converted = compact_messages(converted)
        body = {'model': self.config['model'], 'messages': converted, **self.config.get('request_options', {})}
        if tools:
            body.update(tools=convert_tools(tools), tool_choice=convert_tool_choice(tool_choice))
        else:
            require(tool_choice in ('auto', 'none'), '强制工具选择须提供实际工具')
        size = len(encode(body))
        limits = (("--max-request-chars", self.max_request_chars),
                  ("context_chars", getattr(self.storage.config, "context_chars", 0)))
        for name, limit in limits:
            if limit and size > limit:
                raise RequestContextTooLarge(f'完整请求 {size} 字符超出 {name}={limit}；'
                                 '尚未发送，工作区保留；不截断依据或重复尝试同一超长请求')
        operation = asyncio.create_task(self.request(scope, body))
        if cancellation_token:
            cancellation_token.link_future(operation)
        return await operation

    async def request(self, scope, body):
        """任务/修复轮次确定请求身份，重启与并发交错不改变回放归属。"""
        require(not any(key in body for key in ('functions', 'response_format')),
                '交付使用 SDK 原生 tools，不使用旧 functions 或额外 JSON 输出模式')
        key = digest({'scope': scope, 'body': body})
        for attempt in range(self.config['max_retries'] + 1):
            # 批次已失败时不能绕过语义入口继续 HTTP 重试；已发出的请求仍收回回执。
            check_batch()
            if self.stopped:
                raise RunStopped(self.stopped)
            secret = os.environ.get(self.config['api_key_env'], '')
            if secret:
                self.log.secret = secret
            try:
                row, replay = self.storage.reserve(key, attempt + 1, body, self.limit, can_send=bool(secret))
            except RunStopped as exc:
                self.stopped = str(exc)
                raise
            call_id = row['id']
            transient, payload = None, None
            if replay:
                self.log.write('replay', call=call_id, job=scope)
                raw, status, transient = row['response'], row['status'], row['error']
            else:
                try:
                    self.log.write('request', call=call_id, job=scope, request=body, metrics=request_metrics(body))
                    # 仅真实发送时建连接池；回放不需要网络客户端或密钥。
                    if self._sdk is None:
                        self._sdk = AsyncOpenAI(api_key=secret,
                            base_url=self.config['api_url'].removesuffix('/chat/completions') + '/',
                            http_client=HTTPClient(timeout=self.config['timeout_seconds'], follow_redirects=False,
                                                   trust_env=False), max_retries=0)
                    self._sdk.api_key = secret
                    try:
                        receipt = await self._sdk.chat.completions.with_raw_response.create(**body)
                        response = receipt.http_response
                    except APIStatusError as exc:
                        response = exc.response
                    raw, status = response.content.replace(secret.encode('utf-8'), b'[REDACTED]'), response.status_code
                    try:
                        raw = raw.decode('utf-8')
                    except UnicodeDecodeError:
                        # SQLite 原样保留损坏字节；严格 JSON 入口拒绝它，不能替换正文后接收。
                        pass
                    try:
                        payload = loads(raw, location='接口响应', expected=dict)
                    except ValueError:
                        payload = {}
                    usage = receipt_usage(payload)
                    self.storage.received(call_id, status, raw, usage)
                    self.log.write('received', call=call_id, job=scope, status=status, usage=usage, body=raw)
                except APIConnectionError as exc:
                    cause = exc.__cause__
                    transient = type(cause).__name__ if cause is not None else type(exc).__name__
                    if not isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
                        # 请求可能已送达，留下无回执账目；resume 不应重复发出同一请求。
                        self.log.write('delivery_unknown', call=call_id, job=scope, error=transient)
                        self.stopped = f'请求送达结果未知（{transient}），保留账本等待核查'
                        raise RunStopped(self.stopped) from exc
                    self.storage.failed(call_id, transient)
                    self.log.write('error', call=call_id, job=scope, error=transient)
                finally:
                    self.storage.release_call(call_id)
            if transient or status == 429 or status is not None and 500 <= status < 600:
                if attempt < self.config['max_retries']:
                    if not replay:
                        await asyncio.sleep(min(2 ** attempt, 10))
                    continue
                raise ModelTransportError(f'网络/接口重试用尽：{transient or status}')
            if status in (401, 403):
                self.stopped = f'接口拒绝授权：HTTP {status}'
                raise RunStopped(self.stopped)
            if status != 200:
                raise ModelResponseError(f'接口返回 HTTP {status}，完整响应已保存')
            try:
                # 记账与交付共用同一读取入口；正常请求复用已解析对象，回放再读取原回执。
                # BOM 等合法表示不能导致正文已接收、实际 usage 却被记为未知。
                if payload is None:
                    payload = loads(raw, location='接口响应', expected=dict)
                # 与 SDK BaseClient._process_response_data 的默认转换相同。
                # 使用同一转换回放已保存回执，无须创建客户端、读取密钥或再次联网。
                completion = construct_type(type_=ChatCompletion, value=payload)
                choice = completion.choices[0]
                finish, message = choice.finish_reason, choice.message
                content = message.content
                thought = getattr(message, 'reasoning_content', None)
                # 可选元数据不决定业务交付是否有效；原始值仍完整保存在 HTTP 回执。
                if not isinstance(thought, str):
                    thought = None
                if finish == 'content_filter':
                    self.stopped = '接口返回 content_filter；响应保留，不执行或重试'
                    raise RunStopped(self.stopped)
                # 部分兼容服务以 stop 标记完整工具消息；仍按同一工具契约处理。
                # length / content_filter 等明确未完成或被拒绝的结果不得进入工具执行。
                if finish == 'tool_calls' or finish == 'stop' and message.tool_calls:
                    calls = message.tool_calls
                    if not calls or any(not isinstance(call, ChatCompletionMessageFunctionToolCall)
                            or not isinstance(call.id, str) or not call.id
                            or not isinstance(call.function.name, str) or not call.function.name
                            or not isinstance(call.function.arguments, str) for call in calls):
                        raise ModelDeliveryError('接口未返回完整的原生函数调用，原回执已保存')
                    if len({call.id for call in calls}) != len(calls):
                        raise ModelDeliveryError('接口返回重复工具调用身份，原回执已保存')
                    from tools.JsonRepair import parse_arguments
                    original_calls = [FunctionCall(id=call.id, name=call.function.name,
                                                   arguments=call.function.arguments) for call in calls]
                    content, errors = [], {}
                    for call in calls:
                        # AutoGen 0.7.5 only catches JSONDecodeError. Normalize
                        # transport wrappers/escapes, then use our strict reader
                        # before SDK dispatch: duplicate keys cannot be silently replaced,
                        # and deep nesting/oversized integers must not escape its
                        # tool task and leave the streaming queue waiting forever.
                        try:
                            arguments = parse_arguments(call.function.arguments)
                        except ValueError as exc:
                            errors[call.id] = (
                                f'工具 {call.function.name[:80]!r}（call_id={call.id[:80]!r}）'
                                f'参数不是无歧义 JSON 对象：{str(exc)[:300]}；原回执已保存'
                            )
                            continue
                        # Dispatch the validated representation, not its original
                        # envelope (e.g. a permitted UTF-8 BOM). The ledger retains
                        # the exact raw response; SDK replay receives stable JSON.
                        content.append(FunctionCall(id=call.id, name=call.function.name,
                                                    arguments=encode(arguments, compact=True)))
                    if errors:
                        # Preserve every call, including valid peers, so the next
                        # conversation has matched native tool errors and no
                        # partially dispatched read batch. Reasoning remains
                        # protocol metadata, never part of the error prose.
                        raise ModelDeliveryError('；'.join(errors.values()), calls=original_calls,
                                                 errors=errors, thought=thought)
                    finish = 'function_calls'
                elif finish != 'stop' or message.tool_calls or not isinstance(content, str) or not content.strip():
                    raise ModelDeliveryError(f'接口未返回完整文本或工具调用：finish_reason={finish}')
                usage = receipt_usage(payload)
                # SDK 计数仅接受有效整数；账本仍保留原值，不将未知费用记为实际零费用。
                if not usage_counts_valid(usage):
                    usage = {}
                prompt, completion = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                raise ModelDeliveryError('接口响应结构不合法，原文已保存') from exc
            if not replay:
                self.usage.prompt_tokens += prompt
                self.usage.completion_tokens += completion
            return CreateResult(content=content, finish_reason=finish,
                                usage=RequestUsage(prompt, completion), cached=replay, thought=thought)
        raise AssertionError('不可达的请求分支')

    async def create_stream(self, *args, **kwargs):
        raise NotImplementedError('为原子接收完整响应，本项目不使用流式接口')
        yield  # 保持 AutoGen 的异步生成器接口。

    async def close(self):
        """工作流退出时释放本次运行的连接池。"""
        if self._sdk is not None:
            await self._sdk.close()
            self._sdk = None

    def actual_usage(self):
        return self.usage

    def total_usage(self):
        return self.usage

    def count_tokens(self, messages, **kwargs):
        raise NotImplementedError('未配置模型 tokenizer，不将字符数冒充 token 数')

    def remaining_tokens(self, messages, **kwargs):
        raise NotImplementedError('上下文容量由接口与显式请求限制约束')
