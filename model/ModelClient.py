"""AutoGen 的唯一 HTTP 边界：配置、累计预算、持久请求及断点回放。"""
import asyncio
from contextvars import ContextVar
import json
import math
import os
import re
from urllib.parse import urlsplit

from autogen_core import FunctionCall
from autogen_core.models import ChatCompletionClient, CreateResult, RequestUsage
import httpx

from common.Runtime import RunStopped, digest, encode, require

CALL_SCOPE = ContextVar('novel_call_scope', default=None)


class ModelResponseError(Exception):
    """接口未交付完整的文本或工具响应；不同于业务内容未通过检查。"""


class ModelTransportError(ModelResponseError):
    """接口连接或暂时不可用，已耗尽网络层重试。"""


def compact_tool_schema(value):
    """只去掉Schema标题注释；约束、说明和业务字面值原样保留。"""
    if isinstance(value, list):
        return [compact_tool_schema(item) for item in value]
    if isinstance(value, dict):
        return {key: (item if key in ('default', 'const', 'enum', 'examples') else compact_tool_schema(item))
                for key, item in value.items() if not (key == 'title' and isinstance(item, str))}
    return value


class ModelClient(ChatCompletionClient):
    def __init__(self, config, storage, log, max_calls, max_request_chars=0):
        self.config, self.storage, self.log = config, storage, log
        self.limit = min(max_calls, config['max_calls'])
        self.max_request_chars = max_request_chars
        self.stopped = None
        self.usage = RequestUsage(0, 0)

    @staticmethod
    def read_config(path):
        with open(path, encoding='utf-8') as stream:
            config = json.load(stream)
        require(isinstance(config, dict), '模型配置须为对象')
        allowed = {'api_url', 'model', 'api_key_env', 'timeout_seconds', 'max_retries', 'max_calls', 'request_options'}
        require(not set(config) - allowed, '不支持的模型配置字段：' + str(sorted(set(config) - allowed)))
        require(all(isinstance(config.get(k), str) and config[k].strip()
                    for k in ('api_url', 'model', 'api_key_env')), '缺少模型、接口或密钥环境变量名')
        parsed = urlsplit(config['api_url'])
        require(parsed.scheme in ('http', 'https') and parsed.hostname and not parsed.username
                and not parsed.password and not parsed.query and not parsed.fragment
                and parsed.path.endswith('/chat/completions'), 'api_url 须为完整 Chat Completions 接口')
        require(bool(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', config['api_key_env'])), '密钥环境变量名非法')
        for key in ('max_calls', 'max_retries'):
            require(type(config.get(key)) is int and config[key] >= (1 if key == 'max_calls' else 0), key + ' 非法')
        timeout = config.get('timeout_seconds')
        require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0, 'timeout_seconds 非法')
        options = config.get('request_options', {})
        require(isinstance(options, dict) and not set(options) - {'temperature', 'max_tokens', 'top_p', 'seed'},
                'request_options 仅允许 temperature/max_tokens/top_p/seed')
        for key, value in options.items():
            require(type(value) in (int, float) and math.isfinite(value), key + ' 须为有限数值')
            if key in ('seed', 'max_tokens'):
                require(type(value) is int and (key == 'seed' or value > 0), key + ' 非法')
            if key == 'temperature':
                require(0 <= value <= 2, 'temperature 须在 0 到 2 之间')
            if key == 'top_p':
                require(0 < value <= 1, 'top_p 须在 0 到 1 之间')
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
        require(not json_output and not extra_create_args, '不接受隐式模型参数或服务端 JSON 约束')
        scope = CALL_SCOPE.get()
        require(scope is not None, '模型请求必须属于持久化智能体任务')
        converted = []
        for message in messages:
            if type(message).__name__ == 'FunctionExecutionResultMessage':
                converted.extend({'role': 'tool', 'tool_call_id': result.call_id, 'content': result.content}
                                 for result in message.content)
                continue
            role = {'SystemMessage': 'system', 'UserMessage': 'user', 'AssistantMessage': 'assistant'}.get(type(message).__name__)
            if role == 'assistant' and isinstance(message.content, list):
                require(all(isinstance(call, FunctionCall) for call in message.content), '工具消息类型非法')
                converted.append({'role': role, 'content': None, 'tool_calls': [
                    {'id': call.id, 'type': 'function',
                     'function': {'name': call.name, 'arguments': call.arguments}} for call in message.content]})
            else:
                require(role and isinstance(message.content, str), '仅支持文本及工具消息')
                converted.append({'role': role, 'content': message.content})
            if role == 'assistant' and getattr(message, 'thought', None) is not None:
                # DeepSeek 思考模式的工具会话要求回传此前模型响应的 reasoning_content。
                converted[-1]['reasoning_content'] = message.thought
        body = {'model': self.config['model'], 'messages': converted, **self.config.get('request_options', {})}
        if tools:
            require(tool_choice == 'auto', '工具选择须由智能体决定')
            body.update(tools=[{'type': 'function', 'function': {
                key: compact_tool_schema(value) if key == 'parameters' else value
                for key, value in (tool if isinstance(tool, dict) else tool.schema).items()
                if key in ('name', 'description', 'parameters', 'strict')}} for tool in tools],
                tool_choice='auto', parallel_tool_calls=False)
        if self.max_request_chars and len(encode(body)) > self.max_request_chars:
            raise RunStopped(f'完整请求 {len(encode(body))} 字符超出 --max-request-chars={self.max_request_chars}；'
                             '尚未发送，工作区保留；不截断依据或重复尝试同一超长请求')
        operation = asyncio.create_task(self.request(scope, body))
        if cancellation_token:
            cancellation_token.link_future(operation)
        return await operation

    async def request(self, scope, body):
        """任务/修复轮次确定请求身份，重启与并发交错不改变回放归属。"""
        key = digest({'scope': scope, 'body': body})
        for attempt in range(self.config['max_retries'] + 1):
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
            transient = None
            if replay:
                self.log.write('replay', call=call_id, job=scope)
                raw, status, transient = row['response'], row['status'], row['error']
            else:
                self.log.write('request', call=call_id, job=scope, request=body)
                try:
                    async with httpx.AsyncClient(timeout=self.config['timeout_seconds'], follow_redirects=False,
                                                 trust_env=False) as client:
                        response = await client.post(self.config['api_url'], json=body,
                                                     headers={'Authorization': 'Bearer ' + secret})
                    raw, status = response.text.replace(secret, '[REDACTED]'), response.status_code
                    try:
                        payload = json.loads(raw)
                    except ValueError:
                        payload = {}
                    usage = payload.get('usage', {}) if isinstance(payload, dict) else {}
                    self.storage.received(call_id, status, raw, usage)
                    self.log.write('received', call=call_id, job=scope, status=status, usage=usage, body=raw)
                except httpx.TransportError as exc:
                    transient = type(exc).__name__
                    self.storage.failed(call_id, transient)
                    self.log.write('error', call=call_id, job=scope, error=transient)
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
                payload = json.loads(raw)
                choice = payload['choices'][0]
                finish = choice['finish_reason']
                message = choice['message']
                content = message.get('content')
                thought = message.get('reasoning_content')
                if thought is not None and not isinstance(thought, str):
                    raise ModelResponseError('接口返回的 reasoning_content 不是文本')
                if finish == 'content_filter':
                    self.stopped = '接口返回 content_filter；响应保留，不执行或重试'
                    raise RunStopped(self.stopped)
                if finish == 'tool_calls':
                    calls = message.get('tool_calls')
                    if not body.get('tools') or not isinstance(calls, list) or not calls:
                        raise ModelResponseError('接口返回未授权或空的工具调用')
                    content = []
                    for call in calls:
                        function = call['function']
                        if (call.get('type') != 'function' or not all(isinstance(v, str) and v.strip()
                                for v in (call['id'], function['name'], function['arguments']))):
                            raise ModelResponseError('工具调用结构非法，未执行')
                        content.append(FunctionCall(id=call['id'], name=function['name'], arguments=function['arguments']))
                    if len({call.id for call in content}) != len(content):
                        raise ModelResponseError('工具调用编号重复，未执行')
                elif finish != 'stop' or not isinstance(content, str) or not content.strip():
                    raise ModelResponseError(f'接口未返回完整文本：finish_reason={finish}')
                usage = payload.get('usage') or {}
                prompt, completion = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
                if type(prompt) is not int or type(completion) is not int or min(prompt, completion) < 0:
                    raise ModelResponseError('接口 token 用量非法')
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                raise ModelResponseError('接口响应结构不合法，原文已保存') from exc
            if not replay:
                self.usage.prompt_tokens += prompt
                self.usage.completion_tokens += completion
            return CreateResult(content=content, finish_reason='function_calls' if finish == 'tool_calls' else finish,
                                usage=RequestUsage(prompt, completion), cached=replay, thought=thought)
        raise AssertionError('不可达的请求分支')

    async def create_stream(self, *args, **kwargs):
        raise NotImplementedError('为原子接收完整响应，本项目不使用流式接口')
        yield  # 保持 AutoGen 的异步生成器接口。

    async def close(self):
        """网络连接由每次请求的上下文管理器关闭。"""

    def actual_usage(self):
        return self.usage

    def total_usage(self):
        return self.usage

    def count_tokens(self, messages, **kwargs):
        raise NotImplementedError('未配置模型 tokenizer，不将字符数冒充 token 数')

    def remaining_tokens(self, messages, **kwargs):
        raise NotImplementedError('上下文容量由接口与显式请求限制约束')
