"""运行停止信号、任务失败信号以及可复现的请求序列化。"""
from hashlib import sha256
from contextvars import ContextVar
import asyncio
import json
from urllib.parse import quote

CALL_SCOPE = ContextVar('novel_call_scope', default=None)
CALL_ROLE = ContextVar('novel_restore_call_role', default=None)
BATCH_GATE = ContextVar('novel_semantic_batch_gate', default=None)


class RunStopped(Exception):
    """预算或接口政策停止信号，不能被普通格式纠错吞掉。"""


class JobFailed(Exception):
    """任务在有限修复内未交付；候选和反馈保留。"""


class CandidateError(ValueError):
    """候选内容错误及其工具参数位置；供局部返修使用，不改变验收标准。"""

    def __init__(self, message, issues):
        super().__init__(message)
        self.issues = issues


class BatchStopped(RunStopped):
    """同批已有失败，本问题尚未发送；不替换原始失败原因。"""


class BatchGate:
    def __init__(self, parent=None):
        self.parent, self.error = parent, None

    def fail(self, error):
        if isinstance(error, BatchStopped):
            return
        if self.error is None:
            self.error = error
        if self.parent is not None:
            self.parent.fail(error)

    def check(self):
        if self.parent is not None:
            self.parent.check()
        if self.error is not None:
            raise BatchStopped("同批任务已失败，停止新的模型请求：" + str(self.error)) from self.error


def check_batch():
    """语义请求与 HTTP 重试共用的发送前屏障；已经发送的请求仍保存回执。"""
    batch = BATCH_GATE.get()
    if batch is not None:
        batch.check()


async def gather_batch(awaitables):
    """同批失败时关闭新请求；保留已发送回执，最后传播原始错误。"""
    batch = BatchGate(BATCH_GATE.get())
    token = BATCH_GATE.set(batch)

    async def run(awaitable):
        try:
            return await awaitable
        except BaseException as error:
            batch.fail(error)
            raise

    pending = asyncio.gather(*(run(a) for a in awaitables), return_exceptions=True)
    try:
        try:
            results = await asyncio.shield(pending)
        except asyncio.CancelledError:
            batch.fail(RunStopped("上游任务已中断，停止后续请求并等待已发送回执"))
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    continue
            raise
        if batch.error is not None:
            raise batch.error
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results
    finally:
        BATCH_GATE.reset(token)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value, *, compact=False):
    """固定键顺序；传输可省略 JSON 分隔空格，不修改任何字符串内容。"""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":") if compact else None)


def digest(value):
    return sha256(encode(value).encode("utf-8")).hexdigest()


def storage_encode(value):
    """持久记录沿用原有紧凑字节表示，不改变历史指纹。"""
    return encode(value, compact=True)


def fingerprint(value):
    """记录指纹沿用紧凑编码，不能与 HTTP 请求的 digest 互换。"""
    return sha256(storage_encode(value).encode("utf-8")).hexdigest()


def task_id(*parts):
    return "/".join(quote(str(part), safe="") for part in parts)
