"""运行停止信号、任务失败信号以及可复现的请求序列化。"""
from hashlib import sha256
import json


class RunStopped(Exception):
    """预算或接口政策停止信号，不能被普通格式纠错吞掉。"""


class JobFailed(Exception):
    """任务在有限修复内未交付；候选和反馈保留。"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def digest(value):
    return sha256(encode(value).encode("utf-8")).hexdigest()
