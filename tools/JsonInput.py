"""外部 JSON 的无歧义读取；不修补字段、不猜数据内容、不修改原文件。"""
import json
import math

from pydantic import ValidationError


def validate(model, value, *, location):
    """按共享契约构造对象，输入错误带来源和字段路径，不改写原始资料。"""
    try:
        return model.model_validate(value)
    except ValidationError as error:
        issues = []
        for item in error.errors(include_url=False, include_input=False, include_context=False):
            path = "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in item["loc"])
            issues.append(f"{location}/{path}: {item['msg']}")
        raise ValueError("\n".join(issues)) from error


def loads(raw, *, location="JSON", expected=dict):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复 JSON 字段 {key!r}")
            result[key] = value
        return result

    def finite(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("JSON 数值必须有限：" + text)
        return value

    def constant(text):
        raise ValueError("JSON 不接受非标准常量：" + text)

    def unicode_text(value):
        if isinstance(value, str):
            value.encode("utf-8")  # 拒绝未配对代理码；不得等到指纹或输出时才崩溃。
        elif isinstance(value, dict):
            for key, item in value.items():
                unicode_text(key)
                unicode_text(item)
        elif isinstance(value, list):
            for item in value:
                unicode_text(item)

    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig")
        if not isinstance(raw, str):
            raise ValueError("需要 UTF-8 JSON 文本")
        if raw.startswith("\ufeff"):
            raw = raw[1:]
        result = json.loads(raw, object_pairs_hook=unique, parse_float=finite, parse_constant=constant)
        if expected is not None and not isinstance(result, expected):
            label = "对象" if expected is dict else "数组" if expected is list else "对象或数组"
            raise ValueError("顶层需要 JSON " + label)
        unicode_text(result)
        return result
    except (ValueError, RecursionError) as error:
        message = "JSON 嵌套过深" if isinstance(error, RecursionError) else str(error)
        raise ValueError(f"{location}: {message}") from error
