"""模型交付的完整格式归一；repository 仍使用自己的固定 JSON 输入契约。

字符串修复移植自 Pi packages/ai/src/utils/json-parse.ts 的 repairJson，MIT，
Copyright (c) 2025 Mario Zechner；许可见 LICENSES/pi-MIT.txt。
来源：https://github.com/earendil-works/pi/blob/main/packages/ai/src/utils/json-parse.ts
冗余尾闭括号参考 josdejong/jsonrepair；缺少容器闭括号使用
mangiucugna/json_repair。只接纳能由补齐已打开容器的闭括号完整解析的修复，
不补正文、字段或值，不移动业务字段；原始回执由调用方保存。
"""
import json
import re

import json5
from json_repair import repair_json

from tools.JsonInput import loads


def _closed_containers(text):
    """仅补已打开容器的闭括号；JSON5 随后独立验证所有字段和值。

    已有闭括号必须匹配栈中某个容器，只补其上方缺失的闭括号。
    根值之后的冗余闭括号保留给完整值尾部检查，不删除或移动其他字符。
    """
    result, stack, quote, escaped = [], [], None, False
    for char in text:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack:
                if char not in stack:
                    raise ValueError("工具参数闭括号没有对应容器，不能猜测内容层级")
                while stack[-1] != char:
                    result.append(stack.pop())
                stack.pop()
        result.append(char)
    if quote:
        raise ValueError("工具参数包含未完成正文，不能补写内容")
    return "".join(result) + "".join(reversed(stack))


def _repair_strings(raw, *, single_quotes=False):
    result, quote, index = [], None, 0
    while index < len(raw):
        char = raw[index]
        if quote is None:
            if single_quotes and raw[index:index + 2] in {"//", "/*"}:
                raise ValueError("工具参数不可带 JSON5 注释；正文内的注释标记不受限制")
            result.append(char)
            if char == '"' or single_quotes and char == "'":
                quote = char
        elif char == quote:
            result.append(char)
            quote = None
        elif char == "\\":
            following = raw[index + 1:index + 2]
            digits = raw[index + 2:index + 6]
            if following == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                result.append(raw[index:index + 6])
                index += 5
            elif following and (following in '"\\/bfnrt' or quote == "'" and following == "'"):
                result.append(raw[index:index + 2])
                index += 1
            else:
                result.append("\\\\")
        else:
            result.append(f"\\u{ord(char):04x}" if ord(char) < 32 else char)
        index += 1
    return "".join(result)


def _decode_complete(text):
    """先完整读取，再本地恢复语法；完整值及重复键检查始终保留。

    标准库 raw_decode 只定位结束位置，不使用其会覆盖重复键的解析结果。
    jsonrepair 的同类处理见 src/regular/jsonrepair.ts 的根值结束分支。
    """
    try:
        return loads(text, location="工具参数", expected=None)
    except ValueError as original:
        repaired = _repair_strings(text)
        try:
            _, end = json.JSONDecoder().raw_decode(repaired)
        except (ValueError, RecursionError):
            # JSON5 是完整语法解析器，不补缺失值；这里只放宽引号、逗号等表示。
            # 不接收注释，避免注释内引号干扰既有 Pi 字符串修复的边界。
            # 数值仍经 JSON 读取器，拒绝 Infinity/NaN、十六进制和非有限数；
            # JSON5 会先移除十进制前的正号，这一等价表示可接受。
            repaired = _repair_strings(text, single_quotes=True)
            def number(value, **_):
                return loads(value, location="工具数值", expected=None)
            try:
                value, error, end = json5.parse(repaired, allow_duplicate_keys=False,
                    consume_trailing=False, parse_int=number, parse_float=number, parse_constant=number)
                if error:
                    # 独立解析原文本仅补闭括号，阻止修复库补值、截短正文或覆盖重复键。
                    closed = _closed_containers(repaired)
                    value, error, end = json5.parse(closed, allow_duplicate_keys=False,
                        consume_trailing=False, parse_int=number, parse_float=number, parse_constant=number)
                    if error:
                        raise ValueError("工具参数 JSON5: " + error)
                    fixed = repair_json(repaired, return_objects=True)
                    canonical = lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    if canonical(fixed) != canonical(value):
                        raise ValueError("JSON 修复改变了内容，不能作为纯格式恢复")
                    value = fixed
                    repaired = closed
                normalized = json.dumps(value, ensure_ascii=False, allow_nan=False)
            except RecursionError:
                raise ValueError("工具参数 JSON 嵌套过深") from None
        else:
            normalized = repaired[:end]
        remainder = repaired[end:]
        if remainder.strip() and not re.fullmatch(r"[}\]\s]+", remainder):
            raise original from None
        return loads(normalized, location="工具参数", expected=None)


def parse_arguments(raw):
    """完整传输包装、字符串编码和语法冗余在唯一入口恢复，不改变内容。

    最多进行四次 JSON 解码；正文中的字符串不递归处理。多对象、重复键、
    截断正文等不能因格式恢复而成为成功结果，后续还须通过原工具契约。
    """
    for _ in range(4):
        if not isinstance(raw, str):
            raise ValueError("工具参数须为 JSON 对象文本")
        text = raw.strip().lstrip("\ufeff").strip()
        # StoryBox/CreAgentive 的完整代码块提取，不截取夹杂说明的局部对象。
        fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence[1].strip()
        value = _decode_complete(text)
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            raise ValueError("工具参数顶层须为 JSON 对象")
        raw = value
    raise ValueError("工具参数字符串编码层数过多，须直接交付声明的对象")


