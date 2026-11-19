"""Lossless references for repeated request data, using RFC 6901 JSON Pointer.

This is a transport view, like Pi's transformContext: stored sources, SDK
history, tool arguments and responses are never rewritten. No summarization,
retrieval or semantic equality is involved. References point to earlier data
in this same request, not to a cache the model would have to query.
"""
import json

from jsonpointer import JsonPointer

from common.Runtime import encode
from tools.JsonInput import loads


REFERENCE = "$context_ref"
POLICY = ("\n上下文表示：仅含 $context_ref 的对象引用本请求中已提供的完全相同内容。"
          "值为 message索引#JSON Pointer；索引从0开始，路径相对于该消息content解码后的JSON。"
          "引用可逐级展开，不需补查；归属、时点和来源仍由引用所在位置决定。"
          "这只是输入去重表示，不是工具交付字段。")


def compact_messages(messages, *, min_chars=512):
    """Share identical large JSON values across user messages, preserving order.

    Only explicit task/evidence envelopes participate. Native assistant/tool
    messages keep the provider protocol unchanged. A reserved-key collision
    disables this optional projection rather than reinterpreting story data.
    """
    documents = {}
    for index, message in enumerate(messages):
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        try:
            value = loads(message["content"], location="request context", expected=dict)
        except ValueError:
            continue
        if "context" in value or "prior_lookup_evidence" in value:
            documents[index] = value

    def has_reserved(value):
        if isinstance(value, dict):
            return REFERENCE in value or any(has_reserved(item) for item in value.values())
        return isinstance(value, list) and any(has_reserved(item) for item in value)

    seen, replacements = {}, 0

    def visit(value, index, parts):
        nonlocal replacements
        if isinstance(value, (dict, list, str)):
            canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if len(canonical) >= min_chars:
                if canonical in seen:
                    reference = {REFERENCE: seen[canonical]}
                    if len(encode(reference, compact=True)) < len(canonical):
                        replacements += 1
                        return reference
                else:
                    seen[canonical] = f"{index}#{JsonPointer.from_parts(parts).path}"
        if isinstance(value, dict):
            return {key: visit(item, index, [*parts, key]) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, index, [*parts, str(i)]) for i, item in enumerate(value)]
        return value

    try:
        if not documents or any(has_reserved(value) for value in documents.values()):
            return messages
        result = [dict(message) for message in messages]
        for index, value in documents.items():
            result[index]["content"] = encode(visit(value, index, []), compact=True)
    except RecursionError:
        # Optional projection must not reject a valid, deeper source. The
        # original request still goes through normal size and budget checks.
        return messages
    if not replacements:
        return messages
    # Appending the explanation does not shift the referenced message indices.
    system = next((m for m in result if m.get("role") == "system" and isinstance(m.get("content"), str)), None)
    if system is None:
        return messages
    system["content"] += POLICY
    return result if len(encode(result)) < len(encode(messages)) else messages
