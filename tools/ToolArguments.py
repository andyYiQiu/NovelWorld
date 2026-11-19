"""工具传输包装的归一与完整契约校验；身份和故事语义由业务层处理。"""
from copy import deepcopy

from pydantic import BaseModel, AliasChoices, AliasPath, ValidationError
from jsonpatch import JsonPatchException
from jsonpointer import JsonPointer, JsonPointerException

from tools.JsonRepair import parse_arguments


def _declares_arguments(args_type):
    """真实参数或入口别名使用 arguments 时，不能把它误当传输包装。"""
    for name, field in args_type.model_fields.items():
        if name == "arguments":
            return True
        for alias in (field.alias, field.validation_alias):
            choices = alias.choices if isinstance(alias, AliasChoices) else (alias,)
            if any(choice == "arguments" or isinstance(choice, AliasPath)
                   and choice.path[0] == "arguments" for choice in choices):
                return True
    return False


def _relocate_unique_fields(value, args_type):
    """用 Pydantic 缺失/多余路径定位唯一归属，不跨人物或多项行动搬移。"""
    from tools.JsonPatch import move_fields
    try:
        args_type.model_validate(deepcopy(value), extra="forbid")
        return value
    except ValidationError as error:
        issues = error.errors(include_url=False, include_input=False)
    missing = [e["loc"] for e in issues if e["type"] == "missing"]
    extra = [e["loc"] for e in issues if e["type"] == "extra_forbidden"]
    moves = []
    if len(missing) * len(extra) > 64:
        return value
    for target in missing:
        sources = [source for source in extra if source[-1] == target[-1]
                   and source[:len(target) - 1] == target[:-1]]
        if len(sources) != 1 or sum(path[-1] == target[-1] for path in missing) != 1:
            continue
        source = sources[0]
        current, unique = value, True
        try:
            for part in source[:-1]:
                if isinstance(current, list) and len(current) != 1:
                    unique = False
                current = current[part]
        except (KeyError, IndexError, TypeError):
            continue  # Union error locations may contain labels, not input keys.
        if unique:
            pointer = lambda path: "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in path)
            moves.append({"from": pointer(source), "path": pointer(target)})
    return move_fields(value, moves) if moves else value


def _project_empty_extras(value, args_type):
    """Remove empty, undeclared decoration even beside missing semantic content.

    Pydantic identifies actual forbidden fields; declared content and arbitrary
    JSON values are never traversed or pruned by field name. Missing judgments,
    substantive extras, booleans and numbers cannot become an approval.
    """
    try:
        args_type.model_validate(deepcopy(value), extra="forbid")
        return value
    except ValidationError as error:
        issues = error.errors(include_url=False, include_input=False)
    result = deepcopy(value)
    from tools.JsonPatch import field_annotation
    for issue in issues:
        if issue["type"] != "extra_forbidden":
            continue
        path = issue["loc"]
        if field_annotation(args_type, path[:-1]) is None:
            continue  # Do not treat Union labels as fields or prune declared content.
        try:
            parent = JsonPointer.from_parts([str(p) for p in path[:-1]]).resolve(result)
        except JsonPointerException:
            continue  # Union branch labels are not paths in the candidate.
        if not isinstance(parent, dict) or path[-1] not in parent:
            continue
        extra = parent[path[-1]]
        if extra is None or extra == "" or extra == [] or extra == {}:
            del parent[path[-1]]
    return result


def normalize_arguments(value, args_type, *, on_invalid=None):
    """归一完整工具包装并校验，原始映射和值保持不变。

    入口归一后全量校验、校验副本的做法参考 Pi validation.ts。
    arguments 单层解包是本项目针对真实回执的契约适配，并非 Pi 原版规则。
    缺少固定 result 外层时，仅当完整内容符合该工具唯一参数的完整契约才补外层。
    内部字段仅按完整契约的唯一缺失/多余位置归位；空白多余装饰由程序剥离，
    原始回执仍由接口账本保留。不递归解包，不猜归属，
    不返回会插入默认值或强制转换的 model_dump。
    仍有错误时，向调用方交接归一后的候选和其实际错误；不退回已解决的格式问题。
    """
    candidate = value
    if (isinstance(value, dict) and set(value) == {"arguments"}
            and not _declares_arguments(args_type)):
        wrapped = value["arguments"]
        if isinstance(wrapped, str):
            candidate = parse_arguments(wrapped)
        elif isinstance(wrapped, dict):
            candidate = wrapped
    try:
        args_type.model_validate(deepcopy(candidate), extra="forbid")
        return candidate
    except ValidationError:
        fixed = deepcopy(candidate)
        if set(args_type.model_fields) == {"result"} and isinstance(fixed, dict) and fixed:
            if not {"result", "arguments"}.intersection(fixed):
                fixed = {"result": fixed}
            elif isinstance(fixed.get("result"), dict):
                siblings = {k: v for k, v in fixed.items() if k != "result"}
                if not set(siblings).intersection(fixed["result"]):
                    fixed = {"result": {**fixed["result"], **siblings}}
        try:
            fixed = _relocate_unique_fields(fixed, args_type)
        except (ValueError, JsonPatchException, JsonPointerException):
            pass  # Ambiguous relocation is atomic; other safe cleanup can proceed.
        fixed = _project_empty_extras(fixed, args_type)
        try:
            args_type.model_validate(deepcopy(fixed), extra="forbid")
        except ValidationError as error:
            if on_invalid:
                on_invalid(fixed, error.errors(include_url=False, include_input=False, include_context=False))
            raise
        return fixed


def payload_data(payload):
    # SDK validation inserts defaults. Preserve omission information so a default
    # cannot silently overwrite an already established business value.
    return payload.model_dump(mode="python", exclude_unset=True) if isinstance(payload, BaseModel) else payload

