"""RFC 6902 候选补丁：字段移动与定位明确的局部语义修订。

来源：https://github.com/stefankoegl/python-json-patch
字段归属、权限、完整参数和业务验收仍由调用方检查。
"""
from copy import deepcopy
from types import UnionType
from typing import Annotated, Literal, Union, get_args, get_origin

import jsonpatch
from jsonpointer import JsonPointer, JsonPointerException
from pydantic import BaseModel, ConfigDict, Field, JsonValue


class FieldMove(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(alias="from", description="现有字段的 JSON Pointer 路径")
    path: str = Field(description="正确字段位置的 JSON Pointer 路径；目标不能已有值")


class FieldEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["add", "replace"]
    path: str = Field(description="本次允许修复的精确 JSON Pointer 路径")
    value: JsonValue = Field(description="依据原任务资料补充或修订该字段的完整值")


_SCALAR_ERRORS = {
    "enum", "literal_error", "candidate_error", "string_too_short", "string_too_long",
    "string_pattern_mismatch", "greater_than", "greater_than_equal", "less_than",
    "less_than_equal", "multiple_of", "finite_number", "int_from_float",
    "decimal_max_digits", "decimal_max_places", "decimal_whole_digits",
}


def field_annotation(annotation, location):
    """Pydantic 的 Union 分支标签不是实际字段路径，不用于局部补丁。"""
    description = None
    for index, part in enumerate(location):
        while get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        origin, args = get_origin(annotation), get_args(annotation)
        if origin in (Union, UnionType):
            # A real field may belong to one declared alternative. Branch
            # labels and ambiguous constraints are never patch locations.
            choices = [found for branch in args
                       if (found := field_annotation(branch, location[index:])) is not None]
            return choices[0] if choices and all(c == choices[0] for c in choices) else None
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            field = annotation.model_fields.get(part)
            if field is None:
                return None
            annotation, description = field.rebuild_annotation(), field.description
        elif origin is list and (type(part) is int or isinstance(part, str) and part.isdecimal()):
            annotation, description = args[0], None
        elif origin is tuple and (type(part) is int or isinstance(part, str) and part.isdecimal()):
            position = int(part)
            if len(args) == 2 and args[1] is Ellipsis:
                annotation = args[0]
            elif position < len(args):
                annotation = args[position]
            else:
                return None
            description = None
        elif origin is dict:
            annotation, description = args[1], None
        else:
            return None
    return Annotated[annotation, Field(description=description)]


def amendment_type(args_type, allowed_paths):
    """The SDK receives the unresolved values' real types, in bound order."""
    annotations = []
    for path in allowed_paths:
        annotation = field_annotation(args_type, JsonPointer(path).parts)
        if annotation is None:
            raise ValueError("局部修订位置已不属于当前工具契约")
        annotations.append(annotation)
    return tuple[tuple(annotations)]


def _unannotated(annotation):
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _scalar(annotation):
    annotation = _unannotated(annotation)
    if get_origin(annotation) in (Union, UnionType):
        return all(_scalar(branch) for branch in get_args(annotation))
    return annotation in (str, int, float, bool, type(None)) or get_origin(annotation) is Literal


def _optional_entry(args_type, parts):
    """Only a declared nullable record in a mapping can be revised as a unit."""
    parent = _unannotated(field_annotation(args_type, parts[:-1]))
    annotation = _unannotated(field_annotation(args_type, parts))
    branches = get_args(annotation)
    return (get_origin(parent) is dict and get_origin(annotation) in (Union, UnionType)
            and type(None) in branches and all(branch is type(None) or
                isinstance(branch, type) and issubclass(branch, BaseModel) for branch in branches))


def amendment_plan(candidate, issues, args_type=None):
    """定位全部可修字段；不能定位、根级或混合未知问题仍交原任务处理。"""
    allowed, paths = {}, []
    for issue in issues:
        kind, location = issue.get("type", ""), issue.get("loc", ())
        if (not location or tuple(location) == ("result",) or not (
                kind in ("missing", "candidate_record") or kind in _SCALAR_ERRORS
                or kind.endswith(("_type", "_parsing")))):
            return {}
        if args_type is not None and field_annotation(args_type, location) is None:
            return {}
        path = JsonPointer.from_parts([str(part) for part in location])
        try:
            parent = JsonPointer.from_parts(path.parts[:-1]).resolve(candidate)
        except JsonPointerException:
            return {}
        if not isinstance(parent, dict):
            return {}  # Never insert, replace or reorder array entries.
        exists = path.parts[-1] in parent
        if exists == (kind == "missing"):
            return {}
        if (exists and isinstance(parent[path.parts[-1]], (dict, list)) and not (
                args_type is not None and (_scalar(field_annotation(args_type, path.parts)) or
                    kind == "candidate_record" and _optional_entry(args_type, path.parts)))):
            return {}  # A container replacement could change unrelated facts or array order.
        if path.path not in allowed:
            if any(path.parts[:len(other)] == other or other[:len(path.parts)] == path.parts for other in paths):
                return {}
            paths.append(path.parts)
        allowed[path.path] = "replace" if exists else "add"
        if len(allowed) > 16:
            return {}
    return allowed


def apply_amendments(candidate, edits, allowed_paths, args_type, *, record_paths=(), on_invalid=None):
    """一次交付全部相关修订，在副本合并；原候选及未授权字段保持不变。"""
    edits = [FieldEdit.model_validate(edit) for edit in edits]
    if not 1 <= len(edits) <= 16 or not allowed_paths:
        raise ValueError("局部修订须覆盖本次1至16个已定位字段")
    if len({edit.path for edit in edits}) != len(edits) or {edit.path for edit in edits} != set(allowed_paths):
        raise ValueError("局部修订须恰好覆盖全部允许路径，不能遗漏、重复或修改无关字段")
    paths = []
    for edit in edits:
        pointer = JsonPointer(edit.path)
        if not pointer.parts or pointer.parts == ["result"] or edit.op != allowed_paths[edit.path]:
            raise ValueError("局部修订的操作或位置不在本次允许范围")
        if any(pointer.parts[:len(other)] == other or other[:len(pointer.parts)] == pointer.parts for other in paths):
            raise ValueError("局部修订不能同时替换父字段及其子字段")
        paths.append(pointer.parts)
        parent = JsonPointer.from_parts(pointer.parts[:-1]).resolve(candidate)
        if not isinstance(parent, dict) or ((pointer.parts[-1] in parent) != (edit.op == "replace")):
            raise ValueError("局部修订只能补充缺失对象字段或替换已存在字段，不能操作数组项")
        if (edit.op == "replace" and isinstance(parent[pointer.parts[-1]], (dict, list)) and not (
                _scalar(field_annotation(args_type, pointer.parts)) or
                edit.path in record_paths and _optional_entry(args_type, pointer.parts))):
            raise ValueError("局部修订不能整体替换已有对象或列表，须定位到受影响字段")
    result = jsonpatch.JsonPatch([edit.model_dump() for edit in edits]).apply(deepcopy(candidate))
    from tools.ToolArguments import normalize_arguments
    return normalize_arguments(result, args_type, on_invalid=on_invalid)


def move_fields(candidate, moves):
    """在副本上应用原子移动；任何一步不成立时，原候选完全不变。"""
    result = deepcopy(candidate)
    touched = []
    for raw in moves:
        move = FieldMove.model_validate(raw)
        source, target = JsonPointer(move.source), JsonPointer(move.path)
        if not source.parts or not target.parts:
            raise ValueError("结构修复不能替换整个交付")
        for path in (source.parts, target.parts):
            if any(path[:len(prior)] == prior or prior[:len(path)] == path for prior in touched):
                raise ValueError("结构修复的移动必须独立，不能借助临时字段跨对象转移")
        touched.extend((source.parts, target.parts))
        source_parent = JsonPointer.from_parts(source.parts[:-1]).resolve(result)
        target_parent = JsonPointer.from_parts(target.parts[:-1]).resolve(result)
        if not isinstance(source_parent, dict) or not isinstance(target_parent, dict):
            raise ValueError("结构修复只移动对象字段，不能调整数组项及其身份顺序")
        if target.parts[-1] in target_parent:
            raise ValueError("结构修复不能覆盖已有字段")
        a, b = source.parts[:-1], target.parts[:-1]
        if a[:len(b)] != b and b[:len(a)] != a:
            raise ValueError("结构修复不能在不同对象间转移事实或身份")
        result = jsonpatch.JsonPatch([{"op": "move", "from": move.source, "path": move.path}]).apply(result)
    return result
