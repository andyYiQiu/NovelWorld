"""客观世界状态更新与验收引用检查；复用上游回放，语义判断属于世界智能体。"""
import re
from common.Runtime import encode
from inputs.NovelWorldState import apply_delta, public_state, validate_state
from common.NovelRestoreSchema import WorldState


def scene_fact(state, path):
    """Read one actual state property, preserving its known/unknown structure.

    A reference or position is a fact, not just its final ID. Whole entities
    remain too broad; an unknown reference cannot establish a location/owner.
    Whether the returned fact supports an opportunity still needs review.
    """
    parts = path.split("/")[1:]
    domains = ("characters", "locations", "objects", "relations")
    if not parts or parts[0] not in domains or len(parts) < 3:
        raise ValueError(f"现场依据须定位到一项状态属性，不能引用整个实体：{path}")
    value = resolve_pointer(state, path)

    def known(item):
        if isinstance(item, dict):
            return any(known(v) for k, v in item.items() if k != "unit")
        if isinstance(item, list):
            return not item or any(known(v) for v in item)
        return item is not None and (not isinstance(item, str) or item.strip() not in ("", "unknown"))

    if not known(value):
        raise ValueError(f"现场依据尚无已知内容：{path}；请使用实际已知的状态属性或已提交行动，不能补造事实")
    return value


def apply_patch(state, patch):
    return WorldState.model_validate(public_state(apply_delta(state, patch), include_evidence=False))


def sync_snapshot(state, snapshot):
    """初次载入快照；以后保留已验收的实际后果，只切换节点标识。"""
    if state is None:
        return WorldState.model_validate(snapshot.model_dump(mode="json"))
    return state.model_copy(update={"node_id": snapshot.node_id, "previous_node_id": snapshot.previous_node_id})


def validate_references(state, catalog, *, locations=None):
    assets = {**catalog, "location": locations} if locations is not None else catalog
    validate_state(state.model_dump(mode="json"), assets, "运行世界状态")


def _same_json(left, right):
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right


def validate_result_evidence(judgment, state, action_ids, *, event_action_ids=None):
    """核对已达成结论的引用与实际值；不替模型判断文字是否由依据推出。"""
    indexes = [item.result_index for item in judgment.result_evidence]
    if sorted(indexes) != list(range(len(judgment.achieved_results))):
        raise ValueError("每项 achieved_results 须恰有一条 result_evidence，result_index 从 0 开始")
    cited = {identity for item in judgment.result_evidence for identity in item.action_ids}
    if judgment.achieved and event_action_ids is not None and not cited.intersection(event_action_ids):
        raise ValueError("确认核心事件完成须引用本事件实际发生或本批裁定的行动")
    for index, item in enumerate(judgment.result_evidence):
        if len(item.action_ids) != len(set(item.action_ids)) or set(item.action_ids) - action_ids:
            raise ValueError("结果依据只能引用本组已发生或本批已裁定的行动编号")
        for fact_index, fact in enumerate(item.state_facts):
            location = f"result_evidence[{index}].state_facts[{fact_index}]"
            try:
                value = resolve_pointer(state, fact.path)
            except ValueError as error:
                raise ValueError(f"{location}（current_state）：{error}") from error
            if not _same_json(value, fact.value):
                raise ValueError(f"结果依据的值与 current_state 不一致：{location}；path={fact.path}；"
                                 f"实际值={encode(value)[:1200]}；交付值={encode(fact.value)[:1200]}")


def resolve_pointer(value, path):
    """严格读取已提供资料；不把不存在的路径当作 null。"""
    if not path.startswith("/") or re.search(r"~(?![01])", path):
        raise ValueError("依据须使用合法的 JSON Pointer：" + path)
    traversed = []
    for encoded in path[1:].split("/"):
        segment = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and segment in value:
            value = value[segment]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", segment) and int(segment) < len(value):
            value = value[int(segment)]
        else:
            available = ("可用字段=" + str(list(value)) if isinstance(value, dict) else
                         f"有效数组下标=0..{len(value) - 1}" if isinstance(value, list) and value else
                         "该处没有可继续读取的子项")
            prefix = "/" + "/".join(traversed) if traversed else "evidence 根"
            raise ValueError(f"资料中不存在依据路径：{path}；已存在位置={prefix}；{available}")
        traversed.append(encoded)
    return value


def resolved_views(value):
    views = {}
    for resolved in value.actions:
        for person, view in resolved.observations.items():
            views.setdefault(person, []).append(view)
    people = set(views) | set(value.advance.observations)
    if set(value.final_views) - people:
        raise ValueError("当前视图只能属于本批有个人体验的人物")
    final = {}
    for person in people:
        parts = [*views.get(person, [])]
        if person in value.advance.observations:
            parts.append(value.advance.observations[person])
        if person in value.final_views:
            final[person] = value.final_views[person]
        elif all(view == parts[0] for view in parts):
            final[person] = parts[0]
        else:
            raise ValueError(f"{person} 本批有多份不同体验，须提供 final_views 中的统一当前视图；原体验均须保留")
    return final
