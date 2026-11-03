"""客观世界状态更新与验收引用检查；复用上游回放，语义判断属于世界智能体。"""
import re
from contract.NovelWorldState import apply_delta, public_state
from common.NovelRestoreSchema import WorldState
from contract.RepositoryState import validate_state


def apply_patch(state, patch):
    return WorldState.model_validate(public_state(apply_delta(state, patch), include_evidence=False))


def sync_snapshot(state, snapshot):
    """初次载入快照；以后保留已验收的实际后果，只切换节点标识。"""
    if state is None:
        return WorldState.model_validate(snapshot.model_dump(mode="json"))
    return state.model_copy(update={"node_id": snapshot.node_id, "previous_node_id": snapshot.previous_node_id})


def validate_references(state, catalog):
    validate_state(state.model_dump(mode="json"), catalog, "运行世界状态")


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
    for item in judgment.result_evidence:
        if len(item.action_ids) != len(set(item.action_ids)) or set(item.action_ids) - action_ids:
            raise ValueError("结果依据只能引用本组已发生或本批已裁定的行动编号")
        for fact in item.state_facts:
            value = resolve_pointer(state, fact.path)
            if not _same_json(value, fact.value):
                raise ValueError("结果依据的值与 current_state 不一致：" + fact.path)


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
