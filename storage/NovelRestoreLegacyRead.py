"""方法 1.37 之前运行的只读投影；不迁移数据库、不恢复旧决策算法。

历史需求数值仅按原值展示为文本。旧时间漂移参数仍保留在原数据库，
不进入当前心理模型。只处理已知心理字段，不递归改写故事或证据中的值。
"""
from copy import deepcopy


def character_state(value):
    value = deepcopy(value)
    for need in value.get("needs", {}).values():
        if type(need.get("urgency")) in (int, float):
            need["urgency"] = str(need["urgency"])
        need.pop("drift_per_hour", None)
    return value


def _experience_states(value):
    value["working_states"] = {person: character_state(state)
                               for person, state in value.get("working_states", {}).items()}
    for entry in value.get("subjective_inputs", {}).values():
        entry["working_state"] = character_state(entry["working_state"])


def action_record(value):
    value = deepcopy(value)
    _experience_states(value)
    if value.get("advance"):
        _experience_states(value["advance"])
    return value


def event_draft(value):
    value = deepcopy(value)
    for material in value.get("materials", []):
        _experience_states(material)
    return value
