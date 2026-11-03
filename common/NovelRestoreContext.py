"""模拟与叙述的来源选择；repository 约束与模拟补充分别保留。"""

from common.Runtime import encode


REFERENCE_STATE_POLICY = (
    "target_state 与 next_reference 是上游参考，不是将覆盖运行状态的值。进入下一组保留已发生的实际后果。"
    "same_state_as_target 仅表示两份参考快照的六类内容相同，不表示当前运行状态必须与参考逐字段相同。"
    "有依据的下游物品、状态和环境不因未列在参考中而消失；仅实际违反明确目标或后续必要前提时判承接冲突。")


def action_history(events):
    """保留故事依据；旧审核结论和存储元数据按 action_id 查询，不充作新判断的依据。"""
    def project(record):
        result = {key: value for key, value in record.items()
                  if key not in ("memory_review", "completion", "run_id", "expected_version")}
        if "subjective_inputs" in result:
            result["subjective_inputs"] = {person: {key: value for key, value in access.items()
                if key in ("experiences", "processing")} for person, access in result["subjective_inputs"].items()}
        return result
    result = {}
    for event, actions in events.items():
        result[event] = []
        for action in actions:
            record = project(action)
            if record.get("advance"):
                record["advance"] = project(record["advance"])
            result[event].append(record)
    return result


def interaction_progress(events, eligible, *, group, recent=5):
    """只给世界的机会索引；收到反馈不等于听见、理解或愿意回应。"""
    current = {a["action_id"]: a for records in events.values() for a in records
               if (a["position"]["node_id"], a["position"]["group_order"]) == group}
    ordered = sorted(current.values(), key=lambda a: (a["position"]["turn"], a["position"]["round"]))
    feedback, shared = [], set()
    for action in ordered:
        actor, mode = action["proposal"]["character_id"], action["proposal"]["mode"]
        if mode in ("talk", "action"):
            feedback.append((action["action_id"], "action", action["position"]["turn"],
                             set(action.get("observations", {})) - {actor}))
        advance = action.get("advance") or {}
        if advance.get("shared_id") and advance["shared_id"] not in shared:
            shared.add(advance["shared_id"])
            feedback.append((advance["shared_id"], "shared", action["position"]["turn"],
                             set(advance.get("observations", {}))))
    result = {}
    for event, people in eligible.items():
        actions = [a for a in ordered if a["position"]["event_id"] == event]
        turns = {p: [a for a in ordered if a["proposal"]["character_id"] == p] for p in people}
        opportunities = {}
        for person, own in turns.items():
            last_batch = own[-1]["position"]["turn"] if own else -1
            # 同批是共同前态；本批获得的新反馈不能算作本批已作回应。
            pending = [(identity, kind) for identity, kind, batch, receivers in feedback
                       if person in receivers and batch >= last_batch]
            opportunities[person] = {
                "action_count": len(own), "last_action_id": own[-1]["action_id"] if own else None,
                "feedback_without_later_opportunity": {
                    "count": len(pending), "latest_source_id": pending[-1][0] if pending else None,
                    "source_kind": pending[-1][1] if pending else None}}
        result[event] = {"opportunities": opportunities,
            "recent_actions": [{"action_id": a["action_id"], "proposal": a["proposal"], "result": a["result"]}
                               for a in actions[-recent:]] if recent else []}
    return result


def compact_next_reference(target, following):
    """仅当六类状态逐项完全相同时复用目标；节点标识仍独立提供。"""
    sections = {"characters", "objects", "locations", "relations", "public_state", "constraints"}
    fields = sections | {"node_id", "previous_node_id"}
    if (isinstance(target, dict) and isinstance(following, dict) and set(target) == set(following) == fields
            and all(encode(target[key]) == encode(following[key]) for key in sections)):
        return {"node_id": following["node_id"], "previous_node_id": following["previous_node_id"],
                "same_state_as_target": True}
    return following


def link_components(links):
    """按关联的两端追踪完整链；分叉和环属于同一组，不受阅读距离限制。"""
    neighbors = {}
    for record in links.values():
        link = record["link"]
        left, right = link["from_fragment"], link["to_fragment"]
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    components = []
    while neighbors:
        reached, pending = set(), [next(iter(neighbors))]
        while pending:
            identity = pending.pop()
            if identity not in reached:
                reached.add(identity)
                pending.extend(neighbors.pop(identity, ()))
        components.append(reached)
    return components


def repository_reference(world, event_ids=None):
    """只缩小事件范围，不把草稿的概括当成 repository 的核心要求。"""
    events = None if event_ids is None else set(event_ids)
    groups = [g for g in world["groups"] if events is None or
              any(e["task_id"] in events for e in g["events"])]
    nodes = {g["node_id"] for g in groups}
    selected = {g["group_id"] for g in groups}
    # 保留同节点前序组，供核对参考状态的演变；同组要求不能拆给单个事件。
    last = {node: max(g["order"] for g in groups if g["node_id"] == node) for node in nodes}
    return {"assets": world["assets"], "worlds": world["worlds"],
            "nodes": [n for n in world["nodes"] if n["node_id"] in nodes],
            "state_before": {n: world["state_before"][n] for n in world["state_before"] if n in nodes},
            "groups": groups,
            "preceding_groups": [g for g in world["groups"] if g["node_id"] in nodes
                                 and g["order"] <= last[g["node_id"]] and g["group_id"] not in selected]}


def simulation_source(source, event_ids, action_ids=None):
    """草稿保持完整；actions 是草稿中的故事材料索引，不读取模拟运行日志。"""
    events = set(event_ids)
    selected = None if action_ids is None else set(action_ids)
    drafts = [d for d in source["drafts"] if d["event_id"] in events]
    shared_ids = {identity for draft in drafts for identity in draft.get("shared_material_ids", [])}
    # 已提供完整结构材料时，不重复发送同一内容的阅读投影；片段选择仍保留完整草稿。
    visible_drafts = [{k: v for k, v in d.items() if k != "text"} for d in drafts] if selected is None else drafts
    return {"drafts": visible_drafts,
            "characters": {i: source["characters"][i] for d in drafts for i in d["participant_ids"]},
            "shared_materials": {i: source.get("shared_materials", {})[i] for i in shared_ids},
            "actions": {e: [a for a in actions if selected is None or a["action_id"] in selected]
                        for e, actions in source["actions"].items() if e in events}}


def fragment_index(state):
    """全书轻量索引用于找出未显式连线的影响；实际摘要与待实现计划分开。"""
    return [{**{k: f[k] for k in ("fragment_id", "draft_id", "event_id", "action_ids", "selected_content", "reveal", "withhold")},
             "written": f["fragment_id"] in state["prose"],
             "actual": state["prose_meta"].get(f["fragment_id"])} for f in state["plan"]["fragments"]]
