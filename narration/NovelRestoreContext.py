"""写作阶段的冻结草稿、章节及片段索引；不读取运行中的世界。"""
from common.Runtime import RunStopped, storage_encode as encode


def timeline_records(source):
    """按故事顺序交付完整素材，每份共享经历只交一次。"""
    for identity, character in sorted(source["characters"].items()):
        yield {"source_id": "role/" + identity, "material": character}
    for draft in source["drafts"]:
        yield {"source_id": draft["draft_id"], "material": draft}
        by_turn = {}
        for event in draft["event_ids"]:
            for action in source["actions"][event]:
                by_turn.setdefault(action["position"]["turn"], []).append(action)
        shared = {}
        for identity in draft["shared_material_ids"]:
            item = source["shared_materials"][identity]
            shared.setdefault(item["turn"], []).append(item)
        for turn in sorted(by_turn.keys() | shared.keys()):
            for action in sorted(by_turn.get(turn, ()), key=lambda a: a["position"]["round"]):
                yield {"source_id": action["action_id"], "material": action}
            for item in shared.get(turn, ()):
                yield {"source_id": item["shared_id"], "material": item}


def timeline_batches(branch, source, base, limit):
    """整条记录装入一批，不拆句、不跳过放不下的单项。"""
    context = {"operation": "read_timeline", **base, "branch": branch,
               "candidate_run_id": source["run_id"], "records": []}
    empty_size = len(encode(context))
    size = empty_size
    for record in timeline_records(source):
        record_size = len(encode(record))
        if empty_size + record_size > limit:
            raise RunStopped(f"{branch} 的完整素材 {record['source_id']} 超过叙述上下文预算，请使用较大预算的新版本")
        if context["records"] and size + 1 + record_size > limit:
            yield context
            context = {**context, "records": []}
            size = empty_size
        size += int(bool(context["records"])) + record_size
        context["records"].append(record)
    if context["records"]:
        yield context


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


def narrative_index(source):
    """以已冻结的组草稿及其已发生结果建立全书索引。"""
    fields = ("draft_id", "group_id", "event_ids", "node_order", "group_order", "achieved_results")
    return {"run_id": source["run_id"], "drafts": [{k: d[k] for k in fields} for d in source["drafts"]]}


def simulation_source(source, event_ids, action_ids=None):
    """草稿保持完整；actions 是草稿中的故事材料索引，不读取模拟运行日志。"""
    events = set(event_ids)
    selected = None if action_ids is None else set(action_ids)
    drafts = [d for d in source["drafts"] if events.intersection(d["event_ids"])]
    events = {event for d in drafts for event in d["event_ids"]}
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
