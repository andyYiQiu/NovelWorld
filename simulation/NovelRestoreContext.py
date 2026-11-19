"""世界模拟的行动历史、互动机会及相邻节点参考；不推断人物知情。"""

import re
from copy import deepcopy
from jsonpointer import JsonPointer

from common.Runtime import encode
from common.NovelRestoreSchema import CharacterState, PersonRuntime, ConfirmedAction
from simulation.NovelRestoreObjects import object_assets
from inputs.NovelWorldState import location_ancestors
from simulation.NovelRestoreGoals import goal_evidence_view


REFERENCE_STATE_POLICY = (
    "target_state 与 next_reference 是上游参考，不是将覆盖运行状态的值。进入下一组保留已发生的实际后果。"
    "same_state_as_target 仅表示两份参考快照的四类内容相同，不表示当前运行状态必须与参考逐字段相同。"
    "有依据的下游物品、状态和环境不因未列在参考中而消失；仅实际违反明确目标或后续必要前提时判承接冲突。")

TASK_VIEW_POLICY = ("\n资料为本任务视图，存储原文未改。state_scope 列出状态范围，范围外不表示不存在，"
    "必要时用 world_lookup 补查。state_comparisons 中每项从 base 指定的本层字段出发，"
    "按 JSON Patch 的 add/remove/replace 重建该时点或参考快照；空 patch 表示值完全相同。"
    "进度行动中的 working_state_refs 按人物引用本层资料 /working_states 下已提供的同一批后心理，"
    "不表示新心理，也不合并不同人物或不同批次。"
    "history_scope 标明未默认发送的旧历史，不能把未展示当作未发生；需要时按其中说明补查完整记录。"
    "value_ref 指向所在候选或依据对象内的原值；审核可引用该路径，不合并目标或时点。"
    "target_state 始终只是参考要求，不能当作已发生。完整当前候选仍须独立审核。")


def action_history(events):
    """历史行动的客观证据视图；完整审计记录由世界按 action_id 读取。

    参考 StoryBox Event 的正文/运行字段分离、CreAgentive 的按事件 ID 读取详情。
    不把已存行动中的人物输入、记忆与心理副本再复制进当前任务；原文仍在存储中。
    当前待审候选不得经过此函数，必须完整交给独立审核。
    """
    fields = ("action_id", "position", "proposal", "result", "state_patch", "observations",
              "resolved_targets", "unknown_target_review", "environment_basis", "perceptions")
    result = {}
    for event, actions in events.items():
        result[event] = []
        for action in actions:
            record = {key: action[key] for key in fields if key in action}
            shared_id = (action.get("advance") or {}).get("shared_id") or action.get("shared_id")
            if shared_id:
                record["shared_id"] = shared_id
            result[event].append(record)
    return result


def progress_actions(actions):
    """Progress reads actual consequences and mental changes, not storage receipts.

    The candidate under review is WaveAssessment. Memories and personal inputs
    have their own complete review before this step; nothing in storage changes.
    """
    fields = ("action_id", "position", "proposal", "result", "state_patch", "observations",
              "resolved_targets", "environment_basis", "working_state_before", "working_states", "working_state_refs")
    actions = [row.model_dump(mode="json", include=set(fields) | {"advance"})
               if isinstance(row, ConfirmedAction) else row for row in actions]
    shared = shared_history({"wave": actions})
    return [{**{key: row[key] for key in fields if key in row},
             **({"advance": shared[row["advance"]["shared_id"]]} if row.get("advance") else {})}
            for row in actions]


def _state_patch(base, value, path=""):
    """RFC 6902 add/remove/replace; lists remain atomic, no semantic inference."""
    if encode(base) == encode(value):
        return []
    if not isinstance(base, dict) or not isinstance(value, dict):
        return [{"op": "replace", "path": path, "value": value}]
    changes = []
    for key in sorted(base.keys() | value.keys()):
        pointer = path + "/" + key.replace("~", "~0").replace("/", "~1")
        if key not in value:
            changes.append({"op": "remove", "path": pointer})
        elif key not in base:
            changes.append({"op": "add", "path": pointer, "value": value[key]})
        else:
            changes.extend(_state_patch(base[key], value[key], pointer))
    return changes


def _state_scope(context, extra):
    """Program-prepared local records; global rules/relations stay available.

    StoryBox selects nearby facts; CreAgentive selects character/chapter facts.
    Here explicit identities and references form the scope, including every
    changed record in before/target comparisons. Missing records are retrievable.
    """
    fields = ("current_state", "state_before", "reference_before", "target_state")
    states = {k: context[k] for k in fields if isinstance(context.get(k), dict)
              and all(isinstance(context[k].get(d), dict) for d in ("characters", "objects", "locations"))}
    base_key = next((k for k in fields[:3] if k in states), None)
    if base_key is None:
        return context
    base = states[base_key]
    catalogs = {d: {key for state in states.values() for key in state[d]}
                for d in ("characters", "objects", "locations")}
    known = set().union(*catalogs.values())

    def references(value):
        found = set()
        if isinstance(value, dict):
            found.update(known.intersection(value))
            for item in value.values():
                found.update(references(item))
        elif isinstance(value, list):
            for item in value:
                found.update(references(item))
        elif isinstance(value, str):
            if value in known:
                found.add(value)
            found.update(known.intersection(re.findall(r"[A-Za-z0-9_]+", value)))
            found.update(known.intersection(part.replace("~1", "/").replace("~0", "~")
                                             for part in value.split("/")))
        return found

    seeds = {k: v for k, v in context.items()
             if k not in fields and k not in {"environment", "person_runtime", "object_inventory"}}
    if isinstance(seeds.get("working_states"), dict):
        empty_mental = CharacterState().model_dump(mode="json")
        seeds["working_states"] = {k: v for k, v in seeds["working_states"].items()
                                   if v is not None and v != empty_mental}
    runtime = context.get("person_runtime", {})
    default_runtime = PersonRuntime().model_dump(mode="json")
    # Empty runtime placeholders for every repository character do not make
    # every character relevant. Any non-default body/effect remains a seed.
    active_runtime = {k: v for k, v in runtime.items() if v != default_runtime}
    assets = context.get("environment", {}).get("assets", {})
    selected = references([seeds, extra, active_runtime, assets.get("role", {}), assets.get("object", {})])
    for state in states.values():
        for domain in catalogs:
            selected.update(k for k in base[domain].keys() | state[domain].keys()
                            if encode(base[domain].get(k)) != encode(state[domain].get(k)))
    # Follow exact object ownership/location and character location references,
    # including references in earlier states. This never invents availability.
    while True:
        expanded = set(selected)
        for state in states.values():
            for domain in catalogs:
                for key in selected & state[domain].keys():
                    expanded.update(references(state[domain][key]))
        expanded.update(location_ancestors(assets.get('location', {}), expanded & assets.get('location', {}).keys()))
        if expanded == selected:
            break
        selected = expanded
    if not selected:
        return context  # No reliable scope: keep the source, don't guess relevance.
    result = dict(context)
    for field, state in states.items():
        result[field] = {**state, **{d: {k: v for k, v in state[d].items() if k in selected}
                                   for d in catalogs}}
    if all(result[field] == state for field, state in states.items()):
        return context
    result["state_scope"] = {"included": {d: sorted(ids & selected) for d, ids in catalogs.items()},
                             "omitted_counts": {d: len(ids - selected) for d, ids in catalogs.items()},
                             "lookup": "范围外记录仍存在；用 world_lookup 的 current/asset/index 查询。人物关系保留；地点范围包含上级地点链上的环境与使用条件。"}
    if "person_runtime" in result:
        result["person_runtime"] = {k: v for k, v in runtime.items() if k in selected or k not in known}
    if isinstance(result.get("working_states"), dict):
        result["working_states"] = {k: v for k, v in result["working_states"].items()
                                    if k in selected or k not in known}
    if "environment" in result:
        result["environment"] = {**result["environment"], "assets": {
            kind: ({k: v for k, v in records.items() if k in selected or k not in known}
                   if kind in {"role", "object", "location"} else records)
            for kind, records in assets.items()}}
    return result


def _history_scope(context, extra, recent=5):
    """Default recent history plus explicitly referenced sources; full records remain queryable.

    StoryBox retrieves local/relevant events rather than replaying the entire
    transcript. Only callers with world_lookup use this view. Current candidate
    lists and shared changes without a linked historical action stay complete.
    """
    pairs = (("actions", "shared_advances"), ("previous_actions", "previous_shared_advances"))
    histories = {field: context[field] for field, _ in pairs if isinstance(context.get(field), dict)
                 and all(isinstance(rows, list) and all(isinstance(row, dict) and isinstance(row.get("action_id"), str)
                                                       for row in rows) for rows in context[field].values())}
    if not histories:
        return context
    records = {row["action_id"]: row for history in histories.values() for rows in history.values() for row in rows}
    known = set(records)
    shared_ids = {identity for field, shared in pairs if field in histories
                  for identity in context.get(shared, {})}
    sources = known | shared_ids
    # Exclude the histories themselves: reading every old action_id here would
    # select the complete transcript again. Current memories, goal evidence,
    # pending replies, scene anchors and the independent candidate remain seeds.
    excluded = set(histories) | {shared for field, shared in pairs if field in histories}
    pending = [extra, {k: v for k, v in context.items() if k not in excluded}]
    referenced = set()
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            referenced.update(sources.intersection(value))
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) and value in sources:
            referenced.add(value)
    selected = set(referenced)
    for history in histories.values():
        for rows in history.values():
            selected.update(row["action_id"] for row in rows[-recent:])
            selected.update(row["action_id"] for row in rows if row.get("shared_id") in referenced)
    advances = {identity: row for field, shared in pairs if field in histories
                for identity, row in context.get(shared, {}).items()}
    selected.update(row["source_action_id"] for identity, row in advances.items()
                    if identity in referenced and row.get("source_action_id") in known)

    def batch(row):
        position = row.get("position", {})
        if (isinstance(position, dict) and isinstance(position.get("node_id"), str)
                and type(position.get("group_order")) is int and type(position.get("turn")) is int):
            return position["node_id"], position["group_order"], position["turn"]
        return None  # 不把多个缺少时点的记录误认为同一批。

    if any(batch(row) is None for row in records.values()):
        return context  # 无法可靠重建共享批次归属时，保留原历史而不猜测裁剪。
    batches = {batch(records[identity]) for identity in selected & known} - {None}
    kept_shared = set(referenced)
    for identity, advance in advances.items():
        carrier = advance.get("source_action_id")
        if carrier in known and (carrier in selected or batch(records[carrier]) in batches):
            # 共同经过物理上仅随首条行动保存，但属于整批。保留另一事件
            # 的行动时，也须携带其共同经过及承载来源，不能仅按 shared_id 查。
            selected.add(carrier)
            kept_shared.add(identity)
    result, omitted, shared_omitted = dict(context), {}, {}
    for field, shared in pairs:
        if field not in histories:
            continue
        projected = {event: [row for row in rows if row["action_id"] in selected]
                     for event, rows in histories[field].items()}
        counts = {event: len(rows) - len(projected[event]) for event, rows in histories[field].items()
                  if len(rows) != len(projected[event])}
        if counts:
            result[field], omitted[field] = projected, counts
        if shared in context:
            kept_shared.update(row.get("shared_id") for rows in projected.values() for row in rows)
            preserved = {identity: row for identity, row in context[shared].items()
                         if identity in kept_shared or row.get("source_action_id") not in known}
            if len(preserved) != len(context[shared]):
                result[shared] = preserved
                shared_omitted[shared] = len(context[shared]) - len(preserved)
    if omitted or shared_omitted:
        result["history_scope"] = {"recent_per_event": recent, "omitted": omitted, "shared_omitted": shared_omitted,
            "lookup": "旧正文未删除。用 world_lookup(kind=history,category=event,identity=事件编号) 从 after=0 按 next 分页；"
                      "已有 action_id 可直接用 kind=action 读取。未展示不代表不存在，关键依据不足须补查。"}
    return result


def world_task_context(context, *, extra=(), allow_scope=True):
    """Model-facing task view; original context stays bound to the task receipt.

    Never project current personal candidates or delete prose from stored facts.
    Only objective progress uses a smaller view of its already-reviewed action.
    """
    operation = context.get("operation")
    if operation in {"review_visibility", "review_semantics"}:
        evidence = context.get("evidence", {})
        outside = {k: v for k, v in context.items() if k != "evidence"}
        result = {**context, "evidence": world_task_context(evidence, extra=[extra, outside], allow_scope=allow_scope)}
        candidate = context.get("candidate", {})
        if isinstance(candidate, dict) and isinstance(candidate.get("goal_updates"), dict):
            result["candidate"] = {**candidate, "goal_updates": goal_evidence_view(candidate["goal_updates"], current=True)}
        return result
    if operation not in {"schedule", "perceive", "resolve_wave", "review_memory",
                         "assess_wave", "assess_group", "review_unknown_targets"}:
        return context
    result = dict(context)
    if operation == "assess_wave" and isinstance(result.get("actions"), list):
        result["actions"] = progress_actions(result["actions"])
        # 同批后心理已经在顶层按人物给出。只替换同一人物、完全相同的
        # 这份副本；不同心理、行动前认识及完整审核候选均保留原文。
        shared = result.get("working_states", {})
        for action in result["actions"]:
            states = dict(action.get("working_states", {}))
            refs = dict(action.get("working_state_refs", {}))
            for person, state in list(states.items()):
                pointer = "/working_states/" + person.replace("~", "~0").replace("/", "~1")
                if (person in shared and encode(state) == encode(shared[person])
                        and len(encode(state)) > len(encode(pointer)) + 32):
                    refs[person] = pointer
                    del states[person]
            if refs:
                action["working_states"], action["working_state_refs"] = states, refs
    if allow_scope:
        result = _task_sources(result, extra)
        result = _history_scope(result, extra)
        result = _state_scope(result, extra)
    if isinstance(result.get("goal_records"), dict):
        result["goal_records"] = goal_evidence_view(result["goal_records"])
        if operation in {"assess_wave", "assess_group"}:
            # ``recent`` is a bounded planning trend already represented by
            # status/counters and the complete latest judgment.  Progress
            # assessment and its independent review do not need to replay the
            # same prior summaries for every stable goal.  Stored records and
            # schedule inputs remain unchanged.
            result["goal_records"] = {
                identity: {key: value for key, value in row.items() if key != "recent"}
                for identity, row in result["goal_records"].items()
            }
    if operation in {"assess_wave", "assess_group"} and isinstance(result.get("previous_actions"), dict):
        # Historical personal deliveries have already passed visibility and
        # memory review.  Objective progress reads the action/result/state and
        # shared advance; current mental changes are supplied separately.
        # Full historical records remain available to other task types.
        result["previous_actions"] = {
            event: [{key: value for key, value in action.items()
                     if key not in {"observations", "perceptions"}}
                    for action in actions]
            for event, actions in result["previous_actions"].items()
        }
    if operation in {"assess_wave", "assess_group"} and isinstance(result.get("previous_shared_advances"), dict):
        # 与旧行动使用相同证据边界；不能只精简 action 而把同批个人附件从 advance 再传一次。
        result["previous_shared_advances"] = {
            identity: {key: value for key, value in advance.items()
                       if key not in {"observations", "final_views"}}
            for identity, advance in result["previous_shared_advances"].items()
        }
    base_key = next((key for key in ("current_state", "state_before", "reference_before")
                     if isinstance(result.get(key), dict) and "characters" in result[key]), None)
    if base_key is None:
        return result
    comparisons = {}
    for key in ("state_before", "reference_before", "target_state"):
        value = result.get(key)
        if key == base_key or not isinstance(value, dict) or "characters" not in value:
            continue
        comparison = {"base": "/" + base_key, "patch": _state_patch(result[base_key], value)}
        if len(encode(comparison)) < len(encode(value)):
            comparisons[key] = comparison
            del result[key]
    if comparisons:
        result["state_comparisons"] = comparisons
    return result


def _strings(value):
    """Exact identifiers in task data; no keyword guessing or recursive depth."""
    found, pending = set(), [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            found.add(item)
        elif isinstance(item, dict):
            found.update(item)
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return found


def _task_sources(context, extra=()):
    """Task-specific sources, not audit-record replay. Requires read-only lookup.

    Keep current candidates and objective prose complete. Historical perception
    attachments are retrieved by action ID; they are not current perceptions.
    Subjective records remain person-specific and never become world facts.
    """
    operation = context.get("operation")
    result, omitted = dict(context), []
    if not context.get("retrieval_scope"):
        return result  # Only histories explicitly supplied as queryable archives.
    protected = _strings(extra)
    if operation in {"schedule", "perceive", "resolve_wave", "review_memory", "assess_wave", "assess_group"}:
        for field in ("actions", "previous_actions"):
            history = result.get(field)
            if not isinstance(history, dict):
                continue  # Lists contain this round's candidate, not history.
            if not all(isinstance(rows, list) and all(isinstance(r, dict) and r.get("action_id")
                                                     for r in rows) for rows in history.values()):
                continue
            projected = {event: [row if row["action_id"] in protected else
                                {k: v for k, v in row.items() if k not in {"perceptions", "observations"}}
                                 for row in rows] for event, rows in history.items()}
            if projected != history:
                result[field] = projected
                omitted.append(field + "/个人感知附件")
        for field in ("shared_advances", "previous_shared_advances"):
            history = result.get(field)
            if isinstance(history, dict):
                result[field] = {identity: {k: v for k, v in row.items() if k != "final_views"}
                                 if (isinstance(row, dict) and row.get("source_action_id")
                                     and identity not in protected and row["source_action_id"] not in protected) else row
                                 for identity, row in history.items()}
                if result[field] != history:
                    omitted.append(field + "/旧当前视图")
    if operation == "perceive" and context.get("selected"):
        selected = set(context["selected"])
        for field in ("subjective_worlds", "views"):
            if isinstance(context.get(field), dict):
                result[field] = {p: row for p, row in context[field].items() if p in selected}
                if result[field] != context[field]:
                    omitted.append(field + "/非本次感知者")
    if operation == "review_memory" and context.get("summaries"):
        people = set(context["summaries"])
        for field in ("subjective_worlds", "character_assets", "views_before", "person_runtime"):
            if isinstance(result.get(field), dict):
                scoped = {p: row for p, row in result[field].items() if p in people}
                if scoped != result[field]:
                    result[field] = scoped
                    omitted.append(field + "/非本次记忆接收者")
    # Working states already contain the current needs/beliefs/plan. The last
    # experience supplies immediate provenance; older cited memories and their
    # correction chains are kept verbatim, other originals remain retrievable.
    if operation in {"schedule", "perceive", "resolve_wave", "review_memory"}:
        subjective = result.get("subjective_worlds", {})
        seeds = [extra, {k: v for k, v in context.items() if k not in
                        {"subjective_worlds", "actions", "shared_advances", "previous_actions",
                         "previous_shared_advances", "environment"}},
                 {p: {k: v for k, v in row.items() if k != "history"} for p, row in subjective.items()}]
        cited = _strings(seeds)
        scoped = {}
        for person, row in subjective.items():
            history = row.get("history", [])
            if not history or not all(isinstance(m, dict) and m.get("memory_id") for m in history):
                scoped[person] = row
                continue
            selected = {history[-1]["memory_id"]} | cited
            while True:
                kept = [m for m in history if selected.intersection(
                    {m["memory_id"], m.get("action_id"), m.get("source_action_id"), m.get("source_shared_id"),
                     *m.get("corrects", []), *m.get("related_memory_ids", [])})]
                expanded = selected | _strings(kept)
                if expanded == selected:
                    break
                selected = expanded
            scoped[person] = {**row, "history": kept}
            if len(kept) != len(history):
                scoped[person]["history_scope"] = {"omitted": len(history) - len(kept),
                    "lookup": "world_lookup(kind=memories,identity=" + person + ")，按 next 分页读取原文"}
                omitted.append("subjective_worlds/" + person + "/未引用旧记忆")
        if subjective:
            result["subjective_worlds"] = scoped
    if omitted:
        result["source_scope"] = {"omitted": omitted,
            "lookup": "历史个人体验用 world_lookup(kind=action,identity=action_id)；"
                      "其他人物主观资料用 kind=subjective/memories。当前候选、实际经过及状态变化未删。"}
    return result


def shared_history(events):
    """同批自然经过按 shared_id 仅展示一次，保留完整经过、变化及个人体验。

    首条行动可用于 world_lookup(kind=action) 回读完整保存记录。不能用文本相同
    合并不同 shared_id，也不能悄悄合并同一编号下相互矛盾的记录。
    """
    fields = ("shared_id", "elapsed_seconds", "basis", "result", "state_patch", "observations",
              "related_event_ids", "runtime_changes", "processing", "new_objects", "final_views")
    result, originals = {}, {}
    for actions in events.values():
        for action in actions:
            advance = action.get("advance") or {}
            identity = advance.get("shared_id")
            if not identity:
                continue
            if identity in originals:
                if encode(advance) != originals[identity]:
                    raise ValueError(f"同一共享经过 {identity} 对应不同保存记录")
                continue
            originals[identity] = encode(advance)
            result[identity] = {key: advance[key] for key in fields if key in advance}
            result[identity]["source_action_id"] = action["action_id"]
    return result


def interaction_progress(events, eligible, *, group, recent=5, shared_advances=None):
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
        advance = action.get("advance") or (shared_advances or {}).get(action.get("shared_id"), {})
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


def pending_observations(events, interactions, *, shared_advances=None):
    """重规划审读用实际个人反馈核对尚待回应的机会，不从反馈次数推断人物看见了什么。"""
    pending = {}
    for item in interactions.values():
        for person, opportunity in item["opportunities"].items():
            source = opportunity["feedback_without_later_opportunity"]["latest_source_id"]
            if source:
                pending.setdefault(source, set()).add(person)
    result = {}
    for actions in events.values():
        for action in actions:
            advance = action.get("advance") or (shared_advances or {}).get(action.get("shared_id"), {})
            for source, record in ((action["action_id"], action), (advance.get("shared_id"), advance)):
                if source in pending and source not in result:
                    result[source] = {person: record["observations"][person]["personal"]
                                      for person in sorted(pending[source])}
    return result


def compact_next_reference(target, following):
    """仅当四类状态逐项完全相同时复用目标；节点标识仍独立提供。"""
    sections = {"characters", "objects", "locations", "relations"}
    fields = sections | {"node_id", "previous_node_id"}
    if (isinstance(target, dict) and isinstance(following, dict) and set(target) == set(following) == fields
            and all(encode(target[key]) == encode(following[key]) for key in sections)):
        return {"node_id": following["node_id"], "previous_node_id": following["previous_node_id"],
                "same_state_as_target": True}
    return following


def runtime_context(world, group, *, selected=None):
    """感知与裁定共用真实局面；目标另由规划及验收任务按需添加。"""
    people = {p["role_code"] for e in group.events for p in e["participants"]}
    views = world.storage.get_checkpoint("views", {})
    state = world.storage.current()[1].model_dump(mode="json")
    locations = {i for e in group.events for i in e.get("location_ids", [])}
    locations.update(state['characters'][p]['position']['location_id'] for p in people
                     if state['characters'][p]['position']['location_id'])
    for person, character in state['characters'].items():
        if character['position']['location_id'] in locations:
            people.add(person)
    locations = location_ancestors(world.world['assets']['location'], locations)
    for person in tuple(people):
        for mapping in views.get(person, {}).get("correspondences", []):
            if mapping.get("actual_kind") == "role":
                people.add(mapping["actual_id"])
    actions = {e["task_id"]: [a.model_dump(mode="json", exclude={"completion"})
                               for a in world.storage.event_actions(e["task_id"])] for e in group.events}
    loaded = {row["action_id"] for rows in actions.values() for row in rows}
    shared = {(row.get("advance") or {}).get("shared_id") for rows in actions.values() for row in rows}
    subjective = {}
    for person in sorted(people if selected is None else selected):
        working = world.storage.character_state(person).model_dump(mode="json")
        perception = views.get(person, {}).get("personal", {}).get("content", "")
        query = " ".join([perception, *working["goals"], *working["plan"]])
        memories = world.storage.retrieve_memories(person, query=query, relevant=world.storage.config.memory_candidates)
        subjective[person] = {"character_id": person, "profile": world.storage.manifest["characters"][person],
            "current": {"working_state": working},
            "history": [m.model_dump(mode="json") for m in memories],
            "summaries": world.storage.memory_summaries(person)}
        # 候选记忆的客观来源必须可直接核对；其他历史可使用世界查询。
        for memory in memories:
            # 本任务已读取的来源原文直接复用，不按每人的每条记忆重复查询及解析。
            if memory.action_id in loaded or memory.source_shared_id and memory.source_shared_id in shared:
                continue
            if memory.action_id:
                action = world.storage.get_action(memory.action_id)
            elif memory.source_shared_id:
                row = world.storage.connection.execute("""SELECT payload FROM actions WHERE status='committed'
                    AND json_extract(payload,'$.advance.shared_id')=?""", (memory.source_shared_id,)).fetchone()
                action = ConfirmedAction.model_validate_json(row[0]) if row else None
            else:
                action = None
            if action and action.action_id not in loaded:
                actions.setdefault(action.position.event_id, []).append(action.model_dump(mode="json", exclude={"completion"}))
                loaded.add(action.action_id)
                if action.advance:
                    shared.add(action.advance.shared_id)
    role_assets = {p: world.world["assets"]["role"][p] for p in sorted(people)}
    for person, record in subjective.items():
        # Only remove an exact duplicate. Approved personal profiles or a
        # differing historical source must not be mistaken for this copy.
        if record["profile"] == role_assets[person]:
            del record["profile"]
    object_ids = {i for e in group.events for i in e.get("object_ids", [])}
    for identity, item in state["objects"].items():
        if (set((item['owner_ids'] or []) + (item['custodian_ids'] or [])) & people
                or item['position']['location_id'] in locations):
            object_ids.add(identity)
    for person in people:
        for mapping in views.get(person, {}).get("correspondences", []):
            if mapping.get("actual_kind") == "object":
                object_ids.add(mapping["actual_id"])
    assets = {**world.world["assets"], "role": role_assets,
              "object": {i: row for i, row in object_assets(world.storage).items() if i in object_ids}}
    return {"environment": {"assets": assets, "worlds": world.world["worlds"]},
            "current_state": state,
            "subjective_worlds": subjective,
            "views": views if selected is None else {p: v for p, v in views.items() if p in selected},
            "actions": action_history(actions), "shared_advances": shared_history(actions),
            "person_runtime": world.storage.get_checkpoint("person_runtime", {}),
            "reference_state_policy": REFERENCE_STATE_POLICY,
            "initial_locations": world.storage.get_checkpoint("location_initializations", []),
            "story_seconds": world.storage.get_checkpoint("story_seconds", 0),
            "retrieval_scope": "actions 是已提交行动的客观证据视图，含本组及候选记忆来源；共享经过按 shared_id 在 shared_advances 中只列一次，source_action_id 可回读原始记录。实际发生、状态变化、个人体验和未知对象依据均保留。完整静态人物档案在 environment.assets.role，subjective_worlds 不重复同源档案，保留候选记忆及当前心理；当前感知只在 views 中提供。旧人物输入、心理、记忆、认知处理及审核记录均可用 world_lookup 的 action 查询按 action_id 完整读取，其他个人历史可用 subjective/memories/history 查询。旧批准不证明当前事实，未展示不等于不存在；需要核对的依据应读取后判断。"}



def amendment_view(context, candidate, paths, *, world=False, allow_scope=False):
    parts = [JsonPointer(path).parts for path in paths]
    scope = deepcopy(context)
    local = candidate
    anchors = [p for p in parts if len(p) == 5 and p[:2] == ["result", "decisions"]
               and p[2].isdecimal() and p[3:] == ["opportunity", "scene_anchor"]]
    locations = [p for p in parts if len(p) in (3, 4) and p[:2] == ["result", "initial_locations"]]
    if scope.get("operation") == "schedule" and len(anchors) + len(locations) == len(parts):
        # Selecting an existing anchor needs actual scene facts and the
        # opportunity it supports, not the entire goal bookkeeping archive.
        keys = {"operation", "eligible", "clock", "current_state", "environment", "person_runtime",
                "actions", "shared_advances", "story_seconds", "retrieval_scope"}
        scope = {k: v for k, v in scope.items() if k in keys}
        selected = [p[:3] for p in anchors + locations]
        # Initial-location proposals are separate from the opportunity. Keep
        # their selected actor's opportunity available without the goal archive.
        people = {p[2] for p in locations}
        selected += [["result", "decisions", str(i)] for i, row in enumerate(candidate["result"]["decisions"])
                     if row.get("actor") in people]
        local = {"items": {JsonPointer.from_parts(p).path: JsonPointer.from_parts(p).resolve(candidate)
                           for p in selected}}
    # Each item is an independently attributed goal. A missing verdict needs
    # that goal and the full actual batch, not every other goal's old proofs.
    if (scope.get("operation") == "assess_wave" and scope.get("goal_records")
            and all(len(p) >= 4 and p[:2] == ["result", "goal_updates"] and p[2].isdecimal() for p in parts)):
        identities = sorted(scope["goal_records"])
        indices = sorted({int(p[2]) for p in parts})
        if all(i < len(identities) for i in indices):
            scope["goal_records"] = {identities[i]: scope["goal_records"][identities[i]] for i in indices}
            scope["repair_scope"] = "仅修所列原始位置；实际本批行动与前后态仍完整用于核对，不改其他目标。"
            local = {"items": {f"/result/goal_updates/{i}": candidate["result"]["goal_updates"][i]
                               for i in indices}}
    if world:
        scope = world_task_context(scope, extra=local, allow_scope=allow_scope)
    return scope, local

def personal_repair_view(context):
    """Only affected personal deliveries; the objective resolution stays whole."""
    if not context.get("repair_personal_only"):
        return context
    people = {row["character_id"] for row in context["repair_targets"]}
    result = deepcopy(context)
    for field in ("subjective_worlds", "views"):
        if field in result:
            result[field] = {p: v for p, v in result[field].items() if p in people}
    original = result["unsafe_delivery"]
    for row in [original, *original.get("actions", []), original.get("advance", {})]:
        for field in ("views", "observations", "final_views"):
            if field in row:
                row[field] = {p: v for p, v in row[field].items() if p in people}
    result["repair_scope"] = "仅展示待修人物的交付；客观裁定原文不变，其余人物由程序保留，合并后重新审核。"
    return result
