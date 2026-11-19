"""Stable repository goals and deterministic per-goal feedback bookkeeping.

Planning hypotheses remain separate from actual evidence. Callers commit the
returned records with the action batch; this module never writes state.
"""
from copy import deepcopy

from common.Runtime import encode, task_id
from common.NovelRestoreSchema import GoalUpdate
from simulation.NovelRestoreWorldState import resolve_pointer


def goal_evidence_view(records, *, current=False):
    """Lossless wire references; never merge goals, versions or before/after facts.

    Each reference points to the first original value, so reviewers can cite
    the same path against the unprojected evidence. Stored records stay intact.
    """
    result, known = deepcopy(records), {}
    for identity, row in result.items():
        escaped = identity.replace("~", "~0").replace("/", "~1")
        entries = [(f"/goal_updates/{escaped}", row)] if current else [
            (f"/goal_records/{escaped}/{field}", row.get(field)) for field in ("latest", "completion_evidence")]
        for root, feedback in entries:
            if not isinstance(feedback, dict) or not current and type(feedback.get("version")) is not int:
                continue
            for phase in ("before_facts", "state_facts"):
                for index, fact in enumerate(feedback.get(phase, [])):
                    if "value" not in fact:
                        continue
                    key = encode([None if current else feedback["version"], phase, fact["path"], fact["value"]])
                    path = f"{root}/{phase}/{index}/value"
                    previous = known.setdefault(key, path)
                    if previous != path and len(encode(fact["value"])) > len(encode(previous)) + 16:
                        del fact["value"]
                        fact["value_ref"] = previous
    return result


def reusable_goal_updates(records, before, after):
    """Offer old judgments only when their explicit state evidence is unchanged.

    This is eligibility, not approval: the world must affirm no new semantic
    effect and the independent reviewer still sees each fully assembled update.
    Historical actions are not carried forward as evidence of a new gain.
    """
    result = {}
    for identity, row in records.items():
        previous = row.get("latest")
        if (row.get("review_pending") or not previous or not previous.get("state_facts")
                or previous.get("status") != row.get("status")):
            continue
        try:
            if any(encode(resolve_pointer(snapshot, fact["path"])) != encode(fact["value"])
                   for snapshot in (before, after) for fact in previous["state_facts"]):
                continue
        except ValueError:
            continue
        update = {k: deepcopy(previous[k]) for k in GoalUpdate.model_fields if k in previous}
        update.update(change="unchanged", before_facts=deepcopy(previous["state_facts"]),
                      action_ids=[], exploration_gain=[])
        result[identity] = update
    return result


def initial_goal_records(group):
    definitions = []
    for event in group.events:
        definitions.append((task_id("goal", group.group_id, "event", event["task_id"]),
            {"kind": "event", "event_id": event["task_id"], "requirement": event["todo"],
             "required_results": event.get("required_results")}))
    delta = group.state_delta.model_dump(mode="json", exclude_unset=True)
    def collect(value, path):
        if isinstance(value, list):
            for index, row in enumerate(value):
                # Explicit repository updates, not every unchanged snapshot
                # attribute. The semantic review decides satisfaction, not ==.
                if isinstance(row, dict):
                    fields = row.get("set")
                    entries = ((field, {**row, "set": {field: value}}) for field, value in fields.items()) if fields else [(None, row)]
                    for field, requirement in entries:
                        suffix = [*path, str(index), *(["set", field] if field is not None else [])]
                        definitions.append((task_id("goal", group.group_id, "state", *suffix),
                            {"kind": "state", "source_path": "/state_delta/" + "/".join(suffix),
                             "requirement": deepcopy(requirement)}))
        elif isinstance(value, dict):
            for key, child in value.items():
                collect(child, [*path, key])
    collect(delta, [])
    definitions.append((task_id("goal", group.group_id, "continuity"),
                        {"kind": "continuity", "requirement": "实际持续后果与后续必要前提相容；参考未知不禁止有据细节。"}))
    return {identity: {"definition": definition, "status": "unknown", "event_occurred": False,
                       "completion_evidence": None,
                       "stagnant_batches": 0, "no_opportunity_batches": 0, "ineffective_attempts": 0,
                       "blocked_replans": 0, "recent": [], "latest": None}
            for identity, definition in definitions}


def goals_satisfied(records):
    """批准的逐项记录只触发正式验收，不能替代整组独立判断。"""
    return bool(records) and all(row["status"] == "satisfied" and not row.get("review_pending")
                                 for row in records.values())


def goal_replan_required(records, threshold):
    return any(row.get("review_pending") or row["status"] != "satisfied" and row["stagnant_batches"] >= threshold
               for row in records.values())


def invalidate_group_acceptance(records, *, version, assessment_key, continuity_ok=True):
    """终审否定总通过但未逐项定位时，不猜测哪项条件失败，也不计作新行动。

    已发生的事件不能撤销；其余曾满足的条件恢复待复核，保留原批次证据。
    下一次机会规划读取终审意见，下一批逐项审核重新确定实际状态。
    """
    result = deepcopy(records)
    previously_complete = goals_satisfied(records)
    for row in result.values():
        kind = row["definition"]["kind"]
        if (kind != "event" and row["status"] == "satisfied"
                and (previously_complete or kind == "continuity" and not continuity_ok)):
            row["status"] = "unknown"
            row["review_pending"] = {"version": version, "assessment_key": assessment_key}
    return result


def goal_exhausted(records, config):
    return any(row["status"] != "satisfied" and row.get("blocked_replans", 0) >= config.max_blocked_replans
               for row in records.values())


def validate_goal_updates(records, assessment, before, after, action_ids):
    if set(assessment.goal_updates) != set(records):
        raise ValueError("逐项目标检查须恰好覆盖本组稳定目标，不可增加小目标或漏审停滞项")
    all_satisfied = all(update.status == "satisfied" for update in assessment.goal_updates.values())
    if assessment.goal.achieved != all_satisfied:
        raise ValueError("整组完成判断须与逐项目标一致；新增承接矛盾须记录到承接目标，不可仅修改总标志")
    continuity = [update for identity, update in assessment.goal_updates.items()
                  if records[identity]["definition"]["kind"] == "continuity"]
    if assessment.goal.continuity_ok != all(update.status == "satisfied" for update in continuity):
        raise ValueError("承接判断须与逐项承接目标一致")
    for identity, update in assessment.goal_updates.items():
        if len(update.action_ids) != len(set(update.action_ids)) or set(update.action_ids) - action_ids:
            raise ValueError("目标贡献只能引用本组实际行动")
        for source, snapshot, facts in (("before_facts/state_before", before, update.before_facts),
                                        ("state_facts/current_state", after, update.state_facts)):
            for index, fact in enumerate(facts):
                location = f"goal_updates[{identity!r}].{source}[{index}]"
                try:
                    actual = resolve_pointer(snapshot, fact.path)
                except ValueError as error:
                    raise ValueError(f"{location}：{error}") from error
                if encode(actual) != encode(fact.value):
                    raise ValueError(f"目标贡献的前后值必须与对应实际快照相符：{location}；path={fact.path}；"
                                     f"实际值={encode(actual)[:1200]}；交付值={encode(fact.value)[:1200]}")
        if (update.change != "unchanged" or update.status == "satisfied" or update.exploration_gain) and not (
                update.action_ids or update.state_facts):
            raise ValueError("目标改善、退步、达标或探索收益必须有实际行动或状态依据")
        row = records[identity]
        if row["definition"]["kind"] == "event":
            event = row["definition"]["event_id"]
            judgment = assessment.events.get(event)
            occurred = row["event_occurred"] or bool(judgment and judgment.achieved)
            if (update.status == "satisfied") != occurred:
                raise ValueError("事件历史目标须与已经发生的事件证据一致；当前状态退步不能抹去历史")


def update_goal_records(records, assessment, schedule, *, version, window):
    result = deepcopy(records)
    opportunities = schedule.get("opportunities", {})
    for identity, update in assessment.goal_updates.items():
        row = result[identity]
        row.pop("review_pending", None)
        offered = {event: item for event, item in opportunities.items()
                   if event in assessment.events and identity in item["goal_ids"]}
        improved = update.change == "improved"
        satisfied = update.status == "satisfied"
        row["status"] = update.status
        if row["definition"]["kind"] == "event" and satisfied:
            row["event_occurred"] = True
            if row["completion_evidence"] is None:
                row["completion_evidence"] = {"version": version, **update.model_dump(mode="json")}
        row["stagnant_batches"] = 0 if improved or satisfied else row["stagnant_batches"] + 1
        row["no_opportunity_batches"] = 0 if offered or satisfied else row["no_opportunity_batches"] + 1
        row["ineffective_attempts"] += int(bool(offered) and not improved and not satisfied)
        blocked = schedule.get("replan", False) and offered and all(
            item["path_status"] == "no_known_path" for item in offered.values())
        row["blocked_replans"] = 0 if improved or satisfied else row["blocked_replans"] + int(bool(blocked))
        feedback = {"version": version, **update.model_dump(mode="json"),
                    "schedule_round": schedule.get("round"), "opportunity_event_ids": sorted(offered)}
        row["latest"] = feedback
        # The database wave receipt keeps all prior feedback. This window is
        # a bounded planning view, not deletion of historical evidence.
        row["recent"] = [*row["recent"], {"version": version, "status": update.status,
                          "change": update.change, "offered": bool(offered),
                          "exploration_gain": update.exploration_gain}][-window:]
    return result
