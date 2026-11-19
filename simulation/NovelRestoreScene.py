"""从已审、已提交的互动结果接续机会；不代写人物回应或重放旧行为。"""
from copy import deepcopy

from common.NovelRestoreSchema import RoundSchedule
from simulation.NovelRestoreGoals import goal_replan_required, goals_satisfied


def validate_continuations(assessment, actions, eligible, state, runtimes):
    rows = {a.action_id: a for a in actions}
    people = state["characters"]
    selected = []
    for event, next_step in assessment.scene_continuations.items():
        source = rows.get(next_step.source_action_id)
        if (event not in eligible or next_step.actor not in eligible[event]
                or source is None or source.position.event_id != event
                or next_step.actor not in source.observations):
            raise ValueError("场景接续须由本批本事件中实际获得体验的参与人物承担")
        if (next_step.actor not in people
                or people[next_step.actor]["life_status"] in ("dead", "not_born")
                or not runtimes.get(next_step.actor, {}).get("can_act", True)):
            raise ValueError("场景接续人物当前不能主动行动")
        selected.append(next_step.actor)
    if len(selected) != len(set(selected)):
        raise ValueError("同一人物不能同时接续两个事件")


def continuation_schedule(assessment, actions, previous, active, goals, clock, version, config):
    """仅在整个下批仍可直接接续时派生调度，否则走正常世界调度。"""
    if (not active or not active <= assessment.scene_continuations.keys()
            or goals_satisfied(goals)
            or goal_replan_required(goals, config.goal_stagnation_batches)):
        return None
    sources = {a.action_id: a for a in actions}
    final_views = actions[0].advance.final_views
    actors, opportunities, decisions = {}, {}, {}
    for event in sorted(active):
        continuation = assessment.scene_continuations[event]
        source = sources.get(continuation.source_action_id)
        original = previous.get("opportunities", {}).get(event)
        if (source is None or source.position.event_id != event or not original
                or continuation.actor not in source.observations
                or continuation.actor not in final_views):
            return None
        actors[event] = continuation.actor
        opportunities[event] = {**deepcopy(original),
            "scene_anchor": source.action_id, "perceivers": [continuation.actor],
            "motive": continuation.reason, "possible_responses": ["本人依据实际体验自主回应，结果仍待裁定"],
            "alternatives": [], "change_from_previous": continuation.reason,
            "stop_before": continuation.stop_before}
        decisions[event] = {"pacing": "stay", "after_action_id": source.action_id,
            "focus_anchor": source.action_id, "opportunity": continuation.reason,
            "stop_before": continuation.stop_before}
    result = RoundSchedule(actors=actors, opportunities=opportunities, scene_decisions=decisions)
    return {**result.model_dump(mode="json"), "group_id": previous["group_id"],
        "round": clock["round"], "replan": False,
        "scene_continuation_version": version,
        "scene_origin_round": previous.get("scene_origin_round", previous["round"])}
