"""已提交模拟就是故事材料；只做顺序与显示投影，不生成、概括或改写文字。"""
from itertools import groupby


def story_text(actions, characters, event_id=None):
    """同批行动不宣称先后，共享经过只出现一次。正式组草稿使用完整批次。"""
    lines = []
    sources = {"experience": "亲历", "observation": "观察", "reported": "转述",
               "inference": "推断", "imagination": "想象", "plan": "打算"}

    def name(person):
        return characters[person].get("role_name") or "未命名人物"

    def section(label, text):
        if text:
            lines.extend([label + "：", "", text, ""])

    def perceptions(views, label):
        for person, view in views.items():
            section(f"{label}（{name(person)}）", view.content)
            for obj in view.objects:
                section(f"眼中的对象（{name(person)}）", obj.description)

    def short_plan(person, plan, phase):
        if plan is None:
            return
        section(f"{phase}本人短计划（{name(person)}；打算不等于实现）", plan.purpose)
        for index, step in enumerate(plan.steps):
            status = "本人认为已完成" if step.status == "completed" else "本人拟做"
            section(f"计划步骤 {index + 1}（{status}）", step.description)
        status = {"active": "继续", "completed": "认为已完成", "blocked": "认为受阻", "abandoned": "放弃"}[plan.status]
        section(f"{phase}本人计划判断（{name(person)}）", status)
        if plan.status == "active":
            section(f"{phase}本人拟定下一步（{name(person)}）", plan.steps[plan.next_step].description)
        section(f"{phase}本人计划反馈（{name(person)}）", plan.feedback)

    def psychology(person, state, phase):
        section(f"{phase}当前打算（{name(person)}）", state.intention)
        section(f"{phase}情绪（{name(person)}）", state.emotion)
        for field, label in (("beliefs", "认识"), ("goals", "愿望"), ("plan", "打算")):
            for content in getattr(state, field):
                section(f"{phase}{label}（{name(person)}）", content)
        for need in state.needs.values():
            section(f"{phase}需求感受（{name(person)}）", need.description)
            section(f"{phase}主观紧迫感（{name(person)}）", need.urgency)
        short_plan(person, state.short_plan, phase)

    def personal(experiences, memories, states):
        perceptions(experiences, "个人体验")
        for memory in memories:
            section(f"主观记忆（{name(memory.character_id)}；{sources[memory.source]}）", memory.content)
        for person, state in states.items():
            psychology(person, state, "批后")

    key = lambda a: (a.position.node_order, a.position.group_order, a.position.turn)
    ordered = sorted(actions, key=lambda a: (*key(a), a.position.round))
    for _, batch in groupby(ordered, key=key):
        batch = list(batch)
        selected = [a for a in batch if event_id is None or a.position.event_id == event_id]
        shared = {a.advance.shared_id: a.advance for a in batch if a.advance and a.advance.result
                  and (event_id is None or event_id in a.advance.related_event_ids)}
        if not selected and not shared:
            continue
        lines.extend(["---", ""])
        for action in selected:
            material = action.story_material()
            perceptions(material.perceptions_before, "行动前所感")
            if material.proposal:
                for recall in material.recollections_before:
                    section(f"行动前的个人回想（{name(material.proposal.character_id)}；{sources[recall.source]}）",
                            recall.content)
                if material.working_state_before is not None:
                    psychology(material.proposal.character_id, material.working_state_before, "行动前")
                if material.proposal.short_plan != getattr(material.working_state_before, "short_plan", None):
                    short_plan(material.proposal.character_id, material.proposal.short_plan, "本次选定")
                section(f"本人选择理由（{name(material.proposal.character_id)}）", material.proposal.selection_reason)
                section(f"本人目的（{name(material.proposal.character_id)}）", material.proposal.purpose)
                section(f"预期停点（{name(material.proposal.character_id)}；尚不代表实际到达）", material.proposal.stop_before)
                section(f"本人意图（{name(material.proposal.character_id)}；尚不代表实现）",
                        material.proposal.content)
            label = "同时发生的实际经过" if len(selected) > 1 else "实际经过"
            section(label, material.result)
            personal(material.experiences, material.memories, material.working_states)
        for advance in shared.values():
            section("共同时间经过", advance.result)
            personal({p: v.personal for p, v in advance.observations.items()},
                     advance.memories, advance.working_states)
    return "\n".join(lines).strip()
