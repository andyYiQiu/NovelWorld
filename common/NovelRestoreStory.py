"""已提交模拟就是故事材料；只做顺序与显示投影，不生成、概括或改写文字。"""
from itertools import groupby


def story_text(actions, characters, event_id=None):
    """同批行动不宣称先后；共享经过放在整批之后，相关事件各自引用。"""
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

    def personal(experiences, memories, states):
        perceptions(experiences, "个人体验")
        for memory in memories:
            section(f"主观记忆（{name(memory.character_id)}；{sources[memory.source]}）", memory.content)
        for person, state in states.items():
            section(f"情绪（{name(person)}）", state.emotion)
            for field, label in (("beliefs", "认识"), ("goals", "愿望"), ("plan", "打算")):
                for content in getattr(state, field):
                    section(f"{label}（{name(person)}）", content)
            for need in state.needs.values():
                section(f"需求感受（{name(person)}）", need.description)

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
