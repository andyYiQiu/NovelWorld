"""Bind fixed review coverage and personal repair destinations before model delivery."""
from copy import deepcopy

from pydantic import Field, create_model

from common.NovelRestoreSchema import Identity, Record
from common.Runtime import encode
from simulation.NovelRestoreVisibility import (
    VisibilityReview, PersonalVisibilityReview, EnvironmentVisibilityReview, PersonalPerceptionRepair,
    CharacterFeedback,
)


class PersonalText(Record):
    content: Identity
    object_descriptions: list[Identity] = Field(default_factory=list,
        description="按该目标原 objects 的顺序给出描述；对象身份及数量由程序固定，不新增或删除。")


class ReviewIssueText(Record):
    path: Identity
    reason: Identity


class PersonalReview(Record):
    issues: list[ReviewIssueText] = Field(
        description="完整审读此人的全部候选；没有问题时为空，缺少知情依据时须说明。")


def repair_layout(context):
    """Select all existing deliveries of the affected people, never parse model paths."""
    original = context["unsafe_delivery"]
    affected = {issue["character_id"] for issue in context["visibility_feedback"]}
    groups = [("views", None, original.get("views", {}))]
    groups += [("actions", action["action_id"], action["observations"]) for action in original.get("actions", [])]
    groups += [("advance", None, original.get("advance", {}).get("observations", {})),
               ("final_views", None, original.get("final_views", {}))]
    repair = {"views": {}, "actions": {}, "advance": {}, "final_views": {}}
    targets = []
    for section, action, views in groups:
        destination = repair[section].setdefault(action, {}) if section == "actions" else repair[section]
        for person in sorted(views):
            destination[person] = deepcopy(views[person]["personal"])
            if person in affected:
                targets.append({"section": section, "action_id": action, "character_id": person})
    if not targets or affected - {target["character_id"] for target in targets}:
        raise ValueError("个人返修意见没有对应的实际交付")
    return repair, targets


def visibility_contract(schema, context):
    if schema is CharacterFeedback:
        people = sorted(context["subjective_inputs"])
        payload_type = create_model("CharacterFeedbackInput", __base__=Record,
            messages=(list[list[Identity]], Field(min_length=len(people), max_length=len(people),
                description="按此顺序分别给出本人可知的返修提示，接收者由程序绑定：" + encode(people))))
        def assemble(payload):
            value = payload_type.model_validate(payload)
            return schema(messages=dict(zip(people, value.messages)))
        return payload_type, assemble

    if schema is PersonalPerceptionRepair:
        original, targets = repair_layout(context)
        payload_type = create_model("PersonalTextRepairInput", __base__=Record,
            texts=(list[PersonalText], Field(min_length=len(targets), max_length=len(targets),
                description="按 repair_targets 顺序一次修正完整个人体验；不交付接收者、行动编号或替换路径。")))

        def assemble(payload):
            value = payload_type.model_validate(payload)
            result = deepcopy(original)
            for index, (target, text) in enumerate(zip(targets, value.texts)):
                group = result[target["section"]]
                if target["section"] == "actions":
                    group = group[target["action_id"]]
                personal = group[target["character_id"]]
                if len(text.object_descriptions) != len(personal["objects"]):
                    raise ValueError(f"texts[{index}].object_descriptions 须有 {len(personal['objects'])} 项，按目标原 objects 顺序描述")
                personal["content"] = text.content
                for obj, description in zip(personal["objects"], text.object_descriptions):
                    obj["description"] = description
            return schema.model_validate(result)
        return payload_type, assemble

    if schema in (VisibilityReview, PersonalVisibilityReview, EnvironmentVisibilityReview):
        people = sorted(context["candidates"])
        fields = {"checks": (list[PersonalReview], Field(min_length=len(people), max_length=len(people),
            description="按此顺序逐人完整审读 candidates：" + encode(people)))}
        if schema is not PersonalVisibilityReview:
            fields["world_issues"] = (list[Identity], Field(description="完整核对客观后果后的问题；无问题显式给 []。"))
        environment = sorted(context.get("environment_action_ids", []))
        if schema is EnvironmentVisibilityReview:
            fields["environment_reasons"] = (list[Identity], Field(min_length=len(environment), max_length=len(environment),
                description="按此顺序说明每项环境尝试的现场依据、实际后果及反应边界；有问题同时列入 world_issues：" + encode(environment)))
        payload_type = create_model(schema.__name__ + "Input", __base__=Record, **fields)

        def assemble(payload):
            value = payload_type.model_validate(payload)
            issues = [{"character_id": person, **issue.model_dump()}
                      for person, check in zip(people, value.checks) for issue in check.issues]
            world_issues = getattr(value, "world_issues", [])
            result = {"approved": not (issues or world_issues), "covered_characters": people,
                      "issues": issues, "world_issues": world_issues}
            if schema is EnvironmentVisibilityReview:
                result["environment_checks"] = dict(zip(environment, value.environment_reasons))
            return schema.model_validate(result)
        return payload_type, assemble
    return None
