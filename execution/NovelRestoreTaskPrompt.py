"""按完整业务任务选择提示段；不根据内部字段生成模型问题。"""
from pathlib import Path
import re

from common.Runtime import RunStopped


def load_prompt():
    return (Path(__file__).resolve().parents[1] / "prompts/NovelRestoreTaskPrompt.md").read_text(encoding="utf-8")


def task_instructions(prompt, context):
    """两个后端共用任务规则；短附加说明不能替换方法约束。"""
    rules = task_prompt(prompt if "<!-- tasks:" in prompt else load_prompt(), context)
    return rules + ("\n" + prompt if prompt and "<!-- tasks:" not in prompt else "")


def task_prompt(prompt, context):
    if "<!-- tasks:" not in prompt:
        return prompt
    operation = context.get("operation", "action")
    selected, matched = [], False
    sections = {operation}
    if operation == "schedule" and context.get("replan"):
        sections.add("review_schedule")
    if operation == "review_semantics":
        sections.add("review_schedule" if context.get("reviewed_operation") == "schedule" else "review_assessment")
    if operation == "review_visibility":
        reviewed = context.get("evidence", {}).get("operation")
        sections.add("review_visibility." + str(reviewed))
        if context.get("environment_action_ids"):
            sections.add("review_visibility.environment")
        if "approved_boundaries" in context.get("private_evidence", {}):
            sections.add("review_visibility.boundaries")
    if context.get("repair_personal_only"):
        # Repair already has a fixed world candidate. Do not also instruct the
        # model to perform the original full perception/resolution generation.
        sections = {"repair_personal"}
    parts = re.split(r"(?m)^<!-- tasks: ([^>]+) -->\n", prompt)
    for names, body in zip(parts[1::2], parts[2::2]):
        tasks = set(names.split(","))
        matched |= operation in tasks
        if "*" in tasks or sections.intersection(tasks):
            selected.append(body.strip())
    if not matched:
        raise RunStopped("任务没有对应的提示段：" + str(operation))
    return "\n\n".join(selected)
