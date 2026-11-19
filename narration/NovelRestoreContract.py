"""冻结素材的表达交付；篇幅与记录身份由程序装配。"""
from typing import Annotated, Literal

from pydantic import Field, create_model

from common.Runtime import RunStopped, task_id
from tools.ToolArguments import payload_data
from common.NovelRestoreSchema import (Identity, PositiveInt, Record, NarrativeLink, NarrativeSelection)


Weight = Annotated[int, Field(ge=1, le=10, strict=True)]


class ChapterContent(Record):
    title: Identity
    purpose: Identity
    draft_ids: list[Identity] = Field(min_length=1,
        description="引用本次 drafts 的 draft_id；章内不重复，各章合计须覆盖全部提供的冻结草稿。")
    setting: str = ""
    weight: Weight = Field(default=1, description="本章相对其他章节需要的表达空间；精确篇幅由程序分配。")


class FragmentContent(Record):
    draft_id: Identity = Field(description="本章 drafts 中的一份冻结草稿编号；所有片段须合计覆盖本章全部草稿。")
    event_id: Identity = Field(description="必须属于所选草稿的 event_ids。")
    selected_content: Identity
    action_ids: list[Identity] = Field(min_length=1,
        description="只选本次 actions[event_id] 提供的不同 action_id，不能借用其他事件或虚构行动。")
    viewpoint: Identity = Field(pattern=r"\S", description="实际采用的叙述视角；角色视角的知情范围仍须独立审核。")
    purpose: Identity = Field(pattern=r"\S", description="此片段在全篇的具体叙述目的。")
    reveal: list[str] = Field(default_factory=list)
    withhold: list[str] = Field(default_factory=list)
    knowledge_basis: list[Identity] = Field(default_factory=list)
    viewpoint_role_id: Identity | None = None
    knowledge_action_id: Identity | None = None
    weight: Weight = Field(default=1, description="相对章内其他片段需要的表达空间；程序精确分配本章预算。")


class LinkContent(Record):
    kind: Literal["foreshadow", "suspense", "echo", "retelling"]
    from_fragment: Identity = Field(description="本次 fragments 中的起点 fragment_id；伏笔起点须在 reading_order 中先于终点。")
    to_fragment: Identity = Field(description="本次 fragments 中不同于起点的终点 fragment_id。")
    content: Identity
    acceptance: list[Identity] = Field(min_length=1, description="关联在正文中成立或回收的具体检查条件，不得为空。")


class LinkDelivery(Record):
    links: list[LinkContent] = Field(default_factory=list)


class FragmentTextInput(Record):
    text: Identity = Field(pattern=r"\S")
    revealed: list[Identity] = Field(default_factory=list)
    change: Literal["initial", "wording", "meaning"] = "meaning"
    summary: Identity = Field(pattern=r"\S", description="本次正文的实际内容摘要，供后续检索和全书复核；不是正文替代品。")


class NarrativeLinkInput(NarrativeLink):
    acceptance: list[Identity] = Field(min_length=1,
        description="关联在正文中成立或回收的具体检查条件，不得为空。")
    link_id: Identity = Field(description="更新时沿用本次 links 中的已有 link_id；新增时提供任务内唯一局部别名，正式编号由程序绑定。")


class NarrativeSelectionInput(NarrativeSelection):
    upsert_links: list[NarrativeLinkInput] = Field(default_factory=list,
        description="关联两端必须同时列入 fragment_ids；修改已有关系还须包含原有两端。每个编号唯一，且不能同时删除。")


def distribute(total, weights):
    """Retain the prior deterministic largest-remainder budget calculation."""
    if type(total) is not int or total <= 0 or not weights or any(
            type(weight) is not int or weight <= 0 for weight in weights):
        raise RunStopped("篇幅预算和相对权重必须为正整数")
    if total < len(weights):
        raise ValueError(f"总篇幅预算 {total} 无法分配给 {len(weights)} 个内容单元；请合并章节或片段，并保留全部来源内容")
    rest, denominator = total - len(weights), sum(weights)
    budgets = [1 + rest * weight // denominator for weight in weights]
    order = sorted(range(len(weights)), key=lambda i: (-(rest * weights[i] % denominator), i))
    for index in order[:total - sum(budgets)]:
        budgets[index] += 1
    return budgets

def outline_contract(schema, context, owner, job):
    drafts = {row["draft_id"] for row in context["drafts"]}
    total = context["version"].get("length")
    fields = {"chapters": (list[ChapterContent], Field(min_length=1, max_length=total))}
    if total is None:
        fields["proposed_length"] = (PositiveInt, Field(description="完整呈现冻结素材所需的总篇幅，不为凑长度添删事实。"))
    payload_type = create_model("NarrativeOutlineInput", __base__=Record, **fields)

    def assemble(payload):
        value = payload_type.model_validate(payload_data(payload))
        covered = {identity for row in value.chapters for identity in row.draft_ids}
        if covered != drafts or any(len(row.draft_ids) != len(set(row.draft_ids)) for row in value.chapters):
            raise ValueError("章节须完整覆盖实际冻结草稿，每章来源不得重复")
        budgets = distribute(total if total is not None else value.proposed_length,
                             [row.weight for row in value.chapters])
        return schema(chapters=[{**row.model_dump(exclude={"weight"}),
            "chapter_id": task_id("chapter", job, index), "length_budget": budget}
            for index, (row, budget) in enumerate(zip(value.chapters, budgets), 1)])
    return payload_type, assemble

def layout_contract(schema, context, owner, job):
    drafts = {row["draft_id"]: row for row in context["drafts"]}
    total = context["chapter"]["length_budget"]
    payload_type = create_model("ChapterLayoutInput", __base__=Record,
        fragments=(list[FragmentContent], Field(min_length=1, max_length=total)))

    def assemble(payload):
        value = payload_type.model_validate(payload_data(payload))
        if {row.draft_id for row in value.fragments} != set(drafts):
            raise ValueError("章内片段须覆盖指定的全部冻结草稿")
        for row in value.fragments:
            if row.event_id not in drafts[row.draft_id]["event_ids"]:
                raise ValueError("片段事件必须属于所选冻结草稿")
            actions = {action["action_id"] for action in context["actions"].get(row.event_id, [])}
            if not set(row.action_ids) <= actions or len(row.action_ids) != len(set(row.action_ids)):
                raise ValueError("片段只能选取该事件中实际提供的不同动作")
        budgets = distribute(total, [row.weight for row in value.fragments])
        result = schema(fragments=[{**row.model_dump(exclude={"weight"}),
            "fragment_id": task_id("fragment", job, index), "length_budget": budget}
            for index, (row, budget) in enumerate(zip(value.fragments, budgets), 1)])
        if {row.fragment_id for row in result.fragments} & set(context.get("used_fragment_ids", [])):
            raise ValueError("新片段身份与此前章节重复")
        return result
    return payload_type, assemble

def connections_contract(schema, context, owner, job):
    identities = [row["fragment_id"] for row in context["fragments"]]
    order = context.get("reading_order", identities)

    def assemble(payload):
        value = LinkDelivery.model_validate(payload_data(payload))
        for row in value.links:
            if row.from_fragment not in identities or row.to_fragment not in identities or row.from_fragment == row.to_fragment:
                raise ValueError("叙述关联必须绑定两个实际存在的不同片段")
            if row.kind == "foreshadow" and order.index(row.from_fragment) >= order.index(row.to_fragment):
                raise ValueError("伏笔必须在其回收片段之前")
        return schema(links=[{**row.model_dump(), "link_id": task_id("link", job, index)}
                             for index, row in enumerate(value.links, 1)])
    return LinkDelivery, assemble

def batch_contract(schema, context, owner, job):
    targets = context["targets"]
    payload_type = create_model("NarrativeBatchInput", __base__=Record,
        texts=(list[FragmentTextInput], Field(min_length=len(targets), max_length=len(targets),
            description="按 targets 顺序一次交付各片段完整正文；片段编号由程序绑定。")),
        confirmed_link_ids=(list[Identity], Field(default_factory=list)))
    def assemble(payload):
        value = payload_type.model_validate(payload_data(payload))
        return schema(texts=[{"fragment_id": identity, **text.model_dump()}
                             for identity, text in zip(targets, value.texts)],
                      confirmed_link_ids=value.confirmed_link_ids)
    return payload_type, assemble

def selection_contract(schema, context, owner, job):
    def assemble(payload):
        value = NarrativeSelectionInput.model_validate(payload_data(payload))
        existing = set(context.get("links", {}))
        for removal in value.remove_links:
            if removal.link_id not in existing:
                raise ValueError("删除必须引用本次 links 中的已有关系")
        for index, link in enumerate(value.upsert_links):
            if link.link_id not in existing:
                link.link_id = task_id("link", job, index)
                if link.link_id in existing:
                    raise ValueError("新关联编号与已存关联冲突")
        return schema.model_validate(value.model_dump())
    return NarrativeSelectionInput, assemble

