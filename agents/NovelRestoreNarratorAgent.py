"""叙述角色的一次规划、范围选择或联合正文交付。"""
from pathlib import Path

from pydantic import Field

from agents.NovelRestoreAgent import task_id
from common.NovelRestoreSchema import (NarrativeBatch, NarrativePlan, NarrativeSelection,
    NarrativeOutline, ChapterLayout, ChapterPlan, NarrativeConnections, MemoryApproval, Identity, Record)
from common.Runtime import JobFailed, RunStopped
from storage.NovelRestoreStorage import encode
from common.NovelRestoreContext import link_components, narrative_index, simulation_source


class TimelineChoice(Record):
    branch: Identity
    reason: Identity


class TimelineReading(Record):
    assessment: str = Field(min_length=1, max_length=800)
    source_ids: list[Identity] = Field(min_length=1, max_length=6)


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


class NovelRestoreNarratorAgent:
    def __init__(self, storage, version_id, executor):
        self.storage, self.version_id, self.executor = storage, version_id, executor
        directory = Path(__file__).resolve().parents[1] / "prompts"
        self.plan_prompt = (directory / "NovelRestoreNarrationPlanPrompt.md").read_text(encoding="utf-8")
        self.write_prompt = (directory / "NovelRestoreNarrationWritePrompt.md").read_text(encoding="utf-8")
        self.selection_prompt = (directory / "NovelRestoreTimelineSelectionPrompt.md").read_text(encoding="utf-8")
        self.review_prompt = (directory / "NovelRestoreNarrativeReviewPrompt.md").read_text(encoding="utf-8")

    async def choose_timeline(self, candidates, config):
        base = {"requirements": config.requirements, "version": config.version.model_dump(mode="json"),
                "run_id": self.storage.run_id}
        context = {"operation": "choose_timeline", **base, "candidates": candidates}
        if len(encode(context)) > config.context_chars:
            batches = {branch: list(timeline_batches(branch, source, base, config.context_chars))
                       for branch, source in candidates.items()}
            readings = {}
            for branch, parts in batches.items():
                notes = []
                for index, part in enumerate(parts, 1):
                    source_ids = {r["source_id"] for r in part["records"]}
                    def cited_here(note):
                        if len(set(note.source_ids)) != len(note.source_ids) or not set(note.source_ids) <= source_ids:
                            raise ValueError("评语必须引用本分支本批提供的不同素材编号")
                    note = await self.executor.run_task(task_id("narrator", "read_timeline", branch, index),
                        "narrator", self.selection_prompt, part, TimelineReading, cited_here)
                    notes.append({"part": index, "record_count": len(part["records"]), **note.model_dump(mode="json")})
                readings[branch] = {"run_id": candidates[branch]["run_id"], "readings": notes}
            context["candidates"] = readings
            if len(encode(context)) > config.context_chars:
                raise RunStopped("全部候选的完整分批评语超过择线上下文预算，请使用较大预算的新版本")

        def validate(choice):
            if choice.branch not in candidates:
                raise ValueError("须从实际提供的分支中选定一条时间轴")

        return await self.executor.run_task("narrator/choose_timeline", "narrator", self.selection_prompt,
                                            context, TimelineChoice, validate)

    async def plan(self, source, assignment):
        state = self.storage.narrative(self.version_id)
        drafts = {d["draft_id"]: d for d in source["drafts"]}
        base = {"assignment": assignment, "requirements": state["config"]["requirements"],
                "version": state["config"]["version"]}
        overview = narrative_index(source)["drafts"]
        def validate_outline(result):
            ids = [c.chapter_id for c in result.chapters]
            if len(ids) != len(set(ids)) or {d for c in result.chapters for d in c.draft_ids} != set(drafts):
                raise ValueError("全书章节身份须唯一并覆盖全部草稿")
            if any(len(c.draft_ids) != len(set(c.draft_ids)) for c in result.chapters):
                raise ValueError("同章来源草稿索引不能重复；重述在片段层安排")
            target = base["version"]["length"]
            if target is not None and sum(c.length_budget for c in result.chapters) != target:
                raise ValueError("章节预算须等于版本目标篇幅")
        feedback = []
        for revision in range(state["config"]["semantic_repairs"] + 1):
            outline_context = {"operation": "outline", **base, "drafts": overview, "feedback": feedback}
            outline = await self.task(f"narrator/outline/{revision}", outline_context, NarrativeOutline, validate_outline)
            fragments, chapters = [], []
            for sketch in outline.chapters:
                selected = [drafts[i] for i in sketch.draft_ids]
                def validate_chapter(result):
                    ids = [f.fragment_id for f in result.fragments]
                    if len(ids) != len(set(ids)) or set(ids) & {f.fragment_id for f in fragments}:
                        raise ValueError("片段身份须在全书唯一")
                    if {f.draft_id for f in result.fragments} != set(sketch.draft_ids):
                        raise ValueError("章内片段须覆盖本章指定草稿")
                    for fragment in result.fragments:
                        self.storage.validate_fragment(fragment, drafts)
                    if sum(f.length_budget or 0 for f in result.fragments) != sketch.length_budget:
                        raise ValueError("章内片段篇幅须与章节预算一致")
                layout = await self.task(f"narrator/chapter/{revision}/{sketch.chapter_id}",
                    {"operation": "chapter", **base, "outline": outline.model_dump(), "chapter": sketch.model_dump(),
                     **simulation_source(source, [e for d in selected for e in d["event_ids"]]),
                     "used_fragment_ids": [f.fragment_id for f in fragments], "feedback": feedback}, ChapterLayout, validate_chapter)
                fragments.extend(layout.fragments)
                chapters.append(ChapterPlan(chapter_id=sketch.chapter_id, title=sketch.title, purpose=sketch.purpose,
                    length_budget=sketch.length_budget, setting=sketch.setting, fragment_ids=[f.fragment_id for f in layout.fragments]))
            def bind(connections):
                return NarrativePlan(run_id=self.storage.run_id, version_id=self.version_id,
                    requirements=state["config"]["requirements"], fragments=fragments, chapters=chapters, links=connections.links)
            connections = await self.task(f"narrator/connections/{revision}", {"operation": "connections", **base,
                "chapters": [c.model_dump() for c in chapters], "fragments": [f.model_dump() for f in fragments], "feedback": feedback},
                NarrativeConnections, lambda c: self.storage.validate_narrative_plan(self.version_id, bind(c)))
            plan = bind(connections)
            check = await self.review_plan(f"world/review_plan/{revision}", base, source, plan,
                                           state["config"]["context_chars"])
            if check.approved:
                return plan
            feedback = check.issues
        raise JobFailed("章节片段计划未通过全部事实忠实、内容覆盖、阅读揭示和媒介核对")

    async def review_plan(self, job, base, source, plan, limit):
        base = {"operation": "review_plan", **base}
        layout = plan.model_dump(mode="json")
        full = {**base, **simulation_source(source, source["actions"]),
                "plan": layout}
        if len(encode({**full, "run_id": self.storage.run_id})) <= limit:
            return await self.task(job, full, MemoryApproval, role="world")

        sections = [(c["chapter_id"], set(c["fragment_ids"])) for c in layout["chapters"]]
        checks = [("chapter", chapter, ids) for chapter, ids in sections]
        links = {link["link_id"]: {"link": link} for link in layout["links"]}
        for n, ids in enumerate(link_components(links), 1):
            if not any(ids <= chapter_ids for _, chapter_ids in sections):
                checks.append(("link_chain", str(n), ids))
        reviews = []
        for scope, identity, ids in checks:
            fragments = [f for f in layout["fragments"] if f["fragment_id"] in ids]
            events = {f["event_id"] for f in fragments}
            section = {**layout, "fragments": fragments,
                       "chapters": [c for c in layout["chapters"] if ids.intersection(c["fragment_ids"])],
                       "links": [link for link in layout["links"] if {link["from_fragment"], link["to_fragment"]} <= ids]}
            context = {**base, "scope": scope, "subject": identity, "plan": section,
                       **simulation_source(source, events)}
            result = await self.task(job + "/" + task_id(scope, identity), context, MemoryApproval, role="world")
            if not result.approved:
                return result
            reviews.append({"scope": scope, "subject": identity, "fragment_ids": sorted(ids),
                            "review": result.model_dump(mode="json")})
        overview = {**base, "scope": "whole_plan", "plan": layout, "index": narrative_index(source),
                    "section_reviews": reviews}
        return await self.task(job + "/overview", overview, MemoryApproval, role="world")

    async def task(self, job, context, schema, validate=lambda result: None, *, role="narrator"):
        limit = self.storage.narrative(self.version_id)["config"]["context_chars"]
        if len(encode({**context, "run_id": self.storage.run_id})) > limit:
            raise RunStopped("叙述必要资料超出上下文预算；保留任务，请用更细章节或更大预算的新版本")
        return await self.executor.run_task(job, role, self.review_prompt if role == "world" else self.plan_prompt,
                                            context, schema, validate)

    async def select(self, pending, context):
        return await self.executor.run_task(pending["task_id"] + "/select", "narrator", self.write_prompt,
                                            {"operation": "select", **context}, NarrativeSelection,
                                            lambda result: self.storage.select_narrative_links(
                                                self.version_id, result, pending["required"]))

    async def write(self, pending, context, revision=0):
        selection = NarrativeSelection.model_validate(pending["selection"])
        job = pending["task_id"] + "/write" + ("/" + str(revision) if revision else "")
        return await self.executor.run_task(job, "narrator", self.write_prompt,
                                            {"operation": "write", **context}, NarrativeBatch,
                                            lambda result: self.storage.preview_narrative_batch(
                                                self.version_id, pending["task_id"], pending["epoch"], selection, result))
