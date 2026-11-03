"""叙述角色的一次规划、范围选择或联合正文交付。"""
from pathlib import Path

from common.NovelRestoreSchema import (NarrativeBatch, NarrativePlan, NarrativeSelection,
    NarrativeOutline, ChapterLayout, ChapterPlan, NarrativeConnections, MemoryApproval)
from common.Runtime import JobFailed, RunStopped
from storage.NovelRestoreStorage import encode
from common.NovelRestoreContext import repository_reference, simulation_source


class NovelRestoreNarratorAgent:
    def __init__(self, storage, version_id, executor):
        self.storage, self.version_id, self.executor = storage, version_id, executor
        directory = Path(__file__).resolve().parents[1] / "prompts"
        self.plan_prompt = (directory / "NovelRestoreNarrationPlanPrompt.md").read_text(encoding="utf-8")
        self.write_prompt = (directory / "NovelRestoreNarrationWritePrompt.md").read_text(encoding="utf-8")
        self.review_prompt = (directory / "NovelRestoreWorldPrompt.md").read_text(encoding="utf-8")

    async def plan(self, source, assignment, repository):
        state = self.storage.narrative(self.version_id)
        drafts = {d["draft_id"]: d for d in source["drafts"]}
        base = {"assignment": assignment, "requirements": state["config"]["requirements"],
                "version": state["config"]["version"]}
        overview = [{k: d[k] for k in ("draft_id", "event_id", "node_order", "group_order", "achieved_results")}
                    for d in drafts.values()]
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
            outline = await self.task(f"narrator/outline/{revision}", {"operation": "outline", **base,
                "drafts": overview, "repository": repository_reference(repository),
                "feedback": feedback}, NarrativeOutline, validate_outline)
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
                     **simulation_source(source, [d["event_id"] for d in selected]),
                     "repository": repository_reference(repository, [d["event_id"] for d in selected]),
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
            check = await self.task(f"world/review_plan/{revision}", {"operation": "review_plan", **base,
                **simulation_source(source, source["actions"]), "repository": repository_reference(repository),
                "plan": plan.model_dump(mode="json")}, MemoryApproval, role="world")
            if check.approved:
                return plan
            feedback = check.issues
        raise JobFailed("章节片段计划未通过全部事实忠实、内容覆盖、阅读揭示和媒介核对")

    async def task(self, job, context, schema, validate=lambda result: None, *, role="narrator"):
        limit = self.storage.narrative(self.version_id)["config"]["context_chars"]
        if len(encode(context)) > limit:
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
