"""叙述角色的一次规划、范围选择或联合正文交付。"""
from execution.NovelRestoreTaskPrompt import load_prompt

from pydantic import Field

from common.Runtime import JobFailed, RunStopped, task_id, fingerprint, storage_encode as encode
from common.NovelRestoreSchema import (NarrativeBatch, NarrativePlan, NarrativeSelection,
    NarrativeOutline, ChapterLayout, ChapterPlan, NarrativeConnections, Identity, SemanticPayload)
from narration.NovelRestoreContext import narrative_index, simulation_source, timeline_batches


class TimelineChoice(SemanticPayload):
    branch: Identity
    reason: Identity


class TimelineReading(SemanticPayload):
    assessment: str = Field(min_length=1, max_length=800)
    source_ids: list[Identity] = Field(min_length=1, max_length=6)


class NovelRestoreNarratorAgent:
    def __init__(self, storage, version_id, supervisor):
        self.storage, self.version_id, self.supervisor = storage, version_id, supervisor
        self.prompt = load_prompt()

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
                    note = await self.supervisor.run_task(task_id("narrator", "read_timeline", branch, index),
                        "narrator", self.prompt, part, TimelineReading, cited_here)
                    notes.append({"part": index, "record_count": len(part["records"]), **note.model_dump(mode="json")})
                readings[branch] = {"run_id": candidates[branch]["run_id"], "readings": notes}
            context["candidates"] = readings
            if len(encode(context)) > config.context_chars:
                raise RunStopped("全部候选的完整分批评语超过择线上下文预算，请使用较大预算的新版本")

        def validate(choice):
            if choice.branch not in candidates:
                raise ValueError("须从实际提供的分支中选定一条时间轴")

        return await self.supervisor.run_task("narrator/choose_timeline", "narrator", self.prompt,
                                            context, TimelineChoice, validate)

    async def plan(self, source, assignment, review_plan):
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
        feedback, affected, scope = [], set(), "all"
        outline, layouts = None, {}
        for revision in range(self.supervisor.semantic_attempts()):
            recovery = self.supervisor.semantic_recovery(revision)
            if outline is None or scope == "all":
                outline_context = {"operation": "outline", **base, "drafts": overview, "feedback": feedback}
                outline = await self.task(f"narrator/outline/{revision}", outline_context, NarrativeOutline, validate_outline,
                                          recovery=recovery)
                layouts = {key: value for key, value in layouts.items()
                           if key in {chapter.chapter_id for chapter in outline.chapters}}
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
                materials = {"operation": "chapter", **base, "outline": outline.model_dump(), "chapter": sketch.model_dump(),
                             **simulation_source(source, [e for d in selected for e in d["event_ids"]])}
                signature = fingerprint(materials)
                prior = layouts.get(sketch.chapter_id)
                revise = scope == "all" or scope == "chapters" and sketch.chapter_id in affected
                if prior is not None and prior[0] == signature and not revise:
                    layout = prior[1]
                    validate_chapter(layout)
                else:
                    reserved = {f.fragment_id for identity, (_, previous) in layouts.items()
                                if identity != sketch.chapter_id for f in previous.fragments}
                    layout = await self.task(f"narrator/chapter/{revision}/{sketch.chapter_id}",
                        {**materials, "used_fragment_ids": sorted(reserved | {f.fragment_id for f in fragments}),
                         "feedback": feedback if revise else [],
                         **({"previous_layout": prior[1].model_dump()} if prior else {})}, ChapterLayout, validate_chapter,
                        recovery=recovery)
                    layouts[sketch.chapter_id] = (signature, layout)
                fragments.extend(layout.fragments)
                chapters.append(ChapterPlan(chapter_id=sketch.chapter_id, title=sketch.title, purpose=sketch.purpose,
                    length_budget=sketch.length_budget, setting=sketch.setting, fragment_ids=[f.fragment_id for f in layout.fragments]))
            missing = {(identity, event) for identity, draft in drafts.items() for event in draft["event_ids"]} - {
                (fragment.draft_id, fragment.event_id) for fragment in fragments}
            if missing:
                # 全书覆盖是确定性校验。把缺口交回既有有限规划返修，不让连接任务
                # 因先前章节漏选一个事件而永久失败，也不要求每章各自呈现全部事件。
                feedback = ["全书尚未安排这些实际来源事件：" + encode([
                    {"草稿": identity, "事件": event} for identity, event in sorted(missing)])
                    + "。请在全书范围补足呈现安排；可以跨章拆分或重述，不必每章各自覆盖全部事件。"]
                scope = "chapters"
                affected = {c.chapter_id for c in outline.chapters
                            if set(c.draft_ids) & {draft for draft, _ in missing}}
                continue
            def bind(connections):
                return NarrativePlan(run_id=self.storage.run_id, version_id=self.version_id,
                    requirements=state["config"]["requirements"], fragments=fragments, chapters=chapters, links=connections.links)
            connections = await self.task(f"narrator/connections/{revision}", {"operation": "connections", **base,
                "chapters": [c.model_dump() for c in chapters], "fragments": [f.model_dump() for f in fragments], "feedback": feedback},
                NarrativeConnections, lambda c: self.storage.validate_narrative_plan(self.version_id, bind(c)), recovery=recovery)
            plan = bind(connections)
            check = await review_plan(f"world/review_plan/{revision}", base, source, plan,
                                      state["config"]["context_chars"])
            if check.approved:
                return plan
            feedback = check.issues
            scope = getattr(check, "repair_scope", "all")
            affected = set(getattr(check, "chapter_ids", []))
        raise JobFailed("章节片段计划未通过全部事实忠实、内容覆盖、阅读揭示和媒介核对：" + "; ".join(feedback))

    async def task(self, job, context, schema, validate=lambda result: None, *, recovery=False):
        limit = self.storage.narrative(self.version_id)["config"]["context_chars"]
        if len(encode({**context, "run_id": self.storage.run_id})) > limit:
            raise RunStopped("叙述必要资料超出上下文预算；保留任务，请用更细章节或更大预算的新版本")
        return await self.supervisor.run_task(job, "narrator", self.prompt,
                                            context, schema, validate, recovery=recovery)

    async def select(self, pending, context):
        return await self.supervisor.run_task(pending["task_id"] + "/select", "narrator", self.prompt,
                                            {"operation": "select", **context}, NarrativeSelection,
                                            lambda result: self.storage.select_narrative_links(
                                                self.version_id, result, pending["required"]))

    async def write(self, pending, context, revision=0, *, recovery=False):
        selection = NarrativeSelection.model_validate(pending["selection"])
        job = pending["task_id"] + "/write" + ("/" + str(revision) if revision else "")
        return await self.supervisor.run_task(job, "narrator", self.prompt,
                                            {"operation": "write", **context}, NarrativeBatch,
                                            lambda result: self.storage.preview_narrative_batch(
                                                self.version_id, pending["task_id"], pending["epoch"], selection, result,
                                                allow_unconfirmed=True), recovery=recovery)
