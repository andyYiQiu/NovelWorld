"""世界角色在叙述阶段的委托与复核组件；仅接收冻结草稿，不是独立角色。"""
from execution.NovelRestoreTaskPrompt import load_prompt

from common.Runtime import RunStopped, task_id, fingerprint, storage_encode as encode
from narration.NovelRestoreContext import link_components, narrative_index, simulation_source
from common.NovelRestoreSchema import NarrativeAssignment, NarrativeBatchReview, NarrativeReview, NarrativePlanReview


class NovelRestoreWorldReview:
    def __init__(self, storage, supervisor):
        self.storage, self.supervisor = storage, supervisor
        self.prompt = load_prompt()
        self.review_jobs = {}

    async def check_source(self, job, context, schema, validate):
        # Like reviewed_task: reuse only the exact same input within this
        # revision process. run_task still validates the persisted receipt.
        key = self.review_jobs.setdefault(fingerprint(context), job)
        return await self.supervisor.run_task(key, "world", self.prompt, context, schema, validate)

    async def assign_narration(self, source, requirements, version=None):
        base = {"operation": "assign_narration", "requirements": requirements, "version": version}
        context = {**base, "source": simulation_source(source, source["actions"])}
        limit = self.supervisor.storage.config.context_chars
        if len(encode({**context, "run_id": self.storage.run_id})) > limit:
            context = {**base, "scope": "draft_index", "index": narrative_index(source)}
            if len(encode({**context, "run_id": self.storage.run_id})) > limit:
                raise RunStopped("完整委托索引超过叙述上下文预算，尚未派发写作任务")
        return await self.supervisor.run_task("world/assign_narration", "world", self.prompt, context, NarrativeAssignment)

    async def review_batch(self, identity, revision, context, batch):
        def validate(result):
            written = set(context.get("prose", {})) | {text.fragment_id for text in batch.texts}
            result.validate_context(links=context["links"], targets=context["targets"],
                                    written=written, reading_order=context["reading_order"])
        return await self.check_source(identity + "/review/" + str(revision),
            {"operation": "review_batch", "context": context, "batch": batch.model_dump(mode="json")},
            NarrativeBatchReview, validate)

    async def review_narrative(self, source, state):
        def validate(review):
            if any(set(issue.fragment_ids) - set(state["prose"]) for issue in review.issues):
                raise ValueError("核对问题必须定位到当前版本的实际片段")
        base = {"operation": "review_narrative", "run_id": self.storage.run_id, "requirements": state["config"]["requirements"],
                "assignment": state["assignment"], "version": state["config"]["version"],
                "plan": {key: value for key, value in state["plan"].items() if key != "links"},
                "links": state["links"], "prose_meta": state["prose_meta"]}
        job = "world/review_narrative/" + str(len(state["reviews"]) + 1)
        limit = state["config"]["context_chars"]
        async def check(key, context):
            if len(encode(context)) > limit:
                raise RunStopped("内容复核必要资料超出上下文预算；未截断或跳过核对")
            return await self.check_source(key, context, NarrativeReview, validate)
        # 和分章审核使用同一来源投影；完整结构材料已含全部经过，
        # 不再重复发送由它们渲染的 draft.text，也不改变冻结库。
        full = {**base, "source": {"run_id": source["run_id"],
                                  **simulation_source(source, source["actions"])}, "prose": state["prose"]}
        if len(encode(full)) <= limit:
            return await check(job, full)
        # 长篇逐章审读；跨章的多段关联另读整链正文，摘要不能代替整链核对。
        async def check_part(key, scope, ids, chapter=None):
            fragments = [f for f in state["plan"]["fragments"] if f["fragment_id"] in ids]
            events = {f["event_id"] for f in fragments}
            return await check(key, {
                **base, "scope": scope, "chapter": chapter,
                "plan": {"chapters": [c for c in state["plan"]["chapters"] if ids.intersection(c["fragment_ids"])],
                         "fragments": fragments},
                "prose_meta": {i: value for i, value in state["prose_meta"].items() if i in ids},
                "prose": {i: value for i, value in state["prose"].items() if i in ids},
                "links": {i: row for i, row in state["links"].items()
                          if {row["link"]["from_fragment"], row["link"]["to_fragment"]}.intersection(ids)},
                "source": simulation_source(source, events)})
        checks = []
        chapters = [set(c["fragment_ids"]) for c in state["plan"]["chapters"]]
        for chapter, ids in zip(state["plan"]["chapters"], chapters):
            checks.append(await check_part(job + "/chapter/" + task_id(chapter["chapter_id"]), "chapter", ids, chapter))
        for index, ids in enumerate(link_components(state["links"]), 1):
            # 单条关系已联合核对，同章完整链已在上一步读取，不重复调用。
            if len(ids) > 2 and not any(ids <= chapter for chapter in chapters):
                checks.append(await check_part(job + "/chain/" + str(index), "link_chain", ids))
        checks.append(await check(job + "/overview", {**base, "scope": "whole_book_summary",
            "source": narrative_index(source),
            "section_reviews": [r.model_dump(mode="json") for r in checks]}))
        issues = list({encode(i.model_dump()): i for r in checks for i in r.issues}.values())
        return NarrativeReview(approved=all(r.approved for r in checks), summary="逐章、跨章关联链正文及全书摘要核对", issues=issues)

    async def review_plan(self, job, base, source, plan, limit):
        def validate(review):
            if not set(review.chapter_ids) <= {c.chapter_id for c in plan.chapters}:
                raise ValueError("返修只能定位当前计划实际存在的章节")

        async def check(key, context):
            if len(encode({**context, "run_id": self.storage.run_id})) > limit:
                raise RunStopped("叙述必要资料超出上下文预算；保留任务，请用更细章节或更大预算的新版本")
            return await self.check_source(key, context, NarrativePlanReview, validate)
        base = {"operation": "review_plan", **base}
        layout = plan.model_dump(mode="json")
        full = {**base, **simulation_source(source, source["actions"]),
                "plan": layout}
        if len(encode({**full, "run_id": self.storage.run_id})) <= limit:
            return await check(job, full)

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
            result = await check(job + "/" + task_id(scope, identity), context)
            if not result.approved:
                return result
            reviews.append({"scope": scope, "subject": identity, "fragment_ids": sorted(ids),
                            "review": result.model_dump(mode="json")})
        overview = {**base, "scope": "whole_plan", "plan": layout, "index": narrative_index(source),
                    "section_reviews": reviews}
        return await check(job + "/overview", overview)

