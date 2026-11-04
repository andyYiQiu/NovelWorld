"""写作委托和世界复核只接收冻结的故事草稿，没有查询模拟世界的入口。"""
from pathlib import Path

from agents.NovelRestoreAgent import task_id
from common.NovelRestoreContext import link_components, narrative_index, simulation_source
from common.NovelRestoreSchema import NarrativeAssignment, NarrativeBatchReview, NarrativeReview
from common.Runtime import RunStopped
from storage.NovelRestoreStorage import encode


class NovelRestoreNarrativeReview:
    def __init__(self, storage, executor):
        self.storage, self.executor = storage, executor
        self.prompt = (Path(__file__).resolve().parents[1] / "prompts/NovelRestoreNarrativeReviewPrompt.md").read_text()

    async def assign_narration(self, source, requirements, version=None):
        base = {"operation": "assign_narration", "requirements": requirements, "version": version}
        context = {**base, "source": simulation_source(source, source["actions"])}
        limit = self.executor.storage.config.context_chars
        if len(encode({**context, "run_id": self.storage.run_id})) > limit:
            context = {**base, "scope": "draft_index", "index": narrative_index(source)}
            if len(encode({**context, "run_id": self.storage.run_id})) > limit:
                raise RunStopped("完整委托索引超过叙述上下文预算，尚未派发写作任务")
        return await self.executor.run_task("world/assign_narration", "world", self.prompt, context, NarrativeAssignment)

    async def review_batch(self, identity, revision, context, batch):
        targets = set(context["targets"])
        direct = {i for i, row in context["links"].items()
                  if {row["link"]["from_fragment"], row["link"]["to_fragment"]}.intersection(targets)}
        def validate(result):
            if any(set(issue.fragment_ids) - targets for issue in result.issues):
                raise ValueError("本批拒收问题须定位本批加工片段；后续影响使用 affected_fragment_ids")
            if set(result.preserved_link_ids) - direct:
                raise ValueError("只能保留本批目标的直接关联，不能顺带确认链上其他关系")
            if set(result.affected_fragment_ids) - (set(context["reading_order"]) - targets):
                raise ValueError("影响传播引用未知片段或当前目标")
        return await self.executor.run_task(identity + "/review/" + str(revision), "world", self.prompt,
            {"operation": "review_batch", "context": context, "batch": batch.model_dump(mode="json")},
            NarrativeBatchReview, validate)

    async def review_narrative(self, source, state):
        def validate(review):
            if any(set(issue.fragment_ids) - set(state["prose"]) for issue in review.issues):
                raise ValueError("核对问题必须定位到当前版本的实际片段")
        base = {"operation": "review_narrative", "requirements": state["config"]["requirements"],
                "assignment": state["assignment"], "version": state["config"]["version"],
                "plan": {key: value for key, value in state["plan"].items() if key != "links"},
                "links": state["links"], "prose_meta": state["prose_meta"]}
        job = "world/review_narrative/" + str(len(state["reviews"]) + 1)
        limit = state["config"]["context_chars"]
        async def check(key, context):
            if len(encode(context)) > limit:
                raise RunStopped("内容复核必要资料超出上下文预算；未截断或跳过核对")
            return await self.executor.run_task(key, "world", self.prompt, context, NarrativeReview, validate)
        full = {**base, "source": source, "prose": state["prose"]}
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
