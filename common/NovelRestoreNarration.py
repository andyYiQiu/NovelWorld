"""固定章节位置下的串行联合任务；正文和关系的真值由存储事务维护。"""
from agents.NovelRestoreNarratorAgent import NovelRestoreNarratorAgent
from agents.NovelRestoreWorldAgent import NovelRestoreWorldAgent
from common.NovelRestoreSchema import NarrativePlan, NarrativeSelection
from common.Runtime import JobFailed, RunStopped
from storage.NovelRestoreStorage import encode
from common.NovelRestoreContext import repository_reference, simulation_source, fragment_index, link_components


class NovelRestoreNarration:
    def __init__(self, storage, repository, version_id, executor):
        self.storage, self.version_id = storage, version_id
        self.source = storage.check_narrative_source(version_id)
        self.drafts = {row["draft_id"]: row for row in self.source["drafts"]}
        self.narrator = NovelRestoreNarratorAgent(storage, version_id, executor)
        self.world = NovelRestoreWorldAgent(storage, repository, executor)

    @staticmethod
    def neighbors(plan, identity, window):
        order = plan.reading_order
        index = order.index(identity)
        return order[max(0, index - window):index], order[index + 1:index + window + 1]

    def context(self, state, targets, links):
        plan = NarrativePlan.model_validate(state["plan"])
        fragments = {row.fragment_id: row.model_dump(mode="json") for row in plan.fragments}
        windows = {identity: self.neighbors(plan, identity, state["config"]["window"]) for identity in targets}
        loaded = set(targets)
        for before, after in windows.values():
            loaded.update(before + after)
        # 读取目标所在的完整关联链；读取范围不等于本次修改范围。
        chain = set().union(*(ids for ids in link_components(links) if ids.intersection(targets)))
        related = {key: row for key, row in links.items()
                   if row["link"]["from_fragment"] in chain}
        loaded.update(chain)
        ordered = [identity for identity in plan.reading_order if identity in loaded]
        source_ids = list(dict.fromkeys(fragments[identity]["draft_id"] for identity in ordered))
        event_ids = {self.drafts[identity]["event_id"] for identity in source_ids}
        action_ids = {a for i in ordered for a in fragments[i]["action_ids"]}
        context = {"requirements": state["config"]["requirements"], "version": state["config"]["version"],
                "assignment": state["assignment"],
                "targets": targets, "reading_order": plan.reading_order,
                "chapters": [chapter.model_dump(mode="json") for chapter in plan.chapters],
                "fragment_index": fragment_index(state),
                "windows": {key: {"before": value[0], "after": value[1]} for key, value in windows.items()},
                "fragments": [fragments[identity] for identity in ordered],
                **simulation_source(self.source, event_ids, action_ids),
                "repository": repository_reference(self.world.world, event_ids),
                "action_scope": "所载片段明确引用的行动明细；完整事件过程保留在草稿中",
                "prose": {identity: state["prose"][identity] for identity in ordered if identity in state["prose"]},
                "links": related, "issues": state["issues"], "plan_revision": plan.revision,
                "link_scope": "加工目标所在的完整关联链；核对整链一致性，只修改targets，不能只确认孤立的两端",
                "max_revisions": state["config"]["max_revisions"]}
        context["reader_before"] = {target: [
            {"fragment_id": i, "revealed": state.get("prose_meta", {}).get(i, {}).get("revealed", []),
             "planned_reveal": fragments[i]["reveal"], "written": i in state["prose"]}
            for i in plan.reading_order[:plan.reading_order.index(target)]] for target in targets}
        context["character_knowledge"] = {}
        for identity in ordered:
            fragment = fragments[identity]
            person = fragment.get("viewpoint_role_id")
            if not person:
                continue
            ids = [fragment["knowledge_action_id"]] if fragment.get("knowledge_action_id") else fragment["action_ids"]
            actions = [self.storage.get_action(i) for i in ids]
            # 同一批次整体生效；不得把批内 SQL 行序号当成人物获知先后。
            bases = [a.expected_version for a in actions]
            cutoff = max(self.storage.connection.execute(
                "SELECT MAX(version) FROM actions WHERE status='committed' AND json_extract(payload,'$.expected_version')=?",
                (base,)).fetchone()[0] for base in bases)
            context["character_knowledge"][identity] = [m.model_dump(mode="json") for m in self.storage.retrieve_memories(
                person, query=fragment["selected_content"], through_version=cutoff)]
        limit = state["config"]["context_chars"]
        if len(encode(context)) > limit:
            mandatory = set(targets) | chain
            full_drafts = {fragments[i]["draft_id"] for i in mandatory}
            context["drafts"] = [d if d["draft_id"] in full_drafts else
                {k: d[k] for k in ("draft_id", "event_id", "achieved_results")} for d in context["drafts"]]
            context["actions"] = simulation_source(self.source,
                {self.drafts[i]["event_id"] for i in full_drafts},
                {a for i in mandatory for a in fragments[i]["action_ids"]})["actions"]
            for identity in set(context["prose"]) - mandatory:
                meta = state.get("prose_meta", {}).get(identity, {})
                context["prose"][identity] = {"fragment_id": identity, "summary": meta.get("summary") or state["prose"][identity]["text"],
                                               "representation": "summary"}
            context["neighbor_representation"] = "邻近片段可为摘要；加工目标与完整关联链的来源和已有正文保留"
        if len(encode(context)) > limit:
            raise RunStopped("联合任务必要来源超出上下文预算，未静默截断；请用更细片段或更大预算的新版本")
        return context

    def next_pending(self, state):
        plan = NarrativePlan.model_validate(state["plan"])
        # 一个问题所需的片段一起修；无关问题保留到后续事务，避免整书联合重写。
        required = set(state["issues"][0]["fragment_ids"]) if state["issues"] else set()
        stale = [row["link"] for row in state["links"].values()
                 if not self.storage.link_current(row, state["prose"])]
        if not required and stale:
            required.update((stale[0]["from_fragment"], stale[0]["to_fragment"]))
        if not required:
            required = set(next(([identity] for identity in plan.reading_order if identity not in state["prose"]), []))
        if not required:
            return None
        targets = [identity for identity in plan.reading_order if identity in required]
        return {"task_id": "batch/" + str(state["sequence"]), "epoch": state["epoch"],
                "required": targets, "selection": None}

    async def run(self):
        state = self.storage.narrative(self.version_id)
        if state["stage"] in ("complete", "incomplete"):
            return state["stage"] == "complete"
        if state["assignment"] is None:
            assignment = await self.world.assign_narration(self.source, state["config"]["requirements"], state["config"]["version"])
            self.storage.update_narrative(self.version_id, assignment=assignment.model_dump(mode="json"))
        state = self.storage.narrative(self.version_id)
        if state["plan"] is None:
            self.storage.save_narrative_plan(self.version_id, await self.narrator.plan(
                self.source, state["assignment"], self.world.world))
        while True:
            state = self.storage.narrative(self.version_id)
            if state["stage"] == "complete":
                return True
            if state["issues"] and len(state["reviews"]) > state["config"]["max_review_rounds"]:
                self.storage.update_narrative(self.version_id, stage="incomplete", error="全书内容修订轮数已用尽")
                return False
            pending = state["pending"] or self.next_pending(state)
            if pending is None:
                review = await self.world.review_narrative(self.source, state)
                self.storage.record_narrative_review(self.version_id, state["epoch"], review)
                continue
            if state["pending"] is None:
                self.storage.update_narrative(self.version_id, pending=pending)
            if pending["selection"] is None:
                context = self.context(state, pending["required"], state["links"])
                selection = await self.narrator.select(pending, context)
                pending["selection"] = selection.model_dump(mode="json")
                self.storage.update_narrative(self.version_id, pending=pending)
            selection = NarrativeSelection.model_validate(pending["selection"])
            links = self.storage.select_narrative_links(self.version_id, selection, pending["required"])
            context = self.context(state, selection.fragment_ids, links)
            context["relationship_changes"] = selection.model_dump(mode="json")
            feedback = []
            for revision in range(state["config"]["semantic_repairs"] + 1):
                batch = await self.narrator.write(pending, {**context, "feedback": feedback}, revision)
                review = await self.world.review_batch(pending["task_id"], revision, context, batch)
                if review.approved:
                    break
                feedback = [issue.model_dump(mode="json") for issue in review.issues]
            else:
                raise JobFailed("联合正文尚未通过来源、视角、揭示与关联核对")
            self.storage.commit_narrative_batch(self.version_id, pending["task_id"], pending["epoch"], selection, batch, review)
