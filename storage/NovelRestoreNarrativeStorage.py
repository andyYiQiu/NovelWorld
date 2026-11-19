"""冻结草稿来源与叙述版本的校验、事务和预算作用域；复用同一运行库。"""
import json
from urllib.parse import quote

from common.Runtime import RunStopped, fingerprint, storage_encode
from common.NovelRestoreConfig import NovelRestoreNarrativeConfig
from common.NovelRestoreSchema import (
    NarrativeBatch, NarrativePlan, NarrativeReview, NarrativeSelection, ProseFragment,
)


class NarrativeLimitReached(RunStopped):
    """正文实质修订已达上限，保留未完成结果。"""


class NovelRestoreNarrativeStorage:
    def narrative_source(self):
        """仅使用库中已确认资料，不读取草稿导出文件或原小说。"""
        if self.connection.execute("SELECT json_extract(progress,'$.stage') FROM runtime WHERE singleton=1").fetchone()[0] != "simulation_complete":
            raise ValueError("全部模拟完成后才能开始叙述")
        if self.manifest["format"] < 18:
            raise ValueError("旧运行的草稿未归档当前协议的行动前回想，只能查看历史结果")
        drafts, actions, shared, characters = [], {}, {}, {}
        for node in self.manifest["nodes"]:
            for group in node["groups"]:
                draft = self.get_draft(group["group_id"])
                if (draft is None or draft.run_id != self.run_id or draft.node_id != node["node_id"]
                        or draft.group_id != group["group_id"] or draft.group_order != group["order"]
                        or draft.node_order != node["order"] or draft.event_ids != group["event_ids"]):
                    raise ValueError("叙述来源缺失或组草稿与冻结时间轴不一致")
                drafts.append(draft.model_dump(mode="json", exclude={"materials", "shared_materials"}))
                characters.update({p: {"role_name": name} for p, name in draft.participant_names.items()})
                for event in draft.event_ids:
                    actions[event] = []
                for material in draft.materials:
                    actions[material.position.event_id].append(material.model_dump(mode="json"))
                shared.update({m.shared_id: m.model_dump(mode="json") for m in draft.shared_materials})
        if not drafts:
            raise ValueError("叙述来源必须包含已验收的组草稿")
        return {"run_id": self.run_id, "characters": characters, "drafts": drafts,
                "actions": actions, "shared_materials": shared}

    def create_narrative(self, config, binding, *, selection=None):
        self._writable()
        config = NovelRestoreNarrativeConfig.model_validate(config.model_dump(mode="json"))
        if self.connection.execute("SELECT 1 FROM narratives WHERE key=?", (config.version_id,)).fetchone():
            raise ValueError("叙述版本已存在，请使用 resume 或选择新的版本身份")
        state = {"config": config.model_dump(mode="json"), "binding": binding,
                 "source_hash": fingerprint(self.narrative_source()),
                 "stage": "initialized", "error": None, "epoch": 0, "sequence": 1,
                 "assignment": None, "plan": None, "prose": {}, "links": {},
                 "pending": None, "reviews": [], "issues": [], "prose_meta": {}, "batch_reviews": []}
        if selection is not None:
            state["selection"] = selection
        with self.connection:
            self._put("narratives", config.version_id, state)

    def narrative(self, version_id):
        row = self.connection.execute("SELECT payload FROM narratives WHERE key=?", (version_id,)).fetchone()
        if row is None:
            raise ValueError("叙述版本不存在：" + version_id)
        return json.loads(row[0])

    def update_narrative(self, version_id, **updates):
        self._writable()
        if updates.keys() - {"stage", "error", "assignment", "pending"}:
            raise ValueError("正文与计划须通过专用事务更新")
        state = self.narrative(version_id)
        state.update(updates)
        with self.connection:
            self._put("narratives", version_id, state)

    def check_narrative_source(self, version_id):
        source = self.narrative_source()
        if fingerprint(source) != self.narrative(version_id)["source_hash"]:
            raise ValueError("叙述来源草稿或行动已变化，不能继续旧版本")
        return source

    def validate_narrative_plan(self, version_id, plan):
        plan = NarrativePlan.model_validate(plan.model_dump(mode="json"))
        config = self.narrative(version_id)["config"]
        if (plan.run_id, plan.version_id, plan.requirements, plan.revision) != (
                self.run_id, version_id, config["requirements"], 1):
            raise ValueError("叙述计划身份或版本要求不匹配")
        drafts = {row["draft_id"]: row for row in self.check_narrative_source(version_id)["drafts"]}
        if {(fragment.draft_id, fragment.event_id) for fragment in plan.fragments} != {
                (key, event) for key, draft in drafts.items() for event in draft["event_ids"]}:
            raise ValueError("叙述计划必须覆盖全部来源草稿及其事件")
        for fragment in plan.fragments:
            self.validate_fragment(fragment, drafts)
        spec = config["version"]
        if spec["length"] is not None and sum(f.length_budget for f in plan.fragments) != spec["length"]:
            raise ValueError("片段篇幅预算总和须等于版本目标长度")
        by_id = {f.fragment_id: f for f in plan.fragments}
        if any(c.length_budget != sum(by_id[i].length_budget for i in c.fragment_ids) for c in plan.chapters):
            raise ValueError("每章篇幅预算须等于所含片段预算之和")
        if any(not link.acceptance for link in plan.links):
            raise ValueError("关联须明确正文中的成立和回收条件")
        return plan

    def validate_fragment(self, fragment, drafts):
        draft = drafts.get(fragment.draft_id)
        if not draft or fragment.event_id not in draft["event_ids"] or (
                set(fragment.action_ids) - set(draft["event_action_ids"][fragment.event_id])):
            raise ValueError("片段的草稿、事件或行动引用不一致")
        if not fragment.action_ids or not fragment.purpose or not fragment.viewpoint or fragment.length_budget is None:
            raise ValueError("片段须明确具体行动来源、叙述目的、视角和篇幅预算")
        if fragment.viewpoint_role_id is not None and not any(
                fragment.viewpoint_role_id in d["participant_ids"] for d in drafts.values()):
            raise ValueError("片段视角引用未知人物")
        if fragment.knowledge_action_id:
            if not any(fragment.knowledge_action_id in d["action_ids"] for d in drafts.values()):
                raise ValueError("人物知情时点须指向实际已提交行动")
            if fragment.knowledge_action_id not in fragment.action_ids and not fragment.knowledge_basis:
                raise ValueError("使用内容来源之外的知情时点须说明依据")

    def save_narrative_plan(self, version_id, plan):
        self._writable()
        plan = self.validate_narrative_plan(version_id, plan)
        state = self.narrative(version_id)
        payload = plan.model_dump(mode="json")
        if state["plan"] is not None:
            if state["plan"] != payload:
                raise ValueError("已接收的叙述落位不能覆盖；重新落位请另建版本")
            return
        state.update(plan=payload, stage="writing", epoch=state["epoch"] + 1,
                     links={link.link_id: {"link": link.model_dump(mode="json"), "revisions": {}} for link in plan.links})
        with self.connection:
            self._put("narratives", version_id, state)

    @staticmethod
    def link_current(record, prose):
        link = record["link"]
        ends = [link["from_fragment"], link["to_fragment"]]
        return (all(end in prose for end in ends) and
                record["revisions"] == {end: prose[end]["revision"] for end in ends})

    def select_narrative_links(self, version_id, selection, required=()):
        """校验拟修改范围并在副本上应用关系变更；不触碰当前正文。"""
        selection = NarrativeSelection.model_validate(selection.model_dump(mode="json"))
        state = self.narrative(version_id)
        if state["plan"] is None:
            raise ValueError("先保存完整章节片段落位")
        plan = NarrativePlan.model_validate(state["plan"])
        targets = set(selection.fragment_ids)
        if targets - set(plan.reading_order) or set(required) - targets:
            raise ValueError("联合任务包含未知片段或遗漏必需目标")
        links = state["links"]  # narrative() 每次解码新副本。
        for operation in [*selection.upsert_links, *selection.remove_links]:
            old = links.get(operation.link_id)
            if old and {old["link"]["from_fragment"], old["link"]["to_fragment"]} - targets:
                raise ValueError("修改已有关系必须共同读取其原有两端")
        for removal in selection.remove_links:
            if removal.link_id not in links:
                raise ValueError("不能删除不存在的叙述关系")
            del links[removal.link_id]
        for link in selection.upsert_links:
            if not link.acceptance:
                raise ValueError("关联须明确正文中的成立和回收条件")
            if {link.from_fragment, link.to_fragment} - targets:
                raise ValueError("新建或修改关系必须共同处理其两端")
            old = links.get(link.link_id)
            if old is None or old["link"] != link.model_dump(mode="json"):
                links[link.link_id] = {"link": link.model_dump(mode="json"), "revisions": {}}
        NarrativePlan.model_validate({**state["plan"], "links": [row["link"] for row in links.values()]})
        return links

    def preview_narrative_batch(self, version_id, task_id, epoch, selection, batch, review=None, *, allow_unconfirmed=False):
        state = self.narrative(version_id)
        if state["stage"] in ("complete", "incomplete") or epoch != state["epoch"]:
            raise ValueError("叙述版本已结束或正文依据版本已变化")
        pending = state["pending"]
        if not pending or pending["task_id"] != task_id or pending.get("selection") != selection.model_dump(mode="json"):
            raise ValueError("正文必须绑定已保存的联合任务与选取范围")
        links = self.select_narrative_links(version_id, selection, pending["required"])
        batch = NarrativeBatch.model_validate(batch.model_dump(mode="json"))
        targets = set(selection.fragment_ids)
        if {row.fragment_id for row in batch.texts} != targets:
            raise ValueError("联合交付必须恰好包含本次目标的全部正文")
        internal = {key for key, row in links.items()
                    if {row["link"]["from_fragment"], row["link"]["to_fragment"]} <= targets}
        confirmed = set(batch.confirmed_link_ids)
        if confirmed - internal or (not allow_unconfirmed and confirmed != internal):
            raise ValueError("须共同核对本次目标之间的全部有效关联")
        for text in batch.texts:
            if not text.summary.strip():
                raise ValueError("正文须提供供后续检索和全书复核的内容摘要")
            old = state["prose"].get(text.fragment_id)
            if not old or old["text"] != text.text:
                revision = old["revision"] + 1 if old else 1
                if revision > state["config"]["max_revisions"] + 1:
                    raise NarrativeLimitReached("片段正文修订上限已用尽：" + text.fragment_id)
                state["prose"][text.fragment_id] = ProseFragment(
                    version_id=version_id, plan_revision=state["plan"]["revision"], fragment_id=text.fragment_id,
                    revision=revision, task_id=task_id, text=text.text).model_dump(mode="json")
            # 同文修正摘要/读者揭示同样是已审核的交付，但不消耗正文修订次数。
            state.setdefault("prose_meta", {})[text.fragment_id] = {
                "summary": text.summary, "revealed": text.revealed, "change": text.change}
        if review is not None:
            if not review.approved:
                raise ValueError("联合正文语义检查未通过，不得保存为有效正文")
            review.validate_context(links=links, targets=targets, written=state["prose"],
                                    reading_order=NarrativePlan.model_validate(state["plan"]).reading_order)
            confirmed.update(review.preserved_link_ids)
            state.setdefault("batch_reviews", []).append({"task_id": task_id, "result": review.model_dump(mode="json")})
        for identity in confirmed:
            link = links[identity]["link"]
            links[identity]["revisions"] = {end: state["prose"][end]["revision"]
                                            for end in (link["from_fragment"], link["to_fragment"])}
        issues = [i for i in state["issues"] if set(i["fragment_ids"]) - targets]
        if review is not None:
            issues.extend({"fragment_ids": [i], "description": "本次表达、已有线索呈现或读者揭示变化，需要复核"}
                          for i in dict.fromkeys(review.affected_fragment_ids))
        # 未经语义核对的外部关系保留旧确认，自动进入后续联合任务。
        state.update(links=links, epoch=epoch + 1, sequence=state["sequence"] + 1,
                     pending=None, issues=issues, stage="writing", error=None)
        return state

    def commit_narrative_batch(self, version_id, task_id, epoch, selection, batch, review=None):
        self._writable()
        if review is None or not review.approved:
            raise ValueError("保存联合正文前必须通过世界角色的语义核对")
        self.check_narrative_source(version_id)
        payload = storage_encode({"epoch": epoch, "selection": selection.model_dump(mode="json"), "batch": batch.model_dump(mode="json"),
                          "review": review.model_dump(mode="json")})
        old = self.connection.execute("SELECT payload FROM narrative_batches WHERE version_id=? AND task_id=?",
                                      (version_id, task_id)).fetchone()
        if old is not None:
            if old[0] != payload:
                raise ValueError("联合任务已经绑定另一份结果")
            return
        state = self.preview_narrative_batch(version_id, task_id, epoch, selection, batch, review)
        with self.connection:
            self.connection.execute("INSERT INTO narrative_batches VALUES (?,?,?)", (version_id, task_id, payload))
            self._put("narratives", version_id, state)

    def narrative_status(self, version_id):
        state = self.narrative(version_id)
        order = NarrativePlan.model_validate(state["plan"]).reading_order if state["plan"] else []
        scope = NarrativeScope.scope_name(version_id)
        receipt = self.get_checkpoint("export/narrative/" + version_id, {})
        return {"version_id": version_id, "stage": state["stage"], "error": state["error"],
                "export_error": receipt.get("error"),
                "published": state["stage"] == "complete" and receipt.get("complete_epoch") == state["epoch"] and not receipt.get("pending"),
                "plan_ready": state["plan"] is not None, "fragments": len(order), "written": len(state["prose"]),
                "missing": [identity for identity in order if identity not in state["prose"]],
                "pending_links": [key for key, row in state["links"].items() if not self.link_current(row, state["prose"])],
                "issues": state["issues"], "review_count": len(state["reviews"]),
                "calls": self.call_count(scope)}

    def record_narrative_review(self, version_id, epoch, review):
        self._writable()
        review = NarrativeReview.model_validate(review.model_dump(mode="json"))
        state, status = self.narrative(version_id), self.narrative_status(version_id)
        if state["epoch"] != epoch or state["pending"] or not status["plan_ready"] or status["missing"] or status["pending_links"]:
            raise ValueError("内容核对必须针对完整且关联已落实的当前正文")
        if state["stage"] in ("complete", "incomplete") or len(state["reviews"]) > state["config"]["max_review_rounds"]:
            raise ValueError("内容核对已结束或超过配置上限")
        if any(set(issue.fragment_ids) - set(state["prose"]) for issue in review.issues):
            raise ValueError("内容核对的问题引用了未知片段")
        self.check_narrative_source(version_id)
        state["reviews"].append({"epoch": epoch, "result": review.model_dump(mode="json")})
        state.update(issues=[issue.model_dump(mode="json") for issue in review.issues], error=None,
                     stage="complete" if review.approved else "writing")
        with self.connection:
            self._put("narratives", version_id, state)


class NarrativeScope:
    """同一账本按叙述版本隔离任务及预算；不复制模型请求实现。"""
    def __init__(self, storage, version_id, *, selection_config=None):
        self.storage = storage
        self.prior_usage = {}
        if selection_config is not None:
            self.scope = "selection/" + quote(version_id, safe="")
            self.config = selection_config
            return
        self.scope = self.scope_name(version_id)
        state = storage.narrative(version_id)
        self.config = NovelRestoreNarrativeConfig.model_validate(state["config"])
        self.prior_usage = state.get("selection", {}).get("usage", {})

    @staticmethod
    def scope_name(version_id):
        return "narrative/" + quote(version_id, safe="")

    def get_job(self, key):
        return self.storage.get_job(self.scope + "/" + key)

    def save_job(self, key, payload):
        self.storage.save_job(self.scope + "/" + key, payload)

    def reserve(self, key, occurrence, body, limit, *, can_send, billable=True):
        ceiling = min((n for n in (limit, self.config.max_calls) if n is not None), default=None)
        remaining = None if ceiling is None else ceiling - self.prior_usage.get("calls", 0)
        return self.storage.reserve(key, occurrence, body, remaining, can_send=can_send,
                                    scope=self.scope, budget_config=self.config, prior_usage=self.prior_usage, billable=billable)

    def received(self, call_id, status, raw, usage):
        self.storage.received(call_id, status, raw, usage)

    def release_call(self, call_id):
        self.storage.release_call(call_id)

    def failed(self, call_id, error):
        self.storage.failed(call_id, error)
