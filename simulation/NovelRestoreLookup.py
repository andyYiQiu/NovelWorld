"""世界侧只读查询；按职责投影记录并保留版本，不把结果直接交给人物。"""
from hashlib import sha256
from copy import deepcopy
import json
from typing import Literal

from pydantic import Field, model_validator
from jsonpointer import JsonPointer

from common.Runtime import RunStopped
from common.NovelRestoreSchema import Identity, Record
from common.Runtime import storage_encode as encode
from simulation.NovelRestoreObjects import object_assets, effective_catalog


def lookup_for(supervisor, role, context):
    """两个执行后端共用查阅权限；人物、叙述及限定来源任务不开放世界查询。"""
    if role not in {"world", "world_visibility", "world_semantic"}:
        return None, False
    operation = context.get("operation")
    if role == "world_semantic":
        reviewed = context.get("reviewed_operation")
        if operation != "review_semantics" or reviewed not in {"schedule", "assess_wave", "assess_group"}:
            return None, False
        # A scene reviewer can verify omitted motives/abilities from actual
        # sources, but cannot fetch future repository outcomes as scene facts.
        return getattr(supervisor, "world_lookup", None), reviewed != "schedule"
    reviewed = context.get("evidence", {}).get("operation") if operation == "review_visibility" else operation
    permitted = {"consider_branch", "schedule", "perceive", "resolve_wave", "review_memory",
                 "assess_wave", "assess_group", "review_visibility", "review_unknown_targets"}
    if operation not in permitted or reviewed in {"initial_profiles", "prepare_experience", "process_recall"}:
        return None, False
    allow_timeline = reviewed not in {"consider_branch", "perceive", "resolve_wave", "review_memory",
                                     "review_unknown_targets"}
    return getattr(supervisor, "world_lookup", None), allow_timeline


class WorldQuery(Record):
    kind: Literal["index", "asset", "background", "node", "group", "event", "reference_before",
                  "reference_after", "current", "subjective", "memories", "history", "action", "decision"]
    identity: Identity = Field(description="精确编号，不支持关键词搜索。kind=index 时只能填 assets/background/timeline；"
        "asset 用资产编号；background 用背景目录返回的文件路径；node 用节点编号；group 用组编号；"
        "event/reference_before/reference_after 用事件编号；current 用所属类别中的记录编号，relations 用关系任一端编号；"
        "subjective/memories 用人物编号；history 用 category 对应的编号；action/decision 用已提交行动编号。未知编号先查目录，不编造编号。")
    category: str = Field(default="", description="kind=asset 或 index+assets 时必填 role/object/location/rule/organization；"
        "kind=current 时必填 characters/objects/locations/relations（注意为状态类别，不是资产类别）；"
        "kind=history 时必填 event/person/object/location；其余查询省略。"
        "列物品编号用 kind=index,identity=assets,category=object；查人物当前状态用 kind=current,category=characters,identity=人物编号。")
    after: int = Field(default=0, ge=0, strict=True, description="首次为0；后续原样填写上页 next，行动历史使用版本游标，其他分页使用偏移")
    limit: int = Field(default=20, ge=1, le=50, strict=True)


class CurrentWorldQuery(WorldQuery):
    """Pydantic 工具契约按任务权限收窄；底层 query 仍独立检查权限。"""
    kind: Literal["index", "asset", "background", "current", "subjective", "memories",
                  "history", "action", "decision"]
    identity: Identity = Field(description="精确编号，不支持关键词搜索。kind=index 时只能填 assets/background；"
        "asset 用资产编号；background 用背景目录返回的文件路径；current 用所属类别中的记录编号，relations 用关系任一端编号；"
        "subjective/memories 用人物编号；history 用 category 对应的编号；action/decision 用已提交行动编号。"
        "本任务不能查询时间轴目录、节点、事件目标或参考前后态。未知编号先查目录，不编造编号。")

    @model_validator(mode="after")
    def current_sources_only(self):
        if self.kind == "index" and self.identity == "timeline":
            raise ValueError("当前任务不能查询时间轴目录；kind=index 的 identity 只能填 assets/background")
        return self


def history_page(data):
    """客观历史索引；个人体验与审计附件由明确的行动查询读取。"""
    from simulation.NovelRestoreContext import action_history, shared_history

    events = {"page": [row["action"] for row in data["items"]]}
    actions = action_history(events)["page"]
    shared = {**data.get("shared_advances", {}), **shared_history(events)}
    personal = {"observations", "perceptions", "final_views"}
    return {**data, "items": [{**row, "action": {k: v for k, v in action.items() if k not in personal}}
                              for row, action in zip(data["items"], actions)],
            "shared_advances": {identity: {k: v for k, v in advance.items() if k not in personal}
                                for identity, advance in shared.items()},
            "detail_scope": "客观经过、行动提议与状态变化完整保留；个人体验、记忆和审核附件用 kind=action 按行动编号读取。"}


def assessment_action(data):
    """Objective action evidence for progress assessment and its reviewer.

    The committed row remains the authoritative full record.  This projection
    keeps actual proposal/result/state/shared consequences while excluding
    personal deliveries, memories and audit attachments that are unrelated to
    objective goal progress.
    """
    page = history_page({"items": [{"version": 0, "action": data}], "next": None})
    action = page["items"][0]["action"]
    shared = page.get("shared_advances", {})
    if shared:
        action["advance"] = next(iter(shared.values()))
    return action


def assessment_lookup(context):
    if not isinstance(context, dict):
        return False
    operation = context.get("operation")
    return operation in {"assess_wave", "assess_group"} or (
        operation == "review_semantics"
        and context.get("reviewed_operation") in {"assess_wave", "assess_group"}
    )


def candidate_action_paths(context, identity):
    """Locate supplied candidate records, never turn an uncommitted ID into history."""
    found = []
    def visit(value, parts):
        if isinstance(value, dict):
            if value.get("action_id") == identity or value.get("source_id") == identity:
                found.append(JsonPointer.from_parts(parts).path)
            for key, item in value.items():
                visit(item, [*parts, key])
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, [*parts, str(index)])
    visit(context, ["context"])
    return found


def current_lookup_evidence(records):
    """Project saved planner queries to the same boundary as CurrentWorldQuery."""
    result = []
    for record in records:
        try:
            query = CurrentWorldQuery.model_validate(record["request"])
        except ValueError:
            continue  # Future references remain in the planning task, not scene evidence.
        item = deepcopy(record)
        if query.kind == "action":
            item["data"].pop("completion", None)
        elif query.kind == "history":
            item["data"] = history_page(item["data"])
        result.append(item)
    return result


class WorldLookup:
    """绑定一次交接的世界状态；业务查询使用参数化 SQL 和稳定编号。"""
    def __init__(self, storage, repository):
        self.storage, self.repository = storage, repository

    def stamp(self):
        version, state, progress = self.storage.current()
        # A process failure changes lifecycle diagnostics, not the story queried
        # by this task. Keep logical cursors and all actual sources in the stamp
        # so recovery can reuse receipts without accepting a changed world.
        progress = {key: value for key, value in progress.items() if key not in {"stage", "error"}}
        value = {"run_id": self.storage.run_id, "repository": self.storage.manifest["repository_hash"],
                 "version": version, "state": state.model_dump(mode="json"), "progress": progress,
                 "views": self.storage.get_checkpoint("views", {}),
                 "person_runtime": self.storage.get_checkpoint("person_runtime", {}),
                 "runtime_objects": self.storage.get_checkpoint("runtime_objects", {}),
                 "profiles": self.storage.get_checkpoint("self_profiles", {}),
                 "subjective": [tuple(row) for row in self.storage.connection.execute(
                     "SELECT key,payload FROM checkpoints WHERE key LIKE 'character/%' OR key LIKE 'summaries/%' OR key LIKE 'access/%' ORDER BY key")]}
        return sha256(encode(value).encode()).hexdigest()

    def query(self, request, *, stamp, allow_timeline=True, candidate_context=None):
        request = WorldQuery.model_validate(request)
        if self.stamp() != stamp:
            raise RunStopped("世界查询的状态已变化，不能混用不同批次资料")
        kind, identity = request.kind, request.identity
        source = "repository" if kind in {"index", "asset", "background", "node", "group", "event", "reference_before", "reference_after"} else "simulation"
        version, state, _ = self.storage.current()
        if not allow_timeline and (kind in {"node", "group", "event", "reference_before", "reference_after"}
                                   or kind == "index" and identity == "timeline"):
            raise ValueError("当前实际裁定或感知任务不能查询时间轴目标")
        if kind == "index":
            if identity == "assets":
                if request.category not in self.repository.ASSET_KINDS:
                    raise ValueError(f"资产目录须指定 category，合法值为 {self.repository.ASSET_KINDS}；identity 固定为 assets")
                assets = (object_assets(self.storage) if request.category == "object"
                          else self.repository.assets(request.category))
                items = [{"asset_id": key, "name": row.get(request.category + "_name", row.get("name", "")),
                          "aliases": row.get("aliases", []), "origin": row.get("origin", "repository")}
                         for key, row in assets.items()]
                if any(row["origin"] == "simulation" for row in items):
                    source = "repository_and_simulation"
            elif identity == "background":
                items = self.repository.background_paths()
            elif identity == "timeline":
                items = [{"node_id": n["node_id"], "node_order": n["order"],
                          "groups": [{"group_id": g.group_id, "order": g.order,
                                      "event_ids": [e["task_id"] for e in g.events]}
                                     for g in self.repository.groups(n["node_id"])]} for n in self.repository.nodes()]
            else:
                raise ValueError(f"未知目录 identity={identity!r}；kind=index 的 identity 只能为 assets/background/timeline，不能填关键词或实体编号；列物品用 identity=assets, category=object")
            page = items[request.after:request.after + request.limit]
            data = {"items": page, "next": request.after + len(page) if request.after + len(page) < len(items) else None}
        elif kind == "asset":
            if request.category not in self.repository.ASSET_KINDS:
                raise ValueError(f"资产查询须指定 category，合法值为 {self.repository.ASSET_KINDS}；identity 填该类资产的准确编号")
            dynamic = self.storage.get_checkpoint("runtime_objects", {}).get(identity) if request.category == "object" else None
            if dynamic is not None:
                data = dynamic
                source = "simulation"
            else:
                try:
                    data = self.repository.asset(request.category, identity)
                except KeyError as error:
                    raise ValueError(f"资产不存在：category={request.category}, identity={identity}；先用 kind=index, identity=assets 和相同 category 查询实际编号") from error
        elif kind == "background":
            try:
                data = self.repository.background(identity)
            except KeyError as error:
                raise ValueError("世界背景不存在") from error
        elif kind == "node":
            data = next((n for n in self.repository.nodes() if n["node_id"] == identity), None)
            if data is None:
                raise ValueError("时间节点不存在")
        elif kind == "group":
            data = self.repository.get_event_group(identity)
        elif kind == "event":
            data = self.repository.get_event_context(identity)
        elif kind == "reference_before":
            data = self.repository.get_state_before(identity)
        elif kind == "reference_after":
            try:
                data = self.repository.get_state_after(identity)
            except KeyError as error:
                raise ValueError("事件不存在") from error
        elif kind == "current":
            categories = state.model_dump(mode="json")
            allowed = ("characters", "objects", "locations", "relations")
            located = [key for key in ("characters", "objects", "locations") if identity in categories[key]]
            if request.category not in allowed:
                raise ValueError(f"当前状态查询 category={request.category!r} 无效；合法值为 {allowed}；编号 {identity!r} 的直接记录类别为 {located}")
            category = categories[request.category]
            if request.category == "relations":
                data = [r for r in category if identity in (r["subject_role_id"], r["related_role_id"])]
            else:
                if identity not in category:
                    raise ValueError(f"当前状态 {request.category} 中没有编号 {identity!r}；该编号的直接记录类别为 {located}；不得把未命中当成对象不存在")
                data = category[identity]
        elif kind == "subjective":
            if identity not in self.storage.manifest["catalog"]["role"]:
                raise ValueError("人物不存在")
            view = self.storage.get_checkpoint("views", {}).get(identity)
            data = {"profile": self.storage.manifest["characters"][identity],
                    "current": self.storage.character_state(identity).model_dump(mode="json"),
                    "personal_profile": self.storage.get_checkpoint("self_profiles", {}).get(identity),
                    "available_input": self.storage.get_checkpoint("access/" + identity),
                    "view": view, "runtime": self.storage.get_checkpoint("person_runtime", {}).get(identity, {}),
                    "summaries": self.storage.memory_summaries(identity)}
        elif kind == "memories":
            rows = self.storage.memories(identity, through_version=version, offset=request.after, limit=request.limit + 1)
            page = rows[:request.limit]
            data = {"items": [m.model_dump(mode="json") for m in page],
                    "next": request.after + len(page) if len(rows) > request.limit else None}
        elif kind in ("action", "decision"):
            row = self.storage.connection.execute("SELECT payload FROM actions WHERE action_id=? AND status='committed' AND version<=?", (identity, version)).fetchone()
            if row is None:
                paths = candidate_action_paths(candidate_context, identity) if kind == "action" else []
                if paths:
                    return {"source": "task_candidate", "run_id": self.storage.run_id, "version": version,
                            "request": request.model_dump(mode="json"), "data": {"committed": False,
                                "candidate_paths": paths,
                                "instruction": "该行动尚未提交，以下路径指向原任务已提供的相关候选内容；"
                                               "不能作为新增独立历史证据，也不能从数据库查到已提交结果。请结合原任务来源审读。"}}
                raise ValueError("已提交行动不存在；当前批次候选须依据本任务提供的提议、裁定及体验审读，不能当作历史查询")
            data = json.loads(row[0])
            if kind == "decision":
                from common.Runtime import task_id
                pos = data["position"]
                job = self.storage.get_job(task_id("character", pos["event_id"], pos["round"], pos["turn"])) or {}
                data = {"character_id": data["proposal"]["character_id"], "attempt": data["proposal"],
                        "recorded_candidates": job.get("result"), "scope": "人物主观推演。只有最终尝试另经裁定后才能成为故事事实。"}
            elif assessment_lookup(candidate_context):
                data = assessment_action(data)
        else:
            if request.category == "event":
                condition = "json_extract(payload,'$.position.event_id')=?"
            elif request.category == "person":
                if identity not in self.storage.manifest["catalog"]["role"]:
                    raise ValueError("人物不存在")
                condition = """(json_extract(payload,'$.proposal.character_id')=? OR EXISTS
                    (SELECT 1 FROM json_each(payload,'$.observations') WHERE key=?) OR EXISTS
                    (SELECT 1 FROM json_each(payload,'$.advance.observations') WHERE key=?))"""
            elif request.category in ("object", "location"):
                if identity not in effective_catalog(self.storage)[request.category]:
                    raise ValueError("历史引用的物品或地点不存在")
                events = self.repository.event_ids_for_asset(request.category, identity)
                placeholders = ",".join("?" for _ in events)
                indexed = f" OR json_extract(payload,'$.position.event_id') IN ({placeholders})" if events else ""
                condition = "(EXISTS (SELECT 1 FROM json_tree(payload) WHERE atom=? AND key IN " \
                            "('actual_id','object_id','location_id','id'))" + indexed + ")"
            else:
                raise ValueError("历史查询须指定 category=event/person/object/location，identity 填对应的精确编号")
            values = ((identity, identity, identity) if request.category == "person" else
                      (identity, *events) if request.category in ("object", "location") else (identity,))
            rows = self.storage.connection.execute(
                "SELECT version,payload FROM actions WHERE status='committed' AND version>? AND version<=? AND "
                + condition + " ORDER BY version LIMIT ?", (request.after, version, *values, request.limit + 1)).fetchall()
            more, rows = len(rows) > request.limit, rows[:request.limit]
            items = [{"version": r["version"], "action": json.loads(r["payload"])} for r in rows]
            data = history_page({"items": items,
                                 "next": rows[-1]["version"] if more else None})
        evidence = {"source": source, "run_id": self.storage.run_id, "version": version,
                    "request": request.model_dump(mode="json"), "data": data}
        if kind == "action" and assessment_lookup(candidate_context):
            evidence["detail_scope"] = (
                "目标进度客观证据视图；完整行动、人物体验、记忆与审核附件仍保留在已提交记录中。")
        if not allow_timeline and kind == "action":
            evidence["data"].pop("completion", None)
        return evidence
