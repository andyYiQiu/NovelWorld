"""世界侧只读查询；返回原始记录及版本，不把结果直接交给人物。"""
from hashlib import sha256
import json
from typing import Literal

from pydantic import Field

from common.Runtime import RunStopped
from common.NovelRestoreSchema import Identity, Record
from storage.NovelRestoreStorage import encode
from common.NovelRestoreObjects import object_assets, effective_catalog


class WorldQuery(Record):
    kind: Literal["index", "asset", "background", "node", "group", "event", "reference_before",
                  "reference_after", "current", "subjective", "memories", "history", "action", "decision"]
    identity: Identity = Field(description="精确编号，不支持关键词搜索。kind=index 时只能填 assets/background/timeline；"
        "asset 用资产编号；background 用背景目录返回的文件路径；node 用节点编号；group 用组编号；"
        "event/reference_before/reference_after 用事件编号；current 用所属类别中的记录编号，relations 用关系任一端编号；"
        "subjective/memories 用人物编号；history 用 category 对应的编号；action/decision 用已提交行动编号。未知编号先查目录，不编造编号。")
    category: str = Field(default="", description="kind=asset 或 index+assets 时必填 role/object/location/rule/organization；"
        "kind=current 时必填 characters/objects/locations/relations/public_state/constraints（注意为状态类别，不是资产类别）；"
        "kind=history 时必填 event/person/object/location；其余查询省略。"
        "列物品编号用 kind=index,identity=assets,category=object；查人物当前状态用 kind=current,category=characters,identity=人物编号。")
    after: int = Field(default=0, ge=0, strict=True, description="首次为0；后续原样填写上页 next，行动历史使用版本游标，其他分页使用偏移")
    limit: int = Field(default=20, ge=1, le=50, strict=True)


class WorldLookup:
    """绑定一次交接的世界状态；业务查询使用参数化 SQL 和稳定编号。"""
    def __init__(self, storage, repository):
        self.storage, self.repository = storage, repository

    def stamp(self):
        version, state, progress = self.storage.current()
        value = {"run_id": self.storage.run_id, "repository": self.storage.manifest["repository_hash"],
                 "version": version, "state": state.model_dump(mode="json"), "progress": progress,
                 "views": self.storage.get_checkpoint("views", {}),
                 "person_runtime": self.storage.get_checkpoint("person_runtime", {}),
                 "runtime_objects": self.storage.get_checkpoint("runtime_objects", {}),
                 "profiles": self.storage.get_checkpoint("self_profiles", {}),
                 "subjective": [tuple(row) for row in self.storage.connection.execute(
                     "SELECT key,payload FROM checkpoints WHERE key LIKE 'character/%' OR key LIKE 'summaries/%' OR key LIKE 'access/%' ORDER BY key")]}
        return sha256(encode(value).encode()).hexdigest()

    def query(self, request, *, stamp, allow_timeline=True):
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
            allowed = ("characters", "objects", "locations", "relations", "public_state", "constraints")
            located = [key for key in ("characters", "objects", "locations", "constraints") if identity in categories[key]]
            if request.category not in allowed:
                raise ValueError(f"当前状态查询 category={request.category!r} 无效；合法值为 {allowed}；编号 {identity!r} 的直接记录类别为 {located}")
            category = categories[request.category]
            if request.category == "relations":
                data = [r for r in category if identity in (r["owner_id"], r["other_id"])]
            elif request.category == "public_state":
                data = {k: v[identity] for k, v in category.items() if identity in v}
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
            rows = self.storage.memories(identity, through_version=version)
            page = rows[request.after:request.after + request.limit]
            data = {"items": [m.model_dump(mode="json") for m in page],
                    "next": request.after + len(page) if request.after + len(page) < len(rows) else None}
        elif kind in ("action", "decision"):
            row = self.storage.connection.execute("SELECT payload FROM actions WHERE action_id=? AND status='committed' AND version<=?", (identity, version)).fetchone()
            if row is None:
                raise ValueError("已提交行动不存在")
            data = json.loads(row[0])
            if kind == "decision":
                from agents.NovelRestoreAgent import task_id
                pos = data["position"]
                job = self.storage.get_job(task_id("character", pos["event_id"], pos["round"], pos["turn"])) or {}
                data = {"character_id": data["proposal"]["character_id"], "attempt": data["proposal"],
                        "recorded_candidates": job.get("result"), "scope": "人物主观推演。只有最终尝试另经裁定后才能成为故事事实。"}
            elif not allow_timeline:
                data.pop("completion", None)
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
            if not allow_timeline:
                for item in items:
                    item["action"].pop("completion", None)
            data = {"items": items,
                    "next": rows[-1]["version"] if more else None}
        return {"source": source,
                "run_id": self.storage.run_id, "version": version, "request": request.model_dump(mode="json"), "data": data}
