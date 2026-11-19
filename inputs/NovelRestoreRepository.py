"""公开 repository 的选择入口、共用数据契约和只读 SQLite 适配器。"""
from hashlib import sha256
import json
from pathlib import Path

from common.NovelRestoreSchema import EventGroup, NodeSnapshot, WorldState
from inputs.RepositoryReader import RepositoryReader
from inputs.RepositoryInput import normalize_asset, normalize_event, normalize_state_delta, normalize_node, normalize_group
from inputs.NovelWorldStateSchema import require_state_sections
from inputs.NovelWorldState import public_state, replay_group, state_reference_errors, validate_state, location_ancestors
from tools.JsonInput import loads, validate
from inputs.RepositoryFormat import FILE_MODELS, Node, Group


def require_public_input(value, *, location="repository"):
    """公开资料可以描述故事，但不能携带原书正文、摘录或上游取证附件。"""
    private_fields = {"evidence", "unverified_quote", "source_text", "original_text", "full_text",
                      "raw_text", "source_excerpt", "original_quote", "原文", "原文摘录", "原文证据"}
    def visit(item, path):
        if isinstance(item, dict):
            for key, row in item.items():
                if key.casefold() in private_fields:
                    raise ValueError(f"{path}/{key}: 公开资料不能包含原文证据或正文")
                visit(row, f"{path}/{key}")
        elif isinstance(item, list):
            for index, row in enumerate(item):
                visit(row, f"{path}/{index}")
    visit(value, location)


def validate_event(event, assets, *, location):
    """核对单个公开事件的字段以及人物、物品和地点引用。"""
    key = event.get("task_id")
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"{location}/task_id: 缺少事件编号")
    identity = f"{location} ({key})"
    if not isinstance(event.get("todo"), str) or not event["todo"].strip():
        raise ValueError(f"{identity}: 事件内容为空")
    if event.get("time") is not None and not isinstance(event["time"], str):
        raise ValueError(f"{identity}: time须为文字或null")
    people = event.get("participants")
    if not isinstance(people, list) or any(
            not isinstance(p, dict) or not isinstance(p.get("role_code"), str)
            or p["role_code"] not in assets["role"] for p in people):
        raise ValueError(f"{identity}: 参与者引用非法")
    if any(p.get("original_goal") is not None and not isinstance(p["original_goal"], str) for p in people):
        raise ValueError(f"{identity}: original_goal须为文字或null")
    objects = event.get("object_ids")
    if not isinstance(objects, list) or any(not isinstance(ref, str) or ref not in assets["object"] for ref in objects):
        raise ValueError(f"{identity}: 物品引用非法")
    locations = event.get("location_ids")
    names = event.get("locations")
    if (not isinstance(locations, list) or any(not isinstance(ref, str) or ref not in assets["location"] for ref in locations)
            or names is not None and not isinstance(names, list)
            or [assets["location"][ref]["location_name"] for ref in locations] != (names or [])):
        raise ValueError(f"{identity}: 地点引用非法或名称与编号不一致")


class PublicRepository:
    """两种公开数据来源共用事件组、快照和跨节点状态契约。"""
    ASSET_KINDS = ("role", "object", "location", "rule", "organization")

    @staticmethod
    def parse_group(node_id, node_order, raw, catalog, *, location=None):
        location = location or f"节点 {node_id}/事件组"
        if not isinstance(raw, dict):
            raise ValueError(f"{location}: 事件组须为 JSON 对象")
        require_public_input(raw, location=location)
        if "state_changes" in raw:
            Group.model_validate(raw)
        raw = normalize_group(raw, location=location, locations=catalog.get("location", {}), objects=catalog.get("object", {}))
        require_state_sections(raw.get("state_delta"), location + "/state_delta")
        if "node_id" in raw or "node_order" in raw:
            raise ValueError(f"{location}: 事件组不能覆盖所属节点")
        raw = {**raw, "state_delta": normalize_state_delta(raw.get("state_delta"), location=location + "/state_delta")}
        group = validate(EventGroup, {"node_id": node_id, "node_order": node_order, **raw}, location=location)
        catalogs = {kind: catalog.get(kind, ()) for kind in PublicRepository.ASSET_KINDS}
        errors = state_reference_errors(group.state_delta, catalogs, location + "/state_delta")
        if errors:
            raise ValueError("\n".join(errors))
        people = []
        for index, event in enumerate(group.events):
            event_location = f"{location}/events/{index}"
            event = normalize_event(event, location=event_location, locations=catalog.get("location", {}), objects=catalog.get("object", {}))
            if "order" in event or "state_delta" in event or event.get("epistemic_status") != "事实":
                raise ValueError(f"{event_location}: 公开事件须使用组级顺序、组级变化并已确认为事实")
            validate_event(event, catalog, location=event_location)
            group.events[index] = event
            people.extend(p["role_code"] for p in event["participants"])
        if len(people) != len(set(people)):
            raise ValueError(f"{location}: 同组实际参与人物不能重复")
        return group

    @staticmethod
    def parse_snapshot(node_id, raw, *, location=None):
        location = location or f"state_snapshots/{node_id}"
        require_public_input(raw, location=location)
        require_state_sections(raw, location)
        snapshot = validate(NodeSnapshot, raw, location=location)
        if snapshot.node_id != node_id:
            raise ValueError(f"{location}: 快照内容与节点身份不一致")
        return snapshot

    def group_states(self, group):
        """两个参照状态只从 repository 回放，绝不使用运行状态推算目标。"""
        envelope = self.snapshot(group.node_id).model_dump(include={"node_id", "previous_node_id"})
        before = self.get_state_before(group.events[0]["task_id"])
        after = self.get_state_after_group(group.node_id, group.order)
        return (WorldState.model_validate({**before, **envelope}),
                WorldState.model_validate({**after, **envelope}))

    def validate_contract(self):
        """核对公开档案、事件引用与跨节点状态连续性。"""
        assets = {kind: self.assets(kind) for kind in self.ASSET_KINDS}
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("repository 来源为空")
        for kind, records in assets.items():
            setting = kind in ("rule", "organization")
            id_field = "setting_id" if setting else f"{kind}_code"
            name_field = "term" if setting else f"{kind}_name"
            for key, record in records.items():
                if (not isinstance(record, dict) or record.get(id_field) != key or (setting and record.get("kind") != kind)
                        or not isinstance(record.get(name_field), str) or not record[name_field].strip()):
                    raise ValueError(f"{kind}/{key}: 档案的身份、类型或名称与资产目录不一致")
                FILE_MODELS['setting' if setting else {'role':'role-info','object':'object-info','location':'locations'}[kind]].model_validate(
                    {key:record} if kind == 'location' else record)
                if "source" in record and record["source"] != self.source:
                    raise ValueError(f"{kind}/{key}: 档案来源与 repository 不一致")
        seen_ids = set()
        for records in assets.values():
            if seen_ids.intersection(records):
                raise ValueError('资产编号须全局唯一，不能跨类别重复')
            seen_ids.update(records)
        location_ancestors(assets['location'], assets['location'])
        for key, role in assets["role"].items():
            if {"age", "identity", "biography", "relation"} & role.keys():
                raise ValueError(f"{key}: 静态人物档案不能包含阶段身份、年龄、传记或关系")
        for key, item in assets["object"].items():
            if "significance" in item:
                raise ValueError(f"{key}: 剧情作用不属于静态物品档案")
        worlds = self.worlds()
        if set(worlds) != {"worlds/background/general.json", "worlds/settings/settings.jsonl"}:
            raise ValueError("背景与设定须使用固定 general.json 和 settings.jsonl 路径")
        setting_ids = set()
        settings_by_kind = {"rule": {}, "organization": {}}
        for path, content in worlds.items():
            records = enumerate(content) if isinstance(content, list) else [(None, content)]
            for index, record in records:
                FILE_MODELS['setting' if isinstance(content,list) else 'general'].model_validate(record)
                if isinstance(content,list):
                    identity=record['setting_id']
                    if identity in setting_ids or any(identity in assets[k] for k in ('role','object','location')):
                        raise ValueError('设定编号重复或与资产冲突：' + identity)
                    setting_ids.add(identity)
                    if record['kind'] in settings_by_kind:
                        settings_by_kind[record['kind']][identity] = record
                if isinstance(record, dict) and "source" in record and record["source"] != self.source:
                    location = path if index is None else f"{path}/{index}"
                    raise ValueError(f"{location}: 背景或设定来源与 repository 不一致")
        for kind, records in settings_by_kind.items():
            if records != assets[kind]:
                raise ValueError(f"{kind}: 设定与资产引用索引不一致")
        nodes = self.nodes()
        require_public_input({"assets": assets, "worlds": worlds, "nodes": nodes})

        seen_events, seen_groups = set(), set()
        previous, expected = None, None
        for node in nodes:
            Node.model_validate({k:v for k,v in node.items() if k not in ("path","state_path")})
            key = node["node_id"]
            name = node.get("node_name")
            if not isinstance(name, str) or not name.strip() or not isinstance(node.get("time"), dict):
                raise ValueError(f"{key}: 缺少节点名称/time")
            try:
                snapshot = self.snapshot(key)
            except KeyError as error:
                raise ValueError(f"state_snapshots/{key}: 缺少已声明节点的入口快照") from error
            if snapshot.previous_node_id != previous:
                raise ValueError(f"{key}: previous_node_id与实际顺序不符")
            state = validate_state(snapshot.model_dump(mode="json"), assets, key)
            if expected is not None and public_state(state, False) != public_state(expected, False):
                raise ValueError(f"{key}: 入口快照与前一节点所有组delta回放不一致")
            for order, group in enumerate(self.groups(key), 1):
                if group.order != order or not group.group_id.strip() or group.group_id in seen_groups:
                    raise ValueError(f"{key}: 事件组编号重复或order未从1连续编号")
                seen_groups.add(group.group_id)
                for event in group.events:
                    identity = event.get("task_id")
                    if not isinstance(identity, str) or not identity.strip() or identity in seen_events:
                        raise ValueError(f"{key}: 缺少或重复事件编号")
                    seen_events.add(identity)
                after = replay_group(state, group.state_delta)
                for event in group.events:
                    for person in event['participants']:
                        identity = person['role_code']
                        life = state.characters[identity].life_status
                        if life == 'dead' or life == 'not_born' and after.characters[identity].life_status != 'alive':
                            raise ValueError(f"{group.group_id}/{identity}: 死者或尚未出生者不能列为主动参与者")
                state = after
                validate_state(state.model_dump(mode="json"), assets, group.group_id)
            previous, expected = key, state
        self._check_event_indexes(seen_events)

    def _check_event_indexes(self, seen_events):
        """JSON 的事件索引直接来自这些组；有独立索引的适配器须额外核对。"""


class NovelRestoreRepository(PublicRepository, RepositoryReader):
    TABLES = (
        "repository_info", "world_data", "assets", "time_nodes", "event_groups", "events", "state_snapshots",
        "event_participants", "event_objects", "event_locations", "state_records", "state_changes", "state_references",
    )

    def __init__(self, database):
        self.path = Path(database).expanduser().resolve()
        super().__init__(self.path)
        try:
            tables = {row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if set(self.TABLES) - tables:
                raise ValueError("下游只接受新版四类状态及事件组 repository 数据库；旧库请保留")
            info = self.connection.execute("SELECT source, time_basis FROM repository_info").fetchall()
            if len(info) != 1 or info[0]["time_basis"] != "story":
                raise ValueError("repository 必须有唯一来源且采用故事时间")
            self.source = info[0]["source"]
        except Exception:
            self.close()
            raise

    def _check_event_indexes(self, seen_events):
        # 运行中补查事件、人物或物品会使用关系索引。其内容与顺序必须和已校验的事件一致。
        for table in ("events", "event_groups", "event_participants", "event_objects", "event_locations"):
            if self.connection.execute(f"PRAGMA foreign_key_check({table})").fetchone() is not None:
                raise ValueError(f"{table}: 存在无效的事件或资产引用")
        expected = {name: [] for name in ("event_participants", "event_objects", "event_locations")}
        for row in self.connection.execute("SELECT task_id,data_json FROM events ORDER BY task_id"):
            identity, payload = row
            if identity not in seen_events:
                raise ValueError(f"events: 事件 {identity} 不属于任何正式事件组或编号重复")
            seen_events.remove(identity)
            event = loads(payload, location=f"events/{identity}")
            for position, person in enumerate(event["participants"], 1):
                expected["event_participants"].append((identity, position, person["role_code"], person.get("original_goal"), "role"))
            for table, field, kind in (("event_objects", "object_ids", "object"),
                                       ("event_locations", "location_ids", "location")):
                expected[table].extend((identity, i, key, kind) for i, key in enumerate(event[field], 1))
        if seen_events:
            raise ValueError(f"events: 缺少正式事件索引 {sorted(seen_events)}")
        fields = {"event_participants": "role_id,original_goal,role_type", "event_objects": "object_id,object_type",
                  "event_locations": "location_id,location_type"}
        for table, columns in fields.items():
            rows = self.connection.execute(f"SELECT task_id,position,{columns} FROM {table} ORDER BY task_id,position")
            if list(map(tuple, rows)) != expected[table]:
                raise ValueError(f"{table}: 查询索引与正式事件资料不一致")

    def assets(self, kind):
        if kind not in self.ASSET_KINDS:
            raise ValueError(f"未知资料类型：{kind}")
        return {row[0]: normalize_asset(loads(row[1], location=f"assets/{kind}/{row[0]}"),
                    kind=kind, location=f"assets/{kind}/{row[0]}") for row in self.connection.execute(
            "SELECT asset_id, data_json FROM assets WHERE asset_type=? ORDER BY asset_id", (kind,))}

    def worlds(self):
        return {row[0]: loads(row[1], location=f"world_data/{row[0]}", expected=(dict, list)) for row in self.connection.execute(
            "SELECT file_path, content_json FROM world_data ORDER BY file_path")}

    def nodes(self):
        rows = self.connection.execute("SELECT node_id, node_order, data_json FROM time_nodes ORDER BY node_order").fetchall()
        if not rows or [row["node_order"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("时间节点必须非空，并从 1 连续排序")
        nodes = []
        for row in rows:
            location = f"time_nodes/{row['node_id']}"
            node = loads(row["data_json"], location=location)
            if (type(node.get("order")) is not int
                    or (node.get("node_id"), node["order"]) != (row["node_id"], row["node_order"])):
                raise ValueError(f"{location}: 节点内容与数据库时间索引不一致")
            nodes.append(normalize_node(node))
        return nodes

    def groups(self, node_id):
        node = self.connection.execute("SELECT node_order FROM time_nodes WHERE node_id=?", (node_id,)).fetchone()
        if node is None:
            raise KeyError(f"节点不存在：{node_id}")
        rows = self.connection.execute(
            "SELECT group_id,group_order,data_json FROM event_groups WHERE node_id=? ORDER BY group_order",
            (node_id,)).fetchall()
        if not rows or rows[0]["group_order"] != 1:
            raise ValueError(f"节点 {node_id} 缺少从 1 开始的事件组")
        catalog = {kind: self.assets(kind) for kind in self.ASSET_KINDS}
        groups = []
        for row in rows:
            location = f"event_groups/{row['group_id']}"
            raw = loads(row["data_json"], location=location)
            if (raw.get("group_id"), raw.get("order")) != (row["group_id"], row["group_order"]):
                raise ValueError(f"{location}: 事件组内容与数据库索引不一致")
            group = self.parse_group(node_id, node["node_order"], raw, catalog, location=location)
            indexed = self.connection.execute(
                "SELECT task_id,data_json FROM events WHERE group_id=? AND node_id=? AND event_order=?",
                (group.group_id, node_id, group.order)).fetchall()
            events = {r["task_id"]: normalize_event(loads(r["data_json"], location=f"events/{r['task_id']}"),
                      location=f"events/{r['task_id']}", locations=catalog["location"], objects=catalog["object"]) for r in indexed}
            if len(group.events) != len(events) or {e["task_id"]: e for e in group.events} != events:
                raise ValueError(f"{location}: 事件组与事件索引不一致或仍使用旧事件契约")
            groups.append(group)
        return groups

    def snapshot(self, node_id):
        row = self.connection.execute("SELECT data_json FROM state_snapshots WHERE node_id=?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(f"节点快照不存在：{node_id}")
        return self.parse_snapshot(node_id, loads(row[0], location=f"state_snapshots/{node_id}"))

    def fingerprint(self):
        """对公开表内容绑定，包括 WAL 中可见记录；忽略历史附加表。"""
        digest = sha256()
        for table in self.TABLES:
            cursor = self.connection.execute(f'SELECT * FROM "{table}"')
            columns = [column[0] for column in cursor.description]
            rows = sorted(json.dumps(dict(zip(columns, row)), sort_keys=True, ensure_ascii=False)
                          for row in cursor)
            for value in (table, *rows):
                digest.update(value.encode("utf-8") + b"\n")
        return digest.hexdigest()


def open_repository(path):
    """打开一份只读的公开资料快照，支持 JSON 目录和已有 SQLite 数据库。"""
    if Path(path).expanduser().is_dir():
        from inputs.NovelRestoreJsonRepository import NovelRestoreJsonRepository
        return NovelRestoreJsonRepository(path)
    return NovelRestoreRepository(path)
