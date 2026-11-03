"""公开 repository 的选择入口、共用数据契约和只读 SQLite 适配器。"""
from hashlib import sha256
import json
from pathlib import Path

from common.NovelRestoreSchema import EventGroup, NodeSnapshot, WorldState
from contract.RepositoryReader import RepositoryReader
from contract.RepositoryState import replay_group, validate_state
from contract.NovelWorldStateSchema import require_state_sections
from contract.NovelWorldState import public_state


def validate_event(event, assets):
    """核对单个公开事件的字段以及人物、物品和地点引用。"""
    identity = event.get("task_id")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("缺少事件编号")
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
    grouped = True
    ASSET_KINDS = ("role", "object", "location", "rule", "organization")

    @staticmethod
    def parse_group(node_id, node_order, raw, catalog):
        if public_state(raw, False) != raw:
            raise ValueError("公开事件组不能包含原文证据")
        validate_state(raw.get("state_delta"), catalog, "事件组变化", delta=True)
        if "node_id" in raw or "node_order" in raw:
            raise ValueError("事件组不能覆盖所属节点")
        group = EventGroup(node_id=node_id, node_order=node_order, **raw)
        people = []
        for event in group.events:
            if "order" in event or "state_delta" in event or event.get("epistemic_status") != "事实":
                raise ValueError("公开事件须使用组级顺序、组级变化并已确认为事实")
            validate_event(event, catalog)
            people.extend(p["role_code"] for p in event["participants"])
        if len(people) != len(set(people)):
            raise ValueError("同组实际参与人物不能重复")
        return group

    @staticmethod
    def parse_snapshot(node_id, raw):
        if public_state(raw, False) != raw:
            raise ValueError("公开快照不能包含原文证据")
        require_state_sections(raw, "节点入口快照")
        snapshot = NodeSnapshot.model_validate(raw)
        if snapshot.node_id != node_id:
            raise ValueError("快照内容与节点身份不一致")
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
        for key, role in assets["role"].items():
            if {"age", "identity", "biography", "relation"} & role.keys():
                raise ValueError(f"{key}: 静态人物档案不能包含阶段身份、年龄、传记或关系")
        for key, item in assets["object"].items():
            if "significance" in item:
                raise ValueError(f"{key}: 剧情作用不属于静态物品档案")
        worlds = self.worlds()
        if public_state(assets, False) != assets or public_state(worlds, False) != worlds:
            raise ValueError("公开资料不能包含原文证据")

        seen_events, seen_groups = set(), set()
        previous, expected = None, None
        for node in self.nodes():
            key = node["node_id"]
            name = node.get("node_name")
            if not isinstance(name, str) or not name.strip() or not isinstance(node.get("time"), dict):
                raise ValueError(f"{key}: 缺少节点名称/time")
            snapshot = self.snapshot(key)
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
                state = replay_group(state, group.state_delta)
            previous, expected = key, state


class NovelRestoreRepository(PublicRepository, RepositoryReader):
    TABLES = (
        "repository_info", "world_data", "assets", "time_nodes", "event_groups", "events", "state_snapshots",
        "event_participants", "event_objects", "event_locations", "state_records", "state_changes", "state_references",
    )

    def __init__(self, database):
        self.path = Path(database).expanduser().resolve()
        super().__init__(self.path)
        try:
            self.connection.execute("BEGIN")  # 所有查询和指纹读取同一 SQLite 快照。
            tables = {row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if set(self.TABLES) - tables:
                raise ValueError("下游只接受新版六类状态及事件组 repository 数据库；旧库请保留")
            info = self.connection.execute("SELECT source, time_basis FROM repository_info").fetchall()
            if len(info) != 1 or info[0]["time_basis"] != "story":
                raise ValueError("repository 必须有唯一来源且采用故事时间")
            self.source = info[0]["source"]
        except Exception:
            self.close()
            raise

    def assets(self, kind):
        if kind not in self.ASSET_KINDS:
            raise ValueError(f"未知资料类型：{kind}")
        return {row[0]: json.loads(row[1]) for row in self.connection.execute(
            "SELECT asset_id, data_json FROM assets WHERE asset_type=? ORDER BY asset_id", (kind,))}

    def worlds(self):
        return {row[0]: json.loads(row[1]) for row in self.connection.execute(
            "SELECT file_path, content_json FROM world_data ORDER BY file_path")}

    def nodes(self):
        rows = self.connection.execute("SELECT node_id, node_order, data_json FROM time_nodes ORDER BY node_order").fetchall()
        if not rows or [row["node_order"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("时间节点必须非空，并从 1 连续排序")
        nodes = []
        for row in rows:
            node = json.loads(row["data_json"])
            if (type(node.get("order")) is not int
                    or (node.get("node_id"), node["order"]) != (row["node_id"], row["node_order"])):
                raise ValueError("节点内容与数据库时间索引不一致")
            nodes.append(node)
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
            raw = json.loads(row["data_json"])
            if (raw.get("group_id"), raw.get("order")) != (row["group_id"], row["group_order"]):
                raise ValueError("事件组内容与数据库索引不一致")
            group = self.parse_group(node_id, node["node_order"], raw, catalog)
            indexed = self.connection.execute(
                "SELECT task_id,data_json FROM events WHERE group_id=? AND node_id=? AND event_order=?",
                (group.group_id, node_id, group.order)).fetchall()
            events = {r["task_id"]: json.loads(r["data_json"]) for r in indexed}
            if len(group.events) != len(events) or {e["task_id"]: e for e in group.events} != events:
                raise ValueError("事件组与事件索引不一致或仍使用旧事件契约")
            groups.append(group)
        return groups

    def snapshot(self, node_id):
        row = self.connection.execute("SELECT data_json FROM state_snapshots WHERE node_id=?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(f"节点快照不存在：{node_id}")
        return self.parse_snapshot(node_id, json.loads(row[0]))

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
        from common.NovelRestoreJsonRepository import NovelRestoreJsonRepository
        return NovelRestoreJsonRepository(path)
    return NovelRestoreRepository(path)
