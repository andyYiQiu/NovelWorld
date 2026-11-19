"""只读查询公开 repository 数据库；按已保存的顺序回放状态，不推断小说事实。"""
from contextlib import suppress
from pathlib import Path
import sqlite3

from tools.JsonInput import loads, validate
from tools.RuntimeFiles import check_sqlite_files
from inputs.NovelWorldState import public_state, replay_group
from inputs.NovelWorldStateSchema import AssetState, StateDelta
from .RepositoryInput import normalize_asset, normalize_event, normalize_group, normalize_state_delta, normalize_node
from .RepositoryState import StateHistoryReader


class RepositoryReader(StateHistoryReader):
    """用 with 管理连接；数据库必须已经存在，不创建文件或执行写入。"""

    def __init__(self, database):
        path = Path(database).expanduser().resolve()
        check_sqlite_files(path)
        uri = path.as_uri() + "?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        try:
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA query_only = ON")
            self.connection.execute("BEGIN")  # 指纹、按需历史索引及其他查询共享只读快照。
        except BaseException:
            with suppress(sqlite3.Error):
                self.connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.connection.close()

    def _rows(self, sql, parameters=()):
        return [dict(row) for row in self.connection.execute(sql, parameters)]

    def _event(self, task_id):
        rows = self._rows("SELECT * FROM events WHERE task_id=?", (task_id,))
        if not rows:
            raise KeyError(f"事件不存在：{task_id}")
        return rows[0]

    def asset(self, kind, identity):
        row = self.connection.execute(
            "SELECT data_json FROM assets WHERE asset_type=? AND asset_id=?", (kind, identity)).fetchone()
        if row is None:
            raise KeyError(f"资产不存在：{kind}/{identity}")
        return normalize_asset(loads(row[0], location=f"assets/{kind}/{identity}"),
                               kind=kind, location=f"assets/{kind}/{identity}")

    def background_paths(self):
        return [r[0] for r in self.connection.execute("SELECT file_path FROM world_data ORDER BY file_path")]

    def background(self, path):
        row = self.connection.execute("SELECT content_json FROM world_data WHERE file_path=?", (path,)).fetchone()
        if row is None:
            raise KeyError(f"世界背景不存在：{path}")
        return loads(row[0], location=f"world_data/{path}", expected=(dict, list))

    def get_state_after(self, task_id):
        event = self._event(task_id)
        return self.get_state_after_group(event["node_id"], event["event_order"])

    def event_ids_for_asset(self, kind, identity):
        if kind == "object":
            sql = "SELECT DISTINCT task_id FROM event_objects WHERE object_id=? ORDER BY task_id"
        elif kind == "location":
            sql = "SELECT DISTINCT task_id FROM event_locations WHERE location_id=? ORDER BY task_id"
        else:
            raise ValueError("事件引用查询只支持物品与地点")
        return [r[0] for r in self.connection.execute(sql, (identity,))]

    def get_event_context(self, task_id):
        """返回原事件、所属节点、按原数组顺序排列的完整人物/物品/地点档案。"""
        event = self._event(task_id)
        nodes = self._rows("SELECT data_json FROM time_nodes WHERE node_id=?", (event["node_id"],))
        if not nodes:
            raise ValueError(f"events: {task_id} 引用不存在的节点 {event['node_id']}")
        node = nodes[0]
        people = self._rows("""
            SELECT p.position, p.original_goal, a.data_json
            FROM event_participants p JOIN assets a
              ON a.asset_type=p.role_type AND a.asset_id=p.role_id
            WHERE p.task_id=? ORDER BY p.position""", (task_id,))
        objects = self._rows("""
            SELECT a.data_json FROM event_objects o JOIN assets a
              ON a.asset_type=o.object_type AND a.asset_id=o.object_id
            WHERE o.task_id=? ORDER BY o.position""", (task_id,))
        locations = self._rows("""
            SELECT a.data_json FROM event_locations l JOIN assets a
              ON a.asset_type=l.location_type AND a.asset_id=l.location_id
            WHERE l.task_id=? ORDER BY l.position""", (task_id,))
        result = {
            "event": normalize_event(loads(event["data_json"], location=f"events/{task_id}"),
                                     location=f"events/{task_id}", locations=self.assets("location"), objects=self.assets("object")),
            "node": normalize_node(loads(node["data_json"], location=f"time_nodes/{event['node_id']}")),
            "participants": [{"position": row["position"], "original_goal": row["original_goal"],
                              "role": normalize_asset(loads(row["data_json"]), kind="role",
                                  location=f"events/{task_id}/participants/{row['position']}")} for row in people],
            "objects": [normalize_asset(loads(row["data_json"]), kind="object",
                location=f"events/{task_id}/objects/{i}") for i, row in enumerate(objects)],
            "locations": [normalize_asset(loads(row["data_json"]), kind="location",
                location=f"events/{task_id}/locations/{i}") for i, row in enumerate(locations)],
        }

        result['group'] = self.get_event_group(event['group_id'])
        if event['event_order'] != result['group'].get('order') or result['event'] not in result['group']['events']:
            raise ValueError(f"events: {task_id} 与所属事件组不一致")
        return result

    def get_role_events(self, role_id):
        """只查实际参与的事件；同一事件重复引用人物时仍只返回一条事件。"""
        rows = self._rows("""
            SELECT n.node_order, e.task_id, e.data_json FROM events e
            JOIN time_nodes n ON n.node_id=e.node_id
            WHERE EXISTS (SELECT 1 FROM event_participants p
                          WHERE p.task_id=e.task_id AND p.role_id=?)
            ORDER BY n.node_order, e.event_order, e.task_id""", (role_id,))
        # task_id 仅稳定展示同组结果，不表示组内发生先后。
        locations, objects = self.assets("location"), self.assets("object")
        return [{"node_order": row["node_order"],
                 "event": normalize_event(loads(row["data_json"], location=f"events/{row['task_id']}"),
                                          location=f"events/{row['task_id']}", locations=locations, objects=objects)} for row in rows]

    def _history_sources(self):
        snapshots = self.connection.execute("""SELECT n.node_id,n.node_order,s.data_json FROM time_nodes n
            JOIN state_snapshots s ON s.node_id=n.node_id ORDER BY n.node_order""")
        for node_id, node_order, raw in snapshots:
            location = f"state_snapshots/{node_id}"
            yield {"node_id": node_id, "node_order": node_order}, validate(AssetState, loads(raw, location=location), location=location)
        groups = self.connection.execute("""SELECT n.node_id,n.node_order,g.group_id,g.group_order,g.data_json
            FROM time_nodes n JOIN event_groups g ON g.node_id=n.node_id ORDER BY n.node_order,g.group_order""")
        for node_id, node_order, group_id, group_order, raw in groups:
            location = f"event_groups/{group_id}"
            delta = normalize_group(loads(raw, location=location), location=location)["state_delta"]
            yield ({"node_id": node_id, "node_order": node_order, "group_id": group_id, "group_order": group_order},
                   validate(StateDelta, delta, location=location + "/state_delta"))

    def get_event_group(self, group_id):
        rows = self._rows('SELECT data_json FROM event_groups WHERE group_id=?', (group_id,))
        if not rows:
            raise KeyError(f'事件组不存在：{group_id}')
        location = f"event_groups/{group_id}"
        return normalize_group(loads(rows[0]['data_json'], location=location), location=location,
                               locations=self.assets("location"), objects=self.assets("object"))

    def get_state_before(self, task_id):
        """节点入口状态加本节点更早 order 组的 delta；同组事件共享前态。"""
        event = self._event(task_id)
        return self._state_through(event["node_id"], event["event_order"] - 1)

    def get_state_after_group(self, node_id, event_order):
        """返回一个同时开始组之后的状态，不为单个同时事件制造先后关系。"""
        exists = self._rows("SELECT 1 FROM events WHERE node_id=? AND event_order=?", (node_id, event_order))
        if not exists:
            raise KeyError(f"事件组不存在：{node_id}/{event_order}")
        return self._state_through(node_id, event_order)

    def _state_through(self, node_id, last_order):
        rows = self._rows('SELECT data_json FROM state_snapshots WHERE node_id=?', (node_id,))
        if not rows:
            raise KeyError(f'节点快照不存在：{node_id}')
        location = f"state_snapshots/{node_id}"
        state = validate(AssetState, loads(rows[0]['data_json'], location=location), location=location)
        groups = self._rows('SELECT group_id,data_json FROM event_groups WHERE node_id=? AND group_order<=? ORDER BY group_order',
                            (node_id, last_order))
        for group in groups:
            location = f"event_groups/{group['group_id']}"
            delta = validate(StateDelta, normalize_group(loads(group['data_json'], location=location), location=location)['state_delta'],
                             location=location + "/state_delta")
            state = replay_group(state, delta)
        return public_state(state, False)
