"""只读查询公开 repository 数据库；按已保存的顺序回放状态，不推断小说事实。"""
import json
from pathlib import Path
import sqlite3


class RepositoryReader:
    """用 with 管理连接；数据库必须已经存在，不创建文件或执行写入。"""

    def __init__(self, database):
        uri = Path(database).expanduser().resolve().as_uri() + "?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only = ON")

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
        return json.loads(row[0])

    def background_paths(self):
        return [r[0] for r in self.connection.execute("SELECT file_path FROM world_data ORDER BY file_path")]

    def background(self, path):
        row = self.connection.execute("SELECT content_json FROM world_data WHERE file_path=?", (path,)).fetchone()
        if row is None:
            raise KeyError(f"世界背景不存在：{path}")
        return json.loads(row[0])

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
        node = self._rows("SELECT data_json FROM time_nodes WHERE node_id=?", (event["node_id"],))[0]
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
            "event": json.loads(event["data_json"]),
            "node": json.loads(node["data_json"]),
            "participants": [{"position": row["position"], "original_goal": row["original_goal"],
                              "role": json.loads(row["data_json"])} for row in people],
            "objects": [json.loads(row["data_json"]) for row in objects],
            "locations": [json.loads(row["data_json"]) for row in locations],
        }

        result['group'] = self.get_event_group(event['group_id'])
        return result

    def get_role_events(self, role_id):
        """只查实际参与的事件；同一事件重复引用人物时仍只返回一条事件。"""
        rows = self._rows("""
            SELECT n.node_order, e.data_json FROM events e
            JOIN time_nodes n ON n.node_id=e.node_id
            WHERE EXISTS (SELECT 1 FROM event_participants p
                          WHERE p.task_id=e.task_id AND p.role_id=?)
            ORDER BY n.node_order, e.event_order, e.task_id""", (role_id,))
        # task_id 仅稳定展示同组结果，不表示组内发生先后。
        return [{"node_order": row["node_order"], "event": json.loads(row["data_json"])} for row in rows]

    def get_object_history(self, object_id):
        """返回物品档案、各节点入口状态和逐字段变化，不混淆持有者与所有者。"""
        history = self.get_state_history('objects', object_id)
        assets = self._rows("SELECT data_json FROM assets WHERE asset_type='object' AND asset_id=?", (object_id,))
        if not assets:
            raise KeyError(f"物品不存在：{object_id}")
        return {'object': json.loads(assets[0]['data_json']), **history}

    def get_state_history(self, category, record_key):
        """按类别/稳定ID查快照和组级变化；关系使用[owner,other,relation]作为键。"""
        from .RepositoryState import state_key
        record_key = state_key(tuple(record_key) if isinstance(record_key, list) else record_key)
        snapshots = self._rows("""
            SELECT n.node_order, r.node_id, r.data_json FROM state_records r
            JOIN time_nodes n ON n.node_id=r.node_id
            WHERE r.category=? AND r.record_key=? ORDER BY n.node_order""", (category, record_key))
        changes = self._rows("""
            SELECT n.node_order,g.node_id,g.group_id,g.group_order,c.field,c.value_json,c.basis
            FROM state_changes c JOIN event_groups g ON g.group_id=c.group_id
            JOIN time_nodes n ON n.node_id=g.node_id
            WHERE c.category=? AND c.record_key=? ORDER BY n.node_order,g.group_order,c.field""", (category, record_key))
        return {'snapshots': [{**{k: v for k, v in row.items() if k != 'data_json'}, 'state': json.loads(row['data_json'])} for row in snapshots],
                'changes': [{**{k: v for k, v in row.items() if k != 'value_json'}, 'value': json.loads(row['value_json'])} for row in changes]}

    def get_event_group(self, group_id):
        rows = self._rows('SELECT data_json FROM event_groups WHERE group_id=?', (group_id,))
        if not rows:
            raise KeyError(f'事件组不存在：{group_id}')
        return json.loads(rows[0]['data_json'])

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
        from .RepositoryState import replay_group
        from .NovelWorldState import public_state
        from .NovelWorldStateSchema import AssetState
        rows = self._rows('SELECT data_json FROM state_snapshots WHERE node_id=?', (node_id,))
        if not rows:
            raise KeyError(f'节点快照不存在：{node_id}')
        state = AssetState.model_validate(json.loads(rows[0]['data_json']))
        groups = self._rows('SELECT data_json FROM event_groups WHERE node_id=? AND group_order<=? ORDER BY group_order',
                            (node_id, last_order))
        for group in groups:
            state = replay_group(state, json.loads(group['data_json'])['state_delta'])
        return public_state(state, False)
