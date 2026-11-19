"""直接读取公开 JSON 文件夹，在同一份只读快照上查询资料与故事状态。"""
from copy import deepcopy
from pathlib import Path

from inputs.NovelRestoreRepository import PublicRepository
from inputs.JsonRepositoryData import load_json_repository
from inputs.NovelWorldState import public_state, replay_group
from inputs.NovelWorldStateSchema import AssetState
from inputs.RepositoryState import StateHistoryReader


class NovelRestoreJsonRepository(PublicRepository, StateHistoryReader):
    def __init__(self, path):
        self._data = load_json_repository(path)
        self.path = Path(path).expanduser().resolve()
        self.source = self._data.source
        self._groups = {}
        self._events = {}
        self._by_node = {}
        self._snapshots = {}
        for node in self._data.nodes:
            node_id = node["node_id"]
            self._snapshots[node_id] = self.parse_snapshot(node_id, self._data.snapshots[node_id],
                                                         location=f"timeline/{node['state_path']}")
            groups = self._by_node[node_id] = []
            for raw in self._data.groups[node_id]:
                group = self.parse_group(node_id, node["order"], raw, self._data.assets,
                                         location=f"timeline/{node['path']}/{raw['group_id']}")
                self._groups[group.group_id] = {**{k:v for k,v in raw.items() if k != "state_changes"}, "events": group.events,
                    "state_delta": group.state_delta.model_dump(mode="json", exclude_unset=True)}
                groups.append(group)
                for event in group.events:
                    self._events[event["task_id"]] = (node, group, event)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """文件已在载入快照时关闭。"""

    def fingerprint(self):
        return self._data.fingerprint

    def asset(self, kind, identity):
        return deepcopy(self._data.assets[kind][identity])

    def assets(self, kind):
        if kind not in self.ASSET_KINDS:
            raise ValueError(f"未知资料类型：{kind}")
        return {key: self.asset(kind, key) for key in sorted(self._data.assets[kind])}

    def background_paths(self):
        return sorted(self._data.worlds)

    def background(self, path):
        return deepcopy(self._data.worlds[path])

    def worlds(self):
        return {path: self.background(path) for path in self.background_paths()}

    def nodes(self):
        return deepcopy(self._data.nodes)

    def groups(self, node_id):
        return [group.model_copy(deep=True) for group in self._by_node[node_id]]

    def snapshot(self, node_id):
        return self._snapshots[node_id].model_copy(deep=True)

    def get_event_group(self, group_id):
        return deepcopy(self._groups[group_id])

    def get_event_context(self, task_id):
        node, group, event = self._events[task_id]
        return {
            "event": deepcopy(event), "node": deepcopy(node), "group": self.get_event_group(group.group_id),
            "participants": [{"position": i, "original_goal": p.get("original_goal"),
                              "role": self.asset("role", p["role_code"])}
                             for i, p in enumerate(event["participants"], 1)],
            "objects": [self.asset("object", key) for key in event["object_ids"]],
            "locations": [self.asset("location", key) for key in event["location_ids"]],
        }

    def get_role_events(self, role_id):
        events = [(node["order"], group.order, event_id, event)
                  for event_id, (node, group, event) in self._events.items()
                  if any(p["role_code"] == role_id for p in event["participants"])]
        return [{"node_order": order, "event": deepcopy(event)} for order, _, _, event in sorted(events)]

    def event_ids_for_asset(self, kind, identity):
        if kind not in ("object", "location"):
            raise ValueError("事件引用查询只支持物品与地点")
        field = kind + "_ids"
        return sorted(key for key, (_, _, event) in self._events.items() if identity in event[field])

    def get_state_before(self, task_id):
        node, group, _ = self._events[task_id]
        return self._state_through(node["node_id"], group.order - 1)

    def get_state_after(self, task_id):
        node, group, _ = self._events[task_id]
        return self._state_through(node["node_id"], group.order)

    def get_state_after_group(self, node_id, event_order):
        if not any(group.order == event_order for group in self._by_node[node_id]):
            raise KeyError(f"事件组不存在：{node_id}/{event_order}")
        return self._state_through(node_id, event_order)

    def _state_through(self, node_id, last_order):
        state = AssetState.model_validate(self._data.snapshots[node_id])
        for group in self._by_node[node_id]:
            if group.order > last_order:
                break
            state = replay_group(state, group.state_delta)
        return public_state(state, False)

    def _history_sources(self):
        for node in self._data.nodes:
            node_id = node["node_id"]
            position = {"node_id": node_id, "node_order": node["order"]}
            yield position, self.snapshot(node_id)
            for group in self._by_node[node_id]:
                yield {**position, "group_id": group.group_id, "group_order": group.order}, group.state_delta
