"""从公开节点快照和事件组变化生成只读历史索引。"""
from copy import deepcopy
from functools import cached_property
import json

from inputs.NovelWorldStateSchema import AssetState
from inputs.NovelWorldState import public_state, state_rows


def state_key(key):
    return json.dumps(list(key), ensure_ascii=False, allow_nan=False) if isinstance(key, tuple) else key


class StateHistoryReader:
    """历史是正式节点快照和组变化的投影。只在首次查询时构建，绑定本次只读视图。"""

    @cached_property
    def _history_index(self):
        records = {}
        for position, state in self._history_sources():
            for category, key, row in state_rows(state):
                history = records.setdefault((category, state_key(key)), {"snapshots": [], "changes": []})
                if isinstance(state, AssetState):
                    history["snapshots"].append({**position, "state": public_state(row, False)})
                else:
                    if category == "relations":
                        history["changes"].append({**position, "operation": row.operation})
                    for field, value in sorted((public_state(row.set, False) or {}).items()):
                        history["changes"].append({**position, "field": field, "value": value, "basis": row.basis})
        return records

    def get_state_history(self, category, record_key):
        key = state_key(tuple(record_key) if isinstance(record_key, list) else record_key)
        return deepcopy(self._history_index.get((category, key), {"snapshots": [], "changes": []}))

    def get_object_history(self, object_id):
        return {"object": self.asset("object", object_id), **self.get_state_history("objects", object_id)}
