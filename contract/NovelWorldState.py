"""六类状态的共用操作：稀疏合并、引用遍历、确定性回放。无模型或数据库依赖。"""
from copy import deepcopy

from .NovelWorldStateSchema import AssetState, StateDelta, STATE_KINDS, PUBLIC_KINDS


def relation_key(row):
    return tuple(getattr(row, key) for key in ('owner_id', 'other_id', 'relation'))


def state_rows(value):
    """统一迭代快照/变化，返回类别、稳定标识、记录。"""
    snapshot = isinstance(value, AssetState)
    for kind, (identity, _) in STATE_KINDS.items():
        rows = getattr(value.public_state if kind in PUBLIC_KINDS else value, kind)
        if snapshot and kind != 'relations':
            yield from ((kind, key, row) for key, row in rows.items())
        else:
            yield from ((kind, relation_key(row) if identity is None else getattr(row, identity), row)
                        for row in rows)


def set_rows(value, kind, rows):
    setattr(value.public_state if kind in PUBLIC_KINDS else value, kind, rows)


def unique_evidence(rows):
    records = {(row.chunk, tuple(row.passage_ids), row.quote): row for row in rows}
    return [records[key] for key in sorted(records)]


def merge_deltas(deltas):
    merged, errors = {}, []
    for delta in deltas:
        if delta is None:
            continue
        for kind, key, row in state_rows(delta):
            target = kind, key
            if target not in merged:
                merged[target] = row.model_copy(deep=True)
                continue
            saved = merged[target]
            before, incoming = saved.set.model_dump(mode='json'), row.set.model_dump(mode='json')
            conflicts = {name: [before[name], incoming[name]] for name in sorted(before.keys() & incoming.keys())
                         if before[name] != incoming[name]}
            if conflicts:
                errors.append(f'state_delta.{kind}.{key}: 同时更新冲突（字段及两个值）{conflicts}；修正变化或同时分组')
                continue
            saved.set = type(saved.set).model_validate({**before, **incoming})
            saved.evidence = unique_evidence([*saved.evidence, *row.evidence])
            saved.basis = '\n'.join(dict.fromkeys((saved.basis, row.basis)))
    if errors:
        raise ValueError('\n'.join(errors))
    # 程序已构造全部类别；保留此交付范围供后续局部修订复验，不借此补全模型原始输入。
    result = StateDelta(characters=[], objects=[], locations=[], relations=[], constraints=[],
                        public_state={'environment': [], 'rules': [], 'organizations': []})
    for kind in STATE_KINDS:
        set_rows(result, kind, [row for (category, _), row in sorted(merged.items()) if category == kind])
    return result


def apply_delta(state, delta):
    """整组在同一前态上合并后应用；异常不会修改原快照。列表/嵌套字段整体替换。"""
    delta = merge_deltas([delta])
    records = {(kind, key): row.model_dump(mode='json') for kind, key, row in state_rows(state)}
    for kind, key, row in state_rows(delta):
        before = records.get((kind, key), {})
        identity = dict(zip(('owner_id', 'other_id', 'relation'), key)) if kind == 'relations' else {}
        candidate = {**before, **identity, **row.set.model_dump(mode='json'),
                     'evidence': before.get('evidence', []) + [e.model_dump(mode='json') for e in row.evidence]}
        parsed = STATE_KINDS[kind][1].model_validate(candidate)
        parsed.evidence = unique_evidence(parsed.evidence)
        records[kind, key] = parsed.model_dump(mode='json')
    result = state.model_copy(deep=True)
    for kind, (_, model) in STATE_KINDS.items():
        rows = {key: model.model_validate(row) for (category, key), row in records.items() if category == kind}
        set_rows(result, kind, [rows[key] for key in sorted(rows)] if kind == 'relations' else rows)
    return result


def record_references(value, path=''):
    """遍历显式引用并返回路径/类型/ID；不从名称或描述推断关联。"""
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json')
    typed = {'owner_id': 'role', 'other_id': 'role', 'role_id': 'role',
             'object_id': 'object', 'location_id': 'location', 'organization_id': 'organization'}
    lists = {'role_ids': 'role', 'object_ids': 'object', 'location_ids': 'location',
             'organization_ids': 'organization', 'rule_ids': 'rule'}
    if isinstance(value, dict):
        if set(value) == {'type', 'id'}:
            yield path, value['type'], value['id']
            return
        for key, item in value.items():
            at = f'{path}/{key}'
            if key in typed and isinstance(item, str):
                yield at, typed[key], item
            elif key in lists and isinstance(item, list):
                yield from ((f'{at}/{i}', lists[key], ref) for i, ref in enumerate(item))
            elif key != 'evidence':
                yield from record_references(item, at)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from record_references(item, f'{path}/{i}')


def public_state(value, include_evidence=True):
    excluded = {'unverified_quote'} | (set() if include_evidence else {'evidence'})
    def visit(item):
        if isinstance(item, dict):
            return {key: visit(row) for key, row in item.items() if key not in excluded}
        if isinstance(item, list):
            return [visit(row) for row in item]
        return item
    return visit(value.model_dump(mode='json') if hasattr(value, 'model_dump') else deepcopy(value))
