"""四类状态的共用操作：稀疏合并、引用校验、确定性回放。无模型或数据库依赖。"""
from .NovelWorldStateSchema import (
    AssetState, StateDelta, STATE_KINDS,  require_state_sections,
)


def relation_key(row):
    return tuple(getattr(row, key) for key in ('subject_role_id', 'related_role_id', 'relationship_type'))


def state_rows(value):
    """统一迭代快照/变化，返回类别、稳定标识、记录。"""
    snapshot = isinstance(value, AssetState)
    for kind, (identity, _) in STATE_KINDS.items():
        rows = getattr(value, kind)
        if snapshot and kind != 'relations':
            yield from ((kind, key, row) for key, row in rows.items())
        else:
            yield from ((kind, relation_key(row) if identity is None else getattr(row, identity), row)
                        for row in rows)


def set_rows(value, kind, rows):
    setattr(value, kind, rows)


def merge_deltas(deltas):
    merged, errors = {}, []
    for delta in deltas:
        if delta is None:
            continue
        batch_seen = set()
        for kind, key, row in state_rows(delta):
            if (kind,key) in batch_seen:
                raise ValueError(f"{kind}/{key}: 同一变化交付不得重复对象")
            batch_seen.add((kind,key))
            target = kind, key
            if target not in merged:
                merged[target] = row.model_copy(deep=True)
                continue
            saved = merged[target]
            if kind == 'relations' and (saved.operation != row.operation or row.operation != 'update'):
                raise ValueError(f'{key}: 同批关系操作冲突或重复')
            before, incoming = saved.set.model_dump(mode='json'), row.set.model_dump(mode='json')
            conflicts = {name: [before[name], incoming[name]] for name in sorted(before.keys() & incoming.keys())
                         if before[name] != incoming[name]}
            if conflicts:
                errors.append(f'state_delta.{kind}.{key}: 同时更新冲突（字段及两个值）{conflicts}；修正变化或同时分组')
                continue
            saved.set = type(saved.set).model_validate({**before, **incoming})
            saved.basis = '\n'.join(dict.fromkeys((saved.basis, row.basis)))
    if errors:
        raise ValueError('\n'.join(errors))
    # 程序已构造全部类别；保留此交付范围供后续局部修订复验，不借此补全模型原始输入。
    result = StateDelta(characters=[], objects=[], locations=[], relations=[])
    for kind in STATE_KINDS:
        set_rows(result, kind, [row for (category, _), row in sorted(merged.items()) if category == kind])
    return result


def apply_delta(state, delta):
    """整组在同一前态上合并后应用；异常不会修改原快照。列表/嵌套字段整体替换。"""
    delta = merge_deltas([delta])
    records = {(kind, key): row.model_dump(mode='json') for kind, key, row in state_rows(state)}
    for kind, key, row in state_rows(delta):
        before = records.get((kind, key))
        if kind == 'relations':
            if row.operation == 'add' and before is not None:
                raise ValueError(f'{key}: 新增关系已经存在')
            if row.operation != 'add' and before is None:
                raise ValueError(f'{key}: 修改或删除的关系不存在')
            if row.operation == 'remove':
                del records[kind, key]
                continue
        elif before is None and kind != 'objects':
            raise ValueError(f'{kind}/{key}: 未登记对象')
        if before is None and kind == 'objects':
            before = {field:None for field in STATE_KINDS['objects'][1].model_fields}
            before['position'] = {'location_id':None, 'position_within_place':None}
        identity = dict(zip(('subject_role_id','related_role_id','relationship_type'), key)) if kind == 'relations' else {}
        records[kind, key] = {**(before or {}), **identity, **row.set.model_dump(mode='json')}
    result = state.model_copy(deep=True)
    for kind, (_, model) in STATE_KINDS.items():
        rows = {key: model.model_validate(row) for (category, key), row in records.items() if category == kind}
        set_rows(result, kind, [rows[key] for key in sorted(rows)] if kind == 'relations' else rows)
    validate_physical_links(result)
    return result


def record_references(value, path=''):
    """遍历显式引用并返回路径/类型/ID；不从名称或描述推断关联。"""
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json')
    typed = {'subject_role_id': 'role', 'related_role_id': 'role', 'role_id': 'role',
             'object_id': 'object', 'container_id': 'object', 'location_id': 'location',
             'parent_location_id': 'location', 'entity_id': 'controller'}
    lists = {'owner_ids': 'controller', 'custodian_ids': 'controller',
             'related_entity_ids': 'entity', 'authority_ids': 'controller'}
    if isinstance(value, dict):
        for key, item in value.items():
            at = f'{path}/{key}'
            if key in typed and isinstance(item, str):
                yield at, typed[key], item
            elif key in lists and isinstance(item, list):
                if len(set(item)) != len(item):
                    raise ValueError(f'{at}: 引用编号重复')
                yield from ((f'{at}/{i}', lists[key], ref) for i, ref in enumerate(item))
            else:
                yield from record_references(item, at)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from record_references(item, f'{path}/{i}')


def public_state(value, include_evidence=True):
    """重建公开 JSON 容器并移除私有材料，不修改输入或共享可变容器。"""
    excluded = {'unverified_quote'} | (set() if include_evidence else {'evidence'})
    def visit(item):
        if isinstance(item, dict):
            return {key: visit(row) for key, row in item.items() if key not in excluded}
        if isinstance(item, list):
            return [visit(row) for row in item]
        return item
    return visit(value.model_dump(mode='json') if hasattr(value, 'model_dump') else value)


def state_reference_errors(value, catalogs, label):
    catalogs = {**catalogs, 'controller': set(catalogs.get('role', ())) | set(catalogs.get('organization', ())),
                'entity': set().union(*(set(catalogs.get(k, ())) for k in ('role','object','location','organization')))}
    errors, seen, known_types = [], set(), {}
    for kind, ids in catalogs.items():
        for key in ids:
            known_types.setdefault(key, set()).add(kind)

    def check_reference(category, key, at, noun='编号'):
        if category not in catalogs or key in catalogs[category]:
            return
        actual = known_types.get(key)
        if actual:
            errors.append(f'{at}: 编号 {key} 已登记为{sorted(actual)}，此处需要{category}')
        else:
            errors.append(f'{at}: 未知{category}{noun} {key}')

    for kind, key, row in state_rows(value):
        at = f'{label}.{kind}.{key}'
        identity = (kind, key)
        if identity in seen:
            errors.append(f'{at}: 重复状态记录，须合成一项明确更新')
        seen.add(identity)
        category = {'characters': 'role', 'objects': 'object', 'locations': 'location',
                    'organizations': 'organization', 'rules': 'rule'}.get(kind)
        check_reference(category, key, at)
        if kind == 'relations' and key[0] == key[1]:
            errors.append(f'{at}: 关系端点须为两个已知人物')
        for path, target_type, target in record_references(row):
            check_reference(target_type, target, f'{at}{path}', '引用')
    if isinstance(value, AssetState):
        for person, row in value.characters.items():
            if row.life_status == 'not_born' and (row.position.location_id or row.current_skills or row.action_terms):
                errors.append(f'{person}: 未出生者不能拥有当前位置、技能或行动条款')
    return errors


def validate_state(raw, assets, label, delta=False, comprehensive=True):
    if not isinstance(raw, dict):
        raise ValueError(f'{label}: 状态尚未交付')
    model = StateDelta if delta else AssetState
    if comprehensive:
        require_state_sections(raw, label)
    parsed = model.model_validate(raw)
    catalogs = {kind: assets.get(kind, ()) for kind in ('role', 'object', 'location', 'organization', 'rule')}
    errors = state_reference_errors(parsed, catalogs, label)
    if not delta:
        for kind, catalog in (('characters', 'role'), ('objects', 'object'), ('locations', 'location')):
            if (comprehensive or kind == 'objects') and set(getattr(parsed, kind)) != set(assets[catalog]):
                errors.append(f'{label}.{kind}: 快照须覆盖全部已提取对象')
    if errors:
        raise ValueError('\n'.join(errors))
    if not delta:
        validate_physical_links(parsed, assets.get("location") if isinstance(assets.get("location"), dict) else None)
    return parsed


def replay_group(state, delta):
    return apply_delta(AssetState.model_validate(state), StateDelta.model_validate(delta))


def validate_physical_links(state, places=None):
    """容器结构与已知位置必须自洽；不从未知位置推断事实。"""
    for identity, obj in state.objects.items():
        seen = {identity}
        current = obj
        while current.container_id:
            container = current.container_id
            if container in seen or container not in state.objects:
                raise ValueError(f'{identity}: 容器循环或不存在 {container}')
            seen.add(container)
            parent = state.objects[container]
            if (places is not None and current.position.location_id and parent.position.location_id
                    and current.position.location_id not in location_ancestors(places, [parent.position.location_id])
                    and parent.position.location_id not in location_ancestors(places, [current.position.location_id])):
                raise ValueError(f'{identity}: 物品与容器地点不一致')
            current = parent
        if obj.quantity and isinstance(obj.quantity.amount, (int,float)) and obj.quantity.amount < 0:
            raise ValueError(f'{identity}: 数量不能为负')


def location_ancestors(places, identities):
    result = set()
    for identity in identities:
        seen = set()
        while identity:
            if identity in seen or identity not in places:
                raise ValueError(f'地点上级不存在或层级循环：{identity}')
            seen.add(identity)
            result.add(identity)
            identity = places[identity].get('parent_location_id')
    return result
