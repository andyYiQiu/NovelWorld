"""校验公开库与模拟运行状态中的引用，并回放事件组的客观变化。"""
import json

from .NovelWorldStateSchema import AssetState, StateDelta, require_state_sections
from .NovelWorldState import apply_delta, record_references, state_rows


def state_key(key):
    return json.dumps(list(key), ensure_ascii=False, allow_nan=False) if isinstance(key, tuple) else key


def state_reference_errors(value, catalogs, label):
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
        fields = row.set if isinstance(value, StateDelta) else row
        for field in ('owner', 'holder', 'controller', 'enforcer', 'subject', 'authority', 'container', 'location'):
            ref = getattr(fields, field, None)
            if ref is None:
                continue
            if field == 'location':
                ref = ref.location
            allowed = {'location'} if field == 'location' else {'object'} if field == 'container' else {'role', 'organization'}
            if any(item.type not in allowed for item in ref.references):
                errors.append(f'{at}.{field}: 只允许引用{sorted(allowed)}')
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
    return parsed


def replay_group(state, delta):
    return apply_delta(AssetState.model_validate(state), StateDelta.model_validate(delta))
