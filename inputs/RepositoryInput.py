"""公开输入的等价可选表示归一化；不补必需字段或故事内容。"""

def normalize_state_delta(raw, *, location):
    """1.4 完整 after_values 转为内部稀疏变化；继承不等于未知或清空。"""
    if not isinstance(raw, dict) or 'character_changes' not in raw:
        return raw
    from inputs.RepositoryFormat import Changes, KINDS
    data = Changes.model_validate(raw).model_dump(mode='json')
    result = {}
    for kind, (_, name, identity, _) in KINDS.items():
        result[kind] = []
        for row in data[name]:
            values = {k:v for k,v in row['after_values'].items() if v != {'inherit': True}}
            if row.get('operation') == 'remove' and values:
                raise ValueError(f'{location}/{name}: 删除关系的 after_values 须全部继承')
            if not values and row.get('operation') != 'remove':
                raise ValueError(f'{location}/{name}: 变化须至少更新一个属性')
            result[kind].append({**{k:v for k,v in row.items() if k != 'after_values'},
                                 'set': None if row.get('operation') == 'remove' else values})
    return result


def normalize_asset(raw, *, kind, location):
    """固定输入字段先校验；不将缺失或错误类型静默补成故事材料。"""
    from inputs.RepositoryFormat import FILE_MODELS
    from tools.JsonInput import validate
    from inputs.NovelRestoreRepository import require_public_input
    require_public_input(raw, location=str(location))
    name = {'role':'role-info','object':'object-info','location':'locations',
            'rule':'setting','organization':'setting'}[kind]
    record = {raw.get('location_code', ''):raw} if kind == 'location' and isinstance(raw,dict) else raw
    validate(FILE_MODELS[name], record, location=str(location))
    return raw


def normalize_node(raw):
    """固定目录布局可由节点编号定位；历史显式路径仍交原有校验。"""
    if not isinstance(raw, dict) or not isinstance(raw.get('node_id'), str):
        return raw
    key = raw['node_id']
    return {'path': f'{key}/todolist.json', 'state_path': f'{key}/state_before.json', **raw}


def normalize_event(raw, *, location, locations=None, objects=None):
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: 事件须为 JSON 对象")
    event = {"time": None, "epistemic_status": "事实", **raw}
    people = event.get("participants")
    if isinstance(people, list):
        event["participants"] = [{"original_goal": None, **person} if isinstance(person, dict) else person
                                 for person in people]
    identities = event.get("location_ids")
    if ("locations" not in event and isinstance(identities, list) and locations is not None
            and all(isinstance(key, str) and isinstance(locations.get(key), dict)
                    and isinstance(locations[key].get("location_name"), str)
                    and locations[key]["location_name"].strip() for key in identities)):
        event["locations"] = [locations[key]["location_name"] for key in identities]
    if event.get("location_ids") == [] and event.get("locations") in (None, []):
        event["locations"] = None
    identities = event.get("object_ids")
    if ("items" not in event and isinstance(identities, list) and objects is not None
            and all(isinstance(key, str) and isinstance(objects.get(key), dict)
                    and isinstance(objects[key].get("object_name"), str) for key in identities)):
        event["items"] = [objects[key]["object_name"] for key in identities] or None
    return event


def normalize_group(raw, *, location, locations=None, objects=None):
    events = raw.get("events")
    if not isinstance(events, list):
        raise ValueError(f"{location}/events: 事件组须提供事件数组")
    return {**{k:v for k,v in raw.items() if k != "state_changes"}, "state_delta": normalize_state_delta(raw.get('state_changes', raw.get('state_delta')), location=location + '/state_delta'),
            "events": [normalize_event(event, location=f"{location}/events/{index}", locations=locations, objects=objects)
                              for index, event in enumerate(events)]}
