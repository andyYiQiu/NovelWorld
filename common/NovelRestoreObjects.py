"""下游普通道具登记；复用六类状态、批次事务和世界审核。"""


def object_assets(storage):
    with storage.read_repository() as repository:
        original = repository.assets("object")
    return {**{key: {**value, "origin": "repository"} for key, value in original.items()},
            **storage.get_checkpoint("runtime_objects", {})}


def object_inventory(storage):
    return [{"object_id": key, **{k: value.get(k, []) if k == "aliases" else value.get(k, "")
             for k in ("object_name", "aliases", "description", "origin")}}
            for key, value in object_assets(storage).items()]


def effective_catalog(storage, additions=None):
    catalog = storage.manifest["catalog"]
    return {**catalog, "object": [*catalog["object"], *storage.get_checkpoint("runtime_objects", {}), *(additions or {})]}


def validate_registrations(storage, additions, patches, action_ids):
    """只查登记、来源与同名候选；适用性由世界独立审读，不由字符串决定。"""
    catalog = effective_catalog(storage)
    if not additions:
        return catalog
    assets = object_assets(storage)
    sources = {source for item in additions.values() for source in item.source_action_ids}
    known_actions = set(action_ids)
    known_actions.update(source for source in sources if storage.connection.execute(
        "SELECT 1 FROM actions WHERE action_id=? AND status='committed'", (source,)).fetchone())
    changed = {row.object_id for patch in patches for row in patch.objects}
    for identity, item in additions.items():
        if not identity.startswith("SIM_OBJ_") or identity in assets:
            raise ValueError("新增物品须使用未登记的 SIM_OBJ_ 编号，不能覆盖既有物品：" + identity)
        if identity not in changed:
            raise ValueError("新增物品须同时保存客观物品状态：" + identity)
        if set(item.considered_object_ids) - assets.keys():
            raise ValueError("复用比较引用了不存在的物品")
        if set(item.source_action_ids) - known_actions:
            raise ValueError("新增物品来源须为已提交或本批实际发生")
        names = {s.strip().casefold() for s in [item.object_name, *item.aliases]}
        matches = {key for key, value in assets.items()
                   if names.intersection(s.strip().casefold() for s in [value["object_name"], *value.get("aliases", [])])}
        if matches - set(item.considered_object_ids):
            raise ValueError("须先比较同名或别名的既有物品，并说明不能复用的具体原因：" + str(sorted(matches)))
    return effective_catalog(storage, additions)
