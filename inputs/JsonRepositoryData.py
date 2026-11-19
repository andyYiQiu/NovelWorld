"""读取一本公开 JSON repository，在内存中固定本次读取的数据与指纹。"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath
from tools.JsonInput import loads, validate
from inputs.RepositoryFormat import FILE_MODELS
from inputs.RepositoryInput import normalize_asset, normalize_node


@dataclass(frozen=True)
class JsonRepositoryData:
    source: str
    assets: dict[str, dict[str, dict]]
    worlds: dict[str, dict | list]
    nodes: list[dict]
    groups: dict[str, list[dict]]
    snapshots: dict[str, dict]
    fingerprint: str


def _require(condition, location, message):
    if not condition:
        raise ValueError(f"{location}: {message}")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _parse(raw, path):
    try:
        content = raw.decode("utf-8-sig")
        if path.suffix == ".jsonl":
            records = []
            for number, line in enumerate(content.split("\n"), 1):
                if line.strip():
                    try:
                        records.append(loads(line))
                    except ValueError as exc:
                        raise ValueError(f"第 {number} 行: {exc}") from exc
            return records
        return loads(content)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _ordered(rows, label):
    _require(isinstance(rows, list) and rows, label, "须为非空数组")
    for row in rows:
        _require(isinstance(row, dict) and type(row.get("order")) is int and row["order"] > 0,
                 label, "order 须为正整数")
    ordered = sorted(rows, key=lambda row: row["order"])
    _require([row["order"] for row in ordered] == list(range(1, len(ordered) + 1)),
             label, "order 须从 1 连续且唯一")
    return ordered


def _timeline_file(node, field):
    name = node.get(field)
    _require(_text(name), f"{node['node_id']}.{field}", "缺少 timeline 文件路径")
    path = Path(name)
    _require(not path.is_absolute() and not PureWindowsPath(name).drive and "\\" not in name
             and ".." not in path.parts and path.suffix == ".json",
             name, "须为 timeline 内的相对 JSON 路径")
    return Path("timeline") / path


def load_json_repository(root: Path) -> JsonRepositoryData:
    """只打开目录中声明的公开文件，拒绝 partial 和无法归属的 JSON 资料。"""
    root = Path(root).expanduser()
    _require(not root.is_symlink() and root.is_dir(), root, "需要一本公开 repository 目录")
    root = root.resolve()
    incomplete = root.with_name(root.name + ".incomplete.json")
    _require(not incomplete.is_symlink() and not incomplete.exists(), incomplete, "partial repository 尚未完成")
    for name in ("roles", "objects", "locations", "worlds/background", "worlds/settings", "timeline"):
        _require((root / name).is_dir(), name, "缺少公开资料目录")

    files = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), path, "repository 内不能有符号链接")
        if path.suffix.lower() in (".json", ".jsonl"):
            _require(path.is_file(), path, "需要普通数据文件")
            files.add(path.relative_to(root))
    hashes = {}

    def read(relative):
        relative = Path(relative)
        _require(relative in files, relative, "缺少公开数据文件")
        _require(relative not in hashes, relative, "文件被重复引用")
        raw = (root / relative).read_bytes()
        hashes[relative] = sha256(raw).hexdigest()
        value = _parse(raw, relative)
        from inputs.NovelRestoreRepository import require_public_input
        require_public_input(value, location=str(relative))
        kind = {'role_info.json':'role-info','object_info.json':'object-info','locations.json':'locations',
                'general.json':'general','settings.jsonl':'setting','timeline.json':'timeline',
                'state_before.json':'state-before','todolist.json':'todolist'}.get(relative.name)
        if kind:
            for record in value if kind == 'setting' else [value]:
                validate(FILE_MODELS[kind], record, location=str(relative))
        return value

    timeline = read("timeline/timeline.json")
    timeline = {"time_basis": "story", **timeline}
    _require(timeline.keys() == {"source", "time_basis", "nodes"}
             and _text(timeline["source"]) and timeline["time_basis"] == "story",
             "timeline/timeline.json", "需要唯一来源、故事时间和节点数组")
    source = timeline["source"]
    assets = {kind: {} for kind in ("role", "object", "location", "rule", "organization")}

    def add_asset(kind, key, row, path):
        _require(isinstance(row, dict) and _text(key) and row.get(f"{kind}_code") == key
                 and _text(row.get(f"{kind}_name")), path, f"{kind} 档案编号或名称不合法")
        _require("source" not in row or row["source"] == source, path, "source 与 timeline 不一致")
        _require(key not in assets[kind], path, f"重复 {kind} 编号 {key}")
        assets[kind][key] = normalize_asset(row, kind=kind, location=path)

    for kind in ("role", "object"):
        for directory in sorted((root / (kind + "s")).iterdir()):
            if directory.is_dir():
                key = directory.name
                path = directory.relative_to(root) / f"{kind}_info.json"
                add_asset(kind, key, read(path), path)
    for key, row in read("locations/locations.json").items():
        add_asset("location", key, row, "locations/locations.json")

    worlds = {"worlds/background/general.json": read("worlds/background/general.json")}
    settings = [root / "worlds/settings/settings.jsonl"]

    seen_settings = set()
    for path in settings:
        relative = path.relative_to(root)
        rows = read(relative)
        worlds[relative.as_posix()] = rows
        for row in rows:
            key, kind = row.get("setting_id"), row.get("kind")
            _require(_text(key) and key not in seen_settings, path, f"设定编号缺失或重复 {key}")
            _require(kind in ("custom", "rule", "organization", "concept", "other") and _text(row.get("term")),
                     path, f"设定 {key} 的类型或名称不合法")
            seen_settings.add(key)
            if kind in ("rule", "organization"):
                assets[kind][key] = normalize_asset(row, kind=kind, location=f"{path}/{key}")

    nodes = [normalize_node(node) for node in _ordered(timeline["nodes"], "timeline/timeline.json/nodes")]
    snapshots, groups = {}, {}
    seen_groups, seen_events = set(), set()
    for node in nodes:
        key = node.get("node_id")
        _require(_text(key) and key not in snapshots, "timeline/timeline.json/nodes", f"节点编号缺失或重复 {key}")
        snapshot_path = _timeline_file(node, "state_path")
        snapshot = read(snapshot_path)
        _require(snapshot.get("node_id") == key, snapshot_path, "快照节点编号不匹配")
        snapshots[key] = snapshot
        group_path = _timeline_file(node, "path")
        document = read(group_path)
        _require(document.keys() == {"node_id", "todolist"} and document["node_id"] == key,
                 group_path, "todolist 节点编号或容器字段不匹配")
        groups[key] = _ordered(document["todolist"], f"{group_path}/todolist")
        for group in groups[key]:
            gid, events = group.get("group_id"), group.get("events")
            _require(_text(gid) and gid not in seen_groups, group_path, f"事件组编号缺失或重复 {gid}")
            seen_groups.add(gid)
            _require(isinstance(group.get("state_changes"), dict), f"{group_path}/{gid}", "事件组缺少 state_changes")
            _require(isinstance(events, list) and events, f"{group_path}/{gid}", "事件组缺少事件")
            for event in events:
                eid = event.get("task_id") if isinstance(event, dict) else None
                _require(_text(eid) and eid not in seen_events, f"{group_path}/{gid}", f"事件编号缺失或重复 {eid}")
                seen_events.add(eid)

    extra = files - hashes.keys()
    _require(not extra, root, f"未识别或未引用的 JSON 文件 {sorted(path.as_posix() for path in extra)}")
    manifest = [[path.as_posix(), hashes[path]] for path in sorted(hashes)]
    fingerprint = sha256(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return JsonRepositoryData(source, assets, worlds, nodes, groups, snapshots, fingerprint)
