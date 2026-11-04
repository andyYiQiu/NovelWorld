"""从各节点运行时记录的分叉决定和实际谱系重建树，独立推进每条子线。"""
import asyncio
import fcntl
from hashlib import sha256
from pathlib import Path
import json
import sqlite3

from model.ModelClient import ModelClient
from common.NovelRestoreConfig import Branch, BranchPlan
from exporters.NovelRestoreExporter import NovelRestoreExporter
from storage.NovelRestoreStorage import NovelRestoreStorage


def child_directory(parent, name):
    folder = parent / "branches"
    child = folder / name
    if folder.is_symlink() or child.is_symlink():
        raise ValueError(f"分支目录不能是符号链接：{child}")
    return child


def branch_layout(storage):
    # 老格式的状态页仍能显示历史计划。新运行的拓扑只取已提交的节点决策。
    if storage.config.branch_plan is not None:
        return storage.config.branch_plan
    split = storage.get_checkpoint("branch_split")
    if split:
        return BranchPlan(node_id=split["node_id"], children=[Branch(name=a["name"]) for a in split["alternatives"]])
    return None


def tree_status(directory, errors=None):
    """从各分支已提交的状态重建视图，重复的共同前缀只计费一次。"""
    directory = Path(directory).resolve()
    with NovelRestoreStorage(directory, readonly=True) as root:
        result = root.status()
        plan = branch_layout(root)
        managed = root.config.max_timelines > 1 or plan is not None
        root_id = root.manifest.get("branch_root_id", root.run_id)
        repo_hash = root.manifest["repository_hash"]
        if errors is None:
            errors = root.get_checkpoint("branch_errors", {})
    if not managed:
        return result
    nodes, leaves = [], []
    errors = dict(errors)

    def visit(path, fork, parent_id=None, birth=None):
        key = str(path.relative_to(directory))
        row = {"branch": key, "directory": str(path), "run_id": None, "stage": "not_started", "own_calls": 0}
        nodes.append(row)
        try:
            if birth and parent_id:
                child_directory(path.parent.parent, path.name)
            if (not birth or parent_id) and path.exists():
                with NovelRestoreStorage(path, readonly=True) as branch:
                    if (branch.manifest.get("branch_root_id", branch.run_id) != root_id
                            or branch.manifest["repository_hash"] != repo_hash or (branch.config.branch_plan and branch.config.branch_plan != fork)
                            or (birth and (not parent_id or branch.manifest.get("parent_run_id") != parent_id
                                           or branch.manifest.get("fork_node_id") != birth))
                            or branch.config.run_directory.resolve() != path):
                        raise ValueError("已有目录与预期分支谱系或配置不一致")
                    fork = branch_layout(branch)
                    progress = branch.current()[2]
                    row.update(run_id=branch.run_id, stage=progress["stage"],
                               own_calls=branch.call_count("simulation") - branch.manifest.get("inherited_calls", 0))
                    if branch.config.max_timelines > 1:
                        history = branch.connection.execute(
                            "SELECT payload FROM checkpoints WHERE key GLOB 'branch_decision/*' ORDER BY rowid")
                        row["decisions"] = [json.loads(record[0]) for record in history]
                    if progress.get("error"):
                        row["error"] = progress["error"]
            if key in errors:
                row["error"] = errors[key]
        except (ValueError, OSError, sqlite3.Error, KeyError, TypeError) as error:
            row.update(stage="unavailable", error=str(error))
            errors[key] = str(error)
        if not fork:
            leaves.append(row)
        else:
            for child in fork.children:
                visit(path / "branches" / child.name, child.fork, row["run_id"], fork.node_id)

    visit(directory, plan)
    return {**result, "tree_complete": bool(leaves) and not errors and
            all(row["stage"] == "simulation_complete" for row in leaves),
            "simulation_calls": sum(row["own_calls"] for row in nodes),
            "branches": nodes, "leaves": leaves, "branch_errors": errors}


def _child_limit(limit, used, count, index):
    if limit is None:
        return None
    share, extra = divmod(max(0, limit - used), count)
    return used + share + int(index < extra)


async def explore(workflow, directory):
    directory = Path(directory).resolve()
    with NovelRestoreStorage(directory, readonly=True) as run:
        workers = asyncio.Semaphore(run.config.branch_parallelism)
        request_slots = asyncio.Semaphore(run.config.concurrency)
    errors = {}

    async def visit(path):
        async with workers:
            code, result = await workflow._run_segment(directory=path, request_slots=request_slots)
            if code or result["progress"]["stage"] != "branch_ready":
                return
            with NovelRestoreStorage(path, readonly=True) as parent:
                split = parent.get_checkpoint("branch_split")
                config = parent.config
                used_calls = parent.call_count("simulation")
                used_tokens = parent.usage("simulation")["tokens"] if config.max_tokens is not None else 0
                limit = config.max_calls
                if config.backend == "model":
                    limit = min(limit, ModelClient.read_config(config.model_config_path)["max_calls"])
            if split is None:
                raise ValueError("分支入口缺少本次运行已提交的决策")
            choices = split["alternatives"]
            children = []
            for i, choice in enumerate(choices):
                name = choice["name"]
                dest = path / "branches" / name
                try:
                    dest = child_directory(path, name)
                    calls = _child_limit(limit, used_calls, len(choices), i)
                    tokens = _child_limit(config.max_tokens, used_tokens, len(choices), i)
                    if calls <= used_calls or (tokens is not None and tokens <= used_tokens):
                        raise ValueError("本分支没有剩余模拟预算")
                    seed = int.from_bytes(sha256(f"{config.seed}/{name}".encode()).digest()[:8], "big")
                    capacity = _child_limit(config.max_timelines, 0, len(choices), i)
                    NovelRestoreStorage.fork_node(path, dest, split["node_id"], seed=seed, opportunity=choice,
                                                  max_timelines=capacity, max_calls=calls, max_tokens=tokens)
                    children.append(dest)
                except Exception as error:
                    errors[str(dest.relative_to(directory))] = str(error)
        # 等待后代之前释放自己的名额，单并发也可任意深度分叉。
        await asyncio.gather(*(isolated(child) for child in children))

    async def isolated(path):
        try:
            await visit(path)
        except Exception as error:
            errors[str(path.relative_to(directory))] = str(error)

    with (directory / ".branch-tree.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("该分支树已有调度者") from error
        saved = tree_status(directory, {})
        if saved["tree_complete"]:
            # 已结束的树与单线终态一样，只核对和恢复文件发布，不重新加载模型配置。
            for row in saved["branches"]:
                with NovelRestoreStorage(row["directory"]) as branch:
                    NovelRestoreExporter(branch).export_simulation()
        else:
            await visit(directory)
        # 未能创建的分支没有自己的运行库，在根保留失败原因供以后 status 查看。
        with NovelRestoreStorage(directory) as root:
            if root.get_checkpoint("branch_errors", {}) != errors:
                root.set_checkpoint("branch_errors", errors)
        result = saved if saved["tree_complete"] else tree_status(directory, errors)
        return (0 if result["tree_complete"] else 2), result
