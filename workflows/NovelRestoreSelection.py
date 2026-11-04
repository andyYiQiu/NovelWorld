"""在树根单独比较完整叶时间轴，选定后仅在该叶创建并生成叙述版本。"""
from pathlib import Path

from agents.NovelRestoreAgent import fingerprint, task_id
from agents.NovelRestoreNarratorAgent import NovelRestoreNarratorAgent
from common.NovelRestoreConfig import NovelRestoreNarrativeConfig
from common.Runtime import RunStopped
from storage.NovelRestoreLog import NovelRestoreLog
from storage.NovelRestoreStorage import NarrativeScope, NovelRestoreStorage
from workflows.NovelRestoreBranches import branch_layout, tree_status
from workflows.NovelRestoreWorkflow import NovelRestoreWorkflow


def _key(version_id):
    return task_id("selection", version_id)


def _check_leaf(root, leaf, expected_id, parent_id=None):
    if (leaf.run_id != expected_id or leaf.manifest.get("branch_root_id") != root.run_id
            or not leaf.manifest.get("parent_run_id") or branch_layout(leaf) is not None
            or parent_id is not None and leaf.manifest["parent_run_id"] != parent_id
            or leaf.manifest["repository_hash"] != root.manifest["repository_hash"]
            or leaf.config.run_directory.resolve() != leaf.directory):
        raise ValueError("时间轴的分支谱系、来源或运行目录与树根绑定不一致")


def _read_candidates(root, paths):
    leaves = {row["directory"]: row for row in tree_status(root.directory)["leaves"]}
    inputs, sources, seen = {}, {}, set()
    for path in sorted(paths):
        row = leaves.get(str(path))
        if row is None or row["stage"] == "unavailable":
            raise ValueError(f"候选不属于这棵树的独立叶时间轴或谱系无效：{path}")
        if row["stage"] == "not_started":
            continue
        with NovelRestoreStorage(path, readonly=True) as leaf:
            _check_leaf(root, leaf, row["run_id"])
            if leaf.run_id in seen:
                raise ValueError("一个运行不能多次进入择线候选")
            seen.add(leaf.run_id)
            if leaf.current()[2]["stage"] != "simulation_complete":
                continue
            source = leaf.narrative_source()
            name = row["branch"]
            inputs[name] = {"run_id": leaf.run_id, "parent_run_id": leaf.manifest["parent_run_id"],
                            "source_hash": fingerprint(source)}
            # 草稿正文是逐项材料的确定性投影。保留全部结构材料，包括还没有关联到单独事件的共同经过。
            sources[name] = {k: source[k] for k in ("run_id", "characters", "actions", "shared_materials")}
            sources[name]["drafts"] = [{k: v for k, v in d.items() if k != "text"} for d in source["drafts"]]
    return inputs, sources


def _has_version(storage, version_id):
    return storage.connection.execute("SELECT 1 FROM narratives WHERE key=?", (version_id,)).fetchone() is not None


def _selected_directory(root, branch):
    path = root.directory / branch
    if path != path.resolve() or not path.is_relative_to(root.directory):
        raise ValueError("选定分支的目录已被重定向")
    return path


def selection_status(storage, version_id):
    """根目录的版本状态；返回 None 表示该版本不是分支择线。"""
    state = storage.get_checkpoint(_key(version_id))
    if state is None:
        return None
    calls = storage.call_count(_key(version_id))
    result = {"version_id": version_id, "stage": state["stage"], "error": state.get("error"), "published": False,
              "candidates": state["candidates"], "selected_branch": state.get("branch"),
              "selection_calls": calls, "calls": calls}
    if state.get("branch"):
        name = state["branch"]
        directory = _selected_directory(storage, name)
        result.update(reason=state["reason"], selected_directory=str(directory),
                      selected_run_id=state["candidates"][name]["run_id"])
        with NovelRestoreStorage(directory, readonly=True) as leaf:
            _check_leaf(storage, leaf, result["selected_run_id"], state["candidates"][name]["parent_run_id"])
            if _has_version(leaf, version_id):
                if leaf.narrative(version_id).get("selection") != state["receipt"]:
                    raise ValueError("叶中已有的叙述版本属于其他任务")
                result.update(leaf.narrative_status(version_id))
                result["calls"] += calls
    return result


async def select_and_narrate(root_directory, candidate_directories, config):
    config = NovelRestoreNarrativeConfig.model_validate(config.model_dump(mode="json"))
    paths = sorted(Path(p).expanduser().resolve() for p in candidate_directories)
    key = _key(config.version_id)
    with NovelRestoreStorage(root_directory, narration_only=True) as root:
        if root.run_id != root.manifest.get("branch_root_id", root.run_id) or branch_layout(root) is None:
            raise ValueError("择线须从分支树根运行")
        state = root.get_checkpoint(key)
        serialized = config.model_dump(mode="json")
        directories = [str(p) for p in paths]
        if state and (state["config"] != serialized or state["directories"] != directories):
            raise ValueError("这个择线版本已绑定其他配置或候选集合")
        if not state or not state.get("branch"):
            backend, binding = NovelRestoreWorkflow.execution_inputs(config)
            inputs, sources = _read_candidates(root, paths)
            if not inputs:
                return 2, {"stage": "incomplete", "version_id": config.version_id, "published": False,
                           "error": "没有完成模拟且通过验收的候选时间轴"}
            if state and (state["binding"] != binding or state["candidates"] != inputs):
                raise ValueError("择线依据已改变，请使用新版本")
            state = {"stage": "selecting", "directories": directories, "candidates": inputs,
                     "config": serialized, "binding": binding}
            root.set_checkpoint(key, state)
            scope = NarrativeScope(root, config.version_id, selection_config=config)
            log = NovelRestoreLog(root.directory / "run.log.jsonl")
            client = None
            try:
                if root.connection.execute("SELECT 1 FROM calls WHERE scope=? AND ended IS NULL",
                                           (scope.scope,)).fetchone():
                    raise RunStopped("上次择线请求尚无调用回执，无法确定送达结果")
                executor, client = NovelRestoreWorkflow.executor(scope, backend, log)
                narrator = NovelRestoreNarratorAgent(root, config.version_id, executor)
                choice = await narrator.choose_timeline(sources, config)
                usage = root.usage(scope.scope)
                usage["started"] = root.connection.execute(
                    "SELECT MIN(started) FROM calls WHERE scope=?", (scope.scope,)).fetchone()[0]
                receipt = {"root_run_id": root.run_id, "selected_run_id": inputs[choice.branch]["run_id"],
                           "source_hash": inputs[choice.branch]["source_hash"], "usage": usage}
                state.update(stage="selected", branch=choice.branch, reason=choice.reason, receipt=receipt)
                root.set_checkpoint(key, state)
            except RunStopped as error:
                root.set_checkpoint(key, {**state, "stage": "paused", "error": str(error)})
                return 2, selection_status(root, config.version_id)
            except Exception as error:
                root.set_checkpoint(key, {**state, "stage": "failed", "error": str(error)})
                raise
            finally:
                if client is not None:
                    await client.close()
                log.close()
        selected_directory = _selected_directory(root, state["branch"])
        receipt = state["receipt"]
        with NovelRestoreStorage(selected_directory, narration_only=True) as leaf:
            _check_leaf(root, leaf, receipt["selected_run_id"], state["candidates"][state["branch"]]["parent_run_id"])
            if fingerprint(leaf.narrative_source()) != receipt["source_hash"]:
                raise ValueError("选定的时间轴来源已经改变")
            if _has_version(leaf, config.version_id):
                if leaf.narrative(config.version_id).get("selection") != receipt:
                    raise ValueError("选定分支上同名叙述版本已有其他来源")
            else:
                _, current_binding = NovelRestoreWorkflow.execution_inputs(config)
                if current_binding != state["binding"]:
                    raise ValueError("择线后执行配置已改变，请另建版本")
                leaf.create_narrative(config, state["binding"], selection=receipt)

    code, _ = await NovelRestoreWorkflow._narrate_one(selected_directory, version_id=config.version_id)
    with NovelRestoreStorage(root_directory, readonly=True) as root:
        return code, selection_status(root, config.version_id)


async def resume_selection(root_directory, version_id):
    with NovelRestoreStorage(root_directory, readonly=True) as root:
        state = root.get_checkpoint(_key(version_id))
        if state is None:
            raise ValueError("找不到该版本的择线记录")
    return await select_and_narrate(root_directory, state["directories"],
                                    NovelRestoreNarrativeConfig.model_validate(state["config"]))
