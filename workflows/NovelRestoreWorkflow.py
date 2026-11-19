"""下游运行生命周期；模拟细节属于 Simulation，文件投影属于 Exporter。"""
from hashlib import sha256
import asyncio
from pathlib import Path

from agents.NovelRestoreSupervisorAgent import NovelRestoreSupervisorAgent
from execution.NovelRestoreScriptedBackend import ScriptedResponses
from common.Runtime import RunStopped, fingerprint
from workflows.NovelRestoreSimulation import NovelRestoreSimulation
from workflows.NovelRestoreNarration import NovelRestoreNarration
from common.NovelRestoreConfig import NovelRestoreNarrativeConfig
from storage.NovelRestoreExporter import NovelRestoreExporter
from model.ModelClient import ModelClient
from storage.NovelRestoreLog import NovelRestoreLog
from storage.NovelRestoreNarrativeStorage import NarrativeLimitReached, NarrativeScope
from storage.NovelRestoreStorage import NovelRestoreStorage
from workflows.NovelRestoreBranches import branch_layout, explore, tree_status


class NovelRestoreWorkflow:
    @classmethod
    async def summarize(cls, directory, group_id):
        """显式辅助整理，使用运行预算与执行绑定，不推进模拟或修改其终态。"""
        with NovelRestoreStorage(directory) as storage:
            backend, binding = cls.execution_inputs(storage.config)
            if storage.get_checkpoint("execution_binding") != binding:
                raise ValueError("执行代码、提示或后端配置已变化，不能混用旧运行整理摘要")
            log = NovelRestoreLog(storage.directory / "run.log.jsonl")
            client = None
            try:
                supervisor, client = cls.supervisor(storage, backend, log)
                with storage.read_repository() as repository:
                    storage._check_repository(repository)
                    group = next((g for node in repository.nodes() for g in repository.groups(node["node_id"])
                                  if g.group_id == group_id), None)
                    if group is None:
                        raise ValueError("不存在指定事件组")
                    from agents.NovelRestoreWorldAgent import NovelRestoreWorldAgent
                    failed = await NovelRestoreWorldAgent(storage, repository, supervisor).summarize_group(group)
                return (2 if failed else 0), {**storage.status(), "summary_failures": failed}
            finally:
                await cls.close_execution(client, log)

    @staticmethod
    def supervisor(storage, backend, log):
        if storage.config.backend == "scripted":
            return NovelRestoreSupervisorAgent(storage, responder=ScriptedResponses(storage, backend)), None
        client = ModelClient(backend, storage, log, storage.config.max_calls, storage.config.max_request_chars)
        return NovelRestoreSupervisorAgent(storage, client=client), client

    @staticmethod
    async def close_execution(client, log, **context):
        """收尾不覆盖业务异常，也不推翻已保存的终态。"""
        try:
            if client is not None:
                await client.close()
        except Exception as error:
            log.write("close_failed", **context, error=str(error))
        finally:
            log.close()

    @staticmethod
    def execution_inputs(config):
        config.check_execution()
        path = config.model_config_path if config.backend == "model" else config.script_file
        if config.backend == "model":
            backend = ModelClient.read_config(path)
        else:
            from tools.JsonInput import loads
            backend = loads(path.read_text(), location=str(path), expected=dict)
        if config.backend == "scripted" and (not isinstance(backend, dict) or any(not isinstance(value, list) for value in backend.values())):
            raise ValueError("scripted 文件须为任务身份到响应列表的映射")
        root = Path(__file__).resolve().parents[1]
        files = [root / "requirements.txt", root / "main.py"]
        for directory in ("agents", "common", "inputs", "execution", "simulation",
                          "narration", "model", "storage", "workflows", "tools"):
            files.extend((root / directory).rglob("*.py"))
        files.extend((root / "prompts").rglob("*.md"))
        binding = fingerprint({"config": config.model_dump(mode="json"), "backend": backend,
                               "code": {str(file.relative_to(root)): sha256(file.read_bytes()).hexdigest() for file in files}})
        return backend, binding

    @classmethod
    async def run(cls, *, config=None, directory=None, retry_unknown=False):
        if (config is None) == (directory is None):
            raise ValueError("必须选择新建配置或已有运行目录")
        if retry_unknown and config is not None:
            raise ValueError("仅已有运行的显式续跑可重试未知回执")
        if config is None:
            with NovelRestoreStorage(directory, readonly=True) as saved:
                if saved.manifest.get("fork_node_id"):
                    raise ValueError("请从分支树根目录恢复实验")
                managed = saved.config.max_timelines > 1
        else:
            managed = config.max_timelines > 1
        if not managed:
            return await cls._run_segment(config=config, directory=directory, retry_unknown=retry_unknown)
        if config is not None:
            cls.execution_inputs(config)
            with NovelRestoreStorage.create(config) as saved:
                directory = saved.directory
        return await explore(cls, directory, retry_unknown=retry_unknown)

    @classmethod
    async def _run_segment(cls, *, config=None, directory=None, request_slots=None, retry_unknown=False):
        if config is not None:
            cls.execution_inputs(config)  # 在创建目录前检查，不读取密钥或请求接口。
        storage = NovelRestoreStorage.create(config) if config is not None else NovelRestoreStorage(directory)
        with storage:
            exporter = NovelRestoreExporter(storage)
            stage = storage.current()[2]["stage"]
            if stage in ("simulation_complete", "group_complete", "incomplete"):
                # 未完成也是已结束的故事线；恢复仅重建投影，不能重新调用模型。
                # 新的模拟尝试须走隔离重演，保留这条线的事实和费用记录。
                exporter.export_simulation()
                return (2 if stage == "incomplete" else 0), storage.status()
            backend, binding = cls.execution_inputs(storage.config)
            previous = storage.get_checkpoint("execution_binding")
            if previous is not None and previous != binding:
                raise ValueError("执行代码、提示或后端配置已变化；请保留旧运行并新建运行")
            if stage == "branch_ready":
                exporter.export_simulation()
                return 0, storage.status()
            storage.recover()
            storage.set_checkpoint("execution_binding", binding)
            if retry_unknown:
                storage.allow_unknown_retries()
            log = NovelRestoreLog(storage.directory / "run.log.jsonl")
            client = None
            try:
                supervisor, client = cls.supervisor(storage, backend, log)
                if request_slots is not None:
                    supervisor.semaphore = request_slots
                try:
                    with storage.read_repository() as repository:
                        storage._check_repository(repository)
                        if storage.current()[2]["stage"] in ("paused", "failed"):
                            storage.mark_stage("simulating")
                        completed = await NovelRestoreSimulation(storage, repository, supervisor).run(recovered=True)
                    code = 0 if completed else 2
                except (asyncio.CancelledError, KeyboardInterrupt) as error:
                    reason = f"运行被中断（{type(error).__name__}）；已保留进度和请求账本"
                    # 不触碰 prepared 行动、未收回执请求或已经写入的终态。
                    # 恢复时仍先核对执行指纹，再处理待提交事务。
                    if storage.current()[2]["stage"] not in ("simulation_complete", "group_complete", "branch_ready", "incomplete"):
                        storage.mark_stage("paused", reason)
                    log.write("interrupted", error=reason)
                    raise
                except RunStopped as error:
                    stage = "paused"
                    storage.mark_stage(stage, str(error))
                    log.write(stage, error=str(error))
                    code = 2
                except Exception as error:
                    storage.mark_stage("failed", str(error))
                    log.write("failed", error=str(error))
                    # 准备中的行动由 resume 恢复；导出不得越过该事务边界。
                    if not storage.status()["pending_actions"]:
                        try:
                            exporter.export_simulation()
                        except (OSError, ValueError) as export_error:
                            log.write("export_failed", error=str(export_error))
                    raise
                exporter.export_simulation()
                return code, storage.status()
            finally:
                await cls.close_execution(client, log)

    @classmethod
    async def narrate(cls, directory, *, config=None, version_id=None):
        if (config is None) == (version_id is None):
            raise ValueError("新建叙述版本提供配置，继续版本提供 version_id")
        with NovelRestoreStorage(directory, readonly=True) as storage:
            if storage.manifest.get("fork_node_id"):
                raise ValueError("请从分支树根目录择线并生成或继续叙述版本")
            plan = branch_layout(storage)
        if plan is not None:
            from workflows.NovelRestoreSelection import resume_selection, select_and_narrate
            if version_id is not None:
                return await resume_selection(directory, version_id)
            candidates = [row["directory"] for row in tree_status(directory)["leaves"]
                          if row["stage"] == "simulation_complete" and not row.get("error")]
            if not candidates:
                raise ValueError("没有已完整模拟的候选叶时间轴")
            return await select_and_narrate(directory, candidates, config)
        return await cls._narrate_one(directory, config=config, version_id=version_id)

    @classmethod
    async def _narrate_one(cls, directory, *, config=None, version_id=None):
        with NovelRestoreStorage(directory, narration_only=True) as storage:
            if config is not None:
                backend, binding = cls.execution_inputs(config)
                storage.create_narrative(config, binding)
                version_id = config.version_id
            state = storage.narrative(version_id)
            storage.check_narrative_source(version_id)
            exporter = NovelRestoreExporter(storage)
            if state["stage"] in ("complete", "incomplete"):
                exporter.export_narrative(version_id)
                return (0 if state["stage"] == "complete" else 2), storage.narrative_status(version_id)
            config = NovelRestoreNarrativeConfig.model_validate(state["config"])
            backend, binding = cls.execution_inputs(config)
            if state["binding"] != binding:
                raise ValueError("叙述版本的代码、提示或执行配置已变化；请另建版本")
            scope = NarrativeScope(storage, version_id)
            log = NovelRestoreLog(storage.directory / "run.log.jsonl")
            client = None
            try:
                supervisor, client = cls.supervisor(scope, backend, log)
                try:
                    completed = await NovelRestoreNarration(storage, version_id, supervisor).run()
                    code = 0 if completed else 2
                except (asyncio.CancelledError, KeyboardInterrupt) as error:
                    reason = f"叙述被中断（{type(error).__name__}）；已保留进度和请求账本"
                    if storage.narrative(version_id)["stage"] not in ("complete", "incomplete"):
                        storage.update_narrative(version_id, stage="paused", error=reason)
                    log.write("interrupted", version=version_id, error=reason)
                    raise
                except RunStopped as error:
                    stage = "incomplete" if isinstance(error, NarrativeLimitReached) else "paused"
                    storage.update_narrative(version_id, stage=stage, error=str(error))
                    log.write(stage, version=version_id, error=str(error))
                    code = 2
                except Exception as error:
                    storage.update_narrative(version_id, stage="failed", error=str(error))
                    log.write("narrative_failed", version=version_id, error=str(error))
                    try:
                        exporter.export_narrative(version_id)
                    except (OSError, ValueError) as export_error:
                        log.write("export_failed", version=version_id, error=str(export_error))
                    raise
                exporter.export_narrative(version_id)
                return code, storage.narrative_status(version_id)
            finally:
                await cls.close_execution(client, log, version=version_id)
