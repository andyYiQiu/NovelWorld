"""下游运行生命周期；模拟细节属于 Simulation，文件投影属于 Exporter。"""
from hashlib import sha256
import json
from pathlib import Path

from agents.NovelRestoreAgent import NovelRestoreAgent, ScriptedResponses, fingerprint
from common.Runtime import RunStopped
from common.NovelRestoreSimulation import NovelRestoreSimulation
from common.NovelRestoreNarration import NovelRestoreNarration
from common.NovelRestoreConfig import NovelRestoreNarrativeConfig
from exporters.NovelRestoreExporter import NovelRestoreExporter
from model.ModelClient import ModelClient
from storage.NovelRestoreLog import NovelRestoreLog
from storage.NovelRestoreStorage import NarrativeLimitReached, NarrativeScope, NovelRestoreStorage
from workflows.NovelRestoreBranches import branch_layout, explore, tree_status


class NovelRestoreWorkflow:
    @staticmethod
    def executor(storage, backend, log):
        if storage.config.backend == "scripted":
            return NovelRestoreAgent(storage, responder=ScriptedResponses(storage, backend)), None
        client = ModelClient(backend, storage, log, storage.config.max_calls, storage.config.max_request_chars)
        return NovelRestoreAgent(storage, client=client), client

    @staticmethod
    def execution_inputs(config):
        config.check_execution()
        path = config.model_config_path if config.backend == "model" else config.script_file
        backend = ModelClient.read_config(path) if config.backend == "model" else json.loads(path.read_text())
        if config.backend == "scripted" and (not isinstance(backend, dict) or any(not isinstance(value, list) for value in backend.values())):
            raise ValueError("scripted 文件须为任务身份到响应列表的映射")
        root = Path(__file__).resolve().parents[1]
        files = [root / "requirements.txt", root / "model/ModelClient.py", root / "main.py"]
        files.extend((root / "contract").rglob("*.py"))
        for directory, pattern in (("agents", "NovelRestore*.py"), ("common", "*.py"),
                                   ("storage", "*.py"),
                                   ("prompts", "*.md"), ("workflows", "NovelRestore*.py"),
                                   ("exporters", "NovelRestore*.py")):
            files.extend((root / directory).glob(pattern))
        binding = fingerprint({"config": config.model_dump(mode="json"), "backend": backend,
                               "code": {str(file.relative_to(root)): sha256(file.read_bytes()).hexdigest() for file in files}})
        return backend, binding

    @classmethod
    async def run(cls, *, config=None, directory=None):
        if (config is None) == (directory is None):
            raise ValueError("必须选择新建配置或已有运行目录")
        if config is None:
            with NovelRestoreStorage(directory, readonly=True) as saved:
                if saved.manifest.get("fork_node_id"):
                    raise ValueError("请从分支树根目录恢复实验")
                managed = saved.config.max_timelines > 1
        else:
            managed = config.max_timelines > 1
        if not managed:
            return await cls._run_segment(config=config, directory=directory)
        if config is not None:
            cls.execution_inputs(config)
            with NovelRestoreStorage.create(config) as saved:
                directory = saved.directory
        return await explore(cls, directory)

    @classmethod
    async def _run_segment(cls, *, config=None, directory=None, request_slots=None):
        if config is not None:
            cls.execution_inputs(config)  # 在创建目录前检查，不读取密钥或请求接口。
        storage = NovelRestoreStorage.create(config) if config is not None else NovelRestoreStorage(directory)
        with storage:
            exporter = NovelRestoreExporter(storage)
            if storage.current()[2]["stage"] in ("simulation_complete", "group_complete"):
                exporter.export_simulation()
                return 0, storage.status()
            backend, binding = cls.execution_inputs(storage.config)
            previous = storage.get_checkpoint("execution_binding")
            if previous is not None and previous != binding:
                raise ValueError("执行代码、提示或后端配置已变化；请保留旧运行并新建运行")
            if storage.current()[2]["stage"] == "branch_ready":
                exporter.export_simulation()
                return 0, storage.status()
            storage.recover()
            storage.set_checkpoint("execution_binding", binding)
            log = NovelRestoreLog(storage.directory / "run.log.jsonl")
            client = None
            try:
                executor, client = cls.executor(storage, backend, log)
                if request_slots is not None:
                    executor.semaphore = request_slots
                try:
                    with storage.read_repository() as repository:
                        storage._check_repository(repository)
                        completed = await NovelRestoreSimulation(storage, repository, executor).run()
                    code = 0 if completed else 2
                except RunStopped as error:
                    storage.mark_stage("paused", str(error))
                    log.write("paused", error=str(error))
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
                if client is not None:
                    await client.close()
                log.close()

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
                executor, client = cls.executor(scope, backend, log)
                try:
                    completed = await NovelRestoreNarration(storage, version_id, executor).run()
                    code = 0 if completed else 2
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
                if client is not None:
                    await client.close()
                log.close()
