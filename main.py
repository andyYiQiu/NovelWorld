"""从小说 repository 的 JSON 目录或公开 SQLite 库模拟事件，生成独立叙述版本。"""
import argparse
import asyncio
import json
import sqlite3
import sys

from common.NovelRestoreConfig import NovelRestoreConfig, NovelRestoreNarrativeConfig
from storage.NovelRestoreStorage import NovelRestoreStorage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="从一本书的 JSON 目录或 SQLite 库初始化运行空间")
    initialize.add_argument("--config", required=True, help="JSON 配置文件；相对路径以配置目录为基准")
    status = commands.add_parser("status", help="只读查看运行状态，不恢复或请求模型")
    status.add_argument("--run-directory", required=True)
    status.add_argument("--version", help="只读查看指定叙述版本")
    run = commands.add_parser("run", help="新建运行并模拟至事件组草稿时间轴")
    run.add_argument("--config", required=True)
    run.add_argument("--narrative-config", help="模拟完成后创建并生成指定叙述版本")
    resume = commands.add_parser("resume", help="继续已有模拟、叙述版本或重试文件导出")
    resume.add_argument("--run-directory", required=True)
    resume.add_argument("--version", help="继续指定叙述版本；省略时继续模拟")
    narrate = commands.add_parser("narrate", help="从完整模拟草稿新建叙述版本")
    narrate.add_argument("--run-directory", required=True)
    narrate.add_argument("--config", required=True)
    retry = commands.add_parser("retry", help="从指定事件组入口复制新运行；保留旧现场，不调用模型")
    retry.add_argument("--run-directory", required=True)
    retry.add_argument("--new-run-directory", required=True)
    retry.add_argument("--group-id", required=True)
    retry.add_argument("--preserve-progress", action="store_true", help="保留当前组已提交的行动，从下一轮继续；不重演整组")
    fork = commands.add_parser("fork", help="从已有节点入口的已提交历史建立一棵新实验树")
    fork.add_argument("--run-directory", required=True, help="已有运行目录")
    fork.add_argument("--config", required=True, help="新实验的模拟配置")
    fork.add_argument("--node", required=True, help="从来源运行已保存的该节点入口另开实验")
    args = parser.parse_args(argv)
    try:
        if args.command in ("fork", "retry"):
            if args.command == "fork":
                result = NovelRestoreStorage.fork_experiment(args.run_directory, NovelRestoreConfig.load(args.config), node_id=args.node)
            else:
                result = NovelRestoreStorage.retry_group(args.run_directory, args.new_run_directory, args.group_id,
                                                         preserve_progress=args.preserve_progress)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command in ("run", "resume", "narrate"):
            from workflows.NovelRestoreWorkflow import NovelRestoreWorkflow
            from common.Runtime import JobFailed
            from model.ModelClient import ModelResponseError
            try:
                if args.command == "narrate":
                    code, result = asyncio.run(NovelRestoreWorkflow.narrate(
                        args.run_directory, config=NovelRestoreNarrativeConfig.load(args.config)))
                elif args.command == "resume" and args.version:
                    code, result = asyncio.run(NovelRestoreWorkflow.narrate(args.run_directory, version_id=args.version))
                else:
                    config = NovelRestoreConfig.load(args.config) if args.command == "run" else None
                    narrative = (NovelRestoreNarrativeConfig.load(args.narrative_config)
                                 if args.command == "run" and args.narrative_config else None)
                    if narrative is not None:
                        NovelRestoreWorkflow.execution_inputs(narrative)
                    code, result = asyncio.run(NovelRestoreWorkflow.run(
                        config=config, directory=args.run_directory if args.command == "resume" else None))
                    if narrative is not None and (code == 0 or any(
                            leaf["stage"] == "simulation_complete" for leaf in result.get("leaves", []))):
                        story_code, story = asyncio.run(NovelRestoreWorkflow.narrate(config.run_directory, config=narrative))
                        result = {"simulation": result, "narrative": story} if "leaves" in result else story
                        code = code or story_code
            except (JobFailed, ModelResponseError) as error:
                print(f"main.py: {error}", file=sys.stderr)
                return 1
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return code
        storage = (NovelRestoreStorage.create(NovelRestoreConfig.load(args.config)) if args.command == "init"
                   else NovelRestoreStorage(args.run_directory, readonly=True))
        with storage:
            from workflows.NovelRestoreBranches import branch_layout, tree_status
            if args.command == "status" and args.version and branch_layout(storage):
                from workflows.NovelRestoreSelection import selection_status
                result = selection_status(storage, args.version)
                if result is None:
                    raise ValueError(f"没有这个叙述择线版本：{args.version}")
            elif args.command == "status" and args.version:
                result = storage.narrative_status(args.version)
            elif storage.config.max_timelines > 1 or storage.config.branch_plan:
                result = tree_status(storage.directory)
            else:
                result = storage.status()
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"main.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
