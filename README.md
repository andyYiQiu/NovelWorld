# NovelWorld

NovelWorld 直接读取 NovelHarness 导出的一本小说的 JSON/JSONL 目录，在自己的运行目录中模拟人物与世界，再从已完成的模拟生成一个或多个叙述版本。已有的十三表 SQLite 公开库也可以作为输入。

## 开始使用

在 Linux 或 macOS 上，从 NovelWorld 根目录用 Python 3.11 或更高版本创建虚拟环境并安装依赖。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B novel_restore.py --help
```

在项目根目录创建 `repository/`，将你自己的**单本小说完整导出目录**放入其中，保留上游文件名和目录层级。例如，`repository/祝福_正式115/` 下包含 `timeline/timeline.json`、各时间节点的任务和状态快照，以及 `roles/`、`objects/`、`locations/`、`worlds/` 中的已有资料。每本书或每个版本单独放一个目录。

## 每次实验的配置

`configs/example/` 提供两份模板。模拟需要 `simulation.json`，用于选择输入、运行目录和模型预算。生成叙述时才需要 `narrative.json`，用于选择叙述版本、写作要求和该版本的模型预算。Git 只跟踪模板，具体实验的配置在本地管理。

每次实验从项目根目录复制整套模板，使用新的实验编号。下面的命令创建第一套实验配置 `exp-001`。

```bash
cp -R configs/example configs/exp-001
```

编辑 `configs/exp-001/simulation.json`，将 `run_directory` 从 `../../runs/example` 改为 `../../runs/exp-001`。运行目录须位于输入目录之外且尚未创建。`repository` 可以指向某本书的完整导出目录或已有 SQLite 文件。所有相对路径都以配置文件所在目录为基准。

**运行模拟**。示例模型从环境变量 `DEEPSEEK_API_KEY` 读取密钥。设置后，用本次实验的配置启动。两份配置均可通过 `model_config_path` 选用各自的[模型配置](model/config.example.json)。

```bash
.venv/bin/python -B novel_restore.py run --config configs/exp-001/simulation.json
```

若要先离线检查输入，可先用 `init` 初始化，再用 `status` 查看。准备好模型后用 `resume` 继续该运行。

```bash
.venv/bin/python -B novel_restore.py init --config configs/exp-001/simulation.json
.venv/bin/python -B novel_restore.py status --run-directory runs/exp-001
.venv/bin/python -B novel_restore.py resume --run-directory runs/exp-001
```

**生成叙述**。完整模拟结束后，可以按本次实验的叙述配置生成正文。同一份模拟需要比较不同写法时，可另存一份叙述配置并设置新的 `version_id`。

```bash
.venv/bin/python -B novel_restore.py narrate --run-directory runs/exp-001 --config configs/exp-001/narrative.json
```

`retry --help` 说明如何从指定事件组的检查点创建独立尝试。运行目录中的 `run.sqlite3` 保存状态与进度，`story_progress.md` 展示已提交的模拟材料，叙述版本放在 `narratives/<版本编号>/`。

## 代码和数据边界

| 位置 | 职责 |
| --- | --- |
| `novel_restore.py`、`workflows/` | 命令入口、执行配置绑定、模拟及叙述生命周期 |
| `contract/` | JSON/JSONL 导出读取、公开 SQLite 查询、六类状态协议及回放 |
| `agents/`、`common/` | 世界和人物行动、个人知情范围、故事草稿、叙述规划与核对 |
| `model/`、`prompts/` | 模型适配器、执行配置示例及角色提示 |
| `storage/`、`exporters/` | 运行事务、恢复、预算和文件发布 |

目录输入在内存中建立索引。初始化校验资产引用、时间节点、事件组与相邻快照连续性，遇到上游的 `.incomplete.json` 标记会拒绝运行。模拟只读这份资料，运行进度写在独立的 `run.sqlite3`。继续同一运行时会核对输入指纹，修改原书资料后需新建运行。

沿用下游运行格式 16。历史运行目录可用 `status` 查看。已绑定旧执行代码的未完成运行需要原版本续跑。
