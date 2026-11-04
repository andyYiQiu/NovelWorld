# NovelWorld

NovelWorld 以 NovelHarness 导出的单本小说 repository 为唯一外部故事来源，支持 JSON/JSONL 目录和十三表 SQLite。世界与人物模拟后，每个通过验收的事件组产生一份故事草稿。一条故事线完整通过验收后，即可从其草稿时间轴生成一个或多个叙述版本。

## 开始使用

在 Linux 或 macOS 上，从 NovelWorld 根目录用 Python 3.11 或更高版本创建虚拟环境并安装依赖。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B main.py --help
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
.venv/bin/python -B main.py run --config configs/exp-001/simulation.json
```

若要先离线检查输入，可先用 `init` 初始化，再用 `status` 查看。准备好模型后用 `resume` 继续该运行。

```bash
.venv/bin/python -B main.py init --config configs/exp-001/simulation.json
.venv/bin/python -B main.py status --run-directory runs/exp-001
.venv/bin/python -B main.py resume --run-directory runs/exp-001
```

**生成叙述**。完整模拟结束后，可以按本次实验的叙述配置生成正文。同一份模拟需要比较不同写法时，可另存一份叙述配置并设置新的 `version_id`。

```bash
.venv/bin/python -B main.py narrate --run-directory runs/exp-001 --config configs/exp-001/narrative.json
```

`retry --help` 说明如何从指定事件组的检查点创建独立尝试。运行目录中的 `run.sqlite3` 保存状态与进度，`story_progress.md` 展示已提交的模拟材料，叙述版本放在 `narratives/<版本编号>/`。

## 模拟中探索不同可能

配置只设容量和并发。默认允许整棵树最多保留四条最终时间轴，示例设置相同、同时运行两个分支。若要全程单线，设 `max_timelines` 为 1。新配置不接受预先列出节点和分支名称的 `branch_plan`。

```json
{
  "max_timelines": 4,
  "branch_parallelism": 2
}
```

模拟抵达一个时间节点的首组入口时，如果还有探索额度，世界依据当前状态和已发生的行动判断是否分叉。它可以继续原线，也可以提出不同的、有现场依据的行动机会。每条子线继承分叉点的已提交世界、人物记忆和草稿，独立继续，还可在更晚的节点分叉。各线不合并。一个分支的探索方向只交给该分支的世界调度，人物自己决定回应。运行预算在子线间分配，共同历史只模拟一次。

使用前面的 `run`、`resume`、`status`，目录均指向树根。`max_timelines` 是上限，最终可能只有一条；`concurrency` 控制整棵树同时在途的模拟请求数。要从已有的兼容节点入口开一个新实验，在新配置中设同一 repository、不同运行目录及探索容量，再执行以下命令。将 TN_2 替换成该运行已经保存入口和费用记录的实际节点。

```bash
.venv/bin/python -B main.py fork --run-directory runs/exp-001 --node TN_2 --config configs/exp-002/simulation.json
.venv/bin/python -B main.py resume --run-directory runs/exp-002
```

同一组的事件共享批前状态并共同验收。`story_timeline.json` 按节点和组索引，每个通过验收的组对应一份 `simulation/groups/<组编号>/draft.json` 和可读的 `story.md`。组未通过时可阅读 `story_progress.md` 的已提交经过，此时该组没有正式草稿。事件级行动记录保留在 `simulation/events/` 供诊断。

从树根执行 `narrate`。每个叙述版本只选一条完整、已验收的故事线。择线、章节规划、正文和事实审核均以该阶段允许的正式组草稿为故事资料；版本配置用于篇幅、风格、视角等表达要求。所选线的正文位于其 `narratives/<版本编号>/`。用树根的 `status --version`、`resume --version` 查看或继续该版本。

## 代码和数据边界

| 位置 | 职责 |
| --- | --- |
| `main.py`、`workflows/` | 命令入口、执行配置绑定、模拟及叙述生命周期 |
| `contract/` | JSON/JSONL 导出读取、公开 SQLite 查询、六类状态协议及回放 |
| `agents/`、`common/` | 世界和人物行动、个人知情范围、故事草稿、叙述规划与核对 |
| `model/`、`prompts/` | 模型适配器、执行配置示例及角色提示 |
| `storage/`、`exporters/` | 运行事务、恢复、预算和文件发布 |

叙述配置的 `version_id` 使用 1 至 64 个英文、数字、点、短横或下划线字符，以字母或数字开头。`requirements` 可以省略，或填由标点分隔的表达选项，如“白描，倒叙”。`version` 可明确指定 `person`（如“第三人称”）、`pov`（如“人物限知”）、`style`（如“克制”）、`pacing`（如“紧凑”）、`reveal_policy`（如“逐步揭示”）、媒介及篇幅。各字段接受固定选项，错误值在请求模型前拒绝。

目录输入在内存中建立索引。初始化校验资产引用、时间节点、事件组与相邻快照连续性，遇到上游的 `.incomplete.json` 标记会拒绝运行。模拟只读这份资料，运行进度写在独立的 `run.sqlite3`。继续同一运行时会核对输入指纹，修改原书资料后需新建运行。

新运行使用格式 18。人物先自行决定是否提问，再检索本线自己的记忆。命中并实际形成的回想按当时的主观体验写入行动材料。格式 17 的组草稿按原结构只读查看。旧格式运行保持原状，可查看已有状态和文件；需要继续旧实验时使用原执行版本。
