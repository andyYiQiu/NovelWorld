"""显式配置模拟参数；路径相对于配置文件解析，不读取模型密钥。"""
import re
from pathlib import Path
from typing import Literal
from unicodedata import normalize

from pydantic import Field, field_validator, model_validator

from common.NovelRestoreSchema import Identity, PositiveInt, Record, SemanticPayload
from tools.JsonInput import loads, validate


class ExecutionConfig(Record):
    backend: Literal["model", "scripted"] = "model"
    scripted_protocol: Literal["records"] = "records"
    model_config_path: Path | None = None
    script_file: Path | None = None
    max_calls: PositiveInt | None = None
    format_repairs: int = Field(default=2, ge=0, strict=True)
    supervisor_turns: int = Field(default=2, ge=0, strict=True)
    concurrency: PositiveInt = 4
    max_request_chars: int = Field(default=0, ge=0, strict=True)
    max_tokens: PositiveInt | None = None
    max_elapsed_seconds: PositiveInt | None = None
    semantic_repairs: int = Field(default=2, ge=0, strict=True)

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        config = validate(cls, loads(path.read_bytes(), location=str(path)), location=str(path))
        resolved = {}
        for name, value in config:
            if isinstance(value, Path):
                value = value.expanduser()
                resolved[name] = (value if value.is_absolute() else path.parent / value).resolve()
        return config.model_copy(update=resolved)

    def check_execution(self):
        path = self.model_config_path if self.backend == "model" else self.script_file
        if (self.max_calls is None and "max_calls" not in self.model_fields_set) or path is None or not path.is_file():
            raise ValueError("执行任务必须显式提供 max_calls（正整数或 null）及对应后端的有效配置文件")


# 版本设置只能表达呈现方式，不能提供另一份可被写作者读取的故事正文。
PRESENTATION_TERMS = {
    "genre": ("小说", "现实主义", "悬疑", "抒情", "纪实风格", "文学叙事"),
    "pov": ("全知", "限知", "人物限知", "外部观察", "多视角"),
    "person": ("第一人称", "第二人称", "第三人称"),
    "tense": ("过去时", "现在时"),
    "style": ("白描", "简洁", "克制", "细腻", "口语化", "抒情", "写实"),
    "pacing": ("紧凑", "舒缓", "张弛有度"),
    "reveal_policy": ("顺叙", "倒叙", "插叙", "交错叙述", "逐步揭示", "保留悬念"),
}
PRESENTATION_INSTRUCTIONS = {
    "保留核心故事", "保持故事事实", "保留模拟中已经发生的事实和各人物的知情范围",
    "围绕亲历线索展开", "以第一人称叙述", "以第二人称叙述", "以第三人称叙述",
}.union(*(set(terms) for terms in PRESENTATION_TERMS.values()))


class VersionSpec(Record):
    medium: Literal["novel", "film", "television", "game"] = "novel"
    genre: str = ""
    pov: str = ""
    person: str = ""
    tense: str = ""
    style: str = ""
    pacing: str = ""
    length: PositiveInt | None = None
    length_unit: Literal["characters", "words"] = "characters"
    reveal_policy: str = ""

    @field_validator(*PRESENTATION_TERMS)
    @classmethod
    def presentation_option(cls, value, info):
        options = PRESENTATION_TERMS[info.field_name]
        if value and value not in options:
            raise ValueError("只能选择已定义的表达选项：" + "、".join(options))
        return value


class NovelRestoreNarrativeConfig(ExecutionConfig):
    version_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
    requirements: Identity = "保持故事事实"
    max_calls: PositiveInt | None
    window: int = Field(default=5, ge=0, strict=True)
    max_revisions: int = Field(default=2, ge=0, strict=True)
    max_review_rounds: int = Field(default=2, ge=0, strict=True)
    version: VersionSpec = Field(default_factory=VersionSpec)
    context_chars: PositiveInt = 60000

    @field_validator("requirements")
    @classmethod
    def presentation_only(cls, value):
        terms = [t.strip() for t in re.split(r"[，。、；,;]", value) if t.strip()]
        if not terms or len(terms) > 12 or any(t not in PRESENTATION_INSTRUCTIONS for t in terms):
            raise ValueError("requirements 只接受约定的表达选项，不接受原文或补充情节，可省略并使用 version 配置")
        return value


class Branch(Record):
    name: str = Field(pattern=r"^[\p{L}\p{N}_-]+$", min_length=1, max_length=64)
    fork: "BranchPlan | None" = None


class BranchPlan(Record):
    node_id: Identity
    children: list[Branch] = Field(min_length=2)

    @model_validator(mode="after")
    def distinct_names(self):
        names = [normalize("NFC", child.name).casefold() for child in self.children]
        if len(set(names)) != len(names):
            raise ValueError("同一分叉点的子分支名称必须唯一")
        return self


class BranchOpportunity(Record):
    opportunity: str = Field(min_length=1, max_length=400)
    source_ids: list[Identity] = Field(min_length=1, max_length=6)


class BranchDecision(SemanticPayload):
    reason: str = Field(min_length=1, max_length=600)
    alternatives: list[BranchOpportunity] = Field(description="继续当前线时留空，分叉时给出至少两种有依据的机会")


class NovelRestoreConfig(ExecutionConfig):
    repository: Path
    run_directory: Path
    stop_after_group: Identity | None = None
    branch_plan: BranchPlan | None = None  # 只为读取旧格式的运行配置。
    max_timelines: PositiveInt = 4
    branch_parallelism: PositiveInt = 2
    max_rounds: PositiveInt = 100
    seed: int = Field(default=0, strict=True)
    plan_candidates: int = Field(default=3, ge=1, le=8, strict=True)
    plan_steps: int = Field(default=4, ge=1, le=12, strict=True)  # 旧配置可读；连贯行为不使用步骤配额。
    recent_memories: int = Field(default=5, ge=0, strict=True)
    related_memories: int = Field(default=10, ge=0, strict=True)
    # 旧模拟配置可读；实际写作使用叙述版本自己的 window 和 requirements。
    narrative_window: int = Field(default=5, ge=0, strict=True)
    narrative_requirements: str = ""
    goal_stagnation_batches: PositiveInt = 3
    goal_progress_window: PositiveInt = 3
    max_blocked_replans: PositiveInt = 2
    memory_candidates: PositiveInt = 40
    summary_interval: PositiveInt = 5  # 旧配置可读；摘要只由显式按需任务触发。
    max_world_lookups: PositiveInt = 8
    max_personal_lookups: int = Field(default=2, ge=0, le=8, strict=True)

    def check_paths(self):
        if not (self.repository.is_file() or self.repository.is_dir()):
            raise ValueError(f"repository 目录或数据库不存在：{self.repository}")
        if self.run_directory.exists():
            raise ValueError(f"运行目录已存在，不能覆盖：{self.run_directory}")
        if self.repository.resolve().is_relative_to(self.run_directory.resolve()):
            raise ValueError("输入 repository 不能放在本次新建的运行目录内")
        if self.repository.is_dir() and self.run_directory.resolve().is_relative_to(self.repository.resolve()):
            raise ValueError("运行目录不能放在输入 repository 内")

    def validate_branching(self):
        if self.branch_plan is not None:
            raise ValueError("新实验不接受预设 branch_plan；仅设 max_timelines，运行到节点时由世界决定是否分叉")
        if self.max_timelines > 1 and self.stop_after_group:
            raise ValueError("动态分支实验需要完整叶时间轴，不能同时指定 stop_after_group")
