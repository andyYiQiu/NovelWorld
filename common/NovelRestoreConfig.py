"""显式配置模拟参数；路径相对于配置文件解析，不读取模型密钥。"""
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from common.NovelRestoreSchema import ActionMode, Identity, PositiveInt, Record


class ExecutionConfig(Record):
    backend: Literal["model", "scripted"] = "model"
    model_config_path: Path | None = None
    script_file: Path | None = None
    max_calls: PositiveInt | None = None
    format_repairs: int = Field(default=2, ge=0, strict=True)
    concurrency: PositiveInt = 4
    max_request_chars: int = Field(default=0, ge=0, strict=True)
    max_tokens: PositiveInt | None = None
    max_elapsed_seconds: PositiveInt | None = None
    semantic_repairs: int = Field(default=2, ge=0, strict=True)

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        config = cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        resolved = {}
        for name, value in config:
            if isinstance(value, Path):
                value = value.expanduser()
                resolved[name] = (value if value.is_absolute() else path.parent / value).resolve()
        return config.model_copy(update=resolved)

    def check_execution(self):
        path = self.model_config_path if self.backend == "model" else self.script_file
        if self.max_calls is None or path is None or not path.is_file():
            raise ValueError("执行任务必须提供 max_calls 及对应后端的有效配置文件")


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


class NovelRestoreNarrativeConfig(ExecutionConfig):
    version_id: Identity
    requirements: Identity
    max_calls: PositiveInt
    window: int = Field(default=5, ge=0, strict=True)
    max_revisions: int = Field(default=2, ge=0, strict=True)
    max_review_rounds: int = Field(default=2, ge=0, strict=True)
    version: VersionSpec = Field(default_factory=VersionSpec)
    context_chars: PositiveInt = 60000


class NovelRestoreConfig(ExecutionConfig):
    repository: Path
    run_directory: Path
    stop_after_group: Identity | None = None
    max_rounds: PositiveInt = 100
    seed: int = Field(default=0, strict=True)
    character_action_ratios: dict[str, dict[ActionMode, float]] = Field(default_factory=dict)
    action_ratios: dict[ActionMode, float] = Field(default_factory=lambda: {
        "action": 0.4, "talk": 0.3, "think": 0.2, "idle": 0.1,
    })
    recent_memories: int = Field(default=5, ge=0, strict=True)
    related_memories: int = Field(default=10, ge=0, strict=True)
    narrative_window: int = Field(default=5, ge=0, strict=True)
    narrative_requirements: str = ""
    stagnation_rounds: PositiveInt = 3
    memory_candidates: PositiveInt = 40
    summary_interval: PositiveInt = 5
    max_world_lookups: PositiveInt = 8

    @model_validator(mode="after")
    def valid_ratios(self):
        for ratios in [self.action_ratios, *self.character_action_ratios.values()]:
            if set(ratios) != {"action", "talk", "think", "idle"}:
                raise ValueError("行动比例必须明确提供 action/talk/think/idle 四种比例")
            values = ratios.values()
            if any(not math.isfinite(value) or value < 0 for value in values) or not math.isclose(sum(values), 1, abs_tol=1e-9):
                raise ValueError("行动比例必须非负、有限且总和为 1")
        return self

    def check_paths(self):
        if not (self.repository.is_file() or self.repository.is_dir()):
            raise ValueError(f"repository 目录或数据库不存在：{self.repository}")
        if self.run_directory.exists():
            raise ValueError(f"运行目录已存在，不能覆盖：{self.run_directory}")
        if self.repository.resolve().is_relative_to(self.run_directory.resolve()):
            raise ValueError("输入 repository 不能放在本次新建的运行目录内")
        if self.repository.is_dir() and self.run_directory.resolve().is_relative_to(self.repository.resolve()):
            raise ValueError("运行目录不能放在输入 repository 内")
