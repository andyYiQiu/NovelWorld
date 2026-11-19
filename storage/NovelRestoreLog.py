"""运行日志只作诊断，恢复依据保存在 SQLite。"""
import json
import sys
from datetime import datetime

from tools.RuntimeFiles import regular_opener


class NovelRestoreLog:
    progress = False

    @classmethod
    def report(cls, kind, **values):
        """终端只展示执行位置及用量，不打印请求正文、回执正文或密钥。"""
        if not cls.progress:
            return
        labels = {"task_started": "开始任务", "task_completed": "任务完成",
                  "request": "发送请求", "received": "收到响应", "replay": "复用已保存响应",
                  "delivery_unknown": "请求送达状态未知", "error": "接口异常",
                  "failed": "运行失败", "paused": "已暂停", "interrupted": "已中断",
                  "simulation": "模拟进度"}
        if kind not in labels:
            return
        fields = ("step", "node", "group", "round", "actions", "role", "job", "call", "status", "usage")
        detail = " | ".join(f"{key}={values[key]}" for key in fields if key in values)
        try:
            print(f"[{datetime.now().astimezone():%H:%M:%S}] {labels[kind]} {detail}",
                  file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass  # 终端断开不能使已记账请求或正式行动丢失。

    def __init__(self, path):
        self.secret = ""
        self.stream = None
        self.error = None
        try:
            self.stream = open(path, "a", encoding="utf-8",
                               opener=regular_opener)
        except OSError as error:
            self._disable(error)

    def _disable(self, error):
        """诊断文件不是权威账本；磁盘/权限异常不能阻止 SQLite 回执交接。"""
        message = str(error).replace(self.secret, "[REDACTED]") if self.secret else str(error)
        if self.error is None:
            self.error = message
            try:
                sys.stderr.write("NovelWorld: 诊断日志不可写，已禁用本次文件日志；运行记录仍以 SQLite 为准：" + message + "\n")
            except (OSError, ValueError):
                pass
        stream = self.stream
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def write(self, kind, **values):
        self.report(kind, **values)
        if self.stream is None or self.error is not None:
            return
        line = json.dumps({"kind": kind, **values}, ensure_ascii=False, default=str)
        if self.secret:
            line = line.replace(self.secret, "[REDACTED]")
        try:
            self.stream.write(line + "\n")
            self.stream.flush()
        except (OSError, ValueError) as error:
            self._disable(error)

    def close(self):
        stream = self.stream
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError) as error:
                self._disable(error)
