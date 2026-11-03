"""运行日志只作诊断，恢复依据保存在 SQLite。"""
import json


class NovelRestoreLog:
    def __init__(self, path):
        self.stream = path.open("a", encoding="utf-8")
        self.secret = ""

    def write(self, kind, **values):
        line = json.dumps({"kind": kind, **values}, ensure_ascii=False, default=str)
        if self.secret:
            line = line.replace(self.secret, "[REDACTED]")
        self.stream.write(line + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()
