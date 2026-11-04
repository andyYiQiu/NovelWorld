"""人物主观世界组装、文件定位和记忆检索；数据库读写由存储层负责。"""
from pathlib import Path
import re
from urllib.parse import quote

from common.NovelRestoreSchema import MemoryRecord, SubjectiveWorld, WorldPerception


class MemoryCorruption(ValueError):
    pass


class NovelRestoreSubjectiveWorld:
    def __init__(self, directory, run_id, character_ids, *, legacy=False):
        self.directory = Path(directory)
        self.run_id = run_id
        self.character_ids = frozenset(character_ids)
        self.legacy = legacy

    @staticmethod
    def project(character_id, *, profile, view, working_state, history, summaries):
        """组装个人文件，只保留本人感知，不带世界私有对应关系。"""
        return SubjectiveWorld(character_id=character_id, profile=profile,
            current={"perception": WorldPerception.model_validate(view).personal if view else None,
                     "working_state": working_state},
            history=history, summaries=summaries).model_dump(mode="json")

    def path(self, character_id):
        if character_id not in self.character_ids:
            raise ValueError(f"人物不属于本次运行：{character_id}")
        return self.directory / (quote(character_id, safe="") + (".jsonl" if self.legacy else ".json"))

    def read(self, character_id, *, committed_actions):
        """仅用于历史格式的只读状态查询，不迁移或覆盖旧文件。"""
        result, seen = [], set()
        path = self.path(character_id)
        with path.open("rb") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    if not line.endswith(b"\n"):
                        raise ValueError("末行写入不完整")
                    record = MemoryRecord.model_validate_json(line)
                    if (record.run_id, record.character_id) != (self.run_id, character_id) or record.memory_id in seen:
                        raise ValueError("记忆身份重复或人物绑定不匹配")
                    seen.add(record.memory_id)
                except ValueError as error:
                    raise MemoryCorruption(f"{path}:{number}: {error}") from error
                if record.action_id in committed_actions:
                    result.append(record)
        return result

    @staticmethod
    def _terms(text):
        words = set(re.findall(r"[a-z0-9_]+", text.casefold()))
        for phrase in re.findall(r"[\u3400-\u9fff]+", text):
            words.update(phrase[i:i + 2] for i in range(max(1, len(phrase) - 1)))
        return words

    @classmethod
    def _rank(cls, records, terms, entities=(), single_chars=()):
        """按正文词项与可选实体重合排序，同分优先较新的记录。"""
        scores = []
        entities = set(entities)
        for index, row in enumerate(records):
            content_terms = cls._terms(row.content)
            content_terms.update(t for t in single_chars if t in row.content)
            overlap = len(terms & content_terms) / max(1, len(terms | content_terms))
            score = len(entities.intersection(row.role_ids + row.object_ids + row.location_ids)) + overlap
            if score > 0:
                scores.append((score * (1 + row.importance / 10), index, row))
        scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [row for _, _, row in scores]

    @classmethod
    def search(cls, records, query, limit):
        """主动回想只取问题文字的匹配项，不用时间、评分或纠正关联单独构成命中。"""
        # 常见问句成分不能把“谁跟我提过雨伞放在哪里”当成钥匙问题的答案。
        content = re.sub(r"谁跟我|谁告诉我|谁说过|跟我|告诉我|提过|说过|放在|哪里|哪儿|什么|"
                         r"\b(?:who|what|where|when|how|was|were|the|and|did|does|to|in|is|are)\b",
                         " ", query.casefold())
        terms = cls._terms(content)
        if not terms or limit <= 0:
            return []
        single_chars = {t for t in terms if len(t) == 1 and "\u3400" <= t <= "\u9fff"}
        return cls._rank(records, terms, single_chars=single_chars)[:limit]

    @classmethod
    def select(cls, records, *, query="", related_ids=(), recent=5, relevant=10):
        """世界侧查阅可结合近期经历和纠正链；不用于人物主动提问。"""
        if recent < 0 or relevant < 0:
            raise ValueError("检索数量不能为负")
        records = sorted(records, key=lambda row: row.position.sort_key)
        ranked = cls._rank(records, cls._terms(query), related_ids)
        selected = {row.memory_id for row in records[-recent:]} if recent else set()
        selected.update(row.memory_id for row in ranked[:relevant])
        return cls.with_corrections(records, selected)

    @staticmethod
    def with_corrections(records, selected):
        """从可见记录中带上完整纠正链，不能检出获知时间之后的纠正。"""
        selected = set(selected)
        while True:
            before = len(selected)
            for row in records:
                if row.memory_id in selected or selected.intersection(row.corrects):
                    selected.update([row.memory_id, *row.corrects])
            if len(selected) == before:
                return [row for row in records if row.memory_id in selected]
