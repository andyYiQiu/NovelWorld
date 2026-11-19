"""人物主观世界组装、文件定位和记忆检索；数据库读写由存储层负责。"""
from pathlib import Path
from importlib.metadata import distribution
import json
import logging
import re
from urllib.parse import quote

import jieba
from rank_bm25 import BM25Okapi

from common.NovelRestoreSchema import MemoryRecord, SubjectiveWorld, WorldPerception
from tools.RuntimeFiles import regular_opener


# LangChain BM25Retriever uses an explicit preprocessing function + BM25Okapi.
# https://github.com/langchain-ai/langchain-community/blob/f425a3ed1933173fb3694b81359d1519c4f82d36/libs/community/langchain_community/retrievers/bm25.py
# Independent implementations/data, pinned in requirements.txt:
# https://github.com/dorianbrown/rank_bm25/blob/2550648efdbdcc5ebadbc8e5c8b26f5eb94b2b36/rank_bm25.py
# https://github.com/fxsjy/jieba/blob/1e20c89b66f56c9301b0feed211733ffaa1bd72a/jieba/__init__.py
# https://github.com/stopwords-iso/stopwords-iso
# Scope/cue/importance rules below are NovelWorld method boundaries, not BM25 features.
jieba.setLogLevel(logging.WARNING)
_TOKENIZER = jieba.Tokenizer()
# Only its packaged dataset is used: Python 3.11+ stdlib replaces the package's
# deprecated pkg_resources loader, without adding setuptools to experiment runtime.
_STOPWORD_DATA = json.loads(distribution("stopwordsiso").locate_file(
    "stopwordsiso/stopwords-iso.json").read_text(encoding="utf-8"))
_STOPWORDS = frozenset(_STOPWORD_DATA["zh"] + _STOPWORD_DATA["en"])


def retrieval_tokens(text):
    """标准搜索分词与中英停用词表；不猜测问句的主谓或主题。"""
    return [token for token in _TOKENIZER.cut_for_search(text.casefold(), HMM=False)
            if token not in _STOPWORDS and any(char.isalnum() for char in token)]


def literal_cue(cue):
    """主题词逐字匹配，对英文和编号使用词边界。调用方以 casefold 后的正文比较。"""
    cue = cue.casefold()
    left = r"(?<![a-z0-9_])" if re.match(r"[a-z0-9_]", cue) else ""
    right = r"(?![a-z0-9_])" if re.search(r"[a-z0-9_]$", cue) else ""
    return re.compile(left + re.escape(cue) + right)


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
        with open(path, "rb", opener=regular_opener) as stream:
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
    def _rank(records, query, *, related_ids=(), world_links=None, required_cues=()):
        """BM25 排序合法候选；同分按重要性、时点排序，不让重要性创造命中。"""
        records = list(records)
        if not records:
            return []
        terms = retrieval_tokens(query)
        entities = [("entity", identity) for identity in sorted(set(related_ids))]
        terms.extend(entities)
        corpus = []
        for row in records:
            tokens = retrieval_tokens(row.content)
            if entities:
                linked = set(row.role_ids + row.object_ids + row.location_ids)
                linked.update((world_links or {}).get(row.memory_id, ()))
                tokens.extend(("entity", identity) for identity in sorted(linked))
            corpus.append(tokens)
        query_terms = set(terms)
        # Okapi 在单条或极小语料上可能给出零/负 IDF；实存匹配不以分数正负判断。
        eligible = [index for index, (row, tokens) in enumerate(zip(records, corpus))
                    if (any(cue.search(row.content.casefold()) for cue in required_cues)
                        if required_cues else bool(query_terms.intersection(tokens)))]
        if not eligible:
            return []
        scores = BM25Okapi(corpus).get_scores(terms) if terms and any(corpus) else [0.0] * len(records)
        eligible.sort(key=lambda index: (float(scores[index]), records[index].importance,
                                         records[index].position.sort_key, index), reverse=True)
        return [records[index] for index in eligible]

    @classmethod
    def search(cls, records, query, limit, cues=()):
        """只查主动问题；完整线索限定准入，无命中不补近期记录。"""
        if limit <= 0 or not query.strip():
            return []
        needles = []
        if cues:
            distinct = dict.fromkeys(cue.strip().casefold() for cue in cues if cue.strip())
            if not distinct:
                return []
            needles = [literal_cue(cue) for cue in distinct]
            if any(not needle.search(query.casefold()) for needle in needles):
                raise ValueError("每个主题线索必须直接出自本次问题")
        return cls._rank(records, query, required_cues=needles)[:limit]

    @classmethod
    def select(cls, records, *, query="", related_ids=(), recent=5, relevant=10, world_links=None):
        """世界侧按显式范围查近期、BM25 相关记录及纠正链；不用于人物提问。"""
        if recent < 0 or relevant < 0:
            raise ValueError("检索数量不能为负")
        records = sorted(records, key=lambda row: row.position.sort_key)
        ranked = cls._rank(records, query, related_ids=related_ids, world_links=world_links) if relevant else []
        selected = {row.memory_id for row in records[-recent:]} if recent else set()
        selected.update(row.memory_id for row in ranked[:relevant])
        return cls.with_corrections(records, selected)

    @staticmethod
    def with_corrections(records, selected):
        """从可见记录中带上完整纠正链，不能检出获知时间之后的纠正。"""
        # 双向邻接只索引已提供的记录；普通图遍历避免长链反复扫描全部历史。
        links = {}
        for row in records:
            links.setdefault(row.memory_id, set()).update(row.corrects)
            for previous in row.corrects:
                links.setdefault(previous, set()).add(row.memory_id)
        selected = set(selected)
        pending = list(selected)
        while pending:
            neighbors = links.get(pending.pop(), set()) - selected
            selected.update(neighbors)
            pending.extend(neighbors)
        return [row for row in records if row.memory_id in selected]
