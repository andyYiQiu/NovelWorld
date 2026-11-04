"""仅投影已提交资料；收据保护已有文件，导出失败不重演模拟。"""
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

from common.NovelRestoreSchema import NarrativePlan
from common.NovelRestoreStory import story_text


def file_identity(identity):
    name = quote(identity, safe="")
    return name.replace(".", "%2E") if name in (".", "..") else name


class NovelRestoreExporter:
    def __init__(self, storage):
        self.storage = storage

    def documents(self):
        states = self.storage.event_states()
        documents, nodes, unfinished = {}, [], []
        for node in self.storage.manifest["nodes"]:
            groups = []
            for group in node["groups"]:
                identity = group["group_id"]
                directory = "simulation/groups/" + file_identity(identity)
                draft = self.storage.get_draft(identity)
                if draft is not None:
                    documents[directory + "/draft.json"] = {**draft.model_dump(mode="json"), "accepted": True}
                    documents[directory + "/story.md"] = draft.text + "\n"
                events = []
                for event_id in group["events"]:
                    actions = self.storage.event_actions(event_id)
                    documents["simulation/events/" + file_identity(event_id) + "/actions.jsonl"] = "".join(
                        row.model_dump_json() + "\n" for row in actions)
                    event = {"event_id": event_id, "status": states[event_id]["status"]}
                    events.append(event)
                    if event["status"] != "completed":
                        unfinished.append(event)
                if draft is None:
                    unfinished.append({"group_id": identity, "reason": "事件组尚无验收后的正式草稿"})
                groups.append({"group_id": identity, "order": group["order"], "events": events,
                               "draft": directory + "/draft.json" if draft else None})
            nodes.append({"node_id": node["node_id"], "order": node["order"], "groups": groups})
        documents["story_timeline.json"] = {"run_id": self.storage.run_id, "time_basis": "story", "nodes": nodes}
        documents["completion.json"] = {"run_id": self.storage.run_id, "progress": self.storage.current()[2],
                                         "complete": not unfinished, "unfinished": unfinished, "usage": self.storage.usage()}
        return {name: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
                for name, value in documents.items()}

    def export_simulation(self):
        self.publish(self.documents(), "export")
        self.export_story_progress()

    def story_progress(self):
        """只摘录已提交故事材料；个人体验和记忆不提升为客观事实。"""
        actions = self.storage.event_actions()
        drafts = self.storage.connection.execute("SELECT count(*) FROM drafts").fetchone()[0]
        lines = ["# 已提交故事材料", "", f"已保存 {len(actions)} 次行动；正式事件组草稿 {drafts} 份。", "",
                 "这是持续更新的材料预览，不是最终小说；片段已提交不等于事件组已通过验收。"
                 "个人体验与记忆保留其主观归属，同批材料的排列不表示先后。", ""]
        lines.append(story_text(actions, self.storage.manifest["characters"]))
        if not actions:
            lines.extend(["尚无已提交的故事材料。", ""])
        return "\n".join(lines)

    def export_story_progress(self):
        """逐批刷新一个可重建预览；独立收据沿用原子发布和人工改动保护。"""
        self.publish({"story_progress.md": self.story_progress()}, "export/story_progress")

    def export_narrative(self, version_id):
        state, status = self.storage.narrative(version_id), self.storage.narrative_status(version_id)
        directory = "narratives/" + file_identity(version_id)
        documents = {}
        documents[directory + "/version.json"] = {"requirements": state["config"]["requirements"],
                                                    **state["config"]["version"]}
        documents[directory + "/prose_metadata.json"] = state["prose_meta"]
        if state["plan"] is not None:
            plan = NarrativePlan.model_validate(state["plan"])
            # plan 只导出固定落位，生效关系单独从 links 真值投影。
            documents[directory + "/plan.json"] = plan.model_dump(mode="json", exclude={"links"})
            documents[directory + "/links.json"] = state["links"]
            for identity in plan.reading_order:
                if identity in state["prose"]:
                    documents[directory + "/prose/" + file_identity(identity) + ".md"] = state["prose"][identity]["text"] + "\n"
        if state["stage"] == "complete":
            if (not status["plan_ready"] or status["missing"] or status["pending_links"] or state["issues"]
                    or state["pending"] or not state["reviews"] or not state["reviews"][-1]["result"]["approved"]
                    or state["reviews"][-1]["epoch"] != state["epoch"]):
                raise ValueError("当前正文或关系未通过完整核对，不能导出完整小说")
            medium = state["config"]["version"]["medium"]
            name = {"novel": "novel.md", "film": "screenplay.fountain", "television": "episodes.md", "game": "game-script.md"}[medium]
            documents[directory + "/" + name] = "\n\n".join(
                (". " + (chapter.setting or chapter.title) if medium == "film" else "# " + chapter.title)
                + "\n\n" + "\n\n".join(state["prose"][identity]["text"] for identity in chapter.fragment_ids)
                for chapter in plan.chapters) + "\n"
        documents[directory + "/completion.json"] = {"run_id": self.storage.run_id,
            "version_id": version_id, "complete": state["stage"] == "complete",
            "status": {key: value for key, value in status.items() if key != "published"}, "reviews": state["reviews"],
            "batch_reviews": state["batch_reviews"], "usage": self.storage.usage("narrative/" + quote(version_id, safe=""))}
        encoded = {name: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
                   for name, value in documents.items()}
        self.publish(encoded, "export/narrative/" + version_id,
                     complete_epoch=state["epoch"] if state["stage"] == "complete" else None)

    def publish(self, documents, key, *, complete_epoch=None):
        hashes = {name: sha256(content.encode()).hexdigest() for name, content in documents.items()}
        receipt = self.storage.get_checkpoint(key, {"published": {}, "pending": {}})
        existing = {}
        # 全部预检通过后才开始发布；未登记文件不自动覆盖。
        for name in documents:
            path = self.storage.directory / name
            if any(parent.is_symlink() for parent in [path, *path.parents] if parent != self.storage.directory.parent):
                raise ValueError("导出路径不能经过符号链接：" + str(path))
            if path.exists():
                digest = sha256(path.read_bytes()).hexdigest()
                allowed = {receipt["published"].get(name), receipt["pending"].get(name)} - {None}
                if digest not in allowed:
                    raise ValueError("导出文件已有未登记或人工修改的内容：" + str(path))
                existing[name] = digest
        self.storage.set_checkpoint(key, {"published": existing, "pending": hashes})
        for name, content in documents.items():
            path = self.storage.directory / name
            if path.exists() and sha256(path.read_bytes()).hexdigest() == hashes[name]:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        self.storage.set_checkpoint(key, {"published": hashes, "pending": {}, "complete_epoch": complete_epoch})
