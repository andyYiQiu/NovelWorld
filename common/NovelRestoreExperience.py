"""人物输入的唯一投影入口；原始历史和认知作用说明只在世界侧使用。"""
import asyncio
from copy import deepcopy

from agents.NovelRestoreAgent import fingerprint, task_id
from agents.NovelRestoreCharacterAgent import NovelRestoreCharacterAgent, personal_recalls
from common.Runtime import RunStopped
from common.NovelRestoreSchema import Record, SubjectiveInput, CharacterState, ProcessingTrace, RecalledMemory, MemorySource
from common.NovelRestoreVisibility import is_visible, record_visible, visible_task


class ExperienceBatch(Record):
    people: dict[str, SubjectiveInput]


def advance_needs(states, seconds):
    """只推进主观需求的既定变化率；事件反馈随后可改变本人感受。"""
    result = deepcopy(states)
    for state in result.values():
        for need in state.get("needs", {}).values():
            need["urgency"] = min(10, max(0, need["urgency"] + need["drift_per_hour"] * seconds / 3600))
    return result


class DeliveredRecall(RecalledMemory):
    source: MemorySource


class RecallResult(Record):
    memories: list[DeliveredRecall]
    processing: list[ProcessingTrace]


async def retrieve_for_actor(world, identity, person, access, view, runtime):
    """人物先提出问题；命中原文、真实对应及处理轨迹均留在世界侧。"""
    storage = world.storage
    query = await NovelRestoreCharacterAgent(person, storage, world.executor).recall_query(identity, access, view)
    if not query:
        return
    rows = storage.search_personal_memories(person, query)
    if not rows:
        return
    if (len({r.memory_id for r in rows}) != len(rows)
            or any((r.run_id, r.character_id) != (storage.run_id, person) for r in rows)):
        raise ValueError("检索命中须唯一且属于本次运行的本人")
    hits = [r.model_dump(include={"memory_id", "content", "source"}) for r in rows]
    effects = {e["effect_id"]: e for e in runtime.get("effects", []) if e["stage"] == "retrieval"}
    if not effects:
        access.memories = [RecalledMemory.model_validate(m) for m in hits]
    else:
        known = {m["memory_id"]: m for m in hits}
        def validate(result):
            changed = set()
            for trace in result.processing:
                if (trace.stage != "retrieval" or not trace.effect_ids or set(trace.effect_ids) - effects.keys()
                        or not trace.source_ids or set(trace.source_ids) - known.keys()):
                    raise ValueError("回想变换只能引用本人的有效提取作用和本次命中")
                changed.update(trace.source_ids)
            seen = set()
            for m in result.memories:
                if m.memory_id in seen or m.memory_id not in known:
                    raise ValueError("交付回想须引用不同的本次命中")
                if m.model_dump() != known[m.memory_id] and m.memory_id not in changed:
                    raise ValueError("改变回想内容或来源须绑定已生效的提取作用")
                seen.add(m.memory_id)
            if known.keys() - seen - changed:
                raise ValueError("屏蔽已命中记忆须绑定提取作用")

        objective = storage.current()[1].characters.get(person)
        result = await visible_task(world.executor, task_id("world", "process_recall", identity, person), world.prompt,
            {"operation": "process_recall", "character_id": person, "query": query, "hits": hits,
             "perception": view, "objective_state": objective.model_dump(mode="json") if objective else None,
             "runtime": runtime}, RecallResult, validate,
            lambda r: {person: {"memories": personal_recalls(r.memories)}},
            private_review=lambda r: {"processing": [trace.model_dump() for trace in r.processing]})
        access.memories = [RecalledMemory.model_validate(m.model_dump()) for m in result.memories]
        access.processing.extend(result.processing)


async def prepare_experience(world, identity, people, *, observations=None, runtimes=None, states=None,
                             elapsed_seconds=None, acted=()):
    if not people:
        return {}
    storage = world.storage
    after_action = observations is not None
    observations = observations or {}
    runtimes = runtimes if runtimes is not None else storage.get_checkpoint("person_runtime", {})
    states = states if states is not None else {p: storage.character_state(p).model_dump() for p in people}
    sources = {p: {
        "profile": storage.get_checkpoint("self_profiles", {}).get(p, {}), "working_state": states[p],
        "runtime": runtimes.get(p, {}), "perception": storage.get_checkpoint("views", {}).get(p, {}).get("personal", {}),
        "observations": observations.get(p, []), "memories": [],
    } for p in people}

    initial = {p: storage.character_state(p).model_dump() for p in people}
    expected = advance_needs(initial, elapsed_seconds) if elapsed_seconds is not None else initial
    personal = {p: [{k: o[k] for k in ("source_id", "source_kind", "observation")}
                    for o in sources[p]["observations"]] for p in people}
    for rows in personal.values():
        if len({o["source_id"] for o in rows}) != len(rows):
            raise ValueError("个人体验来源编号重复，不能覆盖已有体验")

    async def finish(accesses):
        if after_action:
            # 只沿用本批此人刚刚实际体验的回想，旁观者不会读取旧批次的 access。
            for p in set(people).intersection(acted):
                old_effects = storage.get_checkpoint("person_runtime", {}).get(p, {}).get("effects", [])
                new_effects = runtimes.get(p, {}).get("effects", [])
                if ([e for e in old_effects if e["stage"] == "retrieval"]
                        == [e for e in new_effects if e["stage"] == "retrieval"]):
                    before = storage.get_checkpoint("access/" + p)
                    if before:
                        accesses[p].memories = SubjectiveInput.model_validate(before).memories
        else:
            def stamp():
                version, state, progress = storage.current()
                return fingerprint({"version": version, "state": state.model_dump(), "progress": progress,
                    "people": {p: storage.character_state(p).model_dump() for p in people},
                    "profiles": storage.get_checkpoint("self_profiles", {}), "views": storage.get_checkpoint("views", {}),
                    "runtime": storage.get_checkpoint("person_runtime", {})})
            before = stamp()
            outcomes = await asyncio.gather(*(retrieve_for_actor(world, identity, p, accesses[p], sources[p]["perception"],
                                                      sources[p]["runtime"]) for p in people), return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            if stamp() != before:
                raise RunStopped("回想期间人物或世界依据已改变，不能批准旧资料")
        for p, access in accesses.items():
            if access.memories:
                record_visible(storage, p, {"memories": personal_recalls(access.memories)}, "approved/recall")
        return accesses

    if storage.get_job(task_id("world", "experience", identity)) is None and all(
            not source["runtime"].get("effects") and source["profile"] == storage.manifest["characters"][p]
            and source["working_state"] == expected[p] and is_visible(storage, p, "working_state", initial[p])
            and (not personal[p] or is_visible(storage, p, "observations", personal[p]))
            for p, source in sources.items()):
        accesses = {p: SubjectiveInput(profile=sources[p]["profile"], working_state=expected[p],
            experiences=[{"source_id": o["source_id"], "source_kind": o["source_kind"], "experience": o["observation"]}
                         for o in personal[p]]) for p in people}
        for p in people:
            if expected[p] != initial[p]:
                record_visible(storage, p, {"working_state": expected[p]}, "deterministic/need-drift")
        return await finish(accesses)

    def validate(result):
        if set(result.people) != set(people):
            raise ValueError("个人输入须恰好覆盖指定人物")
        for person, access in result.people.items():
            source = sources[person]
            if access.memories:
                raise ValueError("感知与编码准备阶段不能自行检索或交付历史记忆")
            events = {o["source_id"]: o for o in source["observations"]}
            if len(events) != len(source["observations"]):
                raise ValueError("个人体验来源编号重复，不能覆盖已有体验")
            effects = {e["effect_id"]: e for e in source["runtime"].get("effects", [])}
            covered = {"retrieval": set(), "encoding": set()}
            for trace in access.processing:
                if (trace.stage not in covered or set(trace.effect_ids) - effects.keys()
                        or any(effects[e]["stage"] != trace.stage for e in trace.effect_ids)):
                    raise ValueError("个人输入变化必须引用此人当前生效、对应阶段的认知作用")
                valid = {"working_state", "profile"} if trace.stage == "retrieval" else set(events)
                if not trace.source_ids or set(trace.source_ids) - valid:
                    raise ValueError("认知处理须绑定本次实际输入来源")
                covered[trace.stage].update(trace.source_ids)
            if access.working_state != CharacterState.model_validate(source["working_state"]):
                if "working_state" not in covered["retrieval"]:
                    raise ValueError("改变可回想的心理/打算须有认知依据")
            if access.profile is not None and access.profile != source["profile"] and "profile" not in covered["retrieval"]:
                raise ValueError("改变可自知资料须有当前提取作用依据")
            if (len(access.experiences) != len(events)
                    or {e.source_id for e in access.experiences} != set(events)):
                raise ValueError(f"人物 {person} 的 experiences 须与本次 observations 来源一一对应；"
                                 f"实际输入来源为 {sorted(events)}；输入为空时 experiences 必须为空列表")
            for experience in access.experiences:
                original = events[experience.source_id]
                if experience.source_kind != original["source_kind"]:
                    raise ValueError("体验的行动或组级来源类型不得改变")
                raw = original["observation"]
                actual = experience.experience.model_dump() if experience.experience else None
                if actual != raw and experience.source_id not in covered["encoding"]:
                    raise ValueError("改变或不保留本次体验须有编码作用依据")

    def project(result):
        return {p: {**({"profile": value.profile} if value.profile is not None
                       and value.profile != sources[p]["profile"] else {}),
                    "perception": next((e.experience.model_dump() for e in value.experiences if e.experience is not None), None),
                    "working_state": value.working_state.model_dump(),
                    "memories": [m.model_dump() for m in value.memories],
                    "observations": [{"source_id": e.source_id, "source_kind": e.source_kind,
                                      "observation": e.experience.model_dump()}
                                     for e in value.experiences if e.experience is not None]}
                for p, value in result.people.items()}
    result = await visible_task(world.executor, task_id("world", "experience", identity), world.prompt,
        {"operation": "prepare_experience", "people": sources,
         "working_state_phase": "before_character_update",
         "memory_limit": 0}, ExperienceBatch, validate, project)
    for person, access in result.people.items():
        if access.profile is None:
            access.profile = sources[person]["profile"]
    return await finish(result.people)
