"""人物输入的唯一投影入口；原始历史和认知作用说明只在世界侧使用。"""
from common.Runtime import RunStopped, fingerprint, task_id
from agents.NovelRestoreCharacterAgent import RecallQuery, personal_recalls
from simulation.NovelRestoreObjects import effective_catalog, local_perception_ids
from common.NovelRestoreSchema import Record, SubjectiveInput, CharacterState, ProcessingTrace, RecalledMemory, MemorySource
from simulation.NovelRestoreVisibility import (
    is_initial_profile, is_visible, record_visible, valid_personal_profile, visible_task, working_state_approval,
)


class ExperienceBatch(Record):
    people: dict[str, SubjectiveInput]



class DeliveredRecall(RecalledMemory):
    source: MemorySource


class RecallResult(Record):
    memories: list[DeliveredRecall]
    processing: list[ProcessingTrace]


async def retrieve_for_actor(world, identity, person, access, view, runtime, request):
    """仅响应行为选择中的实际查询；工具内部完成认知处理后才交付。"""
    storage = world.storage
    query = request.query
    if not query:
        return
    rows = storage.search_personal_memories(person, query, cues=request.cues)
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
        result = await visible_task(world.supervisor, task_id("world", "process_recall", identity, person), world.prompt,
            {"operation": "process_recall", "character_id": person, "query": query, "hits": hits,
             "perception": view, "objective_state": objective.model_dump(mode="json") if objective else None,
             "runtime": runtime}, RecallResult, validate,
            lambda r: {person: {"memories": personal_recalls(r.memories)}},
            private_review=lambda r: {"processing": [trace.model_dump() for trace in r.processing]})
        access.memories = [RecalledMemory.model_validate(m.model_dump()) for m in result.memories]
        access.processing.extend(result.processing)


async def prepare_experience(world, identity, people, *, observations=None, runtimes=None, states=None,
                             acted=(), catalog=None):
    if not people:
        return {}
    storage = world.storage
    catalog = effective_catalog(storage) if catalog is None else catalog
    after_action = observations is not None
    observations = observations or {}
    runtimes = runtimes if runtimes is not None else storage.get_checkpoint("person_runtime", {})
    states = states if states is not None else {p: storage.character_state(p).model_dump() for p in people}
    sources = {p: {
        "profile": storage.get_checkpoint("self_profiles", {}).get(p, {}), "working_state": states[p],
        "runtime": runtimes.get(p, {}), "perception": storage.get_checkpoint("views", {}).get(p, {}).get("personal", {}),
        "observations": observations.get(p, []), "memories": [],
    } for p in people}
    if any(not is_initial_profile(storage, p, source["profile"]) for p, source in sources.items()):
        raise RunStopped("缺少本人已批准的起点档案，不能据此生成体验")

    initial = {p: storage.character_state(p).model_dump() for p in people}
    personal = {p: [{k: o[k] for k in ("source_id", "source_kind", "observation")}
                    for o in sources[p]["observations"]] for p in people}
    for rows in personal.values():
        if len({o["source_id"] for o in rows}) != len(rows):
            raise ValueError("个人体验来源编号重复，不能覆盖已有体验")
    job = task_id("world", "experience", identity)
    review_prefix = task_id("visibility", job) + "/"
    # 本任务产生的批准不能回头成为自己的依据，也不能改变恢复时的任务指纹。
    approvals = {p: approval for p in people
                 if (approval := working_state_approval(storage, p, initial[p])) is not None
                 and not approval["source"].startswith(review_prefix)}
    processing = any(effect["stage"] == "retrieval" or effect["stage"] == "encoding" and personal[p]
                     for p, source in sources.items() for effect in source["runtime"].get("effects", []))
    def copied_inputs():
        return {p: SubjectiveInput(profile=sources[p]["profile"], working_state=sources[p]["working_state"],
            experiences=[{"source_id": o["source_id"], "source_kind": o["source_kind"], "experience": o["observation"]}
                         for o in personal[p]]) for p in people}

    def passthrough_source(person, source):
        """本层无语义变换权时，只核对上游收据，不发起模型任务。"""
        state = CharacterState.model_validate(source["working_state"])
        pristine = (not storage.get_checkpoint("subjective_initialized/" + person)
                    and state == CharacterState())
        missing = []
        if not (is_visible(storage, person, "working_state", source["working_state"])
                or source["working_state"] == initial[person] and person in approvals
                or pristine):
            missing.append("working_state")
        # 行动前会将当前感知交给人物；行动后只交接新观察。
        # 因此后者不需要旧感知的当前版本收据，也不再重复模型审核。
        if (not after_action and source["perception"]
                and not is_visible(storage, person, "perception", source["perception"])):
            missing.append("perception")
        if personal[person] and not is_visible(storage, person, "observations", personal[person]):
            missing.append("observations")
        return missing, pristine

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
        for p, access in accesses.items():
            if access.memories:
                record_visible(storage, p, {"memories": personal_recalls(access.memories)}, "approved/recall")
        return accesses

    if not processing:
        checked = {p: passthrough_source(p, source) for p, source in sources.items()}
        missing = {p: fields for p, (fields, _) in checked.items() if fields}
        if missing:
            details = "; ".join(f"{person}: {', '.join(fields)}" for person, fields in missing.items())
            raise RunStopped("未加工人物输入缺少上游知情批准，不能交给模型重新判断或抄写：" + details)
        accesses = copied_inputs()
        for p in people:
            if not is_visible(storage, p, "working_state", sources[p]["working_state"]):
                source = ("program/pristine-working-state" if checked[p][1]
                          else approvals.get(p, {}).get("source", "approved/current-working-state"))
                record_visible(storage, p, {"working_state": sources[p]["working_state"]}, source)
        return await finish(accesses)

    def validate(result):
        if set(result.people) != set(people):
            raise ValueError("个人输入须恰好覆盖指定人物")
        for person, access in result.people.items():
            source = sources[person]
            if access.memories:
                raise ValueError("感知与编码准备阶段不能自行检索或交付历史记忆")
            events = {o["source_id"]: o for o in source["observations"]}
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
            if access.profile is not None:
                if not valid_personal_profile(access.profile):
                    raise ValueError("可自知资料仅可含本人可获知的事实文字")
                if access.profile != source["profile"] and "profile" not in covered["retrieval"]:
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
                if (actual is not None and
                        not local_perception_ids((o["perceived_id"] for o in actual["objects"]), catalog)):
                    raise ValueError("编码后的体验须使用局部感知编号，不能占用真实资产编号")
                if actual != raw and experience.source_id not in covered["encoding"]:
                    raise ValueError("改变或不保留本次体验须有编码作用依据")

    def project(result):
        return {p: {**({"profile": value.profile} if value.profile is not None
                       and value.profile != sources[p]["profile"] else {}),
                    # 行动前交付的是当前感知；行动后首次初始化才以第一条新体验作为入口。
                    "perception": (next((e.experience.model_dump() for e in value.experiences if e.experience is not None), None)
                                   if after_action else sources[p]["perception"]),
                    "working_state": value.working_state.model_dump(),
                    "memories": [m.model_dump() for m in value.memories],
                    "observations": [{"source_id": e.source_id, "source_kind": e.source_kind,
                                      "observation": e.experience.model_dump()}
                                     for e in value.experiences if e.experience is not None]}
                for p, value in result.people.items()}
    result = await visible_task(world.supervisor, job, world.prompt,
        {"operation": "prepare_experience", "people": sources,
         "working_state_phase": "before_character_update",
         "working_state_approvals": {p: approval for p, approval in approvals.items()
                                     if sources[p]["working_state"] == initial[p]},
         "memory_limit": 0}, ExperienceBatch, validate, project)
    for person, access in result.people.items():
        if access.profile is None:
            access.profile = sources[person]["profile"]
    return await finish(result.people)


def personal_lookup(world, row):
    """任务绑定的本人回想工具；缓存获准返回，恢复不重做认知加工。"""
    storage, person = world.storage, row["character_id"]
    identity = row["action_id"]
    runtime = storage.get_checkpoint("person_runtime", {}).get(person, {})
    def dependencies():
        return fingerprint({"version": storage.current()[0],
            "state": storage.character_state(person).model_dump(mode="json"),
            "profile": storage.get_checkpoint("self_profiles", {}).get(person),
            "runtime": storage.get_checkpoint("person_runtime", {}).get(person, {}),
            "view": storage.get_checkpoint("views", {}).get(person),
            "memories": [m.model_dump(mode="json") for m in storage.memories(person)]})
    original = dependencies()
    async def recall(request):
        request = RecallQuery.model_validate(request)
        if dependencies() != original:
            raise RunStopped("回想依据已改变，不能继续旧人物决策")
        key = task_id("personal_recall", identity, fingerprint(request.model_dump()))
        saved = storage.get_checkpoint(key)
        if saved is None:
            access = SubjectiveInput.model_validate(storage.get_checkpoint("access/" + person))
            access.memories = []
            await retrieve_for_actor(world, key, person, access, row["context"]["perception"], runtime, request)
            if dependencies() != original:
                raise RunStopped("回想期间依据已改变")
            saved = {"dependency": original, "delivery": access.model_dump(mode="json")}
            storage.set_checkpoint(key, saved)
        if saved["dependency"] != original:
            raise RunStopped("回想缓存依据已改变，不能复用旧交付")
        delivered = SubjectiveInput.model_validate(saved["delivery"])
        access = SubjectiveInput.model_validate(storage.get_checkpoint("access/" + person))
        # 同一原记忆的两次获准回想可能不同，不能用源编号覆盖或吞掉实际体验。
        access.memories.extend(m for m in delivered.memories if m not in access.memories)
        access.processing.extend(p for p in delivered.processing if p not in access.processing)
        storage.set_checkpoint("access/" + person, access.model_dump(mode="json"))
        result = personal_recalls(access.memories)
        if result:
            record_visible(storage, person, {"memories": result}, key)
        # 仅返回本次命中；旧回想仍保存在任务与行动中，零命中不拿旧内容填充。
        return {"memories": [result[access.memories.index(m)] for m in delivered.memories]}
    recall.dependency = original
    return recall
