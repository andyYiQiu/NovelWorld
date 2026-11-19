"""在故事起点为人物编制可知档案。档案、取证及审核留在世界侧。"""
from pydantic import Field

from common.Runtime import fingerprint, task_id, gather_batch
from common.Runtime import RunStopped, encode
from common.NovelRestoreSchema import Identity, Record
from simulation.NovelRestoreVisibility import valid_personal_profile, visible_task, record_visible


class ProfileFact(Record):
    text: Identity
    field: Identity = Field(description="程序绑定的本人来源位置；历史路径保留只读校验兼容")
    excerpt: Identity = Field(description="程序保存的完整来源编码，或所选来源内可直接定位的连续文字；不能改写、拼接或借用别人的来源")
    knowledge_basis: Identity


class InitialProfile(Record):
    facts: list[ProfileFact]


class InitialProfiles(Record):
    people: dict[str, InitialProfile]

    def delivery(self):
        return {person: {"profile": {"facts": [fact.text for fact in profile.facts]}}
                for person, profile in self.people.items()}


def contains_excerpt(value, excerpt):
    if isinstance(value, str):
        return excerpt in value
    if isinstance(value, dict):
        value = value.values()
    elif not isinstance(value, list):
        return False
    return any(contains_excerpt(part, excerpt) for part in value)


def _profile_source(sources):
    """Current protocol binds this complete source, without enumerating paths."""
    return {name: sources[name] for name in ("record", "opening_state")}


def profile_sources(sources):
    """程序列出本人资料的可引用路径，支持容器、嵌套字段和列表项。"""
    # source 只包含允许用于本人起点判断的原始资料，不把路径目录当作证据。
    result = {"source": _profile_source(sources)}
    seen = {"source"}
    def visit(value, path):
        if path in seen:
            raise ValueError("初始档案来源路径有歧义，不能选择：" + path)
        seen.add(path)
        if isinstance(value, (str, dict, list)):
            result[path] = value
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + "." + key)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child, path + "." + str(i))
    for namespace in ("record", "opening_state"):
        visit(sources[namespace], namespace)
    return result


def profile_materials(sources):
    """每人一个完整合法资料组，保留起点状态中的语义及字段间关系。

    不逐字段拆分语义任务、不跨字段按同样的文字删来源；知情仍须逐项独立审核。
    """
    source = _profile_source(sources)
    return [("source", source)] if any(source.values()) else []


def profile_review_evidence(candidate, people):
    """完整来源已在审核 evidence.people 中；不为每项认识再次发送同一引文。

    只省略程序能逐字确认的完整来源副本；旧的局部引文及各项知情理由仍保留。
    持久化候选和引用校验不变。参考 Pi 的持久记录/模型视图分离。
    """
    result = candidate.model_dump(mode="json")
    for person, profile in result["people"].items():
        source = encode(_profile_source(people[person]))
        for fact in profile["facts"]:
            if fact["field"] == "source" and fact["excerpt"] == source:
                del fact["excerpt"]
    return result


def validate_profiles(candidate, people):
    if candidate.people.keys() != people.keys():
        raise ValueError("初始档案须覆盖本批所有人物，不得添加其他人物；缺少 "
                         + repr(sorted(people.keys() - candidate.people.keys())) + "，多出 "
                         + repr(sorted(candidate.people.keys() - people.keys())))
    errors, choices = [], {}
    for person, profile in candidate.people.items():
        sources = people[person]
        if sources["opening_state"].get("life_status") == "not_born" and profile.facts:
            raise ValueError("故事起点尚未出生的人物没有可交付的初始个人经历，facts 必须为空")
        seen = set()
        fields = {"source": _profile_source(sources)}
        encoded_source = encode(fields["source"])
        legacy_loaded = False
        for i, fact in enumerate(profile.facts):
            path = "/people/" + person.replace("~", "~0").replace("/", "~1") + f"/facts/{i}"
            if not fact.text.strip() or not fact.knowledge_basis.strip() or not fact.excerpt.strip():
                errors.append({"path": path, "reason": "须给出非空文字、引文及本人此时可知的依据"})
            if fact.text.strip() in seen:
                errors.append({"path": path + "/text", "reason": "同一人物的可知内容重复"})
            seen.add(fact.text.strip())
            field = fact.field if fact.field in {"source", "record", "opening_state"} or fact.field.startswith(
                ("record.", "opening_state.")) else "record." + fact.field
            if field != "source" and not legacy_loaded:
                fields = profile_sources(sources)
                legacy_loaded = True
            if field not in fields:
                errors.append({"path": path + "/field", "value": fact.field,
                               "reason": "不存在此人的该来源路径；从本人 source_fields 选择，不猜测字段名"})
                choices[person] = sorted(fields)
                continue
            value = fields[field]
            matches = False
            if isinstance(value, (dict, list)):
                encoded_value = encoded_source if field == "source" else encode(value)
                matches = fact.excerpt == encoded_value
            if not matches:
                matches = contains_excerpt(value, fact.excerpt)
            if not fact.excerpt.strip() or not matches:
                errors.append({"path": path + "/excerpt", "field": fact.field, "value": fact.excerpt,
                               "reason": "引文不是所选来源的完整编码或其中的连续文字；核对本条 field 与 excerpt，不借用其他人的来源"})
    if errors:
        raise ValueError("初始档案引用或内容校验失败：" + encode({"errors": errors, "source_fields": choices}))


async def initialize_profiles(world):
    storage = world.storage
    raw = storage.manifest["characters"]
    profiles = storage.get_checkpoint("self_profiles", {})
    approvals = storage.get_checkpoint("initial_profile_approvals", {})
    if profiles.keys() != approvals.keys() or profiles.keys() - raw.keys() or any(
            not valid_personal_profile(profile) or approvals[person] != fingerprint([person, profile])
            for person, profile in profiles.items()):
        raise RunStopped("初始可知档案与本人批准记录不符，不能交给人物")
    if storage.get_checkpoint("world_initialized") and profiles.keys() != raw.keys():
        raise RunStopped("已初始化的世界缺少原有个人档案，不能重新编写共同过去")
    # 各线从同一起点继承已批准的档案；恢复时不能用当前时间或之后的节点重新编写它们。
    opening = world.world["opening_snapshot"]["characters"]
    role_fields = ("role_name", "nickname", "gender", "profile", "personality")
    pending = sorted(raw.keys() - profiles.keys())
    # 由请求执行器的 semaphore 限制实际并发；不按八人一组等待最慢者再启动下一组。
    people = {person: {"record": {k: raw[person][k] for k in role_fields if raw[person].get(k)},
                       "opening_state": opening.get(person, {})}
              for person in pending}
    async def prepare(person, sources):
        # 一次完整任务对应一人；不把其他人的档案发给该人的编制或审核任务。
        local = {person: sources}
        job = task_id("world", "initial_profiles", person)
        if sources["opening_state"].get("life_status") == "not_born":
            result = InitialProfiles(people={person: InitialProfile(facts=[])})
            validate_profiles(result, local)
            record_visible(storage, person, result.delivery()[person], job + "/not-born-rule")
            return result
        return await visible_task(world.supervisor, job, world.prompt,
            {"operation": "initial_profiles", "people": local}, InitialProfiles,
            lambda candidate: validate_profiles(candidate, local), InitialProfiles.delivery,
            private_review=lambda candidate: profile_review_evidence(candidate, local))

    results = await gather_batch([prepare(person, sources) for person, sources in people.items()])
    for result in results:
        for person, data in result.delivery().items():
            profiles[person] = data["profile"]
            approvals[person] = fingerprint([person, profiles[person]])
    storage.set_checkpoints({"self_profiles": profiles, "initial_profile_approvals": approvals})


class ProfileAssertion(Record):
    text: Identity = Field(description="本人在起点有理由知道的一项完整认识；可保留充分细节。")
    knowledge_basis: Identity = Field(description="说明此项认识在当时已经可知的实际依据；不抄引文或路径。")


class ProfileContent(Record):
    facts: list[ProfileAssertion]


def profile_contract(schema, context, owner, job):
    from tools.ToolArguments import payload_data
    from pydantic import create_model
    people = sorted(context["people"])
    if len(people) == 1:
        payload_type = ProfileContent
    else:
        payload_type = create_model("InitialProfilesInput", __base__=Record,
            profiles=(list[ProfileContent], Field(min_length=len(people), max_length=len(people),
                description="按此顺序分别交付本人认识：" + encode(people))))

    def assemble(payload):
        value = payload_type.model_validate(payload_data(payload))
        profiles = [value] if len(people) == 1 else value.profiles
        result = {}
        for person, profile in zip(people, profiles):
            source = context["people"][person]
            materials = profile_materials(source)
            if source["opening_state"].get("life_status") == "not_born" or not materials:
                if profile.facts:
                    raise ValueError("没有可知起点资料的人物不能新增认识")
                result[person] = {"facts": []}
                continue
            field, evidence = materials[0]
            result[person] = {"facts": [{**fact.model_dump(), "field": field,
                "excerpt": encode(evidence)} for fact in profile.facts]}
        return schema.model_validate({"people": result})
    return payload_type, assemble

