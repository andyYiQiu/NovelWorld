"""六类客观世界状态及稀疏更新。认知、目标和记忆由下游人物管理。"""
from copy import deepcopy
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, TypeAdapter, create_model, field_validator, model_serializer, model_validator
from pydantic.json_schema import SkipJsonSchema


class Record(BaseModel):
    model_config = {'extra': 'ignore', 'str_strip_whitespace': True}

    @field_validator('*', mode='before')
    @classmethod
    def empty_collections(cls, value, info):
        factory = cls.model_fields[info.field_name].default_factory
        if factory in (dict, list) and (value is None or isinstance(value, (dict, list)) and not value):
            return factory()
        return value

    @field_validator('order', 'event_order', mode='before', check_fields=False)
    @classmethod
    def numeric_start(cls, value):
        if isinstance(value, bool):
            raise ValueError('开始刻度不能是布尔值')
        return value


# 旧模拟任务的类型契约；公开 SQLite 记录不得携带原文证据。
class Evidence(Record):
    model_config = {'json_schema_extra': {'required': ['chunk', 'passage_ids']}}
    chunk: int = Field(ge=1)
    quote: SkipJsonSchema[str] = ''
    unverified_quote: SkipJsonSchema[str | None] = None
    passage_ids: list[str] = Field(default_factory=list,
        description='选择支持此事的原文片段编号；程序回填完整原文，不复抄或概括引文',
        json_schema_extra={'minItems': 1})


def legacy_object_fields(value):
    if not isinstance(value, dict):
        return value
    value = deepcopy(value)
    for name, kind in (('owner', 'role'), ('holder', 'role'), ('location', 'location')):
        ref = value.get(name)
        if isinstance(ref, dict) and 'id' in ref:
            converted = {'status': ref.get('status'), 'references':
                         [{'type': kind, 'id': ref['id']}] if ref.get('id') else []}
            value[name] = {'location': converted, 'detail': None} if name == 'location' else converted
    ref = value.get('location')
    if isinstance(ref, dict) and 'references' in ref:
        value['location'] = {'location': ref, 'detail': None}
    return value


class StateRecord(Record):
    model_config = {'extra': 'forbid', 'str_strip_whitespace': True}
    evidence: list[Evidence] = Field(default_factory=list)


class Reference(Record):
    model_config = {'extra': 'forbid'}
    type: Literal['role', 'object', 'location', 'organization']
    id: str = Field(min_length=1)


class ReferenceState(Record):
    model_config = {'extra': 'forbid'}
    status: Literal['known', 'none', 'unknown'] = Field(default='unknown',
        description='known有明确引用；none有依据确认没有；unknown依据不足。三者不能互相代替')
    references: list[Reference] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def accept_single_reference(cls, value):
        if isinstance(value, dict) and set(value) == {'type', 'id'}:
            return {'status': 'known', 'references': [value]}
        return value

    @model_validator(mode='after')
    def valid_references(self):
        if (self.status == 'known') != bool(self.references):
            raise ValueError('known须提供非空references；none/unknown须为空数组')
        keys = [(r.type, r.id) for r in self.references]
        if len(keys) != len(set(keys)):
            raise ValueError('references不能重复')
        return self


class Position(Record):
    model_config = {'extra': 'forbid'}
    location: ReferenceState = Field(default_factory=ReferenceState)
    detail: str | None = None


class AccessState(Record):
    model_config = {'extra': 'forbid'}
    open_state: Literal['open', 'closed', 'not_applicable', 'unknown'] = 'unknown'
    lock_state: Literal['locked', 'unlocked', 'not_applicable', 'unknown'] = 'unknown'
    seal_state: Literal['sealed', 'unsealed', 'not_applicable', 'unknown'] = 'unknown'


class PhysicalCondition(Record):
    model_config = {'extra': 'forbid'}
    condition_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: Literal['active', 'resolved', 'unknown']
    body_part: str | None = None
    severity: str | None = None
    detail: str | None = None


class Quantity(Record):
    model_config = {'extra': 'forbid'}
    value: float | str | None = None
    unit: str | None = None


class ComponentState(Record):
    model_config = {'extra': 'forbid'}
    object_id: str
    status: Literal['installed', 'detached', 'unknown']
    detail: str | None = None


class Scope(Record):
    model_config = {'extra': 'forbid', 'populate_by_name': True}
    global_: bool = Field(default=False, alias='global', serialization_alias='global')
    role_ids: list[str] = Field(default_factory=list)
    location_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    organization_ids: list[str] = Field(default_factory=list)
    activity: str | None = None
    detail: str | None = None

    @model_serializer(mode='wrap')
    def public_global(self, handler):
        value = handler(self)
        value['global'] = value.pop('global_', self.global_)
        return value

    @model_validator(mode='after')
    def consistent_scope(self):
        if self.global_ and any((self.role_ids, self.location_ids, self.object_ids, self.organization_ids)):
            raise ValueError('global=true不能同时填写局部对象范围')
        return self


class CharacterState(StateRecord):
    age: str | None = None
    identities: list[str] | None = Field(default=None, description='该时点的客观身份；不把已结束身份或单次行为当作持续身份')
    life_status: Literal['alive', 'dead', 'not_born', 'unknown'] = 'unknown'
    location: Position = Field(default_factory=Position)
    physical_conditions: list[PhysicalCondition] | None = Field(default=None,
        description='有据身体疾病、伤势或生理情况；不含恐惧、心情、记忆或人物信念。替换列表保留仍有效项')
    capacity_limitations: list[str] | None = Field(default=None, description='原文明确的身体能力限制，不存心理、认知或记忆困难')
    appearance: str | None = None
    current_name: str | None = None
    current_aliases: list[str] | None = None
    gender: str | None = None

    @model_serializer(mode='wrap')
    def applicable_fields(self, handler):
        value = handler(self)
        for key in ('current_name', 'current_aliases', 'gender'):
            if key not in self.model_fields_set:
                value.pop(key, None)
        return value


class ObjectState(StateRecord):
    @model_validator(mode='before')
    @classmethod
    def legacy_fields(cls, value):
        return legacy_object_fields(value)

    owner: ReferenceState = Field(default_factory=ReferenceState, description='所有权归属，与实际保管或位置分开判断')
    holder: ReferenceState = Field(default_factory=ReferenceState, description='实际持有或保管者，不限于手中拿着；依据保管关系的变化更新')
    location: Position = Field(default_factory=Position, description='物品所在位置；位置变化本身不证明所有权或保管权改变')
    container: ReferenceState = Field(default_factory=ReferenceState)
    condition: str | None = None
    functional_status: str | None = None
    quantity: Quantity | None = None
    access_state: AccessState = Field(default_factory=AccessState)
    components: list[ComponentState] | None = None


class LocationState(StateRecord):
    condition: str | None = Field(default=None, description='地点或设施的物理状况，不用发生于此的人物行为、心情或事件摘要代替')
    operational_status: str | None = None
    passage_status: str | None = None
    access_state: AccessState = Field(default_factory=AccessState)
    owner: ReferenceState = Field(default_factory=ReferenceState, description='有据产权归属，不等于居住者、使用者或管理者')
    controller: ReferenceState = Field(default_factory=ReferenceState)
    availability: str | None = None


RelationStatus = Literal['active', 'inactive', 'unknown']


class RelationIdentity(Record):
    owner_id: str
    other_id: str
    relation: str = Field(min_length=1, description='other相对于owner的单项关系；例如乙欠甲钱，owner甲、other乙的关系是债务人')


class RelationState(StateRecord, RelationIdentity):
    status: RelationStatus = 'unknown'
    detail: str | None = None


class EnvironmentState(StateRecord):
    kind: str = Field(min_length=1)
    scope: Scope
    status: RelationStatus
    value: str | None = None
    detail: str | None = Field(default=None, description='原文支持的适用场景、范围或持续限制；一次观察不能自动作为所有后续阶段的环境')


class RuleState(StateRecord):
    scope: Scope
    status: RelationStatus
    requirements: list[str] | None = None
    exceptions: list[str] | None = None
    enforcer: ReferenceState = Field(default_factory=ReferenceState)


class OrganizationState(StateRecord):
    status: str | None = None
    controller: ReferenceState = Field(default_factory=ReferenceState)
    detail: str | None = None


class ConstraintState(StateRecord):
    subject: ReferenceState
    kind: Literal['eligibility', 'permission', 'prohibition', 'obligation', 'confinement']
    target: Scope
    action: str = Field(min_length=1)
    status: RelationStatus
    conditions: list[str] | None = None
    authority: ReferenceState = Field(default_factory=ReferenceState)
    rule_ids: list[str] = Field(default_factory=list)
    detail: str | None = None


STATE_SECTIONS = ('characters', 'objects', 'locations', 'relations', 'public_state', 'constraints')
PUBLIC_SECTIONS = ('environment', 'rules', 'organizations')


class StateSections(Record):
    """允许内部构造空状态；模型交付的类别必须显式提供，null不表示已核对无变化。"""
    model_config = {'extra': 'forbid'}

    @field_validator('*', mode='before')
    @classmethod
    def empty_collections(cls, value, info):
        if value is None:
            raise ValueError('状态类别不能为null；请依据原文交付记录或明确空集合')
        return super().empty_collections(value, info)


def require_state_sections(value, label):
    """在模型交付归一化/保存前查漏；历史读取和内部状态构造不受影响。"""
    if value is None:
        return  # 整份状态未交由原有完成检查处理，不能在这里制造空状态。
    def supplied(record, fields, path):
        if isinstance(record, BaseModel):
            keys = record.model_fields_set
        else:
            keys = record.keys() if isinstance(record, dict) else ()
        missing = [name for name in fields if name not in keys]
        if missing:
            raise ValueError(f'{path}缺少状态类别{missing}；须回查原文补交，不能默认补空代表已检查')
    supplied(value, STATE_SECTIONS, label)
    public = value.public_state if isinstance(value, BaseModel) else value['public_state']
    supplied(public, PUBLIC_SECTIONS, label + '.public_state')


class PublicState(StateSections):
    model_config = {'json_schema_extra': {'required': list(PUBLIC_SECTIONS)}}
    environment: dict[str, EnvironmentState] = Field(default_factory=dict)
    rules: dict[str, RuleState] = Field(default_factory=dict)
    organizations: dict[str, OrganizationState] = Field(default_factory=dict)


class AssetState(StateSections):
    @model_validator(mode='before')
    @classmethod
    def snapshot_envelope(cls, value):
        return {k: v for k, v in value.items() if k not in {'node_id', 'previous_node_id'}} if isinstance(value, dict) else value

    model_config = {'json_schema_extra': {'required': list(STATE_SECTIONS)}}
    characters: dict[str, CharacterState] = Field(default_factory=dict)
    objects: dict[str, ObjectState] = Field(default_factory=dict)
    locations: dict[str, LocationState] = Field(default_factory=dict)
    relations: list[RelationState] = Field(default_factory=list)
    public_state: PublicState = Field(default_factory=PublicState)
    constraints: dict[str, ConstraintState] = Field(default_factory=dict)


class StatePatch(Record):
    """字段类型复用快照，避免维护两套字段定义；只序列化实际提交的键。"""
    model_config = {'extra': 'forbid'}
    state_type: ClassVar[type[StateRecord]]

    @model_validator(mode='before')
    @classmethod
    def legacy_fields(cls, value):
        return legacy_object_fields(value) if cls.state_type is ObjectState else value

    @model_serializer(mode='wrap')
    def supplied_fields_only(self, handler):
        return {key: value for key, value in handler(self).items() if key in self.model_fields_set}

    @model_validator(mode='after')
    def validate_supplied_fields(self):
        if not self.model_fields_set:
            raise ValueError('set须至少包含一个实际更新字段')
        for key in self.model_fields_set:
            TypeAdapter(self.state_type.model_fields[key].annotation).validate_python(getattr(self, key))
        return self


def patch_type(state_type, excluded=()):
    fields = {key: (field.annotation | SkipJsonSchema[None], Field(default=None, description=field.description))
              for key, field in state_type.model_fields.items()
              if key not in {'evidence', *excluded}}
    result = create_model(state_type.__name__ + 'Patch', __base__=StatePatch, **fields)
    result.state_type = state_type
    return result


CharacterStatePatch = patch_type(CharacterState)
ObjectStatePatch = patch_type(ObjectState)
LocationStatePatch = patch_type(LocationState)
RelationStatePatch = patch_type(RelationState, ('owner_id', 'other_id', 'relation'))
EnvironmentStatePatch = patch_type(EnvironmentState)
RuleStatePatch = patch_type(RuleState)
OrganizationStatePatch = patch_type(OrganizationState)
ConstraintStatePatch = patch_type(ConstraintState)


class DeltaRecord(StateRecord):
    basis: str = Field(min_length=1)


class CharacterDelta(DeltaRecord):
    role_id: str
    set: CharacterStatePatch


class ObjectDelta(DeltaRecord):
    object_id: str
    set: ObjectStatePatch


class LocationDelta(DeltaRecord):
    location_id: str
    set: LocationStatePatch


class RelationDelta(DeltaRecord, RelationIdentity):
    set: RelationStatePatch


class EnvironmentDelta(DeltaRecord):
    condition_id: str
    set: EnvironmentStatePatch


class RuleDelta(DeltaRecord):
    rule_id: str
    set: RuleStatePatch


class OrganizationDelta(DeltaRecord):
    organization_id: str
    set: OrganizationStatePatch


class ConstraintDelta(DeltaRecord):
    constraint_id: str
    set: ConstraintStatePatch


class PublicStateDelta(StateSections):
    model_config = {'json_schema_extra': {'required': list(PUBLIC_SECTIONS)}}
    environment: list[EnvironmentDelta] = Field(default_factory=list)
    rules: list[RuleDelta] = Field(default_factory=list)
    organizations: list[OrganizationDelta] = Field(default_factory=list)


class StateDelta(StateSections):
    model_config = {'json_schema_extra': {'required': list(STATE_SECTIONS)}}
    characters: list[CharacterDelta] = Field(default_factory=list)
    objects: list[ObjectDelta] = Field(default_factory=list)
    locations: list[LocationDelta] = Field(default_factory=list)
    relations: list[RelationDelta] = Field(default_factory=list)
    public_state: PublicStateDelta = Field(default_factory=PublicStateDelta)
    constraints: list[ConstraintDelta] = Field(default_factory=list)


# 同一份目录用于引用校验、回放、存储。relations以三元组识别，其余以稳定ID识别。
STATE_KINDS = {
    'characters': ('role_id', CharacterState), 'objects': ('object_id', ObjectState),
    'locations': ('location_id', LocationState), 'relations': (None, RelationState),
    'environment': ('condition_id', EnvironmentState), 'rules': ('rule_id', RuleState),
    'organizations': ('organization_id', OrganizationState), 'constraints': ('constraint_id', ConstraintState),
}
PUBLIC_KINDS = {'environment', 'rules', 'organizations'}
