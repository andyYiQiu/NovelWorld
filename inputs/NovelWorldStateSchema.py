"""1.4 四类客观状态。输入完整后态归一为运行稀疏更新；不转换故事语义。"""
from typing import ClassVar, Literal
from pydantic import BaseModel, Field, TypeAdapter, create_model, field_validator, model_serializer, model_validator
from pydantic.json_schema import SkipJsonSchema
from inputs.RepositoryFormat import (StrictRecord as Record, Character as CharacterState,
    Object as ObjectState, Location as LocationState, Relation as RelationState, Position,
    Quantity, ActionTerm, UseControl)

StateRecord = Record
RELATION_FIELDS = ('subject_role_id', 'related_role_id', 'relationship_type')
STATE_SECTIONS = ('characters', 'objects', 'locations', 'relations')

class StateSections(Record):
    model_config = {'json_schema_extra': {'required': list(STATE_SECTIONS)}}

    @field_validator('*', mode='before')
    @classmethod
    def empty_collections(cls, value, info):
        if value is None:
            raise ValueError('状态类别不能为 null')
        return value

def require_state_sections(value, label):
    if value is None:
        return
    fields = value.model_fields_set if isinstance(value, BaseModel) else value
    external = ('character_states','object_states','location_states','relationship_states')
    expected = external if isinstance(fields, dict) and 'character_states' in fields else STATE_SECTIONS
    missing = set(expected) - set(fields)
    if missing:
        raise ValueError(f'{label}: 缺少状态类别 {sorted(missing)}')

class AssetState(StateSections):
    characters: dict[str, CharacterState] = Field(default_factory=dict)
    objects: dict[str, ObjectState] = Field(default_factory=dict)
    locations: dict[str, LocationState] = Field(default_factory=dict)
    relations: list[RelationState] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def snapshot_envelope(cls, value):
        value = normalize_snapshot(value)
        return {k:v for k,v in value.items() if k not in ('node_id','previous_node_id')} if isinstance(value,dict) else value

def normalize_snapshot(value):
    if not isinstance(value, dict) or 'character_states' not in value:
        return value
    from inputs.RepositoryFormat import Snapshot
    data = Snapshot.model_validate(value).model_dump(mode='json')
    data['relationship_states'].sort(key=lambda r: tuple(r[k] for k in RELATION_FIELDS))
    names = dict(zip(('character_states','object_states','location_states','relationship_states'),STATE_SECTIONS))
    return {names.get(k,k):v for k,v in data.items()}

def sparse_patch_schema(schema):
    """省略即不更新；不能用内部占位默认值诱导模型显式交付 null。"""
    schema['minProperties'] = 1
    for field in schema.get('properties', {}).values():
        field.pop('default', None)


class StatePatch(Record):
    """至少交付一项实际变化；省略字段表示沿用，null 仅用于原字段允许未知/清空的情况。

    嵌套对象与列表整体替换，须保留仍有效项。字段及约束复用快照，只序列化实际提交的键。
    """
    model_config = {'extra': 'forbid', 'json_schema_extra': sparse_patch_schema}
    state_type: ClassVar[type[StateRecord]]
    _field_adapters: ClassVar[dict[str, TypeAdapter]]

    @model_serializer(mode='wrap')
    def supplied_fields_only(self, handler):
        return {key: value for key, value in handler(self).items() if key in self.model_fields_set}

    @model_validator(mode='after')
    def validate_supplied_fields(self):
        if not self.model_fields_set:
            raise ValueError('set须至少包含一个实际更新字段')
        for key in self.model_fields_set:
            self._field_adapters[key].validate_python(getattr(self, key))
        return self


def patch_type(state_type, excluded=()):
    # Pydantic FieldInfo.rebuild_annotation 保留 min_length 等 metadata；仅取
    # annotation 会让补丁接口丢掉完整状态要求的约束。
    fields = {key: (field.rebuild_annotation() | SkipJsonSchema[None], Field(default=None, description=field.description))
              for key, field in state_type.model_fields.items()
              if key not in excluded}
    result = create_model(state_type.__name__ + 'Patch', __base__=StatePatch, __doc__=StatePatch.__doc__, **fields)
    result.state_type = state_type
    # 编译只做一次；保留实例复验，拒绝构造后被直接改成非法值的字段。
    result._field_adapters = {key: TypeAdapter(state_type.model_fields[key].rebuild_annotation()) for key in fields}
    return result



CharacterStatePatch = patch_type(CharacterState)
ObjectStatePatch = patch_type(ObjectState)
LocationStatePatch = patch_type(LocationState)
RelationStatePatch = patch_type(RelationState, RELATION_FIELDS)

class DeltaRecord(Record):
    basis: str = ''  # 运行裁定可保留原因；repository 的原因在事件组，不要求重复抄写。

class CharacterDelta(DeltaRecord):
    role_id: str
    set: CharacterStatePatch

class ObjectDelta(DeltaRecord):
    object_id: str
    set: ObjectStatePatch

class LocationDelta(DeltaRecord):
    location_id: str
    set: LocationStatePatch

class RelationDelta(DeltaRecord):
    subject_role_id: str
    related_role_id: str
    relationship_type: str
    operation: Literal['add','update','remove']
    set: RelationStatePatch | None = None

    @model_validator(mode='after')
    def operation_values(self):
        if (self.operation == 'remove') != (self.set is None):
            raise ValueError('remove 不更新属性；add/update 必须提供属性')
        if self.operation == 'add' and self.set.model_fields_set != set(RelationStatePatch.model_fields):
            raise ValueError('新增关系须交齐全部属性')
        return self

class StateDelta(StateSections):
    characters: list[CharacterDelta] = Field(default_factory=list)
    objects: list[ObjectDelta] = Field(default_factory=list)
    locations: list[LocationDelta] = Field(default_factory=list)
    relations: list[RelationDelta] = Field(default_factory=list)

STATE_KINDS = {'characters': ('role_id', CharacterState), 'objects': ('object_id', ObjectState),
               'locations': ('location_id', LocationState), 'relations': (None, RelationState)}
