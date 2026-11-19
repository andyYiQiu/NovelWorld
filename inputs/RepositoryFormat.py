"""通用格式 1.4：输入文件的唯一结构定义。所有记录必写字段，不含 detail。"""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, RootModel, create_model, field_validator

Text = Annotated[str, Field(min_length=1)]
ID = Annotated[str, Field(pattern=r'^[A-Za-z][A-Za-z0-9_]*$')]

def record(name, fields):
    return create_model(name, __base__=StrictRecord, **{k:(t,Field(description=d)) for k,t,d in fields})

class StrictRecord(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)

class Inherit(StrictRecord):
    inherit: Literal[True] = Field(description='固定标记 true：本次不更新该属性，继承前态；不属于故事状态。')

    @field_validator('inherit', mode='before')
    @classmethod
    def exact_marker(cls, value):
        if value is not True:
            raise ValueError('继承标记必须为布尔 true')
        return value


# 全局唯一编号直接引用；null 未知，[] 有据确认没有。
Position=record('Position',[
 ('location_id',ID|None,'当前地点编号；未知为 null。'),
 ('position_within_place',Text|None,'地点内部位置；不记录其他事实。')])
UseControl=record('UseControl',[
 ('entity_id',ID,'使用或支配地点的人物或组织编号。'),
 ('relationship',Text,'与地点的具体联系，如居住、拥有、占用或控制；不得互相推定。')])
Quantity=record('Quantity',[('amount',float|Text|None,'当前数量或有据概数，0 不等于未知。'),('unit',Text|None,'该数量的单位。')])
ActionTerm=record('ActionTermState',[
 ('term_type',Literal['eligibility','permission','prohibition','obligation','confinement'],'当前资格、许可、禁止、义务或拘束。'),
 ('governed_action',Text,'具体行动及有据的适用、解除条件；不要求补齐制度程序。'),
 ('related_entity_ids',list[ID],'涉及的地点、物品或人物编号；无需另套范围对象。'),
 ('authority_ids',list[ID]|None,'施加或认可者编号；null 未知，[] 确认无特定主体。')])

Character=record('CharacterState',[
 ('age',Text|None,'当前年龄或年龄段。'),('current_name',Text|None,'该时点现用姓名。'),('current_aliases',list[Text]|None,'当前使用的别称。'),
 ('gender',Text|None,'当前有据性别信息。'),('life_status',Literal['alive','dead','not_born','unknown'],'生死或尚未出生状态。'),
 ('current_roles',list[Text]|None,'当前身份和职责；实际权利义务另由关系或行动条款记录。'),('position',Position,'当前所在位置。'),
 ('physical_conditions',list[Text]|None,'当前身体状况，有据记录伤病、疲劳、部位、程度或症状，不要求逐项填写；不凭表现推断诊断。null 未知，[] 有据确认无异常。'),
 ('action_capabilities',list[Text]|None,'当前身体能完成的行动及身体限制；不记录技能掌握、社会许可或个人意愿，不仅凭伤病推定行动后果。'),
 ('current_skills',list[Text]|None,'此刻已掌握的劳动、生活或专业技能及有据熟练程度；学习、遗忘或失去掌握时更新，暂时身体受限不等于失去技能。null 未知，[] 有据确认没有。'),
 ('appearance',Text|None,'可观察的身体外貌。'),('clothing',list[Text]|None,'当前衣着；被登记为物件时同时用物件 ID 管理其归属。'),
 ('action_terms',list[ActionTerm],'该人物当前成立的资格、许可、禁止、义务和拘束；结束的条款从完整列表移除；不自动意味着本人知情或一定服从。')])
Object=record('ObjectState',[
 ('owner_ids',list[ID]|None,'所有者编号；null 未知，[] 确认无主；不等于保管人。'),
 ('custodian_ids',list[ID]|None,'实际持有或保管者编号；null 未知，[] 确认无人保管。'),
 ('position',Position,'当前位置。'),('container_id',ID|None,'所在容器物品编号；null 未知。'),
 ('physical_condition',Text|None,'当前物理状况，包括有据的破损、污染、开闭锁封或加工状况；不要求逐项填写。'),
 ('functional_availability',Text|None,'当前能否使用及已知功能限制；不能只凭外观推断。'),
 ('quantity',Quantity|None,'当前数量及单位。'),
 ('use_conditions',list[Text]|None,'使用该物品的许可、限制及条件；不重复物理状况。')])
Location=record('LocationState',[
 ('physical_condition',Text|None,'地点及设施的客观物理状况，包括完好程度、开闭锁封、道路通畅程度、积水、坍塌与障碍；不记录用途或社会许可。'),
 ('use_and_control',list[UseControl]|None,'有据的使用与支配联系；null 未知，[] 确认无人使用或支配。'),
 ('uses_and_conditions',list[Text]|None,'地点用途、可进行的活动及使用限制；不拆成当前用途与服务能力。'),
 ('environmental_conditions',list[Text]|None,'该地点当前天气、光照、声音等条件；区域共同条件存上级地点。更新为完整列表，结束的条件从列表移除。')])
Relation=record('RelationState',[
 ('subject_role_id',ID,'关系描述的起点人物。'),('related_role_id',ID,'另一端人物。'),('relationship_type',Text,'另一端相对于起点的关系，如雇主。'),
 ('interaction_pattern',list[Text]|None,'当时有据的亲近、疏远、合作、回避或对立等相处方式；可记录已表达的单方态度但须写明归属，不推断未表达的信任、感情或双方共识。'),
 ('practical_ties',list[Text]|None,'双方现实的利益、依赖、资源安排、责任联系及具体争议；明确谁依赖谁、谁对谁负何责任。具体行动义务以人物 action_terms 为准，此处不重复其条款。')])


RoleInfo=record('RoleInfo',[
 ('role_code',ID,'人物稳定编号。'),('role_name',Text,'中性化档案名称。'),('gender',Text|None,'档案已有的性别信息。'),
 ('nickname',Text|None,'称呼和使用者。'),('personality',Text|None,'相对稳定倾向，不是强制行为。'),
 ('profile',Text|None,'经历、技能形成经历及生活基础，标明成立阶段；当前掌握情况以人物状态 current_skills 为准，不写未来命运。')])
ObjectInfo=record('ObjectInfo',[
 ('object_code',ID,'物件稳定编号。'),('object_name',Text,'中性名称。'),('aliases',list[Text],'其他称呼。'),
 ('description',Text|None,'物件是什么及稳定材质构成；当前损坏另记状态。'),('function',Text|None,'通常用途和一般使用条件。')])
LocationInfo=record('LocationInfo',[
 ('location_code',ID,'地点稳定编号。'),('location_name',Text,'中性名称。'),('description',Text|None,'地点性质及稳定特征。'),
 ('parent_location_id',ID|None,'所属上级地点，如房间所属宅院。'),('spatial_layout',Text|None,'内部空间及相对位置。'),
 ('facilities',list[Text]|None,'固定设施及其用途，当前是否可用另记状态。'),('access_routes',list[Text]|None,'与其他地点连接的道路和出入口，明确方向。'),
 ('visibility_conditions',list[Text]|None,'稳定遮蔽和视听条件；不据此推定人物实际看见。')])
General=record('General',[
 ('description',Text|None,'时代、地理及社会背景概述。'),('livelihood_conditions',list[Text]|None,'通用谋生方式及生活供给条件。'),
 ('social_structure',list[Text]|None,'社会身份、地位及资源分配背景。'),('communication_practices',list[Text]|None,'消息传播渠道及交通联系。')])
Setting=record('Setting',[
 ('setting_id',ID,'设定稳定编号，规则和组织可被状态引用。'),('kind',Literal['custom','rule','organization','concept','other'],'风俗、规则、组织、概念或其他。'),
 ('term',Text,'设定名称。'),('definition',Text|None,'该设定是什么；仅存定义，不承接杂项事实。'),
 ('practices',list[Text]|None,'习俗做法、制度程序或组织通常活动。'),('applicability',list[Text]|None,'通常适用于何人何地何事；当前实施范围另记状态。'),
 ('belief_attribution',list[Text]|None,'相关说法由谁相信或主张；不把信仰变成客观世界规律。')])
NodeTime=record('NodeTime',[
 ('relative',Text,'故事实际发生时序。'),('exact_start',Text|None,'有据起始时刻。'),('exact_end',Text|None,'有据结束时刻。'),('duration',Text|None,'有据持续时长，不自动造成状态变化。')])
Node=record('Node',[
 ('node_id',ID,'节点编号及文件目录。'),('order',int,'从 1 连续的节点序号。'),('node_name',Text,'中性阶段名。'),
 ('time',NodeTime,'阶段时间。'),('interval_from_previous',Text|None,'与前节点间隔；第一节点为空。')])
Timeline=record('Timeline',[('source',ID,'中性输入包身份，与根目录一致。'),('nodes',list[Node],'按 order 排列的非空节点。')])
Participant=record('Participant',[('role_code',ID,'实际参与者编号。'),('original_goal',Text|None,'参考事件中的人物目标，不直接写进人物记忆。')])
Event=record('Event',[
 ('task_id',ID,'事件稳定编号。'),('todo',Text,'需要模拟的事件与处境，不规定全部动作台词。'),('time',Text|None,'组内时间说明。'),
 ('participants',list[Participant],'实际参与者，不包括仅被谈论的人。'),('location_ids',list[ID],'事件关联地点。'),('object_ids',list[ID],'事件关联物件。'),
 ('required_results',list[Text],'必要结果；供目标审核，不代替人物过程。')])
KINDS={
 'characters':('character_states','character_changes','role_id',Character),
 'objects':('object_states','object_changes','object_id',Object),
 'locations':('location_states','location_changes','location_id',Location),
 'relations':('relationship_states','relationship_changes',None,Relation),
}
snapshot_fields=[('node_id',ID,'当前节点编号。'),('previous_node_id',ID|None,'上一节点编号；首节点为 null。')]
change_fields=[]
for kind,(snapshot_name,change_name,identifier,model) in KINDS.items():
 snapshot_fields.append((snapshot_name,list[model] if identifier is None else dict[ID,model],f'节点入口已成立的{kind}状态；不写未来变化。'))
 identities=['subject_role_id','related_role_id','relationship_type'] if identifier is None else []
 values=record(model.__name__+'AfterValues',[(k,f.rebuild_annotation()|Inherit,'填 {"inherit": true} 表示本次不改；其他内容均为实际写入的后态。null 只用于原属性允许的未知值，数组 [] 表示明确写入空数组；结构化属性按自身格式表示未知。'+f.description) for k,f in model.model_fields.items() if k not in identities])
 outer=[(k,model.model_fields[k].rebuild_annotation(),model.model_fields[k].description) for k in identities] if identifier is None else [(identifier,ID,'被更新对象的稳定编号。')]
 if identifier is None:
  outer.append(('operation',Literal['add','update','remove'],'新增、修改或删除关系；新增必须给出全部后态，删除的 after_values 全填继承标记，不写入任何属性。'))
 change=record(model.__name__+'Change',outer+[
  ('after_values',values,'全套属性键必写：继承项填 {"inherit": true}，更新项直接填后态值。嵌套对象或数组整体更新，不在内部混用继承标记。修改时至少一项实际更新；删除关系时不更新属性。变化原因由所属事件组承载。')])
 change_fields.append((change_name,list[change],f'本事件组造成的{kind}变化；没有变化时为空数组。'))
Snapshot=record('StateBefore',snapshot_fields)
Changes=record('StateChanges',change_fields)
Group=record('Group',[('group_id',ID,'事件组编号。'),('order',int,'节点内从 1 连续的组顺序。'),('events',list[Event],'共享同一前态的非空事件数组。'),('state_changes',Changes,'本组造成的状态变化；与节点入口快照分开表达。')])
TodoList=record('TodoList',[('node_id',ID,'所属节点编号。'),('todolist',list[Group],'按顺序模拟的事件组。')])
Locations=RootModel[dict[ID,LocationInfo]]
FILE_MODELS={'role-info':RoleInfo,'object-info':ObjectInfo,'locations':Locations,'general':General,'setting':Setting,'timeline':Timeline,'state-before':Snapshot,'todolist':TodoList}
