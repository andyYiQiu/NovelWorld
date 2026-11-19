"""Explicit routing to domain-owned tool contracts; no business assembly here."""
from common.Runtime import RunStopped
from common.NovelRestoreSchema import (
    CharacterState, CharacterDecision, CharacterMemories, RoundSchedule, ReplanSchedule,
    PerceptionBatch, WaveResolution, NarrativeOutline, ChapterLayout, NarrativeConnections,
    NarrativeBatch, NarrativeSelection, StatePatch, SemanticPayload,
)
from tools.ToolArguments import payload_data


def prepare_contract(schema, context, owner, job):
    from simulation.NovelRestoreCharacterContract import mental_contract, decision_contract, memory_contract
    from simulation.NovelRestoreWorldContract import schedule_contract, perception_contract, resolution_contract
    from simulation.NovelRestoreInitialProfile import InitialProfiles, profile_contract
    from narration.NovelRestoreContract import outline_contract, layout_contract, connections_contract, batch_contract, selection_contract
    from execution.NovelRestoreReview import SemanticReview, semantic_contract, review_contract
    from simulation.NovelRestoreEvidence import evidence_contract
    from simulation.NovelRestoreExperienceContract import experience_contract
    from simulation.NovelRestoreVisibilityContract import visibility_contract

    contracts = {
        CharacterState: mental_contract, CharacterDecision: decision_contract, CharacterMemories: memory_contract,
        RoundSchedule: schedule_contract, ReplanSchedule: schedule_contract,
        PerceptionBatch: perception_contract, WaveResolution: resolution_contract,
        InitialProfiles: profile_contract, SemanticReview: semantic_contract,
        NarrativeOutline: outline_contract, ChapterLayout: layout_contract,
        NarrativeConnections: connections_contract, NarrativeBatch: batch_contract, NarrativeSelection: selection_contract,
    }
    if schema in contracts:
        return contracts[schema](schema, context, owner, job)
    for contract in (evidence_contract, experience_contract, visibility_contract, review_contract):
        prepared = contract(schema, context)
        if prepared is not None:
            return prepared
    # New persistent records cannot silently become model-generated file structures.
    if schema is StatePatch or SemanticPayload in schema.__bases__:
        return schema, lambda payload: schema.model_validate(payload_data(payload))
    raise RunStopped("模型任务尚未声明语义交付与程序装配边界：" + schema.__name__)
