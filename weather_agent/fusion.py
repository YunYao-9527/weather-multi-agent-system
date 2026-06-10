from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Dict, List

from weather_agent.calibration import calibrate_hazard_probs
from weather_agent.models import EvidenceCard, EVIDENCE_FAMILIES, FusionResult, HAZARDS, clamp01, strongest_tier, tier_rank


DEFAULT_AGENT_WEIGHTS: Dict[str, float] = {
    "circulation": 0.9,
    "environment": 1.0,
    "objective_guidance": 0.9,
    "model_consensus": 0.95,
    "radar": 1.15,
    "nowcast": 1.0,
    "data_quality_guard": 0.2,
    "cross_source_consistency": 0.3,
}

AGENT_TO_FAMILY: Dict[str, str] = {
    "circulation": "circulation",
    "environment": "environment",
    "objective_guidance": "objective_guidance",
    "model_consensus": "model_guidance",
    "radar": "radar",
    "nowcast": "nowcast",
    "data_quality_guard": "quality_control",
    "cross_source_consistency": "cross_consistency",
    "llm_situation": "llm_reasoning",
}

FAMILY_WEIGHTS: Dict[str, float] = {
    "radar": 1.15,
    "environment": 0.95,
    "circulation": 0.9,
    "objective_guidance": 0.75,
    "model_guidance": 0.85,
    "nowcast": 1.0,
    "quality_control": 0.25,
    "cross_consistency": 0.3,
    "llm_reasoning": 0.35,
    "human_override": 1.2,
}

TIER_WEIGHTS = {"official": 1.0, "experimental": 0.82, "proxy": 0.45}
NON_INDEPENDENT_FAMILIES = {"quality_control", "cross_consistency", "human_override", "llm_reasoning"}


def _family(card: EvidenceCard) -> str:
    family = str(card.family or AGENT_TO_FAMILY.get(card.agent, card.agent)).strip().lower()
    return family if family in EVIDENCE_FAMILIES else card.agent


def _tier(card: EvidenceCard) -> str:
    tier = str(card.source_tier or ("proxy" if card.proxy_source else "experimental")).strip().lower()
    return tier if tier in TIER_WEIGHTS else "proxy"


def _aggregate_family_hazard(values: List[float]) -> float:
    if not values:
        return 0.0
    top = max(values)
    mid = median(values)
    return clamp01(0.8 * top + 0.2 * mid)


def fuse_evidence(
    cards: List[EvidenceCard],
    weights: Dict[str, float] | None = None,
    data_quality: float = 0.65,
    model_spread_score: float = 0.35,
    proxy_weight_cap: float = 0.65,
    correlation_penalty: float = 0.12,
    hazard_weight_multipliers: Dict[str, float] | None = None,
    probability_calibrators: Dict[str, Dict[str, object]] | None = None,
) -> FusionResult:
    if not cards:
        zeros = {h: 0.0 for h in HAZARDS}
        return FusionResult(
            hazard_prob=zeros,
            hazard_prob_raw=zeros,
            calibrated_prob=zeros,
            lower_confidence_bound=zeros,
            p_release=zeros,
            evidence_readiness_score=0.0,
            conflicts=["no evidence"],
            degraded_mode=True,
            degraded_reasons=["no_evidence_cards"],
            degraded_mode_flags=["no_evidence_cards"],
            veto_reasons=["no_evidence_cards"],
        )

    weight_map = dict(DEFAULT_AGENT_WEIGHTS)
    if weights:
        weight_map.update(weights)
    hazard_mul = {h: 1.0 for h in HAZARDS}
    for h, v in (hazard_weight_multipliers or {}).items():
        if h in hazard_mul:
            hazard_mul[h] = float(v)

    penalties: Dict[str, float] = {c.agent: 1.0 for c in cards}
    source_groups: Dict[str, list[str]] = defaultdict(list)
    for card in cards:
        src = str(card.upstream_sources[0] if card.upstream_sources else card.supporting_features.get("source", "unknown"))
        source_groups[src].append(card.agent)
    for agents in source_groups.values():
        if len(agents) > 1:
            penalty = max(0.7, 1.0 - correlation_penalty * (len(agents) - 1))
            for agent in agents:
                penalties[agent] = min(penalties.get(agent, 1.0), penalty)

    per_agent_contribution: Dict[str, Dict[str, float]] = {c.agent: {h: 0.0 for h in HAZARDS} for c in cards}
    family_inputs: Dict[str, Dict[str, list[float]]] = defaultdict(lambda: {h: [] for h in HAZARDS})
    family_tiers: Dict[str, List[str]] = defaultdict(list)
    family_confidences: Dict[str, List[float]] = defaultdict(list)
    conflicts: list[str] = []
    total_signal_weight = 0.0
    proxy_weight = 0.0

    for card in cards:
        family = _family(card)
        tier = _tier(card)
        diagnostic_only = bool(card.supporting_features.get("diagnostic_only", False))
        conf = clamp01(card.confidence)
        base = float(weight_map.get(card.agent, 1.0)) * conf * penalties.get(card.agent, 1.0) * TIER_WEIGHTS[tier]
        if tier == "proxy":
            base = min(base, proxy_weight_cap)
            proxy_weight += base
        if not diagnostic_only:
            total_signal_weight += base
        family_tiers[family].append(tier)
        family_confidences[family].append(conf)
        for hazard in HAZARDS:
            score = clamp01(float(card.hazard_scores.get(hazard, 0.0))) * hazard_mul[hazard]
            contribution = 0.0 if diagnostic_only else score * base
            per_agent_contribution[card.agent][hazard] = round(contribution, 6)
            if diagnostic_only:
                continue
            family_inputs[family][hazard].append(contribution)

    family_contribution: Dict[str, Dict[str, float]] = {}
    raw_accum: Dict[str, float] = defaultdict(float)
    raw_norm: Dict[str, float] = defaultdict(float)

    for family, hazard_values in family_inputs.items():
        family_weight = float(FAMILY_WEIGHTS.get(family, 0.75))
        family_conf = mean(family_confidences.get(family, [0.0]))
        tier_factor = TIER_WEIGHTS[strongest_tier(family_tiers.get(family, []))]
        family_contribution[family] = {}
        for hazard in HAZARDS:
            aggregated = _aggregate_family_hazard(hazard_values.get(hazard, []))
            family_value = clamp01((aggregated / max(family_weight * tier_factor, 1e-6)) if aggregated else 0.0)
            family_value = clamp01(family_value * (0.7 + 0.3 * family_conf))
            family_contribution[family][hazard] = round(family_value, 4)
            raw_accum[hazard] += family_value * family_weight
            raw_norm[hazard] += family_weight

    hazard_prob_raw = {h: clamp01(raw_accum[h] / raw_norm[h]) if raw_norm[h] > 0 else 0.0 for h in HAZARDS}

    spreads = []
    independent_families = 0
    for hazard in HAZARDS:
        vals = [v.get(hazard, 0.0) for v in family_contribution.values()]
        if vals:
            spreads.append(max(vals) - min(vals))
        independent_count = sum(
            1
            for family, values in family_contribution.items()
            if family not in NON_INDEPENDENT_FAMILIES and values.get(hazard, 0.0) >= 0.2
        )
        if independent_count >= 2:
            independent_families = max(independent_families, independent_count)
    avg_spread = mean(spreads) if spreads else 1.0
    conflict_penalty = clamp01(avg_spread)
    if conflict_penalty >= 0.45:
        conflicts.append(f"family_conflict_spread={conflict_penalty:.2f}")

    proxy_ratio = proxy_weight / max(total_signal_weight, 1e-6)
    agreement_score = clamp01(1.0 - conflict_penalty)
    freshness_values = [clamp01(1.0 - min(float(c.freshness_seconds), 10800.0) / 10800.0) for c in cards]
    freshness_score = mean(freshness_values) if freshness_values else 0.0
    data_quality_score = clamp01(data_quality)
    model_dispersion_score = clamp01(1.0 - model_spread_score)
    evidence_strength = mean([clamp01(c.confidence) for c in cards]) if cards else 0.0
    probability_quality_factor = clamp01(
        max(
            0.2,
            0.30 * evidence_strength
            + 0.22 * agreement_score
            + 0.22 * data_quality_score
            + 0.14 * freshness_score
            + 0.12 * model_dispersion_score
            - 0.22 * proxy_ratio,
        )
    )
    hazard_prob = {h: clamp01(hazard_prob_raw[h] * probability_quality_factor) for h in HAZARDS}
    calibrated_prob = calibrate_hazard_probs(hazard_prob, probability_calibrators) if probability_calibrators else dict(hazard_prob)
    uncertainty_margin = clamp01(0.08 + 0.22 * conflict_penalty + 0.20 * proxy_ratio + 0.12 * (1.0 - data_quality_score))
    lower_bound = {h: clamp01(calibrated_prob[h] - uncertainty_margin) for h in HAZARDS}
    p_release = {h: clamp01(min(calibrated_prob[h], lower_bound[h])) for h in HAZARDS}

    degraded_mode_flags: list[str] = []
    veto_reasons: list[str] = []
    if proxy_ratio > 0.35:
        degraded_mode_flags.append("proxy_share_high")
        if proxy_ratio > 0.5:
            veto_reasons.append("proxy_share_exceeded")
    if data_quality_score < 0.45:
        degraded_mode_flags.append("data_quality_low")
        veto_reasons.append("data_quality_low")
    if freshness_score < 0.35:
        degraded_mode_flags.append("stale_evidence")
        veto_reasons.append("stale_evidence")
    if conflict_penalty > 0.55:
        degraded_mode_flags.append("high_conflict")
        veto_reasons.append("high_conflict")
    if independent_families < 2:
        degraded_mode_flags.append("family_support_weak")

    readiness_breakdown = {
        "evidence_strength": round(evidence_strength, 4),
        "agreement_score": round(agreement_score, 4),
        "data_quality_score": round(data_quality_score, 4),
        "freshness_score": round(freshness_score, 4),
        "model_dispersion_score": round(model_dispersion_score, 4),
        "conflict_penalty": round(conflict_penalty, 4),
        "conflict_control_score": round(clamp01(1.0 - conflict_penalty), 4),
        "prob_quality_factor": round(probability_quality_factor, 4),
        "proxy_dependency_ratio": round(proxy_ratio, 4),
        "independent_family_count": float(independent_families),
    }
    readiness = clamp01(
        0.30 * evidence_strength
        + 0.24 * agreement_score
        + 0.20 * data_quality_score
        + 0.13 * freshness_score
        + 0.13 * model_dispersion_score
        - 0.10 * proxy_ratio
    )
    effective_source_tier = strongest_tier(
        [strongest_tier(family_tiers.get(family, [])) for family, vals in family_contribution.items() if max(vals.values() or [0.0]) >= 0.15]
    )

    return FusionResult(
        hazard_prob={h: round(hazard_prob[h], 4) for h in HAZARDS},
        hazard_prob_raw={h: round(hazard_prob_raw[h], 4) for h in HAZARDS},
        calibrated_prob={h: round(calibrated_prob[h], 4) for h in HAZARDS},
        lower_confidence_bound={h: round(lower_bound[h], 4) for h in HAZARDS},
        p_release={h: round(p_release[h], 4) for h in HAZARDS},
        probability_quality_factor=round(probability_quality_factor, 4),
        per_agent_contribution=per_agent_contribution,
        family_contribution=family_contribution,
        evidence_readiness_score=round(readiness, 4),
        evidence_readiness_breakdown=readiness_breakdown,
        conflicts=conflicts,
        proxy_dependency_ratio=round(proxy_ratio, 4),
        effective_source_tier=effective_source_tier,
        veto_reasons=sorted(set(veto_reasons)),
        degraded_mode=bool(degraded_mode_flags),
        degraded_reasons=sorted(set(degraded_mode_flags)),
        degraded_mode_flags=sorted(set(degraded_mode_flags)),
        independent_family_count=int(independent_families),
    )


def decide_level(max_prob: float) -> str:
    if max_prob >= 0.8:
        return "L3"
    if max_prob >= 0.65:
        return "L2"
    if max_prob >= 0.5:
        return "L1"
    return "L0"
