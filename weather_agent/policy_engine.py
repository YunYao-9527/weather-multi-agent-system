from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from weather_agent.models import DecisionPacket, FusionResult, HazardObject, PolicySnapshot
from weather_agent.storage import RegistryStore


def default_policy_snapshot() -> PolicySnapshot:
    return PolicySnapshot(
        policy_version="policy.national.v1",
        issue_threshold=0.62,
        clear_threshold=0.28,
        lower_bound_threshold=0.45,
        min_duration=2,
        min_area=120.0,
        max_proxy_share=0.35,
        required_independent_families=3,
        stale_source_rules={"critical_families": ["radar", "guidance"], "max_stale_count": 0},
        conflict_veto_rules={"max_conflict_penalty": 0.55, "require_attribution": True},
        quality_veto_rules={"min_readiness": 0.4, "block_on_quality_flags": ["coverage_low", "stale", "fetch_failed"]},
        manual_review_rules={"low_family_count": True, "degraded_mode": True},
        hold_down_rules={"issue_hold_cycles": 2, "clear_hold_cycles": 2},
        coverage_rules={"min_coverage_ratio": 0.05},
        policy_notes=[
            "Conservative release: use p_release and lower confidence bound instead of raw hazard probability.",
            "Proxy contamination above threshold always routes to manual review.",
        ],
    )


class PolicyManager:
    def __init__(self, registry: RegistryStore | None = None):
        self.registry = registry or RegistryStore()
        default = default_policy_snapshot()
        if not self.registry.get("policies", default.policy_version):
            self.register(default)
            self.activate(default.policy_version)

    def register(self, snapshot: PolicySnapshot) -> Dict[str, Any]:
        return self.registry.register("policies", snapshot.policy_version, asdict(snapshot))

    def activate(self, policy_version: str) -> None:
        self.registry.set_active("policies", policy_version)

    def active(self) -> PolicySnapshot:
        active = self.registry.get_active("policies") or {}
        name = active.get("name") or default_policy_snapshot().policy_version
        data = self.registry.get("policies", name) or asdict(default_policy_snapshot())
        return PolicySnapshot(**data)

    def list(self) -> List[Dict[str, Any]]:
        return self.registry.list("policies")

    def validate(self, snapshot: PolicySnapshot) -> List[str]:
        issues: list[str] = []
        if snapshot.clear_threshold >= snapshot.issue_threshold:
            issues.append("clear_threshold must be lower than issue_threshold")
        if snapshot.lower_bound_threshold > snapshot.issue_threshold:
            issues.append("lower_bound_threshold should not exceed issue_threshold")
        if snapshot.max_proxy_share < 0 or snapshot.max_proxy_share > 1:
            issues.append("max_proxy_share must be within [0,1]")
        if snapshot.required_independent_families < 1:
            issues.append("required_independent_families must be >= 1")
        return issues


def _level(prob: float) -> str:
    if prob >= 0.8:
        return "L3"
    if prob >= 0.65:
        return "L2"
    if prob >= 0.5:
        return "L1"
    return "L0"


def decide_packet(
    hazard_object: HazardObject,
    fusion: FusionResult,
    policy: PolicySnapshot,
    *,
    versions_registered: bool = True,
    replay_verified: bool = True,
    truth_chain_valid: bool = True,
    recommended_by: List[str] | None = None,
) -> DecisionPacket:
    hazard = hazard_object.hazard_type
    p_release = float((fusion.p_release or {}).get(hazard, 0.0))
    lower = float((fusion.lower_confidence_bound or {}).get(hazard, 0.0))
    veto_reasons = list(fusion.veto_reasons)
    rationale: list[str] = []

    if float(fusion.proxy_dependency_ratio) > policy.max_proxy_share:
        veto_reasons.append("proxy_share_exceeded")
    if fusion.independent_family_count < policy.required_independent_families:
        veto_reasons.append("insufficient_independent_families")
    if hazard_object.area_km2 < policy.min_area:
        veto_reasons.append("min_area_not_met")
    if hazard_object.evidence_persistence < policy.min_duration:
        veto_reasons.append("min_duration_not_met")
    if not hazard_object.min_area_met:
        rationale.append("object area below policy minimum")
    if not hazard_object.min_duration_met:
        rationale.append("evidence persistence below policy minimum")
    if fusion.degraded_mode:
        veto_reasons.append("degraded_mode")
    if not versions_registered:
        veto_reasons.append("version_unregistered")
    if not replay_verified:
        veto_reasons.append("replay_verification_failed")
    if not truth_chain_valid:
        veto_reasons.append("truth_chain_invalid")

    issue_gate = (
        p_release >= policy.issue_threshold
        and lower >= policy.lower_bound_threshold
        and hazard_object.evidence_persistence >= policy.min_duration
        and hazard_object.area_km2 >= policy.min_area
        and fusion.independent_family_count >= policy.required_independent_families
    )
    clear_gate = p_release < policy.clear_threshold and lower < policy.clear_threshold

    if veto_reasons:
        action = "manual_review"
    elif issue_gate:
        action = "recommend_issue" if hazard_object.lifecycle_state != "active" else "active"
    elif clear_gate:
        action = "recommend_clear"
    else:
        action = "manual_review" if p_release >= policy.clear_threshold else "clear"

    rationale.extend(
        [
            f"p_release={p_release:.3f}",
            f"lower_bound={lower:.3f}",
            f"independent_families={fusion.independent_family_count}",
            f"proxy_ratio={fusion.proxy_dependency_ratio:.3f}",
            f"effective_source_tier={fusion.effective_source_tier}",
            f"object_state={hazard_object.lifecycle_state}",
        ]
    )
    if fusion.conflicts:
        rationale.append("conflicts=" + ",".join(fusion.conflicts))

    return DecisionPacket(
        action=action,
        level=_level(p_release if not veto_reasons else 0.0),
        rationale=rationale,
        veto_triggered=bool(veto_reasons),
        veto_reasons=sorted(set(veto_reasons)),
        policy_version=policy.policy_version,
        object_id=hazard_object.object_id,
        recommended_by=list(recommended_by or sorted(fusion.family_contribution.keys())),
        requires_human_review=action == "manual_review",
        effective_probability_used=round(p_release, 4),
        issue_gate_passed=issue_gate and not veto_reasons,
        clear_gate_passed=clear_gate and not veto_reasons,
    )
