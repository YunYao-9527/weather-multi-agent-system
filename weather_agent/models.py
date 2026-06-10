from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any, Dict, List, Optional


HAZARDS = ("short_rain", "wind", "hail", "tornado")
SOURCE_TIERS = ("official", "experimental", "proxy")
OBJECT_STATES = (
    "candidate",
    "watch",
    "recommend_issue",
    "active",
    "recommend_clear",
    "cleared",
    "manual_review",
)
EVIDENCE_FAMILIES = (
    "radar",
    "environment",
    "circulation",
    "objective_guidance",
    "model_guidance",
    "nowcast",
    "quality_control",
    "cross_consistency",
    "llm_reasoning",
    "human_override",
)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def hazard_template(value: float = 0.0) -> Dict[str, float]:
    return {h: value for h in HAZARDS}


def stable_hash(payload: Any) -> str:
    return sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


def tier_rank(tier: str) -> int:
    ordered = {"proxy": 0, "experimental": 1, "official": 2}
    return ordered.get(str(tier or "proxy").strip().lower(), 0)


def strongest_tier(tiers: List[str] | tuple[str, ...] | set[str] | None) -> str:
    items = [str(t).strip().lower() for t in (tiers or []) if str(t).strip()]
    if not items:
        return "proxy"
    return max(items, key=tier_rank)


@dataclass
class GridSpec:
    grid_id: str = "grid.legacy"
    nx: int = 1
    ny: int = 1
    dx_km: float = 10.0
    dy_km: float = 10.0
    origin_lat: float = 0.0
    origin_lon: float = 0.0
    spatial_ref: str = "EPSG:4326"


@dataclass
class ProviderRecord:
    source_id: str
    source_family: str
    source_tier: str
    provider_version: str
    issue_time: str
    valid_time: str
    ingest_time: str
    latency_sec: float
    freshness_sec: float
    spatial_ref: str
    coverage: Dict[str, Any] = field(default_factory=dict)
    qc_flags: List[str] = field(default_factory=list)
    analysis_time: str = ""
    lead_time: int = 0
    domain_id: str = "domain.legacy"
    grid_spec: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    fingerprint: str = ""
    data_snapshot_id: str = ""
    official_candidate: bool = False
    experimental_note: str = ""


@dataclass
class QCRecord:
    source_id: str
    status: str = "ok"
    stale: bool = False
    missing: bool = False
    coverage_ok: bool = True
    time_alignment_ok: bool = True
    version_registered: bool = True
    flags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass
class Observation:
    """Legacy unified input snapshot kept for compatibility."""

    schema_version: str = "observation.v3"
    timestamp: datetime = field(default_factory=datetime.now)
    city: str = "unknown"
    vertical_velocity: float = 0.0
    low_level_convergence: float = 0.0
    cape: float = 0.0
    dcape: float = 0.0
    shear_0_6km: float = 0.0
    t850_500: float = 0.0
    wbz_km: float = 0.0
    humidity_low: float = 0.0
    radar_dbz_max: float = 0.0
    radar_bow_echo: bool = False
    storm_motion_ms: float = 0.0
    prob_guidance: Dict[str, float] = field(default_factory=dict)
    source_meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class ObservationEnvelope:
    schema_version: str = "observation_envelope.v1"
    envelope_id: str = ""
    analysis_time: str = ""
    valid_time: str = ""
    lead_time: int = 0
    domain_id: str = "domain.legacy"
    grid_spec: GridSpec = field(default_factory=GridSpec)
    feature_planes: Dict[str, Any] = field(default_factory=dict)
    source_registry: List[ProviderRecord] = field(default_factory=list)
    qc_registry: Dict[str, QCRecord] = field(default_factory=dict)
    provenance_manifest: Dict[str, Any] = field(default_factory=dict)
    object_context: Dict[str, Any] = field(default_factory=dict)
    source_tiers_summary: Dict[str, Any] = field(default_factory=dict)
    qc_summary: Dict[str, Any] = field(default_factory=dict)
    source_meta: Dict[str, Any] = field(default_factory=dict)
    legacy_observation: Observation | None = None

    @classmethod
    def from_observation(
        cls,
        obs: Observation,
        domain_id: str | None = None,
        grid_spec: GridSpec | None = None,
        source_registry: List[ProviderRecord] | None = None,
        qc_registry: Dict[str, QCRecord] | None = None,
        feature_planes: Dict[str, Any] | None = None,
        provenance_manifest: Dict[str, Any] | None = None,
        object_context: Dict[str, Any] | None = None,
    ) -> "ObservationEnvelope":
        domain = domain_id or f"city:{obs.city.lower().replace(' ', '_')}"
        grid = grid_spec or GridSpec(grid_id=f"{domain}.default")
        providers = list(source_registry or [])
        qc = dict(qc_registry or {})
        planes = dict(feature_planes or {})
        if not planes:
            planes = {
                "environment": {
                    "vertical_velocity": obs.vertical_velocity,
                    "low_level_convergence": obs.low_level_convergence,
                    "cape": obs.cape,
                    "dcape": obs.dcape,
                    "shear_0_6km": obs.shear_0_6km,
                    "t850_500": obs.t850_500,
                    "wbz_km": obs.wbz_km,
                    "humidity_low": obs.humidity_low,
                },
                "radar": {
                    "radar_dbz_max": obs.radar_dbz_max,
                    "radar_bow_echo": obs.radar_bow_echo,
                    "storm_motion_ms": obs.storm_motion_ms,
                },
                "guidance": {"prob_guidance": dict(obs.prob_guidance)},
            }
        source_meta = dict(obs.source_meta)
        tiers = [p.source_tier for p in providers]
        tier_counts = {tier: tiers.count(tier) for tier in SOURCE_TIERS}
        proxy_share = tier_counts["proxy"] / max(len(providers), 1) if providers else 0.0
        summary = {
            "counts": tier_counts,
            "effective_source_tier": strongest_tier(tiers),
            "proxy_share": round(proxy_share, 4),
        }
        now_iso = obs.timestamp.isoformat()
        envelope_id = stable_hash((obs.city, now_iso, domain, sorted(source_meta.items())))
        qc_summary = {
            "stale_sources": [k for k, v in qc.items() if v.stale],
            "coverage_anomalies": [k for k, v in qc.items() if not v.coverage_ok],
            "version_unregistered": [k for k, v in qc.items() if not v.version_registered],
            "missing_sources": [k for k, v in qc.items() if v.missing],
        }
        return cls(
            envelope_id=envelope_id,
            analysis_time=source_meta.get("analysis_time", now_iso),
            valid_time=source_meta.get("valid_time", now_iso),
            lead_time=int(float(source_meta.get("lead_time", 0) or 0)),
            domain_id=domain,
            grid_spec=grid,
            feature_planes=planes,
            source_registry=providers,
            qc_registry=qc,
            provenance_manifest=dict(provenance_manifest or {}),
            object_context=dict(object_context or {}),
            source_tiers_summary=summary,
            qc_summary=qc_summary,
            source_meta=source_meta,
            legacy_observation=obs,
        )

    def to_observation(self) -> Observation:
        if self.legacy_observation is not None:
            return self.legacy_observation
        env = self.feature_planes.get("environment", {})
        radar = self.feature_planes.get("radar", {})
        guidance = self.feature_planes.get("guidance", {})
        valid_time = datetime.fromisoformat(self.valid_time) if self.valid_time else datetime.now()
        return Observation(
            timestamp=valid_time,
            city=str(self.source_meta.get("city") or self.domain_id),
            vertical_velocity=float(env.get("vertical_velocity", 0.0)),
            low_level_convergence=float(env.get("low_level_convergence", 0.0)),
            cape=float(env.get("cape", 0.0)),
            dcape=float(env.get("dcape", 0.0)),
            shear_0_6km=float(env.get("shear_0_6km", 0.0)),
            t850_500=float(env.get("t850_500", 0.0)),
            wbz_km=float(env.get("wbz_km", 0.0)),
            humidity_low=float(env.get("humidity_low", 0.0)),
            radar_dbz_max=float(radar.get("radar_dbz_max", 0.0)),
            radar_bow_echo=bool(radar.get("radar_bow_echo", False)),
            storm_motion_ms=float(radar.get("storm_motion_ms", 0.0)),
            prob_guidance=dict(guidance.get("prob_guidance", {})),
            source_meta={str(k): str(v) for k, v in self.source_meta.items()},
        )


@dataclass
class EvidenceCard:
    agent: str
    claim: str
    confidence: float
    hazard_scores: Dict[str, float]
    supporting_features: Dict[str, float | str | bool]
    freshness_seconds: int = 0
    schema_version: str = "evidence.v3"
    proxy_source: bool = False
    input_fields: List[str] = field(default_factory=list)
    upstream_sources: List[str] = field(default_factory=list)
    observed_at: Optional[str] = None
    missing_fields: List[str] = field(default_factory=list)
    rule_version: str = "rules.v1"
    model_version: str = "heuristic.v1"
    evidence_id: str = ""
    family: str = ""
    source_tier: str = "experimental"
    provider_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id:
            self.evidence_id = stable_hash((self.agent, self.claim, self.observed_at, self.supporting_features))


@dataclass
class AgentOutput:
    agent: str
    evidence: EvidenceCard


@dataclass
class FusionResult:
    schema_version: str = "fusion.v3"
    hazard_prob: Dict[str, float] = field(default_factory=dict)
    hazard_prob_raw: Dict[str, float] = field(default_factory=dict)
    calibrated_prob: Dict[str, float] = field(default_factory=dict)
    lower_confidence_bound: Dict[str, float] = field(default_factory=dict)
    p_release: Dict[str, float] = field(default_factory=dict)
    probability_quality_factor: float = 1.0
    per_agent_contribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    family_contribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    evidence_readiness_score: float = 0.0
    evidence_readiness_breakdown: Dict[str, float] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    proxy_dependency_ratio: float = 0.0
    effective_source_tier: str = "proxy"
    veto_reasons: List[str] = field(default_factory=list)
    degraded_mode: bool = False
    degraded_reasons: List[str] = field(default_factory=list)
    degraded_mode_flags: List[str] = field(default_factory=list)
    independent_family_count: int = 0


@dataclass
class HazardObject:
    schema_version: str = "hazard_object.v1"
    object_id: str = ""
    parent_object_id: str = ""
    object_type: str = "storm_object"
    hazard_type: str = "short_rain"
    geometry: Dict[str, Any] = field(default_factory=dict)
    centroid: Dict[str, float] = field(default_factory=dict)
    area_km2: float = 0.0
    motion_vector: Dict[str, float] = field(default_factory=dict)
    lifecycle_state: str = "candidate"
    start_time: str = ""
    end_time: str = ""
    last_update_time: str = ""
    confidence: float = 0.0
    support_evidence_ids: List[str] = field(default_factory=list)
    source_tiers: List[str] = field(default_factory=list)
    object_version: int = 1
    evidence_persistence: int = 0
    source_stability: float = 0.0
    motion_stability: float = 0.0
    min_duration_met: bool = False
    min_area_met: bool = False
    merge_from: List[str] = field(default_factory=list)
    split_from: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PolicySnapshot:
    schema_version: str = "policy_snapshot.v1"
    policy_version: str = "policy.national.v1"
    issue_threshold: float = 0.62
    clear_threshold: float = 0.28
    lower_bound_threshold: float = 0.45
    min_duration: int = 2
    min_area: float = 120.0
    max_proxy_share: float = 0.35
    required_independent_families: int = 3
    stale_source_rules: Dict[str, Any] = field(default_factory=dict)
    conflict_veto_rules: Dict[str, Any] = field(default_factory=dict)
    quality_veto_rules: Dict[str, Any] = field(default_factory=dict)
    manual_review_rules: Dict[str, Any] = field(default_factory=dict)
    hold_down_rules: Dict[str, Any] = field(default_factory=dict)
    coverage_rules: Dict[str, Any] = field(default_factory=dict)
    policy_notes: List[str] = field(default_factory=list)


@dataclass
class WarningDecision:
    schema_version: str = "decision.v3"
    issue: bool = False
    level: str = "L0"
    action: str = "clear"
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    affected_area: str = "unknown"
    hazard_prob: Dict[str, float] = field(default_factory=dict)
    hazard_prob_raw: Dict[str, float] = field(default_factory=dict)
    calibrated_prob: Dict[str, float] = field(default_factory=dict)
    lower_confidence_bound: Dict[str, float] = field(default_factory=dict)
    p_release: Dict[str, float] = field(default_factory=dict)
    probability_quality_factor: float = 1.0
    confidence: float = 0.0
    evidence_readiness_score: float = 0.0
    evidence_readiness_breakdown: Dict[str, float] = field(default_factory=dict)
    rationale: List[EvidenceCard] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    per_agent_contribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    family_contribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    degraded_mode: bool = False
    degraded_reasons: List[str] = field(default_factory=list)
    degraded_mode_flags: List[str] = field(default_factory=list)
    requires_human_review: bool = False
    proxy_dependency_ratio: float = 0.0
    effective_source_tier: str = "proxy"
    veto_reasons: List[str] = field(default_factory=list)
    decision_trace: List[str] = field(default_factory=list)
    policy_version: str = ""
    object_id: str = ""


@dataclass
class DecisionPacket:
    schema_version: str = "decision_packet.v1"
    action: str = "clear"
    level: str = "L0"
    rationale: List[str] = field(default_factory=list)
    veto_triggered: bool = False
    veto_reasons: List[str] = field(default_factory=list)
    policy_version: str = ""
    object_id: str = ""
    recommended_by: List[str] = field(default_factory=list)
    requires_human_review: bool = False
    effective_probability_used: float = 0.0
    issue_gate_passed: bool = False
    clear_gate_passed: bool = False
    trace_id: str = ""


@dataclass
class DecisionResult:
    schema_version: str = "decision_result.v2"
    decision: WarningDecision = field(default_factory=WarningDecision)


@dataclass
class AuditRecord:
    schema_version: str = "audit.v3"
    run_id: str = ""
    request_id: str = ""
    trace_id: str = ""
    bundle_id: str = ""
    generated_at: str = ""
    city: str = ""
    config_hash: str = ""
    code_version: str = ""
    policy_version: str = ""
    model_version: str = ""
    feature_version: str = ""
    data_window: Dict[str, Any] = field(default_factory=dict)
    source_health: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    fusion_snapshot: Dict[str, Any] = field(default_factory=dict)
    decision_snapshot: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    replay_verification: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TruthRecord:
    schema_version: str = "truth_record.v1"
    truth_id: str = ""
    truth_layer: str = "point"
    timestamp: str = ""
    city: str = ""
    grid_id: str = ""
    object_id: str = ""
    event_id: str = ""
    labels: Dict[str, int] = field(default_factory=dict)
    label_tier: str = "proxy"
    geometry: Dict[str, Any] = field(default_factory=dict)
    source_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TruthVersion:
    schema_version: str = "truth_version.v1"
    truth_version: str = ""
    created_at: str = ""
    city: str = ""
    period: Dict[str, str] = field(default_factory=dict)
    headline_tier: str = "gold"
    manifest_path: str = ""
    snapshot_path: str = ""
    record_counts: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class EvalManifest:
    schema_version: str = "eval_manifest.v1"
    eval_manifest_id: str = ""
    data_snapshot_id: str = ""
    truth_version: str = ""
    feature_version: str = ""
    code_sha: str = ""
    model_version: str = ""
    calibration_version: str = ""
    policy_version: str = ""
    run_config_hash: str = ""
    generated_at: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayBundle:
    schema_version: str = "replay_bundle.v1"
    bundle_id: str = ""
    run_id: str = ""
    request_id: str = ""
    trace_id: str = ""
    source_snapshot_references: List[Dict[str, Any]] = field(default_factory=list)
    normalized_features: Dict[str, Any] = field(default_factory=dict)
    agent_outputs: List[Dict[str, Any]] = field(default_factory=list)
    fusion_snapshot: Dict[str, Any] = field(default_factory=dict)
    policy_snapshot: Dict[str, Any] = field(default_factory=dict)
    decision_trace: Dict[str, Any] = field(default_factory=dict)
    code_reference: Dict[str, Any] = field(default_factory=dict)
    model_references: Dict[str, Any] = field(default_factory=dict)
    truth_reference: Dict[str, Any] = field(default_factory=dict)
    replay_tolerance: Dict[str, float] = field(default_factory=lambda: {"hazard_prob": 1e-06, "decision": 0.0})
    lineage: Dict[str, Any] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRunRecord:
    schema_version: str = "eval_run.v3"
    run_id: str = ""
    created_at: str = ""
    config_hash: str = ""
    data_window: Dict[str, str] = field(default_factory=dict)
    split_manifest: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    eval_manifest: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionRunRecord:
    schema_version: str = "evolve_run.v2"
    run_id: str = ""
    created_at: str = ""
    config_hash: str = ""
    data_window: Dict[str, str] = field(default_factory=dict)
    split_manifest: Dict[str, Any] = field(default_factory=dict)
    parameter_delta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryProfile:
    schema_version: str = "memory_profile.v2"
    key: str = ""
    profile_version: str = ""
    sample_count: int = 0
    coverage_ratio: float = 0.0
    valid_window: Dict[str, str] = field(default_factory=dict)
    generated_at: str = ""
    code_version: str = ""
    agent_weights: Dict[str, float] = field(default_factory=dict)
    calibrators: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleResult:
    observation: Observation
    decision: WarningDecision
    generated_at: datetime
    audit: AuditRecord | None = None
    envelope: ObservationEnvelope | None = None
    fusion_result: FusionResult | None = None
    hazard_object: HazardObject | None = None
    decision_packet: DecisionPacket | None = None
    replay_bundle: ReplayBundle | None = None
    request_id: str = ""
    trace_id: str = ""


def default_start_end(now: datetime | None = None, window_minutes: int = 120) -> tuple[datetime, datetime]:
    start = now or datetime.now()
    return start, start + timedelta(minutes=window_minutes)
