from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

from weather_agent.agents.cross_source_consistency import CrossSourceConsistencyAgent
from weather_agent.agents.circulation import CirculationAgent
from weather_agent.agents.data_quality_guard import DataQualityGuardAgent
from weather_agent.agents.environment import EnvironmentAgent
from weather_agent.agents.llm_situation import LLMSituationAgent
from weather_agent.agents.model_consensus import ModelConsensusAgent
from weather_agent.agents.nowcast import NowcastAgent
from weather_agent.agents.objective_guidance import ObjectiveGuidanceAgent
from weather_agent.agents.radar import RadarAgent
from weather_agent.base import ForecastAgent
from weather_agent.data_quality import source_health_summary
from weather_agent.fusion import AGENT_TO_FAMILY, decide_level, fuse_evidence
from weather_agent.llm_openai import OpenAIResponsesClient
from weather_agent.models import (
    AuditRecord,
    CycleResult,
    DecisionPacket,
    EvidenceCard,
    FusionResult,
    Observation,
    ObservationEnvelope,
    PolicySnapshot,
    ReplayBundle,
    WarningDecision,
    strongest_tier,
)
from weather_agent.object_engine import HazardObjectEngine
from weather_agent.policy_engine import PolicyManager, decide_packet


def _safe_float(v: object, default: float) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class OrchestratorConfig:
    issue_threshold: float = 0.55
    clear_threshold: float = 0.38
    min_readiness_score: float = 0.45
    min_issue_duration_minutes: int = 20
    min_clear_duration_minutes: int = 20
    min_area_coverage_ratio: float = 0.15
    max_conflict_score_for_auto_issue: float = 0.55
    stale_radar_max_minutes: float = 90.0
    min_issue_prob: float = 0.45
    min_confidence: float = 0.35
    window_minutes: int = 120
    region_name: str = "全市"
    agent_weights: Dict[str, float] | None = None
    probability_calibrators: Dict[str, Dict[str, object]] | None = None
    proxy_weight_cap: float = 0.65
    correlation_penalty: float = 0.12
    hazard_weight_multipliers: Dict[str, float] | None = None
    policy_version: str = "policy.national.v1"
    feature_version: str = "feature_schema.v1"
    model_version: str = "multi_agent_rules.v3"
    enable_llm_agent: bool = False
    llm_agent_mode: str = "shadow"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_sec: float = 20.0
    llm_max_output_tokens: int = 900
    llm_temperature: float = 0.1


@dataclass
class _DecisionState:
    issued: bool = False
    object_id: str = ""


class ForecastOrchestrator:
    def __init__(self, config: OrchestratorConfig | None = None, agents: Iterable[ForecastAgent] | None = None):
        self.config = config or OrchestratorConfig()
        self.agents: List[ForecastAgent] = list(agents) if agents else self._default_agents()
        self._state = _DecisionState()
        self.policy_manager = PolicyManager()
        self.object_engine = HazardObjectEngine()

    def _default_agents(self) -> List[ForecastAgent]:
        agents: List[ForecastAgent] = [
            CirculationAgent(),
            EnvironmentAgent(),
            ObjectiveGuidanceAgent(),
            ModelConsensusAgent(),
            RadarAgent(),
            NowcastAgent(),
            DataQualityGuardAgent(),
            CrossSourceConsistencyAgent(),
        ]
        llm_agent = self._build_llm_agent()
        if llm_agent is not None:
            agents.append(llm_agent)
        return agents

    def _build_llm_agent(self) -> ForecastAgent | None:
        if not self.config.enable_llm_agent:
            return None
        provider = str(self.config.llm_provider or "openai").strip().lower()
        if provider not in {"openai", "deepseek"}:
            return None
        client = OpenAIResponsesClient(
            provider=provider,
            model=str(self.config.llm_model),
            timeout_sec=float(self.config.llm_timeout_sec),
            max_output_tokens=int(self.config.llm_max_output_tokens),
            temperature=float(self.config.llm_temperature),
        )
        if not client.ready:
            return None
        return LLMSituationAgent(client=client, mode=str(self.config.llm_agent_mode))

    def _coerce_envelope(self, obs_or_envelope: Observation | ObservationEnvelope) -> ObservationEnvelope:
        if isinstance(obs_or_envelope, ObservationEnvelope):
            return obs_or_envelope
        return ObservationEnvelope.from_observation(obs_or_envelope)

    def _policy_snapshot(self) -> PolicySnapshot:
        policy = self.policy_manager.active()
        policy.issue_threshold = float(self.config.issue_threshold)
        policy.clear_threshold = float(self.config.clear_threshold)
        policy.lower_bound_threshold = max(float(policy.lower_bound_threshold), float(self.config.min_confidence))
        policy.min_area = max(float(policy.min_area), float(self.config.min_area_coverage_ratio) * 1000.0)
        policy.required_independent_families = min(int(policy.required_independent_families), max(1, min(3, len(self.agents))))
        policy.policy_version = self.config.policy_version or policy.policy_version
        return policy

    def _source_tier_for_family(self, envelope: ObservationEnvelope, family: str) -> str:
        tiers = []
        for provider in envelope.source_registry:
            if provider.source_family in {family, "guidance" if family in {"objective_guidance", "model_guidance"} else family}:
                tiers.append(provider.source_tier)
        if not tiers and family == "radar":
            tiers = [p.source_tier for p in envelope.source_registry if p.source_family == "radar"]
        return strongest_tier(tiers) if tiers else "experimental"

    def _enrich_provenance(self, card: EvidenceCard, obs: Observation, envelope: ObservationEnvelope) -> EvidenceCard:
        if not card.input_fields:
            card.input_fields = sorted(list(card.supporting_features.keys()))
        if not card.upstream_sources:
            src = card.supporting_features.get("source") if isinstance(card.supporting_features, dict) else None
            card.upstream_sources = [str(src or obs.source_meta.get("mode", "unknown"))]
        if not card.observed_at:
            card.observed_at = obs.timestamp.isoformat()
        if not card.family:
            card.family = AGENT_TO_FAMILY.get(card.agent, card.agent)
        if not card.source_tier or card.source_tier == "experimental":
            card.source_tier = self._source_tier_for_family(envelope, card.family)
        source_id = card.upstream_sources[0] if card.upstream_sources else card.agent
        matched = next((p for p in envelope.source_registry if p.source_id == source_id or p.source_family == card.family), None)
        if matched and not card.provider_fingerprint:
            card.provider_fingerprint = matched.fingerprint
        return card

    def _decision_trace(self, fusion: FusionResult, packet: DecisionPacket, source_health: dict) -> list[str]:
        trace = list(packet.rationale)
        trace.extend(
            [
                f"readiness={fusion.evidence_readiness_score:.3f}",
                f"prob_quality_factor={fusion.probability_quality_factor:.3f}",
                f"effective_source_tier={fusion.effective_source_tier}",
                f"proxy_dependency_ratio={fusion.proxy_dependency_ratio:.3f}",
                f"issue_gate_passed={packet.issue_gate_passed}",
                f"clear_gate_passed={packet.clear_gate_passed}",
                f"veto_triggered={packet.veto_triggered}",
            ]
        )
        trace.extend([f"source_health:{x}" for x in source_health.get("warnings", [])])
        trace.extend([f"veto:{x}" for x in packet.veto_reasons])
        return trace

    def _versions_registered(self, envelope: ObservationEnvelope, cards: List[EvidenceCard]) -> bool:
        if any(not qc.version_registered for qc in envelope.qc_registry.values()):
            return False
        if any(not c.rule_version or not c.model_version for c in cards):
            return False
        return True

    def run_cycle(
        self,
        obs_or_envelope: Observation | ObservationEnvelope,
        *,
        request_id: str = "",
        trace_id: str = "",
    ) -> CycleResult:
        envelope = self._coerce_envelope(obs_or_envelope)
        obs = envelope.to_observation()
        request_id = request_id or str(uuid.uuid4())
        trace_id = trace_id or request_id

        evidence_cards: List[EvidenceCard] = []
        for agent in self.agents:
            try:
                card = agent.run_envelope(envelope).evidence
            except Exception as e:
                card = EvidenceCard(
                    agent=getattr(agent, "name", "unknown"),
                    claim=f"agent error: {e}",
                    confidence=0.0,
                    hazard_scores={h: 0.0 for h in ("short_rain", "wind", "hail", "tornado")},
                    supporting_features={"error": str(e)},
                    upstream_sources=["agent_runtime"],
                    missing_fields=["agent_output"],
                    model_version="error",
                    rule_version="error",
                )
            evidence_cards.append(self._enrich_provenance(card, obs, envelope))

        weighted_evidence_cards = [
            card for card in evidence_cards if not bool((card.supporting_features or {}).get("diagnostic_only", False))
        ]

        data_quality = _safe_float(obs.source_meta.get("data_quality_score"), 0.65)
        model_spread = _safe_float(obs.source_meta.get("model_spread_score"), 0.35)
        source_health = source_health_summary(obs.source_meta, envelope)
        fusion = fuse_evidence(
            weighted_evidence_cards,
            self.config.agent_weights,
            data_quality=data_quality,
            model_spread_score=model_spread,
            proxy_weight_cap=self.config.proxy_weight_cap,
            correlation_penalty=self.config.correlation_penalty,
            hazard_weight_multipliers=self.config.hazard_weight_multipliers,
            probability_calibrators=self.config.probability_calibrators,
        )
        if source_health.get("veto_reasons"):
            fusion.veto_reasons = sorted(set(list(fusion.veto_reasons) + list(source_health.get("veto_reasons", []))))
        if source_health.get("degraded_flags"):
            fusion.degraded_mode = True
            fusion.degraded_reasons = sorted(set(list(fusion.degraded_reasons) + list(source_health.get("degraded_flags", []))))
            fusion.degraded_mode_flags = sorted(set(list(fusion.degraded_mode_flags) + list(source_health.get("degraded_flags", []))))

        policy = self._policy_snapshot()
        hazard_object = self.object_engine.build_object(envelope, fusion, policy)
        packet = decide_packet(
            hazard_object,
            fusion,
            policy,
            versions_registered=self._versions_registered(envelope, evidence_cards),
            replay_verified=True,
            truth_chain_valid=True,
            recommended_by=sorted(fusion.family_contribution.keys()),
        )

        hysteresis_prob = max(fusion.calibrated_prob.values()) if fusion.calibrated_prob else 0.0
        if self._state.issued and hysteresis_prob >= float(self.config.clear_threshold) and not packet.veto_triggered:
            packet.action = "active"
            packet.issue_gate_passed = True
            packet.clear_gate_passed = False
            packet.level = decide_level(max(fusion.p_release.values()) if fusion.p_release else hysteresis_prob)

        max_prob = max(fusion.p_release.values()) if fusion.p_release else 0.0
        issue = packet.action in {"recommend_issue", "active"}
        if packet.requires_human_review:
            issue = False
        level = packet.level if issue else ("L0" if packet.action != "manual_review" else "L0")
        start_time = obs.timestamp + timedelta(minutes=10)
        end_time = start_time + timedelta(minutes=self.config.window_minutes)
        decision_trace = self._decision_trace(fusion, packet, source_health)

        decision = WarningDecision(
            issue=issue,
            level=level,
            action=packet.action if not packet.requires_human_review else "manual_review",
            start_time=start_time,
            end_time=end_time,
            affected_area=self.config.region_name,
            hazard_prob=dict(fusion.hazard_prob),
            hazard_prob_raw=dict(fusion.hazard_prob_raw),
            calibrated_prob=dict(fusion.calibrated_prob),
            lower_confidence_bound=dict(fusion.lower_confidence_bound),
            p_release=dict(fusion.p_release),
            probability_quality_factor=float(fusion.probability_quality_factor),
            confidence=float(fusion.evidence_readiness_score),
            evidence_readiness_score=float(fusion.evidence_readiness_score),
            evidence_readiness_breakdown=dict(fusion.evidence_readiness_breakdown),
            rationale=evidence_cards,
            conflicts=list(fusion.conflicts),
            per_agent_contribution=dict(fusion.per_agent_contribution),
            family_contribution=dict(fusion.family_contribution),
            degraded_mode=bool(fusion.degraded_mode),
            degraded_reasons=list(fusion.degraded_reasons),
            degraded_mode_flags=list(fusion.degraded_mode_flags),
            requires_human_review=packet.requires_human_review,
            proxy_dependency_ratio=float(fusion.proxy_dependency_ratio),
            effective_source_tier=str(fusion.effective_source_tier),
            veto_reasons=list(packet.veto_reasons),
            decision_trace=decision_trace,
            policy_version=policy.policy_version,
            object_id=hazard_object.object_id,
        )

        self._state.issued = issue
        self._state.object_id = hazard_object.object_id

        config_payload = json.dumps(self.config.__dict__, ensure_ascii=False, sort_keys=True, default=str)
        bundle_id = _sha(f"{request_id}:{trace_id}:{obs.timestamp.isoformat()}:{config_payload}")
        replay_bundle = ReplayBundle(
            bundle_id=bundle_id,
            run_id="",
            request_id=request_id,
            trace_id=trace_id,
            source_snapshot_references=[asdict(p) for p in envelope.source_registry],
            normalized_features=envelope.feature_planes,
            agent_outputs=[asdict(card) for card in evidence_cards],
            fusion_snapshot=asdict(fusion),
            policy_snapshot=asdict(policy),
            decision_trace={"packet": asdict(packet), "trace": decision_trace},
            code_reference={"code_version": "orchestrator.v3", "feature_version": self.config.feature_version},
            model_references={
                "model_version": self.config.model_version,
                "calibration_version": "memory.calibrators",
                "llm_agent_enabled": bool(self.config.enable_llm_agent),
                "llm_agent_mode": str(self.config.llm_agent_mode),
                "llm_model": str(self.config.llm_model),
            },
            truth_reference={},
            lineage={
                "source_lineage": [p.source_id for p in envelope.source_registry],
                "transform_lineage": ["provider.normalize", "observation_adapter", "family_fusion"],
                "feature_lineage": {"feature_version": self.config.feature_version},
                "model_lineage": {
                    "model_version": self.config.model_version,
                    "agent_count": len(self.agents),
                    "llm_agent_enabled": bool(self.config.enable_llm_agent),
                    "llm_agent_mode": str(self.config.llm_agent_mode),
                    "llm_model": str(self.config.llm_model),
                },
                "policy_lineage": {"policy_version": policy.policy_version},
                "human_lineage": {"requires_human_review": packet.requires_human_review},
            },
            manifest={"envelope_id": envelope.envelope_id, "domain_id": envelope.domain_id},
        )

        audit = AuditRecord(
            run_id=_sha(f"{obs.city}-{obs.timestamp.isoformat()}-{request_id}-{trace_id}"),
            request_id=request_id,
            trace_id=trace_id,
            bundle_id=bundle_id,
            generated_at=datetime.now().isoformat(),
            city=obs.city,
            config_hash=_sha(config_payload),
            code_version="orchestrator.v3",
            policy_version=policy.policy_version,
            model_version=self.config.model_version,
            feature_version=self.config.feature_version,
            data_window={
                "observation_time": obs.timestamp.isoformat(),
                "decision_start": start_time.isoformat(),
                "decision_end": end_time.isoformat(),
            },
            source_health=source_health,
            provenance={
                c.agent: {
                    "family": c.family,
                    "source_tier": c.source_tier,
                    "input_fields": c.input_fields,
                    "upstream_sources": c.upstream_sources,
                    "observed_at": c.observed_at,
                    "missing_fields": c.missing_fields,
                    "rule_version": c.rule_version,
                    "model_version": c.model_version,
                    "proxy_source": bool(c.proxy_source or c.supporting_features.get("proxy_source", False)),
                    "provider_fingerprint": c.provider_fingerprint,
                }
                for c in evidence_cards
            },
            fusion_snapshot={
                "hazard_prob": decision.hazard_prob,
                "hazard_prob_raw": decision.hazard_prob_raw,
                "calibrated_prob": decision.calibrated_prob,
                "lower_confidence_bound": decision.lower_confidence_bound,
                "p_release": decision.p_release,
                "probability_quality_factor": decision.probability_quality_factor,
                "readiness": fusion.evidence_readiness_score,
                "readiness_breakdown": fusion.evidence_readiness_breakdown,
                "per_agent_contribution": fusion.per_agent_contribution,
                "family_contribution": fusion.family_contribution,
                "conflicts": fusion.conflicts,
                "veto_reasons": fusion.veto_reasons,
            },
            decision_snapshot={
                "issue": issue,
                "level": level,
                "action": decision.action,
                "requires_human_review": decision.requires_human_review,
                "hazard_prob": decision.hazard_prob,
                "hazard_prob_raw": decision.hazard_prob_raw,
                "p_release": decision.p_release,
                "decision_trace": decision_trace,
                "object_id": hazard_object.object_id,
            },
            lineage=replay_bundle.lineage,
            replay_verification={"bundle_id": bundle_id, "tolerance": replay_bundle.replay_tolerance, "status": "ready"},
        )
        replay_bundle.run_id = audit.run_id
        decision_packet = packet
        decision_packet.trace_id = trace_id
        return CycleResult(
            observation=obs,
            decision=decision,
            generated_at=datetime.now(),
            audit=audit,
            envelope=envelope,
            fusion_result=fusion,
            hazard_object=hazard_object,
            decision_packet=decision_packet,
            replay_bundle=replay_bundle,
            request_id=request_id,
            trace_id=trace_id,
        )
