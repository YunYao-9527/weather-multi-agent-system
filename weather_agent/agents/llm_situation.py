from __future__ import annotations

from weather_agent.base import ForecastAgent
from weather_agent.llm_openai import LLMSituationEvidencePayload, OpenAIResponsesClient
from weather_agent.models import AgentOutput, EvidenceCard, Observation, ObservationEnvelope, clamp01


class LLMSituationAgent(ForecastAgent):
    name = "llm_situation"

    def __init__(self, *, client: OpenAIResponsesClient, mode: str = "shadow") -> None:
        self.client = client
        self.mode = (mode or "shadow").strip().lower()
        if self.mode not in {"shadow", "assist"}:
            self.mode = "shadow"

    def _to_evidence(self, payload: LLMSituationEvidencePayload, envelope: ObservationEnvelope) -> EvidenceCard:
        provider = str(getattr(self.client, "provider", "openai") or "openai").strip().lower()
        factors = [str(x).strip() for x in payload.key_factors if str(x).strip()]
        cautions = [str(x).strip() for x in payload.caution_flags if str(x).strip()]
        supporting = {
            "source": "deepseek_chat_completions" if provider == "deepseek" else "openai_responses",
            "provider": provider,
            "llm_model": self.client.model,
            "llm_mode": self.mode,
            "action_bias": payload.action_bias,
            "key_factors": " | ".join(factors[:8]),
            "caution_flags": " | ".join(cautions[:8]),
            "diagnostic_only": self.mode != "assist",
            "proxy_source": False,
        }
        return EvidenceCard(
            agent=self.name,
            claim=payload.claim,
            confidence=clamp01(payload.confidence),
            hazard_scores=payload.hazard_scores.model_dump(),
            supporting_features=supporting,
            proxy_source=False,
            upstream_sources=["deepseek_chat_completions" if provider == "deepseek" else "openai_responses"],
            observed_at=envelope.valid_time or envelope.analysis_time,
            rule_version="llm_situation_agent.v1",
            model_version=f"{provider}.{('chat_completions' if provider == 'deepseek' else 'responses')}:{self.client.model}",
            family="llm_reasoning",
            source_tier="experimental",
        )

    def run_envelope(self, envelope: ObservationEnvelope) -> AgentOutput:
        payload = self.client.generate_situation_evidence(envelope)
        return AgentOutput(agent=self.name, evidence=self._to_evidence(payload, envelope))

    def run(self, obs: Observation) -> AgentOutput:
        return self.run_envelope(ObservationEnvelope.from_observation(obs))
