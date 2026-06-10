from __future__ import annotations

from weather_agent.base import ForecastAgent
from weather_agent.models import AgentOutput, EvidenceCard, Observation, clamp01, hazard_template


class DataQualityGuardAgent(ForecastAgent):
    name = "data_quality_guard"

    def run(self, obs: Observation) -> AgentOutput:
        quality = clamp01(float(obs.source_meta.get("data_quality_score", 0.62)))
        freshness = clamp01(float(obs.source_meta.get("radar_freshness_score", 0.6)))

        # Diagnostic-only agent: it does not directly vote hazard probabilities.
        scores = hazard_template()
        for h in scores:
            scores[h] = 0.0

        if quality >= 0.75:
            claim = "数据时效与覆盖较好，可支持业务研判，但仍需结合冲突诊断"
        elif quality >= 0.55:
            claim = "数据质量中等，建议提高人工复核权重"
        else:
            claim = "数据质量偏弱或时效不足，需降级运行并人工复核"

        card = EvidenceCard(
            agent=self.name,
            claim=claim,
            confidence=clamp01(0.30 + 0.40 * quality + 0.20 * freshness),
            hazard_scores=scores,
            supporting_features={
                "data_quality_score": quality,
                "radar_freshness_score": freshness,
                "model_count": float(obs.source_meta.get("model_count", 0.0)),
                "model_spread_score": float(obs.source_meta.get("model_spread_score", 0.0)),
                "diagnostic_only": True,
                "source": "quality_guard",
                "proxy_source": False,
            },
            proxy_source=False,
            upstream_sources=["quality_guard"],
            rule_version="data_quality_guard.v3",
            model_version="rule_based.v3",
        )
        return AgentOutput(agent=self.name, evidence=card)
