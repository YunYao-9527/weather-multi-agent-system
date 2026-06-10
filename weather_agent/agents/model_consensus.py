from __future__ import annotations

from weather_agent.base import ForecastAgent
from weather_agent.models import AgentOutput, EvidenceCard, Observation, clamp01, hazard_template


class ModelConsensusAgent(ForecastAgent):
    name = "model_consensus"

    def run(self, obs: Observation) -> AgentOutput:
        model_count = float(obs.source_meta.get("model_count", 1.0))
        spread_score = clamp01(float(obs.source_meta.get("model_spread_score", 0.4)))
        coverage = clamp01(model_count / 3.0)
        agreement = clamp01(1.0 - spread_score)

        # Consensus signal = environmental support * ensemble agreement * model coverage
        base_env = clamp01((obs.low_level_convergence * 0.35) + (obs.cape / 3000.0) * 0.35 + (obs.shear_0_6km / 30.0) * 0.3)
        signal = clamp01(base_env * (0.55 + 0.45 * agreement) * (0.6 + 0.4 * coverage))

        scores = hazard_template()
        scores["short_rain"] = clamp01(signal * 0.8)
        scores["wind"] = clamp01(signal * 0.7)
        scores["hail"] = clamp01(signal * 0.45)

        claim = "中尺度模式结论总体一致，可用于时空落区细化" if agreement >= 0.6 else "中尺度模式分歧偏大，需降低单一模式权重"
        card = EvidenceCard(
            agent=self.name,
            claim=claim,
            confidence=clamp01(0.35 + signal * 0.45 + agreement * 0.2),
            hazard_scores=scores,
            supporting_features={
                "consensus_signal": round(signal, 3),
                "proxy_cape": obs.cape,
                "proxy_shear": obs.shear_0_6km,
                "model_count": model_count,
                "model_spread_score": round(spread_score, 3),
                "source": "open_meteo_models",
                "proxy_source": False,
            },
            proxy_source=False,
            upstream_sources=["open_meteo_models"],
            rule_version="model_consensus.v2",
            model_version="heuristic_model.v2",
        )
        return AgentOutput(agent=self.name, evidence=card)
