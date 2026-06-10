from __future__ import annotations

from weather_agent.base import ForecastAgent
from weather_agent.models import AgentOutput, EvidenceCard, Observation, clamp01, hazard_template


class CrossSourceConsistencyAgent(ForecastAgent):
    name = "cross_source_consistency"

    def run(self, obs: Observation) -> AgentOutput:
        env_signal = clamp01(
            0.30 * clamp01(obs.cape / 2200.0)
            + 0.25 * clamp01(obs.shear_0_6km / 22.0)
            + 0.25 * clamp01(obs.low_level_convergence)
            + 0.20 * clamp01((60.0 - abs(obs.vertical_velocity) * 25.0) / 60.0)
        )
        obj_signal = clamp01(max(float(v) for v in obs.prob_guidance.values()) if obs.prob_guidance else 0.0)
        radar_signal = clamp01((obs.radar_dbz_max - 30.0) / 25.0)

        agreement = clamp01(1.0 - abs(env_signal - obj_signal))
        tri_consistency = clamp01(1.0 - (abs(env_signal - radar_signal) + abs(obj_signal - radar_signal)) / 2.0)
        consistency = clamp01(0.55 * agreement + 0.45 * tri_consistency)

        blend = clamp01(0.5 * obj_signal + 0.3 * env_signal + 0.2 * radar_signal)
        scores = hazard_template()
        scores["short_rain"] = clamp01(blend * 0.8)
        scores["wind"] = clamp01(blend * 0.75)
        scores["hail"] = clamp01(blend * 0.55)
        scores["tornado"] = clamp01(blend * 0.25)

        if consistency >= 0.72:
            claim = "模式、客观概率与雷达信号一致性较高，本轮结论稳定性较好"
        elif consistency >= 0.5:
            claim = "多源信号存在一定分歧，建议关注冲突项并滚动更新"
        else:
            claim = "多源信号分歧较大，自动结论不宜直接发布"

        card = EvidenceCard(
            agent=self.name,
            claim=claim,
            confidence=clamp01(0.4 + 0.5 * consistency),
            hazard_scores=scores,
            supporting_features={
                "env_signal": round(env_signal, 3),
                "obj_signal": round(obj_signal, 3),
                "radar_signal": round(radar_signal, 3),
                "consistency": round(consistency, 3),
                "source": "cross_source_diagnostics",
                "proxy_source": False,
            },
            proxy_source=False,
            upstream_sources=["cross_source_diagnostics"],
            rule_version="cross_source_consistency.v3",
            model_version="rule_based.v3",
        )
        return AgentOutput(agent=self.name, evidence=card)
