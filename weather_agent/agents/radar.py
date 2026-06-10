from __future__ import annotations

from weather_agent.base import ForecastAgent
from weather_agent.models import AgentOutput, EvidenceCard, Observation, clamp01, hazard_template
from weather_agent.rules import thresholds_for


class RadarAgent(ForecastAgent):
    name = "radar"

    def run(self, obs: Observation) -> AgentOutput:
        th = thresholds_for(obs.timestamp)
        dbz_factor = clamp01((obs.radar_dbz_max - 35.0) / 30.0)
        bow_bonus = 0.2 if obs.radar_bow_echo else 0.0
        motion_factor = clamp01(obs.storm_motion_ms / 20.0)

        wind = clamp01(dbz_factor * 0.55 + bow_bonus + motion_factor * 0.2)
        hail = clamp01(dbz_factor * 0.35 + (0.25 if obs.radar_dbz_max >= th.hail_dbz_trigger else 0.0))
        rain = clamp01(dbz_factor * 0.6 + (0.1 if obs.storm_motion_ms <= 8 else 0.0))

        scores = hazard_template()
        scores["short_rain"] = rain
        scores["wind"] = wind
        scores["hail"] = hail

        source = str(obs.source_meta.get("radar_source", "unknown"))
        feature_kind = str(obs.source_meta.get("radar_feature_kind", "unknown"))
        proxy_source = str(obs.source_meta.get("radar_proxy_source", "1")) == "1"
        provider = str(obs.source_meta.get("radar_provider_used", "unknown"))
        claim = "雷达回波显示对流已组织化发展，需关注短临预警" if dbz_factor > 0.5 else "雷达信号较弱，重点观察后续触发"
        card = EvidenceCard(
            agent=self.name,
            claim=claim,
            confidence=clamp01(0.5 + dbz_factor * 0.4),
            hazard_scores=scores,
            supporting_features={
                "radar_dbz_max": obs.radar_dbz_max,
                "radar_bow_echo": obs.radar_bow_echo,
                "storm_motion_ms": obs.storm_motion_ms,
                "hail_dbz_trigger": th.hail_dbz_trigger,
                "source": source,
                "provider": provider,
                "radar_feature_kind": feature_kind,
                "proxy_source": proxy_source,
            },
            proxy_source=proxy_source,
            upstream_sources=[source],
            rule_version="radar_agent.v2",
            model_version="heuristic_radar.v3",
        )
        return AgentOutput(agent=self.name, evidence=card)
