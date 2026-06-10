from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from math import sqrt
from typing import Dict, List

from weather_agent.models import FusionResult, HazardObject, ObservationEnvelope, PolicySnapshot, stable_hash
from weather_agent.storage import ObjectRepository


def _distance_km(a: Dict[str, float], b: Dict[str, float]) -> float:
    dx = float(a.get("lon", 0.0)) - float(b.get("lon", 0.0))
    dy = float(a.get("lat", 0.0)) - float(b.get("lat", 0.0))
    return sqrt(dx * dx + dy * dy) * 111.0


def _level_state(p_release: float, lower_bound: float, persistence: int, policy: PolicySnapshot) -> str:
    if p_release < policy.clear_threshold:
        return "recommend_clear"
    if p_release >= policy.issue_threshold and lower_bound >= policy.lower_bound_threshold:
        if persistence >= policy.min_duration:
            return "recommend_issue"
        return "watch"
    if p_release >= policy.clear_threshold:
        return "watch"
    return "candidate"


class HazardObjectEngine:
    def __init__(self, repository: ObjectRepository | None = None):
        self.repository = repository or ObjectRepository()

    def _dominant_hazard(self, fusion: FusionResult) -> tuple[str, float, float]:
        hazard = max(fusion.p_release or {"short_rain": 0.0}, key=lambda k: float((fusion.p_release or {}).get(k, 0.0)))
        p_release = float((fusion.p_release or {}).get(hazard, 0.0))
        lower = float((fusion.lower_confidence_bound or {}).get(hazard, 0.0))
        return hazard, p_release, lower

    def _make_geometry(self, centroid: Dict[str, float], area_km2: float) -> Dict[str, object]:
        r = max(5.0, sqrt(max(area_km2, 1.0) / 3.14159))
        lon = float(centroid.get("lon", 0.0))
        lat = float(centroid.get("lat", 0.0))
        return {
            "type": "Polygon",
            "coordinates": [[
                [lon - r / 111.0, lat - r / 111.0],
                [lon + r / 111.0, lat - r / 111.0],
                [lon + r / 111.0, lat + r / 111.0],
                [lon - r / 111.0, lat + r / 111.0],
                [lon - r / 111.0, lat - r / 111.0],
            ]],
        }

    def _match_existing(self, hazard_type: str, centroid: Dict[str, float], valid_time: str) -> dict | None:
        active = self.repository.active()
        candidates = []
        for item in active:
            if item.get("hazard_type") != hazard_type:
                continue
            dist = _distance_km(item.get("centroid", {}), centroid)
            candidates.append((dist, item))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        dist, best = candidates[0]
        return best if dist <= 80.0 else None

    def build_object(self, envelope: ObservationEnvelope, fusion: FusionResult, policy: PolicySnapshot) -> HazardObject:
        hazard_type, p_release, lower = self._dominant_hazard(fusion)
        valid_time = envelope.valid_time or envelope.analysis_time or datetime.now().isoformat()
        lat = float(envelope.object_context.get("lat") or envelope.source_meta.get("lat") or 0.0)
        lon = float(envelope.object_context.get("lon") or envelope.source_meta.get("lon") or 0.0)
        centroid = {"lat": lat, "lon": lon}
        area_ratio = float(envelope.source_meta.get("area_coverage_ratio", 0.12) or 0.12)
        area_km2 = max(25.0, area_ratio * 1500.0)
        evidence_ids: List[str] = []
        existing = self._match_existing(hazard_type, centroid, valid_time)
        if existing:
            evidence_persistence = int(existing.get("evidence_persistence", 0) or 0) + 1
            object_version = int(existing.get("object_version", 1) or 1) + 1
            object_id = str(existing.get("object_id"))
            parent_object_id = str(existing.get("parent_object_id") or existing.get("object_id"))
            history = list(existing.get("history", []))
        else:
            persist_minutes = float(envelope.source_meta.get("signal_persist_minutes", 0.0) or 0.0)
            evidence_persistence = max(1, int(round(persist_minutes / 20.0))) or 1
            object_version = 1
            object_id = f"obj_{stable_hash((envelope.domain_id, hazard_type, valid_time, centroid))}"
            parent_object_id = ""
            history = []
        lifecycle_state = _level_state(p_release, lower, evidence_persistence, policy)
        motion_vector = {
            "speed_ms": float(envelope.feature_planes.get("radar", {}).get("storm_motion_ms", 0.0)),
            "direction_deg": float(envelope.object_context.get("motion_direction_deg", 225.0)),
        }
        motion_stability = 1.0 if motion_vector["speed_ms"] >= 3.0 else 0.5
        source_stability = max(0.0, 1.0 - float(envelope.source_tiers_summary.get("proxy_share", 0.0)))
        obj = HazardObject(
            object_id=object_id,
            parent_object_id=parent_object_id,
            object_type="hazard_object",
            hazard_type=hazard_type,
            geometry=self._make_geometry(centroid, area_km2),
            centroid=centroid,
            area_km2=round(area_km2, 3),
            motion_vector=motion_vector,
            lifecycle_state=lifecycle_state,
            start_time=str(existing.get("start_time")) if existing else valid_time,
            end_time=valid_time,
            last_update_time=valid_time,
            confidence=round(p_release, 4),
            support_evidence_ids=evidence_ids,
            source_tiers=list({p.source_tier for p in envelope.source_registry}),
            object_version=object_version,
            evidence_persistence=evidence_persistence,
            source_stability=round(source_stability, 4),
            motion_stability=round(motion_stability, 4),
            min_duration_met=evidence_persistence >= policy.min_duration,
            min_area_met=area_km2 >= policy.min_area,
            history=history + [{"version": object_version, "state": lifecycle_state, "time": valid_time, "confidence": round(p_release, 4)}],
        )
        self.repository.save(asdict(obj))
        return obj
