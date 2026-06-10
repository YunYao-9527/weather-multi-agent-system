from __future__ import annotations

from typing import Dict, List

from weather_agent.models import ObservationEnvelope


def _safe_float(v: object, default: float) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def source_health_summary(source_meta: Dict[str, str], envelope: ObservationEnvelope | None = None) -> dict:
    model_count = _safe_float(source_meta.get("model_count"), 0.0)
    model_coverage = _safe_float(source_meta.get("model_coverage_score"), 0.0)
    model_spread = _safe_float(source_meta.get("model_spread_score"), 1.0)
    radar_age = _safe_float(source_meta.get("radar_age_minutes"), 999.0)
    radar_freshness = _safe_float(source_meta.get("radar_freshness_score"), 0.0)
    data_quality = _safe_float(source_meta.get("data_quality_score"), 0.0)

    warnings: List[str] = []
    veto_reasons: List[str] = []
    degraded_flags: List[str] = []

    if model_count < 1:
        warnings.append("missing_model_data")
        degraded_flags.append("missing_model_data")
    if model_coverage < 0.34:
        warnings.append("low_model_coverage")
    if model_spread > 0.65:
        warnings.append("high_model_spread")
    if radar_age > 90:
        warnings.append("radar_stale")
        veto_reasons.append("critical_source_stale")
    if radar_freshness < 0.4:
        warnings.append("radar_freshness_low")
    if data_quality < 0.5:
        warnings.append("data_quality_low")
        degraded_flags.append("data_quality_low")

    provider_qc: dict = {}
    if envelope is not None:
        for source_id, qc in envelope.qc_registry.items():
            provider_qc[source_id] = {
                "status": qc.status,
                "stale": qc.stale,
                "coverage_ok": qc.coverage_ok,
                "time_alignment_ok": qc.time_alignment_ok,
                "flags": list(qc.flags),
            }
            if qc.stale:
                veto_reasons.append(f"{source_id}:stale")
            if not qc.coverage_ok:
                veto_reasons.append(f"{source_id}:coverage_anomaly")
            if qc.flags:
                degraded_flags.extend([f"{source_id}:{flag}" for flag in qc.flags])

    return {
        "model_count": model_count,
        "model_coverage_score": model_coverage,
        "model_spread_score": model_spread,
        "radar_age_minutes": radar_age,
        "radar_freshness_score": radar_freshness,
        "data_quality_score": data_quality,
        "warnings": sorted(set(warnings)),
        "veto_reasons": sorted(set(veto_reasons)),
        "degraded_flags": sorted(set(degraded_flags)),
        "degraded_mode": bool(warnings or provider_qc),
        "provider_qc": provider_qc,
    }
