from __future__ import annotations

import datetime as dt
import os
from urllib.parse import quote

from weather_agent.models import Observation, ObservationEnvelope
from weather_agent.adapters.rainviewer import fetch_radar_dbz, latest_radar_tile_descriptor
from weather_agent.providers import (
    LocalRadarProvider,
    NowCoastRadarProvider,
    OpenMeteoGuidanceProvider,
    ProxyObjectiveGuidanceProvider,
    RainViewerRadarProvider,
)


def _provider_priority() -> list[str]:
    raw = (os.getenv("AGENT_RADAR_PROVIDER_PRIORITY", "") or "").strip()
    if not raw:
        return ["local_grid", "nowcoast", "rainviewer"]
    out = []
    for p in raw.split(","):
        name = p.strip().lower()
        if name and name not in out:
            out.append(name)
    return out or ["local_grid", "nowcoast", "rainviewer"]


def _fetch_radar(lat: float, lon: float) -> dict:
    provider = (os.getenv("AGENT_RADAR_PROVIDER", "auto") or "auto").strip().lower()
    chain = [provider] if provider not in {"", "auto"} else _provider_priority()
    attempts: list[str] = []
    now = dt.datetime.now()

    for name in chain:
        try:
            if name == "local_grid":
                fp = os.getenv("AGENT_RADAR_GRID_FILE", "data/radar_grids/latest.json")
                max_dist = float(os.getenv("AGENT_RADAR_GRID_MAX_DISTANCE_KM", "180") or 180)
                provider_impl = LocalRadarProvider()
                raw = provider_impl.fetch(lat=lat, lon=lon, path=fp, max_distance_km=max_dist)
                attempts.append(f"{name}:{raw.get('status')}")
                normalized = provider_impl.normalize(raw, now=now, domain_id="domain.live")
                normalized["selected_provider"] = name
                normalized["requested_provider"] = provider
                normalized["attempts"] = attempts
                if raw.get("status") == "ok":
                    return normalized
                continue

            if name == "nowcoast":
                provider_impl = NowCoastRadarProvider()
                raw = provider_impl.fetch(lat=lat, lon=lon)
                attempts.append(f"{name}:{raw.get('status')}")
                normalized = provider_impl.normalize(raw, now=now, domain_id="domain.live")
                normalized["selected_provider"] = name
                normalized["requested_provider"] = provider
                normalized["attempts"] = attempts
                if raw.get("status") == "ok":
                    return normalized
                continue

            if name == "rainviewer":
                provider_impl = RainViewerRadarProvider()
                raw = fetch_radar_dbz(lat, lon)
                attempts.append(f"{name}:proxy")
                normalized = provider_impl.normalize(raw, now=now, domain_id="domain.live")
                normalized["selected_provider"] = name
                normalized["requested_provider"] = provider
                normalized["attempts"] = attempts
                normalized["status"] = "fallback_proxy"
                return normalized
        except Exception as e:
            attempts.append(f"{name}:error:{e}")

    fallback = RainViewerRadarProvider().normalize({"dbz": 0.0, "bow_echo": False, "source": "none"}, now=now, domain_id="domain.live")
    fallback["attempts"] = attempts
    fallback["status"] = "all_failed"
    fallback["selected_provider"] = ""
    fallback["requested_provider"] = provider
    fallback["coverage"]["coverage_ratio"] = 0.0
    fallback["qc_flags"] = ["fetch_failed", "proxy_source"]
    return fallback


def _build_radar_visualization_template(radar_payload: dict) -> str:
    data = dict(radar_payload.get("data") or {})
    frame_path = str(data.get("frame_path", "") or "").strip()
    frame_time = data.get("frame_time")
    if not frame_path:
        try:
            descriptor = latest_radar_tile_descriptor(timeout=12)
        except Exception:
            descriptor = {}
        frame_path = str(descriptor.get("frame_path", "") or "").strip()
        frame_time = descriptor.get("frame_time", frame_time)
    if not frame_path:
        return ""
    query = f"frame_path={quote(frame_path, safe='')}"
    if frame_time is not None:
        query += f"&frame_time={quote(str(frame_time), safe='')}"
    return f"/api/v1/tiles/radar/{{z}}/{{x}}/{{y}}.png?{query}"


def build_live_envelope(city: str = "Tianjin") -> ObservationEnvelope:
    now = dt.datetime.now()
    guidance_provider = OpenMeteoGuidanceProvider()
    guidance_raw = guidance_provider.fetch(city=city, now=now)
    guidance_payload = guidance_provider.normalize(guidance_raw)
    geo = guidance_raw["geo"]
    lat = float(geo["latitude"])
    lon = float(geo["longitude"])

    radar_payload = _fetch_radar(lat, lon)
    radar_visual_template = _build_radar_visualization_template(radar_payload)
    proxy_guidance_provider = ProxyObjectiveGuidanceProvider()
    proxy_guidance_payload = proxy_guidance_provider.normalize(
        proxy_guidance_provider.fetch(guidance=guidance_payload["data"]["prob_guidance"], now=now),
        domain_id=guidance_payload["domain_id"],
    )

    source_registry = [
        guidance_provider.to_record(guidance_payload),
        proxy_guidance_provider.to_record(proxy_guidance_payload),
    ]
    qc_registry = {
        guidance_provider.source_id: guidance_provider.validate(guidance_payload),
        proxy_guidance_provider.source_id: proxy_guidance_provider.validate(proxy_guidance_payload),
    }

    radar_provider_name = str(radar_payload.get("selected_provider") or "")
    radar_provider = {
        "local_grid": LocalRadarProvider(),
        "nowcoast": NowCoastRadarProvider(),
        "rainviewer": RainViewerRadarProvider(),
        "": RainViewerRadarProvider(),
    }[radar_provider_name if radar_provider_name in {"local_grid", "nowcoast", "rainviewer"} else ""]
    source_registry.append(radar_provider.to_record(radar_payload))
    qc_registry[radar_provider.source_id] = radar_provider.validate(radar_payload)

    radar_freshness_score = max(0.0, min(1.0, 1.0 - float(radar_payload.get("freshness_sec", 99999.0)) / 5400.0))
    model_spread_score = float(guidance_payload["data"]["model_spread_score"])
    model_coverage_score = float(guidance_payload["coverage"]["coverage_ratio"])
    data_quality_score = max(0.0, min(1.0, 0.45 * model_coverage_score + 0.35 * radar_freshness_score + 0.20 * (1.0 - model_spread_score)))

    obs = Observation(
        timestamp=now,
        city=geo.get("name", city),
        vertical_velocity=float(guidance_payload["data"]["vertical_velocity"]),
        low_level_convergence=float(guidance_payload["data"]["low_level_convergence"]),
        cape=float(guidance_payload["data"]["cape"]),
        dcape=float(guidance_payload["data"]["dcape"]),
        shear_0_6km=float(guidance_payload["data"]["shear_0_6km"]),
        t850_500=float(guidance_payload["data"]["t850_500"]),
        wbz_km=float(guidance_payload["data"]["wbz_km"]),
        humidity_low=float(guidance_payload["data"]["humidity_low"]),
        radar_dbz_max=float((radar_payload.get("data") or {}).get("dbz", 0.0)),
        radar_bow_echo=bool((radar_payload.get("data") or {}).get("bow_echo", False)),
        storm_motion_ms=float(guidance_payload["data"]["storm_motion_ms"]),
        prob_guidance=dict(guidance_payload["data"]["prob_guidance"]),
        source_meta={
            "mode": "live",
            "city": str(geo.get("name", city)),
            "lat": str(lat),
            "lon": str(lon),
            "timezone": str(guidance_payload["data"]["timezone"]),
            "analysis_time": str(guidance_payload["analysis_time"]),
            "valid_time": str(guidance_payload["valid_time"]),
            "lead_time": str(guidance_payload["lead_time"]),
            "models": ",".join(guidance_payload["data"]["models"]),
            "radar_source": str((radar_payload.get("data") or {}).get("source", radar_provider.source_id)),
            "radar_provider_requested": str(radar_payload.get("requested_provider", "auto")),
            "radar_provider_used": radar_provider_name or "rainviewer",
            "radar_status": str(radar_payload.get("status", "unknown")),
            "radar_proxy_source": "1" if radar_provider.source_tier == "proxy" else "0",
            "radar_feature_kind": str((radar_payload.get("data") or {}).get("feature_kind", "grid")),
            "radar_tile_url_template": radar_visual_template,
            "radar_visual_source": "rainviewer_proxy" if radar_visual_template else "unavailable",
            "radar_attempts": "|".join([str(x) for x in radar_payload.get("attempts", [])]),
            "model_count": str(len(guidance_payload["data"]["models"])),
            "model_coverage_score": f"{model_coverage_score:.3f}",
            "model_spread_score": f"{model_spread_score:.3f}",
            "radar_age_minutes": f"{float(radar_payload.get('freshness_sec', 0.0)) / 60.0:.1f}",
            "radar_freshness_score": f"{radar_freshness_score:.3f}",
            "data_quality_score": f"{data_quality_score:.3f}",
            "area_coverage_ratio": f"{max(0.05, float(radar_payload.get('coverage', {}).get('coverage_ratio', 0.12))):.3f}",
            "signal_persist_minutes": "30",
            "grid_spec": str(guidance_payload["grid_spec"].get("grid_id")),
            "source_tier_summary": ",".join([p.source_tier for p in source_registry]),
        },
    )
    feature_planes = {
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
            "provider_snapshot": radar_payload,
        },
        "guidance": {
            "prob_guidance": dict(obs.prob_guidance),
            "guidance_snapshot": guidance_payload,
            "proxy_guidance_snapshot": proxy_guidance_payload,
        },
    }
    return ObservationEnvelope.from_observation(
        obs,
        domain_id=guidance_payload["domain_id"],
        feature_planes=feature_planes,
        source_registry=source_registry,
        qc_registry=qc_registry,
        provenance_manifest={
            "providers": [
                {"source_id": p.source_id, "source_tier": p.source_tier, "provider_version": p.provider_version}
                for p in source_registry
            ],
            "data_contract": {
                "analysis_time": guidance_payload["analysis_time"],
                "valid_time": guidance_payload["valid_time"],
                "lead_time": guidance_payload["lead_time"],
                "domain_id": guidance_payload["domain_id"],
                "grid_spec": guidance_payload["grid_spec"],
            },
        },
        object_context={"lat": lat, "lon": lon, "city": geo.get("name", city)},
    )


def build_live_observation(city: str = "Tianjin") -> Observation:
    return build_live_envelope(city).to_observation()
