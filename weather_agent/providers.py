from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import asdict
from statistics import median
from typing import Any, Dict, List

from weather_agent.adapters.noaa_nowcoast import fetch_nowcoast_radar_grid
from weather_agent.adapters.objective_guidance import hazard_probs_from_features
from weather_agent.adapters.open_meteo import fetch_model_hourly, geocode_city, nearest_hour_index
from weather_agent.adapters.radar_local_grid import LocalGridConfig, fetch_local_radar_grid
from weather_agent.adapters.rainviewer import fetch_radar_dbz
from weather_agent.adapters.utils import clamp, shear_from_winds
from weather_agent.models import GridSpec, ProviderRecord, QCRecord, SOURCE_TIERS, stable_hash


MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _wind_to_ms(v: float, unit: str | None) -> float:
    u = (unit or "").strip().lower()
    if u in {"km/h", "kmh", "kph"}:
        return float(v) / 3.6
    if u in {"mph"}:
        return float(v) * 0.44704
    return float(v)


class BaseProvider(ABC):
    source_id = "provider"
    source_family = "unknown"
    source_tier = "experimental"
    provider_version = "v1"

    @abstractmethod
    def fetch(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, raw: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def validate(self, payload: Dict[str, Any]) -> QCRecord:
        qc_flags = list(payload.get("qc_flags", []))
        freshness = float(payload.get("freshness_sec", 0.0))
        coverage_ratio = float(payload.get("coverage", {}).get("coverage_ratio", 1.0) or 1.0)
        stale = freshness > float(payload.get("stale_threshold_sec", 5400.0))
        coverage_ok = coverage_ratio >= 0.05
        if stale:
            qc_flags.append("stale")
        if not coverage_ok:
            qc_flags.append("coverage_low")
        version_registered = bool(payload.get("provider_version"))
        status = "ok" if not qc_flags else "degraded"
        return QCRecord(
            source_id=str(payload.get("source_id", self.source_id)),
            status=status,
            stale=stale,
            coverage_ok=coverage_ok,
            version_registered=version_registered,
            flags=sorted(set(qc_flags)),
            metrics={
                "freshness_sec": freshness,
                "coverage_ratio": coverage_ratio,
                "latency_sec": float(payload.get("latency_sec", 0.0)),
            },
            message=";".join(sorted(set(qc_flags))) or "ok",
        )

    def fingerprint(self, payload: Dict[str, Any]) -> str:
        return stable_hash((self.source_id, payload.get("issue_time"), payload.get("valid_time"), payload.get("data")))

    def health_status(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if payload is None:
            return {
                "source_id": self.source_id,
                "source_family": self.source_family,
                "source_tier": self.source_tier,
                "provider_version": self.provider_version,
                "status": "registered",
            }
        qc = self.validate(payload)
        return {
            "source_id": self.source_id,
            "source_family": self.source_family,
            "source_tier": self.source_tier,
            "provider_version": self.provider_version,
            "status": qc.status,
            "flags": qc.flags,
            "metrics": qc.metrics,
        }

    def to_record(self, payload: Dict[str, Any]) -> ProviderRecord:
        normalized = dict(payload)
        normalized["fingerprint"] = self.fingerprint(normalized)
        return ProviderRecord(
            source_id=str(normalized.get("source_id", self.source_id)),
            source_family=str(normalized.get("source_family", self.source_family)),
            source_tier=str(normalized.get("source_tier", self.source_tier)),
            provider_version=str(normalized.get("provider_version", self.provider_version)),
            issue_time=str(normalized.get("issue_time", "")),
            valid_time=str(normalized.get("valid_time", "")),
            ingest_time=str(normalized.get("ingest_time", "")),
            latency_sec=float(normalized.get("latency_sec", 0.0)),
            freshness_sec=float(normalized.get("freshness_sec", 0.0)),
            spatial_ref=str(normalized.get("spatial_ref", "EPSG:4326")),
            coverage=dict(normalized.get("coverage", {})),
            qc_flags=list(normalized.get("qc_flags", [])),
            analysis_time=str(normalized.get("analysis_time", "")),
            lead_time=int(normalized.get("lead_time", 0) or 0),
            domain_id=str(normalized.get("domain_id", "domain.legacy")),
            grid_spec=dict(normalized.get("grid_spec", {})),
            status=str(normalized.get("status", "ok")),
            fingerprint=str(normalized.get("fingerprint", "")),
            data_snapshot_id=str(normalized.get("data_snapshot_id", "")),
            official_candidate=bool(normalized.get("official_candidate", False)),
            experimental_note=str(normalized.get("experimental_note", "")),
        )


class OpenMeteoGuidanceProvider(BaseProvider):
    source_id = "openmeteo_guidance"
    source_family = "guidance"
    source_tier = "experimental"
    provider_version = "openmeteo.v1"

    def fetch(self, *, city: str, now: dt.datetime | None = None) -> Dict[str, Any]:
        now = now or dt.datetime.now()
        geo = geocode_city(city)
        lat = float(geo["latitude"])
        lon = float(geo["longitude"])
        timezone = geo.get("timezone") or "Asia/Shanghai"
        model_rows: list[dict] = []
        model_used: list[str] = []
        for model in MODELS:
            try:
                js = fetch_model_hourly(lat, lon, model=model, timezone=timezone)
                h = js["hourly"]
                units = js.get("hourly_units", {}) if isinstance(js.get("hourly_units"), dict) else {}
                idx = nearest_hour_index(h["time"], now)
                s850 = _wind_to_ms(_safe(h.get("wind_speed_850hPa", [0])[idx]), units.get("wind_speed_850hPa"))
                d850 = _safe(h.get("wind_direction_850hPa", [0])[idx])
                s500 = _wind_to_ms(_safe(h.get("wind_speed_500hPa", [0])[idx]), units.get("wind_speed_500hPa"))
                d500 = _safe(h.get("wind_direction_500hPa", [0])[idx])
                shear = shear_from_winds(s850, d850, s500, d500)
                model_rows.append(
                    {
                        "model": model,
                        "cape": _safe(h.get("cape", [0])[idx]),
                        "cin": abs(_safe(h.get("convective_inhibition", [0])[idx])),
                        "lifted_index": _safe(h.get("lifted_index", [0])[idx]),
                        "humidity": _safe(h.get("relative_humidity_2m", [0])[idx]) / 100.0,
                        "gust": _wind_to_ms(_safe(h.get("wind_gusts_10m", [0])[idx]), units.get("wind_gusts_10m")),
                        "wind10": _wind_to_ms(_safe(h.get("wind_speed_10m", [0])[idx]), units.get("wind_speed_10m")),
                        "omega700": _safe(h.get("vertical_velocity_700hPa", [0])[idx]),
                        "t850": _safe(h.get("temperature_850hPa", [0])[idx]),
                        "t500": _safe(h.get("temperature_500hPa", [0])[idx]),
                        "wbz_km": _safe(h.get("freezing_level_height", [0])[idx]) / 1000.0,
                        "precip_prob": _safe(h.get("precipitation_probability", [0])[idx]),
                        "precip": _safe(h.get("precipitation", [0])[idx]),
                        "showers": _safe(h.get("showers", [0])[idx]),
                        "shear": shear,
                    }
                )
                model_used.append(model)
            except Exception:
                continue
        if not model_rows:
            raise RuntimeError("No model data available from Open-Meteo")
        return {"geo": geo, "now": now, "timezone": timezone, "model_rows": model_rows, "model_used": model_used}

    def normalize(self, raw: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        now = raw["now"]
        rows = list(raw["model_rows"])
        model_used = list(raw["model_used"])
        geo = raw["geo"]

        def med(key: str) -> float:
            return float(median([r[key] for r in rows]))

        def spread(key: str) -> float:
            vals = [r[key] for r in rows]
            return float(max(vals) - min(vals)) if vals else 0.0

        cape = med("cape")
        cin = med("cin")
        omega = med("omega700")
        humidity = med("humidity")
        gust = med("gust")
        shear = med("shear")
        wbz_km = med("wbz_km")
        t850_500 = med("t850") - med("t500")
        low_level_convergence = clamp((max(0.0, -omega) / 2.0) * 0.6 + (1.0 - clamp(cin / 350.0, 0.0, 1.0)) * 0.4, 0.0, 1.0)
        dcape = clamp((1.0 - humidity) * 1200.0 + gust * 20.0, 0.0, 1800.0)
        probs = [
            hazard_probs_from_features(
                r["cape"],
                r["lifted_index"],
                r["humidity"],
                r["gust"],
                r["shear"],
                r["wbz_km"],
                r["precip_prob"],
            )
            for r in rows
        ]
        prob_guidance = {k: float(median([p[k] for p in probs])) for k in ("short_rain", "wind", "hail", "tornado")}
        model_count = len(model_used)
        cape_spread = spread("cape")
        shear_spread = spread("shear")
        wbz_spread = spread("wbz_km")
        model_spread_score = clamp((cape_spread / 2200.0) * 0.45 + (shear_spread / 22.0) * 0.4 + (wbz_spread / 2.0) * 0.15, 0.0, 1.0)
        coverage_ratio = clamp(model_count / max(len(MODELS), 1), 0.0, 1.0)
        return {
            "source_id": self.source_id,
            "source_family": self.source_family,
            "source_tier": self.source_tier,
            "provider_version": self.provider_version,
            "analysis_time": now.isoformat(),
            "issue_time": now.isoformat(),
            "valid_time": now.isoformat(),
            "ingest_time": dt.datetime.now().isoformat(),
            "lead_time": 0,
            "latency_sec": 0.0,
            "freshness_sec": 0.0,
            "spatial_ref": "EPSG:4326",
            "coverage": {"coverage_ratio": coverage_ratio, "models": model_used, "domain_type": "point_domain"},
            "grid_spec": asdict(GridSpec(grid_id=f"city:{geo.get('name', 'city')}.analysis")),
            "data": {
                "city": geo.get("name"),
                "lat": float(geo["latitude"]),
                "lon": float(geo["longitude"]),
                "timezone": raw["timezone"],
                "vertical_velocity": omega,
                "low_level_convergence": low_level_convergence,
                "cape": cape,
                "dcape": dcape,
                "shear_0_6km": shear,
                "t850_500": t850_500,
                "wbz_km": wbz_km,
                "humidity_low": humidity,
                "storm_motion_ms": med("wind10"),
                "prob_guidance": prob_guidance,
                "model_spread_score": model_spread_score,
                "model_coverage_score": coverage_ratio,
                "cape_spread": cape_spread,
                "shear_spread": shear_spread,
                "wbz_spread": wbz_spread,
                "models": model_used,
            },
            "qc_flags": [],
            "domain_id": f"city:{str(geo.get('name') or 'city').lower().replace(' ', '_')}",
            "official_candidate": False,
            "experimental_note": "Open-Meteo guidance is experimental and cannot be treated as official national production input.",
        }


class LocalRadarProvider(BaseProvider):
    source_id = "local_radar_grid"
    source_family = "radar"
    source_tier = "experimental"
    provider_version = "local_grid.v1"

    def fetch(self, *, lat: float, lon: float, path: str, max_distance_km: float) -> Dict[str, Any]:
        return fetch_local_radar_grid(lat, lon, LocalGridConfig(path=path, max_distance_km=max_distance_km))

    def normalize(self, raw: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        now = kwargs.get("now") or dt.datetime.now()
        frame_time = raw.get("frame_time")
        freshness_sec = max(0.0, (now.timestamp() - float(frame_time)) if frame_time else 999999.0)
        return {
            "source_id": self.source_id,
            "source_family": self.source_family,
            "source_tier": self.source_tier,
            "provider_version": self.provider_version,
            "analysis_time": now.isoformat(),
            "issue_time": now.isoformat(),
            "valid_time": dt.datetime.fromtimestamp(float(frame_time)).isoformat() if frame_time else now.isoformat(),
            "ingest_time": dt.datetime.now().isoformat(),
            "lead_time": 0,
            "latency_sec": 0.0,
            "freshness_sec": freshness_sec,
            "spatial_ref": "EPSG:4326",
            "coverage": {"coverage_ratio": 0.7 if raw.get("status") == "ok" else 0.0, "provider": "local_grid"},
            "grid_spec": asdict(GridSpec(grid_id="radar.local_grid")),
            "data": dict(raw),
            "qc_flags": [] if raw.get("status") == "ok" else ["fetch_failed"],
            "domain_id": kwargs.get("domain_id", "domain.legacy"),
            "official_candidate": True,
            "experimental_note": "Local radar grid is wired as official_candidate until formal upstream governance is connected.",
        }


class NowCoastRadarProvider(BaseProvider):
    source_id = "nowcoast_radar_grid"
    source_family = "radar"
    source_tier = "experimental"
    provider_version = "nowcoast.v1"

    def fetch(self, *, lat: float, lon: float) -> Dict[str, Any]:
        return fetch_nowcoast_radar_grid(lat, lon)

    def normalize(self, raw: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        now = kwargs.get("now") or dt.datetime.now()
        frame_time = raw.get("frame_time")
        freshness_sec = max(0.0, (now.timestamp() - float(frame_time)) if frame_time else 999999.0)
        return {
            "source_id": self.source_id,
            "source_family": self.source_family,
            "source_tier": self.source_tier,
            "provider_version": self.provider_version,
            "analysis_time": now.isoformat(),
            "issue_time": now.isoformat(),
            "valid_time": dt.datetime.fromtimestamp(float(frame_time)).isoformat() if frame_time else now.isoformat(),
            "ingest_time": dt.datetime.now().isoformat(),
            "lead_time": 0,
            "latency_sec": 0.0,
            "freshness_sec": freshness_sec,
            "spatial_ref": "EPSG:4326",
            "coverage": {"coverage_ratio": 0.65 if raw.get("status") == "ok" else 0.0, "provider": "nowcoast"},
            "grid_spec": asdict(GridSpec(grid_id="radar.nowcoast")),
            "data": dict(raw),
            "qc_flags": [] if raw.get("status") == "ok" else ["fetch_failed"],
            "domain_id": kwargs.get("domain_id", "domain.legacy"),
            "official_candidate": True,
            "experimental_note": "NowCoast radar is an experimental external feed here, not registered as official production truth.",
        }


class RainViewerRadarProvider(BaseProvider):
    source_id = "rainviewer_proxy"
    source_family = "radar"
    source_tier = "proxy"
    provider_version = "rainviewer.v1"

    def fetch(self, *, lat: float, lon: float) -> Dict[str, Any]:
        return fetch_radar_dbz(lat, lon)

    def normalize(self, raw: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        now = kwargs.get("now") or dt.datetime.now()
        return {
            "source_id": self.source_id,
            "source_family": self.source_family,
            "source_tier": self.source_tier,
            "provider_version": self.provider_version,
            "analysis_time": now.isoformat(),
            "issue_time": now.isoformat(),
            "valid_time": now.isoformat(),
            "ingest_time": dt.datetime.now().isoformat(),
            "lead_time": 0,
            "latency_sec": 0.0,
            "freshness_sec": 3600.0,
            "spatial_ref": "EPSG:4326",
            "coverage": {"coverage_ratio": 0.4, "provider": "rainviewer"},
            "grid_spec": asdict(GridSpec(grid_id="radar.rainviewer")),
            "data": dict(raw),
            "qc_flags": ["proxy_source"],
            "domain_id": kwargs.get("domain_id", "domain.legacy"),
            "official_candidate": False,
            "experimental_note": "RainViewer is proxy-only and must be excluded from formal headline evaluation and release gating.",
        }


class ProxyObjectiveGuidanceProvider(BaseProvider):
    source_id = "proxy_objective_guidance"
    source_family = "objective_guidance"
    source_tier = "proxy"
    provider_version = "proxy_guidance.v1"

    def fetch(self, *, guidance: Dict[str, float], now: dt.datetime | None = None) -> Dict[str, Any]:
        return {"guidance": dict(guidance), "now": now or dt.datetime.now()}

    def normalize(self, raw: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        now = raw["now"]
        return {
            "source_id": self.source_id,
            "source_family": self.source_family,
            "source_tier": self.source_tier,
            "provider_version": self.provider_version,
            "analysis_time": now.isoformat(),
            "issue_time": now.isoformat(),
            "valid_time": now.isoformat(),
            "ingest_time": dt.datetime.now().isoformat(),
            "lead_time": 0,
            "latency_sec": 0.0,
            "freshness_sec": 0.0,
            "spatial_ref": "EPSG:4326",
            "coverage": {"coverage_ratio": 1.0},
            "grid_spec": asdict(GridSpec(grid_id="guidance.proxy")),
            "data": {"prob_guidance": dict(raw.get("guidance", {}))},
            "qc_flags": ["proxy_source"],
            "domain_id": kwargs.get("domain_id", "domain.legacy"),
            "official_candidate": False,
            "experimental_note": "Feature-engineered objective guidance remains proxy-only until formal operational products are connected.",
        }


class ProviderRegistry:
    def __init__(self) -> None:
        self.providers: Dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        self.providers[provider.source_id] = provider

    def list_definitions(self) -> List[Dict[str, Any]]:
        return [provider.health_status() for provider in self.providers.values()]

    def provider(self, source_id: str) -> BaseProvider:
        return self.providers[source_id]


def default_provider_registry() -> ProviderRegistry:
    reg = ProviderRegistry()
    reg.register(OpenMeteoGuidanceProvider())
    reg.register(LocalRadarProvider())
    reg.register(NowCoastRadarProvider())
    reg.register(RainViewerRadarProvider())
    reg.register(ProxyObjectiveGuidanceProvider())
    return reg
