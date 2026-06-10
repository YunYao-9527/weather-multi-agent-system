from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Dict

from zoneinfo import ZoneInfo

from weather_agent.adapters.hail_reports import load_hail_reports
from weather_agent.adapters.noaa_isd import fetch_station_hourly, nearest_stations
from weather_agent.adapters.open_meteo import geocode_city


QUALIFIED_LABEL_TIERS = ("gold", "silver")
ALL_LABEL_TIERS = ("gold", "silver", "proxy")


def _hour_key(dt: datetime) -> str:
    return dt.replace(tzinfo=None, minute=0, second=0, microsecond=0).isoformat()


def _iter_hour_keys(start_date: date, end_date: date, tz: ZoneInfo) -> list[str]:
    keys: list[str] = []
    cur = datetime.combine(start_date, datetime.min.time(), tzinfo=tz)
    stop = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=tz)
    while cur < stop:
        keys.append(_hour_key(cur))
        cur += timedelta(hours=1)
    return keys


def _artifact_path(city: str, start_date: date, end_date: date, root: str | Path = "runs/truth_labels") -> Path:
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    key = f"{city.lower()}_{start_date.isoformat()}_{end_date.isoformat()}".replace(" ", "_")
    return p / f"truth_{key}.json"


@dataclass
class TruthConfig:
    rain_threshold_mm_h: float = 20.0
    gust_threshold_ms: float = 17.2
    sustained_wind_threshold_ms: float = 13.9
    hail_report_dir: str = "data/hail_reports"
    radius_km: float = 300.0
    top_k_stations: int = 6
    gold_min_station_count: int = 2
    silver_min_station_count: int = 1
    min_coverage_for_headline: float = 0.6


def classify_label_tier(record: dict, cfg: TruthConfig | None = None) -> str:
    cfg = cfg or TruthConfig()
    station_count = int(record.get("station_count", 0) or 0)
    hail_report_flag = int(record.get("sources", {}).get("hail_report", 0) or 0)
    if station_count >= cfg.gold_min_station_count:
        return "gold"
    if station_count >= cfg.silver_min_station_count or hail_report_flag == 1:
        return "silver"
    return "proxy"


def _default_hourly_record() -> dict:
    return {
        "rain_mm_h_max": 0.0,
        "wind_ms_max": 0.0,
        "gust_ms_max": 0.0,
        "hail_station_code": 0,
        "station_ids": set(),
        "station_values": {},
    }


def build_truth_label_artifact(
    city: str,
    start_date: date,
    end_date: date,
    timezone: str | None = None,
    cfg: TruthConfig | None = None,
    force_rebuild: bool = False,
) -> dict:
    cfg = cfg or TruthConfig()
    geo = geocode_city(city)
    tz_name = timezone or (geo.get("timezone") or "Asia/Shanghai")
    tz = ZoneInfo(tz_name)
    lat = float(geo["latitude"])
    lon = float(geo["longitude"])

    out_path = _artifact_path(city, start_date, end_date)
    if out_path.exists() and not force_rebuild:
        cached = json.loads(out_path.read_text(encoding="utf-8"))
        cached["artifact_path"] = str(out_path)
        return cached

    stations = nearest_stations(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        top_k=cfg.top_k_stations,
        radius_km=cfg.radius_km,
    )
    station_catalog = {
        s["station_id"]: {
            "station_id": s["station_id"],
            "name": s.get("name", ""),
            "lat": float(s.get("lat", 0.0) or 0.0),
            "lon": float(s.get("lon", 0.0) or 0.0),
            "distance_km": float(s.get("distance_km", 0.0) or 0.0),
        }
        for s in stations
        if s.get("station_id")
    }

    hourly: Dict[str, dict] = {}
    used = []
    for st in stations:
        sid = st["station_id"]
        rows = fetch_station_hourly(sid, start_date, end_date)
        if not rows:
            continue
        used.append(sid)
        for r in rows:
            ts_utc: datetime = r["timestamp_utc"].replace(tzinfo=ZoneInfo("UTC"))
            ts_local = ts_utc.astimezone(tz)
            key = _hour_key(ts_local)

            rec = hourly.setdefault(key, _default_hourly_record())
            rain = float(r["rain_mm_h"])
            gust = float(r["wind_ms"])
            gust_obs = float(r.get("gust_ms", 0.0))
            hail_station_code = int(r["hail_code"])

            rec["rain_mm_h_max"] = max(float(rec["rain_mm_h_max"]), rain)
            rec["wind_ms_max"] = max(float(rec["wind_ms_max"]), gust)
            rec["gust_ms_max"] = max(float(rec["gust_ms_max"]), gust_obs)
            rec["hail_station_code"] = max(int(rec["hail_station_code"]), hail_station_code)
            rec["station_ids"].add(sid)

            sval = rec["station_values"].setdefault(
                sid,
                {"rain_mm_h_max": 0.0, "wind_ms_max": 0.0, "gust_ms_max": 0.0, "hail_station_code": 0},
            )
            sval["rain_mm_h_max"] = max(float(sval["rain_mm_h_max"]), rain)
            sval["wind_ms_max"] = max(float(sval["wind_ms_max"]), gust)
            sval["gust_ms_max"] = max(float(sval["gust_ms_max"]), gust_obs)
            sval["hail_station_code"] = max(int(sval["hail_station_code"]), hail_station_code)

    hail = load_hail_reports(
        city=city,
        start_date=start_date,
        end_date=end_date,
        timezone=tz_name,
        source_dir=cfg.hail_report_dir,
    )
    hail_by_hour: Dict[str, int] = hail.get("by_hour", {})
    for key, hail_flag in hail_by_hour.items():
        if not hail_flag:
            continue
        rec = hourly.setdefault(key, _default_hourly_record())
        rec["hail_station_code"] = max(int(rec["hail_station_code"]), 1)

    labels_by_hour: Dict[str, dict] = {}
    tier_counts = {k: 0 for k in ALL_LABEL_TIERS}
    for k, rec in hourly.items():
        rain = float(rec.get("rain_mm_h_max", 0.0))
        wind = float(rec.get("wind_ms_max", 0.0))
        gust = float(rec.get("gust_ms_max", 0.0))
        hail_station = int(rec.get("hail_station_code", 0))
        hail_file = int(hail_by_hour.get(k, 0))
        hail_flag = 1 if (hail_station or hail_file) else 0

        station_ids = sorted(list(rec.get("station_ids", set())))
        station_values = rec.get("station_values", {})
        station_event_flags = {}
        for sid in station_ids:
            val = station_values.get(sid) or {}
            sid_rain = float(val.get("rain_mm_h_max", 0.0))
            sid_wind = float(val.get("wind_ms_max", 0.0))
            sid_gust = float(val.get("gust_ms_max", 0.0))
            sid_hail = int(val.get("hail_station_code", 0))
            wind_flag = 1 if (sid_gust >= cfg.gust_threshold_ms or sid_wind >= cfg.sustained_wind_threshold_ms) else 0
            station_event_flags[sid] = {
                "short_rain": 1 if sid_rain >= cfg.rain_threshold_mm_h else 0,
                "wind": wind_flag,
                "hail": 1 if sid_hail else 0,
                "rain_mm_h_max": round(sid_rain, 3),
                "wind_ms_max": round(sid_wind, 3),
                "gust_ms_max": round(sid_gust, 3),
            }
        wind_flag_total = 1 if (gust >= cfg.gust_threshold_ms or wind >= cfg.sustained_wind_threshold_ms) else 0

        label = {
            "short_rain": 1 if rain >= cfg.rain_threshold_mm_h else 0,
            "wind": wind_flag_total,
            "hail": hail_flag,
            "tornado": 0,
            "rain_mm_h_max": round(rain, 3),
            "wind_ms_max": round(wind, 3),
            "gust_ms_max": round(gust, 3),
            "station_count": len(station_ids),
            "station_ids": station_ids,
            "station_event_flags": station_event_flags,
            "sources": {
                "station": 1 if station_ids else 0,
                "hail_report": 1 if hail_file else 0,
                "hail_station_code": 1 if hail_station else 0,
            },
        }
        label["label_tier"] = classify_label_tier(label, cfg)
        tier_counts[label["label_tier"]] = tier_counts.get(label["label_tier"], 0) + 1
        labels_by_hour[k] = label

    total_hours = len(_iter_hour_keys(start_date, end_date, tz))
    coverage_ratio = len(labels_by_hour) / max(total_hours, 1)
    tier_ratio = {k: round(v / max(len(labels_by_hour), 1), 4) for k, v in tier_counts.items()}

    dataset = {
        "meta": {
            "city": city,
            "city_resolved": geo.get("name", city),
            "lat": lat,
            "lon": lon,
            "timezone": tz_name,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "thresholds": {
                "rain_threshold_mm_h": cfg.rain_threshold_mm_h,
                "gust_threshold_ms": cfg.gust_threshold_ms,
                "sustained_wind_threshold_ms": cfg.sustained_wind_threshold_ms,
            },
            "station_info": {
                "candidate_count": len(stations),
                "used_count": len(used),
                "used_station_ids": used,
                "station_catalog": station_catalog,
            },
            "hail_info": {
                "files": hail.get("files", []),
                "records": int(hail.get("records", 0)),
                "source_dir": cfg.hail_report_dir,
            },
            "label_tiering": {
                "qualified_tiers": list(QUALIFIED_LABEL_TIERS),
                "all_tiers": list(ALL_LABEL_TIERS),
                "tier_counts": tier_counts,
                "tier_ratio": tier_ratio,
            },
            "label_hours": len(labels_by_hour),
            "label_coverage_ratio": round(coverage_ratio, 4),
            "headline_ready": bool(coverage_ratio >= float(cfg.min_coverage_for_headline)),
            "headline_min_coverage_required": float(cfg.min_coverage_for_headline),
            "generated_at": datetime.now().isoformat(),
            "fallback_note": "proxy labels are auxiliary only and cannot be used for headline evaluation/evolution",
        },
        "labels_by_hour": labels_by_hour,
    }

    payload = json.dumps(dataset, ensure_ascii=False, sort_keys=True)
    dataset["meta"]["sha256"] = sha256(payload.encode("utf-8")).hexdigest()
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset["artifact_path"] = str(out_path)
    return dataset
