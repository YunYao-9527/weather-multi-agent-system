from __future__ import annotations

import csv
import io
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List

import requests

INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
ACCESS_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv"

# WMO present weather codes commonly linked to hail
HAIL_WEATHER_CODES = {27, 87, 88, 89, 90, 93, 94, 96, 99}
_METAR_GUST_RE = re.compile(r"\b\d{3}(?:\d{2,3})G(\d{2,3})(MPS|KT)\b")


def _download_text(url: str, path: Path, timeout: int = 60) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    path.write_text(r.text, encoding="utf-8")
    return r.text


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_stations(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    top_k: int = 8,
    radius_km: float = 300.0,
    cache_dir: str | Path = "data/cache/noaa_isd",
) -> List[dict]:
    cache = Path(cache_dir) / "isd-history.csv"
    text = _download_text(INVENTORY_URL, cache)
    reader = csv.DictReader(io.StringIO(text))

    cand = []
    for row in reader:
        try:
            s_lat = float(row["LAT"])
            s_lon = float(row["LON"])
        except Exception:
            continue
        end_str = (row.get("END") or "").strip()
        if end_str and end_str < start_date.strftime("%Y%m%d"):
            continue
        dist = _haversine_km(lat, lon, s_lat, s_lon)
        if dist <= radius_km:
            cand.append(
                {
                    "station_id": f"{row.get('USAF','').strip()}{row.get('WBAN','').strip()}",
                    "name": row.get("STATION NAME", ""),
                    "lat": s_lat,
                    "lon": s_lon,
                    "distance_km": round(dist, 2),
                    "end": end_str,
                }
            )

    if not cand:
        # Fallback: return nearest stations globally if radius has no hits
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            try:
                s_lat = float(row["LAT"])
                s_lon = float(row["LON"])
            except Exception:
                continue
            dist = _haversine_km(lat, lon, s_lat, s_lon)
            cand.append(
                {
                    "station_id": f"{row.get('USAF','').strip()}{row.get('WBAN','').strip()}",
                    "name": row.get("STATION NAME", ""),
                    "lat": s_lat,
                    "lon": s_lon,
                    "distance_km": round(dist, 2),
                    "end": (row.get("END") or "").strip(),
                }
            )

    cand = [c for c in cand if c["station_id"] and len(c["station_id"]) == 11]
    cand.sort(key=lambda x: x["distance_km"])
    return cand[:top_k]


def _parse_wnd_speed_ms(wnd: str) -> float:
    # WND format: "dir,dir_qc,type,speed_tenth_mps,speed_qc"
    if not wnd:
        return 0.0
    parts = wnd.split(",")
    if len(parts) < 4:
        return 0.0
    try:
        sp = int(parts[3])
    except Exception:
        return 0.0
    if sp >= 9999:
        return 0.0
    return max(0.0, sp / 10.0)


def _parse_precip_rate_mm_h(row: dict) -> float:
    vals = []
    for key in ("AA1", "AA2", "AA3"):
        raw = row.get(key) or ""
        if not raw:
            continue
        parts = raw.split(",")
        if len(parts) < 2:
            continue
        try:
            period_h = int(parts[0])
            amt_tenth_mm = int(parts[1])
        except Exception:
            continue
        if period_h <= 0 or period_h >= 98:
            continue
        if amt_tenth_mm >= 9999:
            continue
        mm = amt_tenth_mm / 10.0
        vals.append(mm / period_h)
    return max(vals) if vals else 0.0


def _weather_codes(row: dict) -> List[int]:
    out: List[int] = []
    for key in ("MW1", "AW1", "AW2", "AY1", "AY2"):
        raw = row.get(key)
        if not raw:
            continue
        try:
            code = int(str(raw).split(",")[0])
            out.append(code)
        except Exception:
            continue
    return out


def _parse_gust_ms(row: dict) -> float:
    # Prefer explicit GUST column when available (tenths m/s).
    raw = row.get("GUST", "")
    if raw not in (None, ""):
        try:
            v = int(str(raw).split(",")[0])
            if 0 <= v < 9999:
                return v / 10.0
        except Exception:
            pass

    # Fallback: parse METAR in REM, e.g. "33010G18MPS" or "32015G25KT"
    rem = str(row.get("REM", "") or "")
    if rem:
        m = _METAR_GUST_RE.search(rem)
        if m:
            try:
                val = float(m.group(1))
                unit = m.group(2)
                if unit == "KT":
                    return val * 0.514444
                return val
            except Exception:
                return 0.0
    return 0.0


def fetch_station_hourly(
    station_id: str,
    start_date: date,
    end_date: date,
    cache_dir: str | Path = "data/cache/noaa_isd",
) -> List[dict]:
    years = range(start_date.year, end_date.year + 1)
    rows: List[dict] = []
    for y in years:
        fp = Path(cache_dir) / str(y) / f"{station_id}.csv"
        if not fp.exists():
            url = ACCESS_URL.format(year=y, station=station_id)
            try:
                txt = _download_text(url, fp, timeout=90)
            except Exception:
                continue
        else:
            txt = fp.read_text(encoding="utf-8", errors="ignore")

        reader = csv.DictReader(io.StringIO(txt))
        for r in reader:
            d = r.get("DATE", "")
            if not d:
                continue
            try:
                ts = datetime.fromisoformat(d)
            except Exception:
                continue
            if ts.date() < start_date or ts.date() > end_date:
                continue
            wind_ms = _parse_wnd_speed_ms(r.get("WND", ""))
            gust_ms = _parse_gust_ms(r)
            rain_rate = _parse_precip_rate_mm_h(r)
            codes = _weather_codes(r)
            hail = 1 if any(c in HAIL_WEATHER_CODES for c in codes) else 0
            rows.append(
                {
                    "timestamp_utc": ts,
                    "wind_ms": wind_ms,
                    "gust_ms": gust_ms,
                    "rain_mm_h": rain_rate,
                    "hail_code": hail,
                    "weather_codes": codes,
                }
            )
    return rows
