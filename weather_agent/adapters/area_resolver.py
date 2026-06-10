from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Tuple

import requests

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "weather-agent/0.2 (contact: local-demo)"


def _risk_radius_km(hazard_prob: Dict[str, float]) -> float:
    mx = max([float(v) for v in hazard_prob.values()] or [0.0])
    if mx >= 0.8:
        return 80.0
    if mx >= 0.65:
        return 60.0
    if mx >= 0.5:
        return 40.0
    return 25.0


def _point_offset_km(lat: float, lon: float, dist_km: float, bearing_deg: float) -> Tuple[float, float]:
    # Equirectangular small-distance approximation for demo purpose.
    rad = math.radians(bearing_deg)
    dlat = (dist_km / 111.0) * math.cos(rad)
    dlon = (dist_km / (111.0 * max(math.cos(math.radians(lat)), 1e-6))) * math.sin(rad)
    return lat + dlat, lon + dlon


def _sample_points(lat: float, lon: float, radius_km: float) -> List[Tuple[float, float]]:
    pts = [(lat, lon)]
    for b in (0, 45, 90, 135, 180, 225, 270, 315):
        pts.append(_point_offset_km(lat, lon, radius_km, b))
    for b in (0, 90, 180, 270):
        pts.append(_point_offset_km(lat, lon, radius_km * 0.5, b))
    return pts


def _pick_first(addr: dict, keys: List[str]) -> str:
    for k in keys:
        v = addr.get(k)
        if v:
            return str(v)
    return ""


def _reverse_admin(lat: float, lon: float) -> Dict[str, str]:
    r = requests.get(
        NOMINATIM_REVERSE_URL,
        params={
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "accept-language": "zh-CN,zh",
            "zoom": 10,
            "addressdetails": 1,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    r.raise_for_status()
    js = r.json()
    addr = js.get("address") or {}

    province = _pick_first(addr, ["state", "province", "region"])
    city = _pick_first(addr, ["city", "municipality", "county", "state_district"])
    district = _pick_first(addr, ["city_district", "district", "county", "town", "suburb", "borough"])

    if city and district and district not in city:
        label = f"{city}{district}"
    elif city:
        label = city
    elif province:
        label = province
    else:
        label = js.get("display_name", "unknown")

    return {
        "province": province,
        "city": city,
        "district": district,
        "label": label,
    }


def resolve_affected_area(lat: float, lon: float, hazard_prob: Dict[str, float], city_hint: str = "") -> Dict[str, object]:
    radius_km = _risk_radius_km(hazard_prob)
    points = _sample_points(lat, lon, radius_km)

    labels: List[str] = []
    details: List[Dict[str, object]] = []
    for p_lat, p_lon in points:
        try:
            adm = _reverse_admin(p_lat, p_lon)
            label = adm.get("label") or ""
            if label:
                labels.append(label)
            details.append(
                {
                    "lat": round(p_lat, 5),
                    "lon": round(p_lon, 5),
                    "label": label,
                    "city": adm.get("city", ""),
                    "district": adm.get("district", ""),
                }
            )
        except Exception:
            continue

    if labels:
        cnt = Counter(labels)
        top = [k for k, _ in cnt.most_common(3)]
        if len(top) == 1:
            area_text = f"{top[0]}及周边"
        else:
            area_text = "、".join(top) + "等区域"
        return {
            "status": "auto",
            "area_text": area_text,
            "radius_km": radius_km,
            "samples": len(details),
            "matched_labels": top,
            "sample_details": details[:8],
        }

    fallback = f"{city_hint or '目标区域'}周边{int(radius_km)}km"
    return {
        "status": "fallback",
        "area_text": fallback,
        "radius_km": radius_km,
        "samples": len(details),
        "matched_labels": [],
        "sample_details": details[:8],
    }
