from __future__ import annotations

import json
from math import cos, radians
from typing import Any, Dict

import requests

NOWCOAST_MAPSERVER = "https://nowcoast.noaa.gov/arcgis/rest/services/radar_meteo_imagery_nexrad_time/MapServer"


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _extract_dbz(res_json: dict) -> float | None:
    results = res_json.get("results")
    if not isinstance(results, list) or not results:
        return None

    for r in results:
        attrs = r.get("attributes") or {}
        # ArcGIS identify value keys vary by layer/version.
        for key in ("Pixel Value", "pixel value", "value", "Value", "GRAY_INDEX"):
            v = _safe_float(attrs.get(key))
            if v is not None:
                if v > 120:  # often encoded by x10
                    v = v / 10.0
                return max(0.0, min(80.0, float(v)))

        # sometimes value may be string in top-level
        v = _safe_float(r.get("value"))
        if v is not None:
            if v > 120:
                v = v / 10.0
            return max(0.0, min(80.0, float(v)))
    return None


def fetch_nowcoast_radar(lat: float, lon: float, layer_id: int = 3, timeout: int = 20) -> Dict[str, Any]:
    """
    Query NOAA nowCOAST radar MapServer identify endpoint.
    Returns dbz when coverage exists; otherwise marks no_coverage.
    """
    geom = {"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}
    params = {
        "f": "json",
        "geometry": json.dumps(geom),
        "geometryType": "esriGeometryPoint",
        "sr": 4326,
        "layers": f"all:{int(layer_id)}",
        "tolerance": 2,
        "mapExtent": f"{lon - 1.0},{lat - 1.0},{lon + 1.0},{lat + 1.0}",
        "imageDisplay": "800,600,96",
        "returnGeometry": "false",
    }
    url = f"{NOWCOAST_MAPSERVER}/identify"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()

    dbz = _extract_dbz(js)
    if dbz is None:
        return {
            "dbz": 0.0,
            "bow_echo": False,
            "frame_time": None,
            "source": "noaa_nowcoast_nexrad",
            "status": "no_coverage_or_no_value",
            "layer_id": int(layer_id),
        }

    return {
        "dbz": float(dbz),
        "bow_echo": False,
        "frame_time": None,
        "source": "noaa_nowcoast_nexrad",
        "status": "ok",
        "layer_id": int(layer_id),
        "proxy_source": False,
        "feature_kind": "grid",
    }


def _offset_latlon(lat: float, lon: float, offset_km: float, bearing_deg: float) -> tuple[float, float]:
    # simple local approximation for small offsets
    if bearing_deg == 0:
        return lat + offset_km / 111.0, lon
    if bearing_deg == 180:
        return lat - offset_km / 111.0, lon
    lon_step = offset_km / max(111.0 * cos(radians(lat)), 1e-6)
    if bearing_deg == 90:
        return lat, lon + lon_step
    if bearing_deg == 270:
        return lat, lon - lon_step
    return lat, lon


def fetch_nowcoast_radar_grid(lat: float, lon: float, layer_id: int = 3, timeout: int = 20) -> Dict[str, Any]:
    """Sample nowCOAST at center + 4-neighborhood and summarize nearby reflectivity."""
    points = [(lat, lon)]
    for b in (0, 90, 180, 270):
        points.append(_offset_latlon(lat, lon, offset_km=12.0, bearing_deg=b))

    vals = []
    for p_lat, p_lon in points:
        out = fetch_nowcoast_radar(p_lat, p_lon, layer_id=layer_id, timeout=timeout)
        if out.get("status") == "ok":
            vals.append(float(out.get("dbz", 0.0)))

    if not vals:
        return {
            "dbz": 0.0,
            "dbz_max_nearby": 0.0,
            "dbz_mean_nearby": 0.0,
            "bow_echo": False,
            "frame_time": None,
            "source": "noaa_nowcoast_nexrad",
            "status": "no_coverage_or_no_value",
            "layer_id": int(layer_id),
            "proxy_source": False,
            "feature_kind": "grid",
            "sample_points": len(points),
        }

    dbz_center = vals[0]
    dbz_max = max(vals)
    dbz_mean = sum(vals) / max(len(vals), 1)
    bow_echo = bool(dbz_max >= 45.0 and dbz_mean >= 36.0)
    return {
        "dbz": round(float(dbz_center), 2),
        "dbz_max_nearby": round(float(dbz_max), 2),
        "dbz_mean_nearby": round(float(dbz_mean), 2),
        "bow_echo": bow_echo,
        "frame_time": None,
        "source": "noaa_nowcoast_nexrad",
        "status": "ok",
        "layer_id": int(layer_id),
        "proxy_source": False,
        "feature_kind": "grid",
        "sample_points": len(vals),
    }
