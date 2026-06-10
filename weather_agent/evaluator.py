from __future__ import annotations

import datetime as dt
import json
import random
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

from weather_agent.adapters.objective_guidance import hazard_probs_from_features
from weather_agent.adapters.open_meteo import fetch_archive_hourly, geocode_city
from weather_agent.adapters.utils import clamp, shear_from_winds
from weather_agent.models import EvalManifest, HAZARDS, Observation
from weather_agent.orchestrator import ForecastOrchestrator, OrchestratorConfig
from weather_agent.policy_engine import PolicyManager
from weather_agent.registry import _code_hash
from weather_agent.spatial_eval import (
    AdminResolverCache,
    aggregate_admin_metrics,
    aggregate_grid_metrics,
)
from weather_agent.truth_labels import QUALIFIED_LABEL_TIERS, TruthConfig, build_truth_label_artifact


def _hash_payload(payload: Dict[str, Any]) -> str:
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _safe(v: object, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _hourly_value(hourly: dict, key: str, idx: int) -> float:
    values = hourly.get(key)
    if not isinstance(values, list) or idx >= len(values):
        return 0.0
    return _safe(values[idx])


def _wind_to_ms(v: float, unit: str | None) -> float:
    u = (unit or "").strip().lower()
    if u in {"km/h", "kmh", "kph"}:
        return float(v) / 3.6
    if u in {"mph"}:
        return float(v) * 0.44704
    return float(v)


def _normalize_row_units(row: dict, units: dict | None) -> dict:
    units = units if isinstance(units, dict) else {}
    out = dict(row)
    for key in ("wind_gusts_10m", "wind_speed_10m", "wind_speed_850hPa", "wind_speed_500hPa"):
        out[key] = _wind_to_ms(float(out.get(key, 0.0)), units.get(key))
    return out


def _truth_labels(precip: float, gust: float) -> Dict[str, int]:
    return {
        "short_rain": 1 if precip >= 20.0 else 0,
        "wind": 1 if gust >= 17.2 else 0,
    }


def _proxy_truth_all(row: dict) -> Dict[str, int]:
    rain_wind = _truth_labels(row.get("precipitation", 0.0), row.get("wind_gusts_10m", 0.0))
    wbz_km = float(row.get("freezing_level_height", 0.0)) / 1000.0
    hail = 1 if (float(row.get("cape", 0.0)) >= 1200.0 and wbz_km <= 4.2 and float(row.get("wind_gusts_10m", 0.0)) >= 14.0) else 0
    tornado = 1 if (
        float(row.get("wind_gusts_10m", 0.0)) >= 20.0
        and float(row.get("wind_speed_500hPa", 0.0)) >= 18.0
        and float(row.get("cape", 0.0)) >= 800.0
    ) else 0
    return {
        "short_rain": rain_wind["short_rain"],
        "wind": rain_wind["wind"],
        "hail": hail,
        "tornado": tornado,
    }


def _classification_metrics(pred_probs: Sequence[float], y_true: Sequence[int], threshold: float = 0.5) -> Dict[str, float]:
    if not pred_probs:
        return {
            "sample_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "brier": 0.0,
            "positives": 0,
        }

    tp = fp = fn = 0
    for p, y in zip(pred_probs, y_true):
        pred = 1 if p >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    brier = mean([(float(p) - int(y)) ** 2 for p, y in zip(pred_probs, y_true)])
    positives = int(sum(int(y) for y in y_true))
    sample_count = len(pred_probs)
    prevalence = positives / max(sample_count, 1)
    return {
        "sample_count": len(pred_probs),
        "positives": positives,
        "event_prevalence": round(prevalence, 4),
        "has_positive": bool(positives > 0),
        "no_positive_warning": "test split has zero positive labels; event metrics are not statistically informative" if positives == 0 else "",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "brier": round(brier, 4),
    }


def _roc_auc(pred_probs: Sequence[float], y_true: Sequence[int]) -> float:
    pairs = sorted([(float(p), int(y)) for p, y in zip(pred_probs, y_true)], key=lambda x: x[0], reverse=True)
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.0
    tp = fp = 0
    prev_p = None
    area = 0.0
    last_tpr = last_fpr = 0.0
    for p, y in pairs:
        if prev_p is not None and p != prev_p:
            tpr = tp / pos
            fpr = fp / neg
            area += (fpr - last_fpr) * (tpr + last_tpr) / 2.0
            last_tpr, last_fpr = tpr, fpr
        tp += int(y == 1)
        fp += int(y == 0)
        prev_p = p
    tpr = tp / pos
    fpr = fp / neg
    area += (fpr - last_fpr) * (tpr + last_tpr) / 2.0
    return round(area, 4)


def _pr_auprc(pred_probs: Sequence[float], y_true: Sequence[int]) -> float:
    pairs = sorted([(float(p), int(y)) for p, y in zip(pred_probs, y_true)], key=lambda x: x[0], reverse=True)
    pos = sum(y for _, y in pairs)
    if pos == 0:
        return 0.0
    tp = fp = 0
    prev_recall = 0.0
    area = 0.0
    for _, y in pairs:
        tp += int(y == 1)
        fp += int(y == 0)
        recall = tp / pos if pos else 0.0
        precision = tp / max(tp + fp, 1)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return round(area, 4)


def _reliability_curve(pred_probs: Sequence[float], y_true: Sequence[int], bins: int = 10) -> dict:
    if not pred_probs:
        return {"ece": 0.0, "curve": []}
    cnt = [0] * bins
    pos = [0] * bins
    conf = [0.0] * bins
    n = len(pred_probs)
    ece = 0.0
    curve = []
    for p, y in zip(pred_probs, y_true):
        idx = min(bins - 1, int(float(p) * bins))
        cnt[idx] += 1
        pos[idx] += int(y)
        conf[idx] += float(p)
    for i in range(bins):
        if cnt[i] == 0:
            continue
        mean_conf = conf[i] / cnt[i]
        event_rate = pos[i] / cnt[i]
        w = cnt[i] / n
        ece += w * abs(mean_conf - event_rate)
        curve.append({"bin": i, "count": cnt[i], "mean_confidence": round(mean_conf, 4), "event_rate": round(event_rate, 4)})
    return {"ece": round(ece, 4), "curve": curve}


def _event_metrics(pred_probs: Sequence[float], y_true: Sequence[int], threshold: float = 0.5) -> Dict[str, float]:
    if not pred_probs:
        return {
            "pod": 0.0,
            "far": 0.0,
            "csi": 0.0,
            "spatial_hit_rate": 0.0,
            "coverage_bias": 0.0,
        }

    tp = fp = fn = 0
    for p, y in zip(pred_probs, y_true):
        pred = 1 if p >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1

    pod = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (tp + fp) if (tp + fp) else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    coverage_bias = (tp + fp) / (tp + fn) if (tp + fn) else 0.0
    positives = int(sum(int(y) for y in y_true))
    return {
        "pod": round(pod, 4),
        "far": round(far, 4),
        "csi": round(csi, 4),
        "spatial_hit_rate": round(pod, 4),
        "coverage_bias": round(coverage_bias, 4),
        "event_metrics_applicable": bool(positives > 0),
    }


def _level_from_prob(prob: float) -> int:
    if prob >= 0.75:
        return 3
    if prob >= 0.6:
        return 2
    if prob >= 0.45:
        return 1
    return 0


def _stability_metrics(pred_probs: Sequence[float]) -> Dict[str, float]:
    n = len(pred_probs)
    if n <= 1:
        return {"level_jitter_rate": 0.0, "upgrade_downgrade_consistency": 1.0}

    levels = [_level_from_prob(float(p)) for p in pred_probs]
    jitter = sum(1 for i in range(1, n) if levels[i] != levels[i - 1]) / (n - 1)

    inconsistent = 0
    checks = 0
    for i in range(1, n):
        dp = float(pred_probs[i]) - float(pred_probs[i - 1])
        dl = levels[i] - levels[i - 1]
        if abs(dp) < 0.05:
            continue
        checks += 1
        if (dp > 0 and dl < 0) or (dp < 0 and dl > 0):
            inconsistent += 1

    consistency = 1.0 - (inconsistent / checks) if checks else 1.0
    return {
        "level_jitter_rate": round(jitter, 4),
        "upgrade_downgrade_consistency": round(consistency, 4),
    }


def _lead_time_minutes(pred_probs: Sequence[float], y_true: Sequence[int], timestamps: Sequence[dt.datetime], threshold: float = 0.5) -> Dict[str, float]:
    if not pred_probs or not timestamps:
        return {"avg_lead_time_min": 0.0, "lead_time_hit_events": 0, "total_events": 0}

    onsets = [i for i in range(len(y_true)) if y_true[i] == 1 and (i == 0 or y_true[i - 1] == 0)]
    if not onsets:
        return {"avg_lead_time_min": 0.0, "lead_time_hit_events": 0, "total_events": 0}

    leads: list[float] = []
    for onset in onsets:
        start = max(0, onset - 6)
        hit_idx = None
        for j in range(start, onset + 1):
            if float(pred_probs[j]) >= threshold:
                hit_idx = j
                break
        if hit_idx is None:
            continue
        delta = timestamps[onset] - timestamps[hit_idx]
        leads.append(max(0.0, delta.total_seconds() / 60.0))

    return {
        "avg_lead_time_min": round(mean(leads), 2) if leads else 0.0,
        "lead_time_hit_events": len(leads),
        "total_events": len(onsets),
    }


def _duration_metrics(pred_probs: Sequence[float], y_true: Sequence[int], threshold: float = 0.5) -> Dict[str, float]:
    def _segments(bits: Sequence[int]) -> list[tuple[int, int]]:
        out = []
        start = None
        for i, bit in enumerate(bits):
            if bit and start is None:
                start = i
            elif not bit and start is not None:
                out.append((start, i - 1))
                start = None
        if start is not None:
            out.append((start, len(bits) - 1))
        return out

    pred_bits = [1 if float(p) >= threshold else 0 for p in pred_probs]
    true_bits = [int(y) for y in y_true]
    pred_segs = _segments(pred_bits)
    true_segs = _segments(true_bits)
    if not true_segs:
        return {"duration_error": 0.0, "clear_delay": 0.0, "event_hit": 0, "false_alarm": len(pred_segs), "miss": 0}
    hits = false_alarm = miss = 0
    duration_errors = []
    clear_delays = []
    for ts in true_segs:
        overlaps = [ps for ps in pred_segs if not (ps[1] < ts[0] or ps[0] > ts[1])]
        if not overlaps:
            miss += 1
            continue
        hits += 1
        best = overlaps[0]
        duration_errors.append(abs((best[1] - best[0]) - (ts[1] - ts[0])))
        clear_delays.append(max(0, best[1] - ts[1]))
    for ps in pred_segs:
        if not any(not (ps[1] < ts[0] or ps[0] > ts[1]) for ts in true_segs):
            false_alarm += 1
    return {
        "duration_error": round(mean(duration_errors), 4) if duration_errors else 0.0,
        "clear_delay": round(mean(clear_delays), 4) if clear_delays else 0.0,
        "event_hit": hits,
        "false_alarm": false_alarm,
        "miss": miss,
    }


def _bootstrap_ci(
    pred_probs: Sequence[float],
    y_true: Sequence[int],
    timestamps: Sequence[dt.datetime],
    threshold: float = 0.5,
    rounds: int = 200,
) -> Dict[str, list[float]]:
    n = len(pred_probs)
    if n < 8:
        return {}

    rng = random.Random(20260401)
    f1_list: list[float] = []
    brier_list: list[float] = []
    csi_list: list[float] = []
    lead_list: list[float] = []

    groups: dict[str, list[int]] = {}
    for i, ts in enumerate(timestamps):
        groups.setdefault(ts.date().isoformat(), []).append(i)
    group_keys = sorted(groups)
    for _ in range(rounds):
        sampled_keys = [group_keys[rng.randrange(len(group_keys))] for _ in range(len(group_keys))]
        idx = [i for key in sampled_keys for i in groups[key]]
        p = [float(pred_probs[i]) for i in idx]
        y = [int(y_true[i]) for i in idx]
        t = [timestamps[i] for i in idx]
        cls = _classification_metrics(p, y, threshold)
        evt = _event_metrics(p, y, threshold)
        lead = _lead_time_minutes(p, y, t, threshold)
        f1_list.append(float(cls["f1"]))
        brier_list.append(float(cls["brier"]))
        csi_list.append(float(evt["csi"]))
        lead_list.append(float(lead["avg_lead_time_min"]))

    def _q(vals: Sequence[float], q: float) -> float:
        arr = sorted(vals)
        pos = int((len(arr) - 1) * q)
        return float(arr[pos])

    return {
        "f1": [round(_q(f1_list, 0.025), 4), round(_q(f1_list, 0.975), 4)],
        "brier": [round(_q(brier_list, 0.025), 4), round(_q(brier_list, 0.975), 4)],
        "csi": [round(_q(csi_list, 0.025), 4), round(_q(csi_list, 0.975), 4)],
        "avg_lead_time_min": [round(_q(lead_list, 0.025), 2), round(_q(lead_list, 0.975), 2)],
    }


def _metric_bundle(
    pred_probs: Sequence[float],
    y_true: Sequence[int],
    timestamps: Sequence[dt.datetime],
    threshold: float = 0.5,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out.update(_classification_metrics(pred_probs, y_true, threshold))
    out.update(_event_metrics(pred_probs, y_true, threshold))
    out.update(_lead_time_minutes(pred_probs, y_true, timestamps, threshold))
    out.update(_duration_metrics(pred_probs, y_true, threshold))
    out.update(_stability_metrics(pred_probs))
    out["roc_auc"] = _roc_auc(pred_probs, y_true)
    out["auprc"] = _pr_auprc(pred_probs, y_true)
    rel = _reliability_curve(pred_probs, y_true)
    out["reliability"] = rel["curve"]
    out["ece"] = rel["ece"]
    out["calibration_curve"] = rel["curve"]
    out["bootstrap_ci"] = _bootstrap_ci(pred_probs, y_true, timestamps, threshold)
    return out


def _quality_bucket(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _improvement_summary(base: Dict[str, float], enh: Dict[str, float]) -> Dict[str, float]:
    base_brier = float(base.get("brier", 0.0))
    enh_brier = float(enh.get("brier", 0.0))
    base_f1 = float(base.get("f1", 0.0))
    enh_f1 = float(enh.get("f1", 0.0))

    brier_reduction = base_brier - enh_brier
    brier_reduction_pct = (brier_reduction / base_brier) if base_brier > 1e-8 else 0.0
    f1_gain = enh_f1 - base_f1
    f1_gain_pct = (f1_gain / base_f1) if base_f1 > 1e-8 else 0.0
    return {
        "brier_reduction": round(brier_reduction, 4),
        "brier_reduction_pct": round(brier_reduction_pct, 4),
        "f1_gain": round(f1_gain, 4),
        "f1_gain_pct": round(f1_gain_pct, 4),
    }


def _hour_key(ts: dt.datetime) -> str:
    return ts.replace(minute=0, second=0, microsecond=0).isoformat()


def _build_obs_from_archive(city_name: str, timestamp: dt.datetime, row: dict) -> Observation:
    shear = shear_from_winds(row["wind_speed_850hPa"], row["wind_direction_850hPa"], row["wind_speed_500hPa"], row["wind_direction_500hPa"])
    humidity = row["relative_humidity_2m"] / 100.0
    cin = abs(row["convective_inhibition"])
    omega = row["vertical_velocity_700hPa"]

    low_level_convergence = clamp((max(0.0, -omega) / 2.0) * 0.6 + (1.0 - clamp(cin / 350.0, 0.0, 1.0)) * 0.4, 0.0, 1.0)
    dcape = clamp((1.0 - humidity) * 1200.0 + row["wind_gusts_10m"] * 20.0, 0.0, 1800.0)
    wbz_km = row["freezing_level_height"] / 1000.0

    precip_prob_pct = clamp((row["precipitation"] / 20.0) * 100.0, 0.0, 100.0)
    probs = hazard_probs_from_features(
        row["cape"],
        row["lifted_index"],
        humidity,
        row["wind_gusts_10m"],
        shear,
        wbz_km,
        precip_prob_pct,
    )

    return Observation(
        timestamp=timestamp,
        city=city_name,
        vertical_velocity=omega,
        low_level_convergence=low_level_convergence,
        cape=row["cape"],
        dcape=dcape,
        shear_0_6km=shear,
        t850_500=row["temperature_850hPa"] - row["temperature_500hPa"],
        wbz_km=wbz_km,
        humidity_low=humidity,
        radar_dbz_max=0.0,
        radar_bow_echo=False,
        storm_motion_ms=row["wind_speed_10m"],
        prob_guidance=probs,
        source_meta={
            "mode": "archive-backtest",
            "model_count": "1",
            "model_coverage_score": "0.33",
            "model_spread_score": "0.25",
            "radar_freshness_score": "0.50",
            "data_quality_score": "0.62",
        },
    )


def _build_time_split(times: Sequence[dt.datetime], train_ratio: float = 0.6, calibration_ratio: float = 0.2) -> dict:
    n = len(times)
    if n == 0:
        return {
            "counts": {"train": 0, "calibration": 0, "test": 0},
            "index_ranges": {"train": [0, -1], "calibration": [0, -1], "test": [0, -1]},
            "windows": {},
        }

    train_n = max(1, int(n * train_ratio))
    calib_n = max(1, int(n * calibration_ratio))
    if train_n + calib_n >= n:
        calib_n = max(1, n // 5)
        train_n = max(1, n - calib_n - 1)
    test_n = max(1, n - train_n - calib_n)

    train_idx = list(range(0, train_n))
    calib_idx = list(range(train_n, train_n + calib_n))
    test_idx = list(range(train_n + calib_n, train_n + calib_n + test_n))

    def _window(indexes: Sequence[int]) -> dict:
        if not indexes:
            return {"start": None, "end": None}
        return {
            "start": times[indexes[0]].isoformat(),
            "end": times[indexes[-1]].isoformat(),
        }

    return {
        "counts": {"train": len(train_idx), "calibration": len(calib_idx), "test": len(test_idx)},
        "index_ranges": {
            "train": [train_idx[0], train_idx[-1]] if train_idx else [0, -1],
            "calibration": [calib_idx[0], calib_idx[-1]] if calib_idx else [0, -1],
            "test": [test_idx[0], test_idx[-1]] if test_idx else [0, -1],
        },
        "windows": {
            "train": _window(train_idx),
            "calibration": _window(calib_idx),
            "test": _window(test_idx),
        },
        "indices": {"train": train_idx, "calibration": calib_idx, "test": test_idx},
    }


def _aggregate_hazard_metrics(metrics_by_hazard: Dict[str, Dict[str, object]]) -> dict:
    keys = ["precision", "recall", "f1", "brier", "pod", "far", "csi", "avg_lead_time_min", "level_jitter_rate", "upgrade_downgrade_consistency"]
    out: dict = {}
    for k in keys:
        vals = [float(metrics_by_hazard[h].get(k, 0.0)) for h in HAZARDS if int(metrics_by_hazard[h].get("sample_count", 0)) > 0]
        out[k] = round(mean(vals), 4) if vals else 0.0
    out["sample_count"] = int(sum(int(metrics_by_hazard[h].get("sample_count", 0)) for h in HAZARDS))
    return out


def _save_eval_reports(result: dict) -> dict:
    out_dir = Path("runs/evaluations")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    city = str(result.get("city", "city")).replace(" ", "_").lower()
    base = out_dir / f"eval_{city}_{ts}"

    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    html_path = base.with_suffix(".html")

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        f"# Evaluation Report - {result.get('city', '-')}",
        "",
        f"- Generated At: {dt.datetime.now().isoformat()}",
        f"- Period: {result.get('period', {}).get('start')} ~ {result.get('period', {}).get('end')}",
        f"- Samples(test): {result.get('split_manifest', {}).get('counts', {}).get('test', 0)}",
        f"- Truth policy: {result.get('truth_labels', {}).get('truth_policy')}",
        f"- Qualified coverage ratio: {result.get('truth_labels', {}).get('qualified_coverage_ratio')}",
        "",
        "## Improvement",
        json.dumps(result.get("improvements", {}), ensure_ascii=False, indent=2),
    ]
    md_payload = "\n".join(summary_lines)
    md_path.write_text(md_payload, encoding="utf-8")

    html_payload = """
<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\" /><title>Evaluation Report</title>
<style>body{font-family:Arial,sans-serif;background:#f6f8fa;margin:20px;}pre{background:#fff;border:1px solid #ddd;padding:12px;border-radius:8px;overflow:auto;}</style>
</head><body>
<h1>Evaluation Report</h1>
<pre>{payload}</pre>
</body></html>
""".replace("{payload}", json.dumps(result, ensure_ascii=False, indent=2))
    html_path.write_text(html_payload, encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
    }


def evaluate_recent(
    city: str = "Tianjin",
    days: int = 3,
    enhanced_weights: Dict[str, float] | None = None,
    enhanced_calibrators: Dict[str, Dict[str, object]] | None = None,
    force_rebuild_truth: bool = False,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    truth_policy: str = "prefer",  # prefer | require | off
    min_truth_coverage: float = 0.6,
    min_total_positive_labels: int = 1,
    headline_tiers: Sequence[str] = QUALIFIED_LABEL_TIERS,
) -> dict:
    if any(str(t).strip().lower() == "proxy" for t in headline_tiers):
        raise RuntimeError("proxy truth cannot be used in headline evaluation")
    geo = geocode_city(city)
    lat = float(geo["latitude"])
    lon = float(geo["longitude"])
    timezone = geo.get("timezone") or "Asia/Shanghai"

    if end_date is None:
        end_date = dt.date.today()
    if start_date is None:
        start_date = end_date - dt.timedelta(days=max(1, days))

    archive = fetch_archive_hourly(
        lat,
        lon,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        timezone=timezone,
    )

    hourly = archive["hourly"]
    hourly_units = archive.get("hourly_units", {}) if isinstance(archive.get("hourly_units"), dict) else {}
    times = [dt.datetime.fromisoformat(t) for t in hourly["time"]]
    split = _build_time_split(times)
    test_indices = split["indices"]["test"]

    fields = [
        "cape",
        "convective_inhibition",
        "lifted_index",
        "relative_humidity_2m",
        "wind_gusts_10m",
        "wind_speed_10m",
        "vertical_velocity_700hPa",
        "temperature_850hPa",
        "temperature_500hPa",
        "freezing_level_height",
        "precipitation",
        "showers",
        "wind_speed_850hPa",
        "wind_speed_500hPa",
        "wind_direction_850hPa",
        "wind_direction_500hPa",
    ]

    baseline = ForecastOrchestrator(OrchestratorConfig(region_name="replay"))
    enhanced = ForecastOrchestrator(
        OrchestratorConfig(
            region_name="replay",
            agent_weights=enhanced_weights,
            probability_calibrators=enhanced_calibrators,
        )
    )

    truth_art = {"meta": {}, "labels_by_hour": {}, "artifact_path": None}
    truth_map = {}
    truth_version = ""
    if truth_policy in {"prefer", "require"}:
        truth_art = build_truth_label_artifact(
            city=city,
            start_date=start_date,
            end_date=end_date,
            timezone=timezone,
            cfg=TruthConfig(min_coverage_for_headline=min_truth_coverage),
            force_rebuild=force_rebuild_truth,
        )
        truth_map = truth_art.get("labels_by_hour", {})
        truth_version = f"truth.{city.lower()}.{start_date.isoformat()}.{end_date.isoformat()}.{str(truth_art.get('meta', {}).get('sha256', 'na'))[:8]}"
    station_catalog = truth_art.get("meta", {}).get("station_info", {}).get("station_catalog", {}) if truth_art else {}
    admin_cache = AdminResolverCache()

    qualified_tiers = {str(t).strip().lower() for t in headline_tiers}
    samples: list[dict] = []

    for i, ts in enumerate(times):
        row = {k: _hourly_value(hourly, k, i) for k in fields}
        row = _normalize_row_units(row, hourly_units)
        obs = _build_obs_from_archive(geo.get("name", city), ts, row)
        b_probs = baseline.run_cycle(obs).decision.hazard_prob
        e_probs = enhanced.run_cycle(obs).decision.hazard_prob

        trow = truth_map.get(_hour_key(ts))
        tier = str((trow or {}).get("label_tier", "proxy")).lower()
        qualified = bool(trow and tier in qualified_tiers)

        proxy_truth = _proxy_truth_all(row)
        label = {h: int((trow or {}).get(h, proxy_truth[h])) for h in HAZARDS}

        samples.append(
            {
                "timestamp": ts,
                "data_quality": float(obs.source_meta.get("data_quality_score", 0.62)),
                "baseline": {h: float(b_probs.get(h, 0.0)) for h in HAZARDS},
                "enhanced": {h: float(e_probs.get(h, 0.0)) for h in HAZARDS},
                "truth": label,
                "truth_row": trow or {},
                "proxy_truth": proxy_truth,
                "label_tier": tier,
                "qualified": qualified,
            }
        )

    test_samples = [samples[i] for i in test_indices]
    qualified_test = [s for s in test_samples if s["qualified"]]
    qualified_coverage_ratio = len(qualified_test) / max(len(test_samples), 1)
    positive_counts = {
        h: int(sum(int(s["truth"][h]) for s in qualified_test))
        for h in HAZARDS
    }
    total_positive_labels = int(sum(positive_counts.values()))
    if truth_policy == "require" and qualified_coverage_ratio < float(min_truth_coverage):
        raise RuntimeError(
            f"qualified truth coverage too low in test split: {qualified_coverage_ratio:.3f} < min_truth_coverage={float(min_truth_coverage):.3f}"
        )
    if truth_policy == "require" and total_positive_labels < int(min_total_positive_labels):
        raise RuntimeError(
            f"insufficient positive labels in qualified test split: total_positive_labels={total_positive_labels} < min_total_positive_labels={int(min_total_positive_labels)}"
        )

    baseline_metrics_by_hazard: dict = {}
    enhanced_metrics_by_hazard: dict = {}
    improvements: dict = {}

    for h in HAZARDS:
        b_probs = [float(s["baseline"][h]) for s in qualified_test]
        e_probs = [float(s["enhanced"][h]) for s in qualified_test]
        y = [int(s["truth"][h]) for s in qualified_test]
        t = [s["timestamp"] for s in qualified_test]

        b_metrics = _metric_bundle(b_probs, y, t)
        e_metrics = _metric_bundle(e_probs, y, t)
        admin_base = aggregate_admin_metrics(
            qualified_test,
            hazard=h,
            city_lat=lat,
            city_lon=lon,
            station_catalog=station_catalog,
            cache=admin_cache,
            prob_key="baseline",
        )
        admin_enh = aggregate_admin_metrics(
            qualified_test,
            hazard=h,
            city_lat=lat,
            city_lon=lon,
            station_catalog=station_catalog,
            cache=admin_cache,
            prob_key="enhanced",
        )
        grid_base = aggregate_grid_metrics(
            qualified_test,
            hazard=h,
            city_lat=lat,
            city_lon=lon,
            station_catalog=station_catalog,
            prob_key="baseline",
        )
        grid_enh = aggregate_grid_metrics(
            qualified_test,
            hazard=h,
            city_lat=lat,
            city_lon=lon,
            station_catalog=station_catalog,
            prob_key="enhanced",
        )
        b_metrics.update(admin_base)
        e_metrics.update(admin_enh)
        b_metrics.update(grid_base)
        e_metrics.update(grid_enh)
        b_metrics["spatial_hit_rate"] = admin_base.get("admin_hit_rate", b_metrics.get("spatial_hit_rate", 0.0))
        b_metrics["coverage_bias"] = admin_base.get("admin_coverage_bias", b_metrics.get("coverage_bias", 0.0))
        e_metrics["spatial_hit_rate"] = admin_enh.get("admin_hit_rate", e_metrics.get("spatial_hit_rate", 0.0))
        e_metrics["coverage_bias"] = admin_enh.get("admin_coverage_bias", e_metrics.get("coverage_bias", 0.0))
        baseline_metrics_by_hazard[h] = b_metrics
        enhanced_metrics_by_hazard[h] = e_metrics
        improvements[h] = _improvement_summary(b_metrics, e_metrics)

    # auxiliary metrics can include proxy labels for visibility only
    aux_metrics: dict = {}
    for h in HAZARDS:
        e_probs = [float(s["enhanced"][h]) for s in test_samples]
        y_aux = [int(s["truth"][h] if s["qualified"] else s["proxy_truth"][h]) for s in test_samples]
        t = [s["timestamp"] for s in test_samples]
        aux_metrics[h] = _metric_bundle(e_probs, y_aux, t)

    # data quality slice on qualified test samples
    quality_slices: dict = {}
    for bucket in ("high", "medium", "low"):
        sub = [s for s in qualified_test if _quality_bucket(float(s["data_quality"])) == bucket]
        quality_slices[bucket] = {
            "sample_count": len(sub),
            "hazards": {
                h: _metric_bundle(
                    [float(s["enhanced"][h]) for s in sub],
                    [int(s["truth"][h]) for s in sub],
                    [s["timestamp"] for s in sub],
                )
                for h in HAZARDS
            },
        }

    aggregate_base = _aggregate_hazard_metrics(baseline_metrics_by_hazard)
    aggregate_enh = _aggregate_hazard_metrics(enhanced_metrics_by_hazard)
    tier_counts = {k: 0 for k in ("gold", "silver", "proxy")}
    for s in test_samples:
        tier_counts[s["label_tier"]] = tier_counts.get(s["label_tier"], 0) + 1

    result = {
        "city": geo.get("name", city),
        "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "samples": len(samples),
        "split_manifest": {
            "strategy": "time_ordered",
            "counts": split["counts"],
            "index_ranges": split["index_ranges"],
            "windows": split["windows"],
        },
        "baseline": {
            "hazards": baseline_metrics_by_hazard,
            "aggregate": aggregate_base,
        },
        "enhanced": {
            "hazards": enhanced_metrics_by_hazard,
            "aggregate": aggregate_enh,
            "slices": {
                "by_data_quality": quality_slices,
            },
        },
        "auxiliary_proxy_analysis": {
            "note": "proxy labels are for auxiliary analysis only",
            "hazards": aux_metrics,
        },
        "improvements": {
            "hazards": improvements,
            "note": "improvement summary reports delta metrics only; no fixed 10% pass/fail gate",
        },
        "metric_guide": {
            "precision": "预测为事件的样本中，实际为事件的比例",
            "recall": "实际事件被识别出来的比例",
            "f1": "precision 与 recall 的调和均值",
            "brier": "概率预报均方误差，越低越好",
            "pod": "事件命中率（Probability of Detection）",
            "far": "空报率（False Alarm Ratio）",
            "csi": "综合成功指数（Critical Success Index）",
            "lead_time": "提前量（分钟）",
            "admin_hit_rate": "行政区级命中率（按行政区标签交集）",
            "grid_hit_rate": "网格级命中率（网格单元 TP/真值单元）",
        },
        "truth_labels": {
            "source": "station+hail_reports" if truth_policy in {"prefer", "require"} else "off",
            "truth_policy": truth_policy,
            "headline_tiers": sorted(list(qualified_tiers)),
            "truth_version": truth_version,
            "qualified_test_hours": len(qualified_test),
            "qualified_coverage_ratio": round(qualified_coverage_ratio, 4),
            "min_required_coverage": float(min_truth_coverage),
            "tier_counts_test": tier_counts,
            "used_proxy_auxiliary": len(qualified_test) < len(test_samples),
            "artifact_path": truth_art.get("artifact_path"),
            "sha256": truth_art.get("meta", {}).get("sha256"),
            "station_info": truth_art.get("meta", {}).get("station_info", {}),
            "label_hours": truth_art.get("meta", {}).get("label_hours", 0),
            "label_coverage_ratio": truth_art.get("meta", {}).get("label_coverage_ratio", 0.0),
            "label_tiering": truth_art.get("meta", {}).get("label_tiering", {}),
            "fallback_note": truth_art.get("meta", {}).get("fallback_note", ""),
            "hail_report_files": truth_art.get("meta", {}).get("hail_info", {}).get("files", []),
        },
        "statistical_validity": {
            "hazard_positive_counts": positive_counts,
            "total_positive_labels": total_positive_labels,
            "min_total_positive_labels": int(min_total_positive_labels),
            "event_metrics_reliable": bool(total_positive_labels >= int(min_total_positive_labels)),
            "warning": ""
            if total_positive_labels >= int(min_total_positive_labels)
            else "qualified test split has too few positive labels; event metrics are not reliable for headline conclusion",
        },
    }
    policy_version = PolicyManager().active().policy_version
    result["business_actions"] = {
        "recommend_issue_accuracy": round(mean([float(result["enhanced"]["hazards"][h].get("precision", 0.0)) for h in HAZARDS]), 4) if HAZARDS else 0.0,
        "recommend_clear_stability": round(mean([float(result["enhanced"]["hazards"][h].get("upgrade_downgrade_consistency", 0.0)) for h in HAZARDS]), 4) if HAZARDS else 0.0,
        "manual_review_trigger_quality": round(1.0 - mean([float(result["enhanced"]["hazards"][h].get("far", 0.0)) for h in HAZARDS]), 4) if HAZARDS else 0.0,
        "asymmetric_cost": {
            "miss_cost": 5.0,
            "false_alarm_cost": 1.0,
        },
    }
    manifest = EvalManifest(
        eval_manifest_id=_hash_payload({"city": city, "period": [start_date.isoformat(), end_date.isoformat()], "tiers": list(qualified_tiers)}),
        data_snapshot_id=truth_art.get("meta", {}).get("sha256", "snapshot.na"),
        truth_version=truth_version,
        feature_version="feature_schema.v1",
        code_sha=_code_hash(),
        model_version="multi_agent_rules.v3",
        calibration_version="memory.calibrators",
        policy_version=policy_version,
        run_config_hash=_hash_payload({"city": city, "days": days, "truth_policy": truth_policy, "headline_tiers": list(qualified_tiers)}),
        generated_at=dt.datetime.now().isoformat(),
        tags={"city": city, "truth_policy": truth_policy},
    )
    result["eval_manifest"] = manifest.__dict__
    admin_cache.save()
    result["reports"] = _save_eval_reports(result)
    return result
