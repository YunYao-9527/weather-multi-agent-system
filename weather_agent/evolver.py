from __future__ import annotations

import datetime as dt
from collections import defaultdict
from statistics import mean
from typing import Dict

from weather_agent.calibration import fit_beta_calibrator, fit_histogram_calibrator
from weather_agent.adapters.open_meteo import fetch_archive_hourly, geocode_city
from weather_agent.evaluator import (
    _build_obs_from_archive,
    _build_time_split,
    _hour_key,
    _hourly_value,
    _normalize_row_units,
    _proxy_truth_all,
)
from weather_agent.models import HAZARDS
from weather_agent.orchestrator import ForecastOrchestrator, OrchestratorConfig
from weather_agent.truth_labels import QUALIFIED_LABEL_TIERS, TruthConfig, build_truth_label_artifact


def optimize_agent_weights(
    city: str = "Tianjin",
    days: int = 5,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    truth_policy: str = "prefer",  # prefer | require | off
    min_truth_coverage: float = 0.6,
    min_total_positive_labels: int = 1,
    force_rebuild_truth: bool = False,
    min_train_samples: int = 24,
    min_calibration_samples: int = 16,
    calibrator_method: str = "histogram",  # histogram | beta
    headline_tiers: tuple[str, ...] = QUALIFIED_LABEL_TIERS,
) -> dict:
    geo = geocode_city(city)
    lat = float(geo["latitude"])
    lon = float(geo["longitude"])
    timezone = geo.get("timezone") or "Asia/Shanghai"

    if end_date is None:
        end_date = dt.date.today()
    if start_date is None:
        start_date = end_date - dt.timedelta(days=max(2, days))

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

    truth_art = {"meta": {}, "labels_by_hour": {}, "artifact_path": None}
    truth_map = {}
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

    qualified_tiers = {str(t).strip().lower() for t in headline_tiers}
    samples: list[dict] = []

    for i, ts in enumerate(times):
        row = {k: _hourly_value(hourly, k, i) for k in fields}
        row = _normalize_row_units(row, hourly_units)
        obs = _build_obs_from_archive(geo.get("name", city), ts, row)
        trow = truth_map.get(_hour_key(ts))
        tier = str((trow or {}).get("label_tier", "proxy")).lower()
        qualified = bool(trow and tier in qualified_tiers)

        proxy = _proxy_truth_all(row)
        truth = {h: int((trow or {}).get(h, proxy[h])) for h in HAZARDS}
        samples.append(
            {
                "timestamp": ts,
                "obs": obs,
                "row": row,
                "truth": truth,
                "label_tier": tier,
                "qualified": qualified,
            }
        )

    train_samples = [samples[i] for i in split["indices"]["train"] if samples[i]["qualified"]]
    calib_samples = [samples[i] for i in split["indices"]["calibration"] if samples[i]["qualified"]]
    test_samples = [samples[i] for i in split["indices"]["test"] if samples[i]["qualified"]]
    train_positive_total = int(sum(int(s["truth"][h]) for s in train_samples for h in HAZARDS))
    test_positive_total = int(sum(int(s["truth"][h]) for s in test_samples for h in HAZARDS))

    qualified_test_ratio = len(test_samples) / max(len(split["indices"]["test"]), 1)
    if truth_policy == "require" and qualified_test_ratio < float(min_truth_coverage):
        raise RuntimeError(
            f"qualified truth coverage too low in test split: {qualified_test_ratio:.3f} < min_truth_coverage={float(min_truth_coverage):.3f}"
        )
    if truth_policy == "require" and test_positive_total < int(min_total_positive_labels):
        raise RuntimeError(
            f"insufficient positive labels in qualified test split: total_positive_labels={test_positive_total} < min_total_positive_labels={int(min_total_positive_labels)}"
        )

    orchestrator = ForecastOrchestrator(OrchestratorConfig(region_name="training"))
    default_weights = {a.name: 1.0 for a in orchestrator.agents}

    fallback_reason = ""
    if truth_policy == "off":
        fallback_reason = "truth_policy=off: headline training disabled to avoid proxy contamination"
        weights = default_weights
        qualities = {k: 0.0 for k in default_weights}
        calibrators = {}
    elif len(train_samples) < int(min_train_samples):
        fallback_reason = f"insufficient qualified train samples: {len(train_samples)} < {int(min_train_samples)}"
        weights = default_weights
        qualities = {k: 0.0 for k in default_weights}
        calibrators = {}
    elif train_positive_total < int(min_total_positive_labels):
        fallback_reason = (
            f"insufficient positive labels in qualified train split: total_positive_labels={train_positive_total} < "
            f"min_total_positive_labels={int(min_total_positive_labels)}"
        )
        weights = default_weights
        qualities = {k: 0.0 for k in default_weights}
        calibrators = {}
    else:
        agent_errors = defaultdict(list)
        for sample in train_samples:
            cards = [a.run(sample["obs"]).evidence for a in orchestrator.agents]
            truth = sample["truth"]
            for c in cards:
                errs = []
                for h in HAZARDS:
                    errs.append((float(c.hazard_scores.get(h, 0.0)) - int(truth[h])) ** 2)
                agent_errors[c.agent].append(mean(errs))

        qualities = {}
        for agent, errs in agent_errors.items():
            brier = mean(errs) if errs else 1.0
            qualities[agent] = max(0.0, 1.0 - brier)

        raw = {a: 0.6 + qualities.get(a, 0.0) * 1.2 for a in default_weights}
        avg = mean(raw.values()) if raw else 1.0
        normalized = {a: raw[a] / avg for a in raw}

        # sample-size-aware shrinkage to avoid unstable month/city overfitting
        shrink = min(1.0, len(train_samples) / 200.0)
        weights = {a: round(1.0 + shrink * (normalized[a] - 1.0), 4) for a in normalized}

        if len(calib_samples) < int(min_calibration_samples):
            calibrators = {}
            fallback_reason = f"insufficient qualified calibration samples: {len(calib_samples)} < {int(min_calibration_samples)}"
        else:
            tuned = ForecastOrchestrator(OrchestratorConfig(region_name="calibration", agent_weights=weights))
            hazard_pairs = defaultdict(list)
            for sample in calib_samples:
                probs = tuned.run_cycle(sample["obs"]).decision.hazard_prob
                truth = sample["truth"]
                for h in HAZARDS:
                    hazard_pairs[h].append((float(probs.get(h, 0.0)), int(truth[h])))

            calibrators = {}
            method = (calibrator_method or "histogram").strip().lower()
            for h in HAZARDS:
                pairs = hazard_pairs[h]
                if len(pairs) < int(min_calibration_samples):
                    continue
                if method == "beta":
                    calibrators[h] = fit_beta_calibrator(pairs)
                else:
                    calibrators[h] = fit_histogram_calibrator(pairs, bins=12)

    return {
        "city": geo.get("name", city),
        "month": (end_date or dt.date.today()).month,
        "trained_period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "samples": len(samples),
        "truth_policy": truth_policy,
        "headline_tiers": sorted(list(qualified_tiers)),
        "qualified_counts": {
            "train": len(train_samples),
            "calibration": len(calib_samples),
            "test": len(test_samples),
        },
        "qualified_positive_labels": {
            "train_total": train_positive_total,
            "test_total": test_positive_total,
        },
        "split_manifest": {
            "strategy": "time_ordered",
            "counts": split["counts"],
            "index_ranges": split["index_ranges"],
            "windows": split["windows"],
        },
        "truth_coverage_ratio_test": round(qualified_test_ratio, 4),
        "min_required_coverage": float(min_truth_coverage),
        "truth_artifact_path": truth_art.get("artifact_path"),
        "truth_tiering": truth_art.get("meta", {}).get("label_tiering", {}),
        "proxy_excluded_from_headline_training": True,
        "fallback_reason": fallback_reason,
        "qualities": {k: round(float(v), 4) for k, v in qualities.items()},
        "weights": weights,
        "calibrator_method": calibrator_method,
        "calibrators": calibrators,
    }
