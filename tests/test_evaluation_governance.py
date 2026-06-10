import datetime as dt

import pytest

from weather_agent import evaluator, evolver


def _fake_archive(start: dt.datetime, hours: int) -> dict:
    times = [(start + dt.timedelta(hours=i)).isoformat() for i in range(hours)]

    def seq(v: float):
        return [v for _ in range(hours)]

    return {
        "hourly": {
            "time": times,
            "cape": seq(1200.0),
            "convective_inhibition": seq(-50.0),
            "lifted_index": seq(-3.0),
            "relative_humidity_2m": seq(70.0),
            "wind_gusts_10m": seq(12.0),
            "wind_speed_10m": seq(8.0),
            "vertical_velocity_700hPa": seq(-0.6),
            "temperature_850hPa": seq(18.0),
            "temperature_500hPa": seq(-7.0),
            "freezing_level_height": seq(3600.0),
            "precipitation": seq(5.0),
            "showers": seq(0.0),
            "wind_speed_850hPa": seq(15.0),
            "wind_speed_500hPa": seq(28.0),
            "wind_direction_850hPa": seq(220.0),
            "wind_direction_500hPa": seq(245.0),
        }
    }


def _mk_truth(times: list[dt.datetime], qualified_idx: set[int]) -> dict:
    labels = {}
    for i, ts in enumerate(times):
        if i not in qualified_idx:
            continue
        labels[evaluator._hour_key(ts)] = {
            "short_rain": 1 if i % 3 == 0 else 0,
            "wind": 1 if i % 4 == 0 else 0,
            "hail": 0,
            "tornado": 0,
            "label_tier": "gold",
        }
    return {
        "meta": {
            "sha256": "abc",
            "label_hours": len(labels),
            "label_coverage_ratio": round(len(labels) / max(len(times), 1), 4),
            "label_tiering": {"tier_counts": {"gold": len(labels), "silver": 0, "proxy": 0}},
            "station_info": {"used_count": 1, "candidate_count": 1},
            "hail_info": {"files": []},
            "fallback_note": "proxy labels are auxiliary only",
        },
        "labels_by_hour": labels,
        "artifact_path": "runs/truth_labels/fake.json",
    }


def _mk_truth_all_zero(times: list[dt.datetime], qualified_idx: set[int]) -> dict:
    labels = {}
    for i, ts in enumerate(times):
        if i not in qualified_idx:
            continue
        labels[evaluator._hour_key(ts)] = {
            "short_rain": 0,
            "wind": 0,
            "hail": 0,
            "tornado": 0,
            "label_tier": "gold",
        }
    return {
        "meta": {
            "sha256": "abc",
            "label_hours": len(labels),
            "label_coverage_ratio": round(len(labels) / max(len(times), 1), 4),
            "label_tiering": {"tier_counts": {"gold": len(labels), "silver": 0, "proxy": 0}},
            "station_info": {"used_count": 1, "candidate_count": 1},
            "hail_info": {"files": []},
            "fallback_note": "proxy labels are auxiliary only",
        },
        "labels_by_hour": labels,
        "artifact_path": "runs/truth_labels/fake.json",
    }


def test_time_split_is_time_ordered_and_non_overlapping():
    start = dt.datetime(2025, 3, 1, 0, 0)
    times = [start + dt.timedelta(hours=i) for i in range(50)]
    split = evaluator._build_time_split(times)

    train = split["indices"]["train"]
    calib = split["indices"]["calibration"]
    test = split["indices"]["test"]

    assert train
    assert calib
    assert test
    assert max(train) < min(calib)
    assert max(calib) < min(test)
    assert len(set(train) & set(calib)) == 0
    assert len(set(calib) & set(test)) == 0
    assert len(set(train) & set(test)) == 0


def test_evaluate_require_blocks_low_qualified_coverage(monkeypatch):
    start = dt.datetime(2025, 3, 1, 0, 0)
    hours = 30
    times = [start + dt.timedelta(hours=i) for i in range(hours)]

    monkeypatch.setattr(evaluator, "geocode_city", lambda city: {"name": city, "latitude": 39.1, "longitude": 117.2, "timezone": "Asia/Shanghai"})
    monkeypatch.setattr(evaluator, "fetch_archive_hourly", lambda *args, **kwargs: _fake_archive(start, hours))

    # split(test)=6 hours for 30-sample with default split; only 2 qualified in test => 0.33 coverage
    qualified = {24, 25}
    monkeypatch.setattr(evaluator, "build_truth_label_artifact", lambda **kwargs: _mk_truth(times, qualified))

    with pytest.raises(RuntimeError):
        evaluator.evaluate_recent(
            city="Tianjin",
            days=3,
            start_date=dt.date(2025, 3, 1),
            end_date=dt.date(2025, 3, 2),
            truth_policy="require",
            min_truth_coverage=0.8,
        )


def test_evaluate_require_blocks_when_no_positive_labels(monkeypatch):
    start = dt.datetime(2025, 3, 1, 0, 0)
    hours = 30
    times = [start + dt.timedelta(hours=i) for i in range(hours)]

    monkeypatch.setattr(evaluator, "geocode_city", lambda city: {"name": city, "latitude": 39.1, "longitude": 117.2, "timezone": "Asia/Shanghai"})
    monkeypatch.setattr(evaluator, "fetch_archive_hourly", lambda *args, **kwargs: _fake_archive(start, hours))

    qualified = {24, 25, 26, 27, 28, 29}
    monkeypatch.setattr(evaluator, "build_truth_label_artifact", lambda **kwargs: _mk_truth_all_zero(times, qualified))

    with pytest.raises(RuntimeError):
        evaluator.evaluate_recent(
            city="Tianjin",
            days=3,
            start_date=dt.date(2025, 3, 1),
            end_date=dt.date(2025, 3, 2),
            truth_policy="require",
            min_truth_coverage=0.6,
            min_total_positive_labels=1,
        )


def test_evaluate_prefer_keeps_proxy_only_auxiliary(monkeypatch):
    start = dt.datetime(2025, 3, 1, 0, 0)
    hours = 30
    times = [start + dt.timedelta(hours=i) for i in range(hours)]

    monkeypatch.setattr(evaluator, "geocode_city", lambda city: {"name": city, "latitude": 39.1, "longitude": 117.2, "timezone": "Asia/Shanghai"})
    monkeypatch.setattr(evaluator, "fetch_archive_hourly", lambda *args, **kwargs: _fake_archive(start, hours))

    qualified = {24, 25, 26}
    monkeypatch.setattr(evaluator, "build_truth_label_artifact", lambda **kwargs: _mk_truth(times, qualified))

    result = evaluator.evaluate_recent(
        city="Tianjin",
        days=3,
        start_date=dt.date(2025, 3, 1),
        end_date=dt.date(2025, 3, 2),
        truth_policy="prefer",
        min_truth_coverage=0.2,
    )

    assert result["split_manifest"]["counts"]["test"] == 6
    assert result["truth_labels"]["qualified_test_hours"] == 3
    assert result["truth_labels"]["used_proxy_auxiliary"] is True
    assert result["enhanced"]["hazards"]["short_rain"]["sample_count"] == 3
    assert "overall_target_10pct_met" not in result["improvements"]
    assert "reports" in result


def test_evolver_does_not_train_on_proxy_only(monkeypatch):
    start = dt.datetime(2025, 3, 1, 0, 0)
    hours = 40
    times = [start + dt.timedelta(hours=i) for i in range(hours)]

    monkeypatch.setattr(evolver, "geocode_city", lambda city: {"name": city, "latitude": 39.1, "longitude": 117.2, "timezone": "Asia/Shanghai"})
    monkeypatch.setattr(evolver, "fetch_archive_hourly", lambda *args, **kwargs: _fake_archive(start, hours))

    def _proxy_only_truth(**kwargs):
        labels = {
            evolver._hour_key(ts): {
                "short_rain": 0,
                "wind": 0,
                "hail": 0,
                "tornado": 0,
                "label_tier": "proxy",
            }
            for ts in times
        }
        return {"meta": {"label_tiering": {"tier_counts": {"proxy": len(labels)}}}, "labels_by_hour": labels, "artifact_path": "x"}

    monkeypatch.setattr(evolver, "build_truth_label_artifact", _proxy_only_truth)

    result = evolver.optimize_agent_weights(
        city="Tianjin",
        days=5,
        start_date=dt.date(2025, 3, 1),
        end_date=dt.date(2025, 3, 2),
        truth_policy="prefer",
        min_train_samples=5,
    )

    assert result["proxy_excluded_from_headline_training"] is True
    assert result["qualified_counts"]["train"] == 0
    assert "insufficient qualified train samples" in result["fallback_reason"]
