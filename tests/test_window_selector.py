import datetime as dt

from weather_agent.window_selector import _scan_one_window


def _labels_for_window(start_date: dt.date, days: int):
    labels = {}
    cur = dt.datetime.combine(start_date, dt.time(0, 0))
    stop = cur + dt.timedelta(days=days)
    while cur < stop:
        labels[cur.isoformat()] = {
            "label_tier": "gold",
            "short_rain": 0,
            "wind": 0,
            "hail": 0,
            "tornado": 0,
        }
        cur += dt.timedelta(hours=1)
    return labels


def test_scan_one_window_pass_with_positive_labels():
    start = dt.date(2025, 3, 1)
    end = dt.date(2025, 3, 3)
    labels = _labels_for_window(start, 3)
    # put one positive in approximate test segment (last 20%)
    labels["2025-03-03T12:00:00"]["wind"] = 1
    out = _scan_one_window(
        labels_by_hour=labels,
        start_date=start,
        end_date=end,
        min_truth_coverage=0.6,
        min_total_positive_labels=1,
        min_train_positive_labels=0,
        min_calibration_positive_labels=0,
        headline_tiers=("gold", "silver"),
    )
    assert out["pass_for_require_eval"] is True
    assert out["total_positive_labels"] >= 1


def test_scan_one_window_fail_without_positive_labels():
    start = dt.date(2025, 3, 1)
    end = dt.date(2025, 3, 3)
    labels = _labels_for_window(start, 3)
    out = _scan_one_window(
        labels_by_hour=labels,
        start_date=start,
        end_date=end,
        min_truth_coverage=0.6,
        min_total_positive_labels=1,
        min_train_positive_labels=0,
        min_calibration_positive_labels=0,
        headline_tiers=("gold", "silver"),
    )
    assert out["pass_for_require_eval"] is False
    assert out["pass_positive"] is False
