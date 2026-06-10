from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from weather_agent.evaluator import _build_time_split, _hour_key
from weather_agent.truth_labels import QUALIFIED_LABEL_TIERS, TruthConfig, build_truth_label_artifact


HAZARDS = ("short_rain", "wind", "hail", "tornado")


@dataclass
class WindowScanConfig:
    city: str
    search_start: dt.date
    search_end: dt.date
    window_days: int = 3
    step_days: int = 1
    min_truth_coverage: float = 0.6
    min_total_positive_labels: int = 1
    min_train_positive_labels: int = 1
    min_calibration_positive_labels: int = 1
    headline_tiers: tuple[str, ...] = QUALIFIED_LABEL_TIERS
    top_k: int = 20
    force_rebuild_truth: bool = False
    timezone: str | None = None


def _iter_hour_times(start_date: dt.date, end_date: dt.date) -> list[dt.datetime]:
    out = []
    cur = dt.datetime.combine(start_date, dt.time(0, 0))
    stop = dt.datetime.combine(end_date + dt.timedelta(days=1), dt.time(0, 0))
    while cur < stop:
        out.append(cur)
        cur += dt.timedelta(hours=1)
    return out


def _window_end(start: dt.date, window_days: int) -> dt.date:
    return start + dt.timedelta(days=max(1, int(window_days)) - 1)


def _scan_one_window(
    labels_by_hour: dict,
    start_date: dt.date,
    end_date: dt.date,
    min_truth_coverage: float,
    min_total_positive_labels: int,
    min_train_positive_labels: int,
    min_calibration_positive_labels: int,
    headline_tiers: Sequence[str],
) -> dict:
    tiers = {str(x).strip().lower() for x in headline_tiers}
    times = _iter_hour_times(start_date, end_date)
    split = _build_time_split(times)
    test_indices = split["indices"]["test"]

    positives = {h: 0 for h in HAZARDS}
    train_pos = {h: 0 for h in HAZARDS}
    calib_pos = {h: 0 for h in HAZARDS}
    test_pos = {h: 0 for h in HAZARDS}
    qualified_train_hours = 0
    qualified_calib_hours = 0
    qualified_test_hours = 0
    tier_counts = {"gold": 0, "silver": 0, "proxy": 0}

    def _accumulate(indexes, split_name: str):
        nonlocal qualified_train_hours, qualified_calib_hours, qualified_test_hours
        for i in indexes:
            ts = times[i]
            row = labels_by_hour.get(_hour_key(ts))
            if not row:
                continue
            tier = str(row.get("label_tier", "proxy")).lower()
            if split_name == "test":
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if tier not in tiers:
                continue
            if split_name == "train":
                qualified_train_hours += 1
            elif split_name == "calibration":
                qualified_calib_hours += 1
            else:
                qualified_test_hours += 1
            for h in HAZARDS:
                v = int(row.get(h, 0) or 0)
                positives[h] += v
                if split_name == "train":
                    train_pos[h] += v
                elif split_name == "calibration":
                    calib_pos[h] += v
                else:
                    test_pos[h] += v

    _accumulate(split["indices"]["train"], "train")
    _accumulate(split["indices"]["calibration"], "calibration")
    _accumulate(test_indices, "test")

    test_hours = len(test_indices)
    train_hours = len(split["indices"]["train"])
    calib_hours = len(split["indices"]["calibration"])
    qualified_train_coverage = qualified_train_hours / max(train_hours, 1)
    qualified_calib_coverage = qualified_calib_hours / max(calib_hours, 1)
    qualified_coverage = qualified_test_hours / max(test_hours, 1)
    total_positive_labels = sum(test_pos.values())
    train_total_positive_labels = sum(train_pos.values())
    calib_total_positive_labels = sum(calib_pos.values())
    event_hours = 0
    for i in test_indices:
        ts = times[i]
        row = labels_by_hour.get(_hour_key(ts))
        if not row:
            continue
        tier = str(row.get("label_tier", "proxy")).lower()
        if tier not in tiers:
            continue
        if any(int(row.get(h, 0) or 0) == 1 for h in HAZARDS):
            event_hours += 1

    pass_coverage = qualified_coverage >= float(min_truth_coverage)
    pass_positive = total_positive_labels >= int(min_total_positive_labels)
    passed = bool(pass_coverage and pass_positive)
    pass_evolve = bool(
        pass_coverage
        and pass_positive
        and train_total_positive_labels >= int(min_train_positive_labels)
        and calib_total_positive_labels >= int(min_calibration_positive_labels)
    )

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "window_days": int((end_date - start_date).days + 1),
        "split_train_hours": train_hours,
        "split_calibration_hours": calib_hours,
        "split_test_hours": test_hours,
        "qualified_train_hours": qualified_train_hours,
        "qualified_calibration_hours": qualified_calib_hours,
        "qualified_test_hours": qualified_test_hours,
        "qualified_train_coverage_ratio": round(qualified_train_coverage, 4),
        "qualified_calibration_coverage_ratio": round(qualified_calib_coverage, 4),
        "qualified_test_coverage_ratio": round(qualified_coverage, 4),
        "hazard_positive_counts_train": train_pos,
        "hazard_positive_counts_calibration": calib_pos,
        "hazard_positive_counts_test": test_pos,
        "hazard_positive_counts": test_pos,
        "total_positive_labels": int(total_positive_labels),
        "train_total_positive_labels": int(train_total_positive_labels),
        "calibration_total_positive_labels": int(calib_total_positive_labels),
        "event_hours_in_test": int(event_hours),
        "tier_counts_test": tier_counts,
        "pass_coverage": pass_coverage,
        "pass_positive": pass_positive,
        "pass_for_require_eval": passed,
        "pass_for_require_evolve": pass_evolve,
        "score": round(total_positive_labels * 100.0 + event_hours * 20.0 + qualified_coverage * 10.0, 4),
    }


def scan_candidate_windows(cfg: WindowScanConfig) -> dict:
    if cfg.search_end < cfg.search_start:
        raise ValueError("search_end must be >= search_start")
    if cfg.window_days < 1:
        raise ValueError("window_days must be >= 1")
    if cfg.step_days < 1:
        raise ValueError("step_days must be >= 1")

    truth = build_truth_label_artifact(
        city=cfg.city,
        start_date=cfg.search_start,
        end_date=cfg.search_end,
        timezone=cfg.timezone,
        cfg=TruthConfig(min_coverage_for_headline=cfg.min_truth_coverage),
        force_rebuild=cfg.force_rebuild_truth,
    )
    labels_by_hour = truth.get("labels_by_hour", {})

    rows = []
    cur = cfg.search_start
    while True:
        win_end = _window_end(cur, cfg.window_days)
        if win_end > cfg.search_end:
            break
        rows.append(
            _scan_one_window(
                labels_by_hour=labels_by_hour,
                start_date=cur,
                end_date=win_end,
                min_truth_coverage=cfg.min_truth_coverage,
                min_total_positive_labels=cfg.min_total_positive_labels,
                min_train_positive_labels=cfg.min_train_positive_labels,
                min_calibration_positive_labels=cfg.min_calibration_positive_labels,
                headline_tiers=cfg.headline_tiers,
            )
        )
        cur += dt.timedelta(days=cfg.step_days)

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            int(x["pass_for_require_eval"]),
            float(x["score"]),
            int(x["total_positive_labels"]),
            float(x["qualified_test_coverage_ratio"]),
        ),
        reverse=True,
    )
    passed = [r for r in rows_sorted if bool(r["pass_for_require_eval"])]
    passed_evolve = [r for r in rows_sorted if bool(r["pass_for_require_evolve"])]
    top = rows_sorted[: max(int(cfg.top_k), 1)]
    top_passed = passed[: max(int(cfg.top_k), 1)]

    return {
        "generated_at": dt.datetime.now().isoformat(),
        "city": cfg.city,
        "search_window": {"start": cfg.search_start.isoformat(), "end": cfg.search_end.isoformat()},
        "scan_config": {
            "window_days": cfg.window_days,
            "step_days": cfg.step_days,
            "min_truth_coverage": cfg.min_truth_coverage,
            "min_total_positive_labels": cfg.min_total_positive_labels,
            "min_train_positive_labels": cfg.min_train_positive_labels,
            "min_calibration_positive_labels": cfg.min_calibration_positive_labels,
            "headline_tiers": list(cfg.headline_tiers),
            "top_k": cfg.top_k,
            "force_rebuild_truth": cfg.force_rebuild_truth,
        },
        "truth_artifact_path": truth.get("artifact_path"),
        "truth_meta": {
            "label_hours": truth.get("meta", {}).get("label_hours", 0),
            "label_coverage_ratio": truth.get("meta", {}).get("label_coverage_ratio", 0.0),
            "label_tiering": truth.get("meta", {}).get("label_tiering", {}),
        },
        "summary": {
            "window_count_scanned": len(rows_sorted),
            "window_count_passed": len(passed),
            "window_count_passed_evolve": len(passed_evolve),
            "best_window": passed[0] if passed else None,
            "best_window_evolve": passed_evolve[0] if passed_evolve else None,
            "note": "pass_for_require_eval means both coverage and positive-label constraints are satisfied",
        },
        "top_windows": top,
        "top_passed_windows": top_passed,
        "top_passed_evolve_windows": passed_evolve[: max(int(cfg.top_k), 1)],
    }


def _save_report(payload: dict, output_dir: str | Path = "runs/window_scan") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    city = str(payload.get("city", "city")).replace(" ", "_").lower()
    json_path = out / f"window_scan_{city}_{ts}.json"
    md_path = out / f"window_scan_{city}_{ts}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Window Scan - {payload.get('city')}",
        "",
        f"- Search: {payload.get('search_window', {}).get('start')} ~ {payload.get('search_window', {}).get('end')}",
        f"- Scanned: {payload.get('summary', {}).get('window_count_scanned', 0)}",
        f"- Passed: {payload.get('summary', {}).get('window_count_passed', 0)}",
        f"- Truth artifact: {payload.get('truth_artifact_path')}",
        "",
        "| start | end | pass | coverage | positives(total) | short_rain | wind | hail | tornado |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload.get("top_windows", []):
        p = r.get("hazard_positive_counts", {})
        lines.append(
            f"| {r.get('start_date')} | {r.get('end_date')} | {'Y' if r.get('pass_for_require_eval') else 'N'} | "
            f"{r.get('qualified_test_coverage_ratio')} | {r.get('total_positive_labels')} | "
            f"{p.get('short_rain', 0)} | {p.get('wind', 0)} | {p.get('hail', 0)} | {p.get('tornado', 0)} |"
        )
    best = payload.get("summary", {}).get("best_window")
    if best:
        lines.extend(
            [
                "",
                "## Recommended Evaluate Payload",
                "```json",
                json.dumps(
                    {
                        "city": payload.get("city"),
                        "start_date": best.get("start_date"),
                        "end_date": best.get("end_date"),
                        "truth_policy": "require",
                        "min_truth_coverage": payload.get("scan_config", {}).get("min_truth_coverage", 0.6),
                        "min_total_positive_labels": payload.get("scan_config", {}).get("min_total_positive_labels", 1),
                        "headline_tiers": payload.get("scan_config", {}).get("headline_tiers", ["gold", "silver"]),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
            ]
        )
    best_e = payload.get("summary", {}).get("best_window_evolve")
    if best_e:
        lines.extend(
            [
                "",
                "## Recommended Evolve Payload",
                "```json",
                json.dumps(
                    {
                        "city": payload.get("city"),
                        "start_date": best_e.get("start_date"),
                        "end_date": best_e.get("end_date"),
                        "truth_policy": "require",
                        "min_truth_coverage": payload.get("scan_config", {}).get("min_truth_coverage", 0.6),
                        "min_total_positive_labels": payload.get("scan_config", {}).get("min_total_positive_labels", 1),
                        "headline_tiers": payload.get("scan_config", {}).get("headline_tiers", ["gold", "silver"]),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def run_window_scan(cfg: WindowScanConfig, output_dir: str | Path = "runs/window_scan") -> dict:
    payload = scan_candidate_windows(cfg)
    reports = _save_report(payload, output_dir=output_dir)
    return {"reports": reports, "result": payload}


def _parse_tiers(v: str | None) -> tuple[str, ...]:
    if not v:
        return QUALIFIED_LABEL_TIERS
    out = []
    for x in str(v).split(","):
        t = x.strip().lower()
        if t and t not in out:
            out.append(t)
    return tuple(out) if out else QUALIFIED_LABEL_TIERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-select historical windows with positive truth labels for robust evaluation")
    parser.add_argument("--city", default="Tianjin")
    parser.add_argument("--search-start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--search-end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--step-days", type=int, default=1)
    parser.add_argument("--min-truth-coverage", type=float, default=0.6)
    parser.add_argument("--min-total-positive-labels", type=int, default=1)
    parser.add_argument("--min-train-positive-labels", type=int, default=1)
    parser.add_argument("--min-calibration-positive-labels", type=int, default=1)
    parser.add_argument("--headline-tiers", default="gold,silver")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", default="runs/window_scan")
    parser.add_argument("--force-rebuild-truth", action="store_true")
    args = parser.parse_args()

    cfg = WindowScanConfig(
        city=args.city,
        search_start=dt.date.fromisoformat(args.search_start),
        search_end=dt.date.fromisoformat(args.search_end),
        window_days=int(args.window_days),
        step_days=int(args.step_days),
        min_truth_coverage=float(args.min_truth_coverage),
        min_total_positive_labels=int(args.min_total_positive_labels),
        min_train_positive_labels=int(args.min_train_positive_labels),
        min_calibration_positive_labels=int(args.min_calibration_positive_labels),
        headline_tiers=_parse_tiers(args.headline_tiers),
        top_k=int(args.top_k),
        force_rebuild_truth=bool(args.force_rebuild_truth),
    )
    out = run_window_scan(cfg, output_dir=args.output_dir)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
