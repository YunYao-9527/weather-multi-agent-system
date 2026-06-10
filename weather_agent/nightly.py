from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from weather_agent.evaluator import evaluate_recent


def _load_cases(path: str | Path = "data/cases/historical_cases.json") -> list[dict]:
    fp = Path(path)
    if not fp.exists():
        return []
    payload = json.loads(fp.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _parse_gate(case: dict) -> dict:
    gate = case.get("gate") if isinstance(case.get("gate"), dict) else {}
    return {
        "min_qualified_coverage": float(gate.get("min_qualified_coverage", 0.6)),
        "min_test_samples": int(gate.get("min_test_samples", 24)),
        "max_brier": float(gate.get("max_brier", 0.35)),
    }


def _run_case(case: dict) -> dict:
    start_date = dt.date.fromisoformat(case["start_date"])
    end_date = dt.date.fromisoformat(case["end_date"])
    result = evaluate_recent(
        city=case["city"],
        start_date=start_date,
        end_date=end_date,
        truth_policy="require",
        min_truth_coverage=_parse_gate(case)["min_qualified_coverage"],
        headline_tiers=("gold", "silver"),
    )
    agg = result.get("enhanced", {}).get("aggregate", {})
    gate = _parse_gate(case)
    qualified_cov = float(result.get("truth_labels", {}).get("qualified_coverage_ratio", 0.0))
    test_count = int(result.get("split_manifest", {}).get("counts", {}).get("test", 0))
    brier = float(agg.get("brier", 0.0))

    gate_checks = {
        "coverage_ok": qualified_cov >= gate["min_qualified_coverage"],
        "test_samples_ok": test_count >= gate["min_test_samples"],
        "brier_ok": brier <= gate["max_brier"],
    }
    gate_pass = all(gate_checks.values())

    return {
        "case_id": case.get("case_id"),
        "city": case.get("city"),
        "period": result.get("period"),
        "samples": result.get("samples"),
        "split_test_count": test_count,
        "qualified_coverage": qualified_cov,
        "enhanced_aggregate": agg,
        "improvements": result.get("improvements", {}),
        "reports": result.get("reports", {}),
        "gate": {
            "rules": gate,
            "checks": gate_checks,
            "pass": gate_pass,
        },
    }


def _latest_previous_json(path: Path, current_name: str) -> Path | None:
    files = sorted([p for p in path.glob("nightly_*.json") if p.name != current_name])
    if not files:
        return None
    return files[-1]


def _index_by_case(items: list[dict]) -> dict[str, dict]:
    out = {}
    for it in items:
        cid = str(it.get("case_id", ""))
        if cid:
            out[cid] = it
    return out


def _build_trend(current_items: list[dict], previous_payload: dict | None) -> dict:
    if not previous_payload:
        return {"has_previous": False, "by_case": {}}
    prev_map = _index_by_case(previous_payload.get("results", []))
    by_case = {}
    for it in current_items:
        cid = str(it.get("case_id", ""))
        prev = prev_map.get(cid)
        if not prev:
            continue
        cur_agg = it.get("enhanced_aggregate", {})
        prv_agg = prev.get("enhanced_aggregate", {})
        by_case[cid] = {
            "f1_delta": round(float(cur_agg.get("f1", 0.0)) - float(prv_agg.get("f1", 0.0)), 4),
            "brier_delta": round(float(cur_agg.get("brier", 0.0)) - float(prv_agg.get("brier", 0.0)), 4),
            "coverage_delta": round(float(it.get("qualified_coverage", 0.0)) - float(prev.get("qualified_coverage", 0.0)), 4),
            "gate_pass_changed": bool(it.get("gate", {}).get("pass", False)) != bool(prev.get("gate", {}).get("pass", False)),
        }
    return {"has_previous": True, "by_case": by_case}


def run_nightly(cases_path: str | Path = "data/cases/historical_cases.json", *, enforce_gate: bool = False) -> dict:
    cases = _load_cases(cases_path)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs/nightly")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for case in cases:
        try:
            items.append(_run_case(case))
        except Exception as e:
            items.append({"case_id": case.get("case_id"), "city": case.get("city"), "error": str(e), "gate": {"pass": False}})

    case_pass = [bool(it.get("gate", {}).get("pass", False)) for it in items if "error" not in it]
    overall_pass = all(case_pass) if case_pass else False

    payload = {
        "generated_at": dt.datetime.now().isoformat(),
        "case_count": len(cases),
        "results": items,
        "summary": {
            "successful_cases": len([x for x in items if "error" not in x]),
            "failed_cases": len([x for x in items if "error" in x]),
            "gate_pass_cases": len([x for x in items if bool(x.get("gate", {}).get("pass", False))]),
            "overall_gate_pass": overall_pass,
        },
    }

    json_path = out_dir / f"nightly_{ts}.json"
    md_path = out_dir / f"nightly_{ts}.md"

    prev_json = _latest_previous_json(out_dir, json_path.name)
    prev_payload = None
    if prev_json and prev_json.exists():
        try:
            prev_payload = json.loads(prev_json.read_text(encoding="utf-8"))
        except Exception:
            prev_payload = None
    payload["trend"] = _build_trend(items, prev_payload)
    payload["previous_run"] = str(prev_json) if prev_json else None

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Nightly Regression {ts}",
        "",
        f"- Cases: {len(cases)}",
        f"- Overall gate pass: {overall_pass}",
        f"- Previous run: {payload.get('previous_run') or 'none'}",
        "",
        "| case_id | city | coverage | test_count | f1 | brier | gate | note |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for it in items:
        if "error" in it:
            lines.append(
                f"| {it.get('case_id')} | {it.get('city')} | - | - | - | - | FAIL | {str(it.get('error')).replace('|', '/')} |"
            )
            continue
        agg = it.get("enhanced_aggregate", {})
        lines.append(
            f"| {it.get('case_id')} | {it.get('city')} | {it.get('qualified_coverage')} | {it.get('split_test_count')} | {agg.get('f1', 0.0)} | {agg.get('brier', 0.0)} | {'PASS' if it.get('gate', {}).get('pass') else 'FAIL'} | - |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    out = {"json": str(json_path), "markdown": str(md_path), "summary": payload}
    if enforce_gate and not overall_pass:
        raise RuntimeError(f"nightly gate failed: see {json_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly regression runner")
    parser.add_argument("--cases-path", default="data/cases/historical_cases.json")
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()
    out = run_nightly(cases_path=args.cases_path, enforce_gate=bool(args.enforce_gate))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
