import json
from pathlib import Path

from weather_agent import nightly


def _fake_eval(*args, **kwargs):
    return {
        "period": {"start": "2025-03-01", "end": "2025-03-03"},
        "samples": 72,
        "split_manifest": {"counts": {"test": 30}},
        "truth_labels": {"qualified_coverage_ratio": 0.7},
        "enhanced": {"aggregate": {"f1": 0.42, "brier": 0.2}},
        "improvements": {"hazards": {}},
        "reports": {"json": "x.json"},
    }


def test_nightly_outputs_and_gate(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(nightly, "evaluate_recent", _fake_eval)
    monkeypatch.chdir(tmp_path)

    cases = [
        {
            "case_id": "c1",
            "city": "Tianjin",
            "start_date": "2025-03-01",
            "end_date": "2025-03-03",
            "gate": {"min_qualified_coverage": 0.6, "min_test_samples": 24, "max_brier": 0.3},
        }
    ]
    p = Path("cases.json")
    p.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

    out = nightly.run_nightly(cases_path=str(p), enforce_gate=True)
    assert Path(out["json"]).exists()
    assert Path(out["markdown"]).exists()
    summary = out["summary"]["summary"]
    assert summary["overall_gate_pass"] is True
