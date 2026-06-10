from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

from weather_agent.models import CycleResult


def _convert(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return _convert(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(v) for v in obj]
    return obj


def cycle_to_dict(result: CycleResult) -> Dict[str, Any]:
    raw = _convert(result)
    raw["decision"]["system_confidence"] = raw["decision"].get("confidence", 0.0)
    raw["decision"]["system_status_score"] = raw["decision"].get("evidence_readiness_score", 0.0)
    return raw
