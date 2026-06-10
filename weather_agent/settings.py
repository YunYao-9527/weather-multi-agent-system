from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RuntimeSettings:
    profile: str = "dev"
    issue_threshold: float = 0.55
    clear_threshold: float = 0.38
    min_readiness_score: float = 0.45
    min_issue_duration_minutes: int = 20
    min_clear_duration_minutes: int = 20
    min_area_coverage_ratio: float = 0.15
    max_conflict_score_for_auto_issue: float = 0.55
    stale_radar_max_minutes: float = 90.0
    proxy_weight_cap: float = 0.65
    correlation_penalty: float = 0.12
    window_minutes: int = 120
    min_truth_coverage: float = 0.6
    memory_min_samples: int = 24
    radar_provider_priority: str = "local_grid,nowcoast,rainviewer"
    radar_grid_file: str = "data/radar_grids/latest.json"
    radar_grid_max_distance_km: float = 180.0
    active_policy_version: str = "policy.national.v1"
    feature_version: str = "feature_schema.v1"
    model_version: str = "multi_agent_rules.v3"
    metadata_store: str = "sqlite"
    replay_root: str = "runs"
    llm_agent_enabled: bool = False
    llm_agent_mode: str = "shadow"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_sec: float = 20.0
    llm_max_output_tokens: int = 900
    llm_temperature: float = 0.1


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def load_settings(profile: str | None = None) -> RuntimeSettings:
    p = profile or os.getenv("AGENT_PROFILE", "dev")
    cfg_path = Path("config") / f"{p}.json"
    settings = RuntimeSettings(profile=p)
    if cfg_path.exists():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                if hasattr(settings, k):
                    setattr(settings, k, v)
        except Exception:
            pass
    settings.llm_agent_enabled = _env_bool("AGENT_LLM_ENABLED", settings.llm_agent_enabled)
    settings.llm_agent_mode = os.getenv("AGENT_LLM_MODE", settings.llm_agent_mode)
    settings.llm_provider = os.getenv("AGENT_LLM_PROVIDER", settings.llm_provider)
    settings.llm_model = os.getenv("AGENT_LLM_MODEL", settings.llm_model)
    settings.llm_timeout_sec = _env_float("AGENT_LLM_TIMEOUT_SEC", settings.llm_timeout_sec)
    settings.llm_max_output_tokens = _env_int("AGENT_LLM_MAX_OUTPUT_TOKENS", settings.llm_max_output_tokens)
    settings.llm_temperature = _env_float("AGENT_LLM_TEMPERATURE", settings.llm_temperature)
    return settings
