from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from weather_agent.adapters.area_resolver import resolve_affected_area
from weather_agent.adapters.live_snapshot import build_live_envelope, build_live_observation
from weather_agent.adapters.open_meteo import geocode_city
from weather_agent.adapters.rainviewer import fetch_weather_maps_manifest, latest_radar_frame
from weather_agent.evaluator import evaluate_recent
from weather_agent.evolver import optimize_agent_weights
from weather_agent.memory import MemoryManager
from weather_agent.migrations import run_all_migrations
from weather_agent.models import CycleResult, Observation, ObservationEnvelope, PolicySnapshot
from weather_agent.object_engine import HazardObjectEngine
from weather_agent.orchestrator import ForecastOrchestrator, OrchestratorConfig
from weather_agent.planner import build_explicit_plan
from weather_agent.policy_engine import PolicyManager
from weather_agent.providers import default_provider_registry
from weather_agent.registry import ExperimentRegistry, RunMeta
from weather_agent.replay import ReplayStore
from weather_agent.serialize import cycle_to_dict
from weather_agent.settings import load_settings
from weather_agent.storage import ManifestObjectStore, ObjectRepository
from weather_agent.truth_factory import TruthFactory
from weather_agent.truth_labels import TruthConfig, build_truth_label_artifact
from weather_agent.window_selector import WindowScanConfig, run_window_scan


API_TOKEN = os.getenv("AGENT_API_TOKEN", "agent-dev-token")
ENABLE_AUTH = os.getenv("AGENT_ENABLE_AUTH", "1") != "0"
CORS_ORIGINS = [s.strip() for s in os.getenv("AGENT_CORS_ORIGINS", "*").split(",") if s.strip()]
SETTINGS = load_settings()
os.environ.setdefault("AGENT_RADAR_PROVIDER_PRIORITY", str(SETTINGS.radar_provider_priority))
os.environ.setdefault("AGENT_RADAR_GRID_FILE", str(SETTINGS.radar_grid_file))
os.environ.setdefault("AGENT_RADAR_GRID_MAX_DISTANCE_KM", str(SETTINGS.radar_grid_max_distance_km))


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, detail: Any = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


def _request_id(req: Request) -> str:
    return getattr(req.state, "request_id", str(uuid.uuid4()))


def _trace_id(req: Request) -> str:
    return getattr(req.state, "trace_id", _request_id(req))


def ok_response(req: Request, data: Any, message: str = "success") -> JSONResponse:
    return JSONResponse(
        {
            "code": "OK",
            "message": message,
            "request_id": _request_id(req),
            "trace_id": _trace_id(req),
            "data": data,
        }
    )


def err_response(req: Request, code: str, message: str, status_code: int, detail: Any = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "code": code,
            "message": message,
            "request_id": _request_id(req),
            "trace_id": _trace_id(req),
            "detail": detail,
        },
    )


async def auth_dependency(authorization: Optional[str] = Header(default=None)) -> None:
    if not ENABLE_AUTH:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise AppError("AUTH_001", "Missing Bearer token", status_code=401)
    token = authorization.split(" ", 1)[1].strip()
    if token != API_TOKEN:
        raise AppError("AUTH_002", "Invalid token", status_code=401)


class ObservationPayload(BaseModel):
    vertical_velocity: float
    low_level_convergence: float = Field(ge=0.0, le=1.0)
    cape: float = Field(ge=0.0)
    dcape: float = Field(ge=0.0)
    shear_0_6km: float = Field(ge=0.0)
    t850_500: float = Field(default=24.0)
    wbz_km: float = Field(ge=0.0)
    humidity_low: float = Field(ge=0.0, le=1.0)
    radar_dbz_max: float = Field(ge=0.0)
    radar_bow_echo: bool = False
    storm_motion_ms: float = Field(ge=0.0)
    prob_guidance: Dict[str, float] = Field(default_factory=dict)


class ManualForecastRequest(BaseModel):
    city: str = "Tianjin"
    area: Optional[str] = None
    auto_area: bool = False
    timestamp: datetime | None = None
    min_issue_prob: float = Field(default=float(SETTINGS.issue_threshold), ge=0.0, le=1.0)
    window_minutes: int = Field(default=int(SETTINGS.window_minutes), ge=10, le=360)
    save_run: bool = False
    observation: ObservationPayload


class LiveForecastRequest(BaseModel):
    city: str = "Tianjin"
    area: Optional[str] = None
    auto_area: bool = True
    min_issue_prob: float = Field(default=float(SETTINGS.issue_threshold), ge=0.0, le=1.0)
    window_minutes: int = Field(default=int(SETTINGS.window_minutes), ge=10, le=360)
    save_run: bool = True


class EvaluateRequest(BaseModel):
    city: str = "Tianjin"
    days: int = Field(default=3, ge=1, le=14)
    force_rebuild_truth: bool = False
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None  # YYYY-MM-DD
    truth_policy: str = "prefer"  # prefer | require | off
    min_truth_coverage: float = Field(default=float(SETTINGS.min_truth_coverage), ge=0.0, le=1.0)
    min_total_positive_labels: int = Field(default=1, ge=0, le=100000)
    headline_tiers: list[str] = Field(default_factory=lambda: ["gold", "silver"])


class EvolveRequest(BaseModel):
    city: str = "Tianjin"
    days: int = Field(default=5, ge=2, le=30)
    force_rebuild_truth: bool = False
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None  # YYYY-MM-DD
    truth_policy: str = "prefer"  # prefer | require | off
    min_truth_coverage: float = Field(default=float(SETTINGS.min_truth_coverage), ge=0.0, le=1.0)
    min_total_positive_labels: int = Field(default=1, ge=0, le=100000)
    min_train_samples: int = Field(default=24, ge=1, le=100000)
    min_calibration_samples: int = Field(default=16, ge=1, le=100000)
    calibrator_method: str = "histogram"  # histogram | beta
    headline_tiers: list[str] = Field(default_factory=lambda: ["gold", "silver"])


class TruthBuildRequest(BaseModel):
    city: str = "Tianjin"
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    force_rebuild: bool = False


class ReplayRequest(BaseModel):
    run_id: Optional[str] = None


class ReplayBundleRequest(BaseModel):
    bundle_id: str


class CompareRunsRequest(BaseModel):
    baseline_run_id: str
    enhanced_run_id: str


class WindowScanRequest(BaseModel):
    city: str = "Tianjin"
    search_start: str
    search_end: str
    window_days: int = Field(default=3, ge=1, le=15)
    step_days: int = Field(default=1, ge=1, le=7)
    min_truth_coverage: float = Field(default=float(SETTINGS.min_truth_coverage), ge=0.0, le=1.0)
    min_total_positive_labels: int = Field(default=5, ge=0, le=100000)
    min_train_positive_labels: int = Field(default=1, ge=0, le=100000)
    min_calibration_positive_labels: int = Field(default=1, ge=0, le=100000)
    headline_tiers: list[str] = Field(default_factory=lambda: ["gold", "silver"])
    top_k: int = Field(default=10, ge=1, le=100)
    force_rebuild_truth: bool = False


class BatchInferRequest(BaseModel):
    area: Optional[str] = None
    min_issue_prob: float = Field(default=float(SETTINGS.issue_threshold), ge=0.0, le=1.0)
    window_minutes: int = Field(default=int(SETTINGS.window_minutes), ge=10, le=360)
    save_run: bool = False
    observations: list[ManualForecastRequest]


class PolicyValidateRequest(BaseModel):
    policy_version: str = "policy.national.v1"
    issue_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    clear_threshold: float = Field(default=0.28, ge=0.0, le=1.0)
    lower_bound_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    min_duration: int = Field(default=2, ge=1, le=24)
    min_area: float = Field(default=120.0, ge=0.0)
    max_proxy_share: float = Field(default=0.35, ge=0.0, le=1.0)
    required_independent_families: int = Field(default=3, ge=1, le=8)


class PolicyActivateRequest(BaseModel):
    policy_version: str


class TruthValidateRequest(BaseModel):
    truth_version: str
    headline_tier: str = "gold"


class TruthCompareRequest(BaseModel):
    left_truth_version: str
    right_truth_version: str


class ReplayBundleCompareRequest(BaseModel):
    left_bundle_id: str
    right_bundle_id: str


class LLMRuntimeRequest(BaseModel):
    mode: str = "off"  # off | shadow | assist
    model: Optional[str] = None
    provider: Optional[str] = None
    timeout_sec: Optional[float] = Field(default=None, ge=1.0, le=120.0)
    max_output_tokens: Optional[int] = Field(default=None, ge=64, le=4096)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    api_key: Optional[str] = None
    clear_api_key: bool = False


API_VERSION = "0.3.1"

app = FastAPI(title="Severe Convection Multi-Agent API", version=API_VERSION)

try:
    _MIGRATION_STATUS = run_all_migrations(create_backup=False)
except Exception:
    _MIGRATION_STATUS = {"error": "migration_failed"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

web_dir = Path(__file__).resolve().parent.parent / "web"
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")


@app.get("/", include_in_schema=False)
def root_page():
    if not web_dir.exists():
        raise AppError("WEB_404", "Web assets not found", 404)
    return FileResponse(web_dir / "index.html")


@app.get("/web/", include_in_schema=False)
def web_index_page():
    if not web_dir.exists():
        raise AppError("WEB_404", "Web assets not found", 404)
    return FileResponse(web_dir / "index.html")


@app.get("/web/ops", include_in_schema=False)
def web_ops_page():
    if not web_dir.exists():
        raise AppError("WEB_404", "Web assets not found", 404)
    return FileResponse(web_dir / "ops.html")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    request.state.trace_id = request.state.request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Trace-ID"] = request.state.trace_id
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return err_response(request, exc.code, exc.message, exc.status_code, exc.detail)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return err_response(request, "REQ_001", "Invalid request payload", 422, exc.errors())


@app.exception_handler(Exception)
async def general_handler(request: Request, exc: Exception):
    return err_response(request, "SYS_001", "Internal server error", 500, str(exc))


@app.get("/")
def root():
    index = web_dir / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"message": "UI not found"}, status_code=404)


@app.get("/health")
def health():
    return {"status": "ok", "service": "weather-agent-api", "version": API_VERSION}


def _llm_runtime_status() -> dict[str, Any]:
    provider = str(SETTINGS.llm_provider or "openai").strip().lower()
    key_env = "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY"
    key_present = bool((os.getenv(key_env) or "").strip())
    enabled = bool(SETTINGS.llm_agent_enabled)
    configured_mode = str(SETTINGS.llm_agent_mode or "shadow").strip().lower()
    effective_mode = configured_mode if enabled else "off"
    supported_provider = provider in {"openai", "deepseek"}
    ready = enabled and supported_provider and key_present
    if not enabled:
        status = "disabled"
        note = "LLM evidence agent is turned off."
    elif not supported_provider:
        status = "unsupported_provider"
        note = f"Provider '{provider}' is not wired in this runtime."
    elif not key_present:
        status = "key_missing"
        note = f"{key_env} is missing, so the LLM agent cannot be activated."
    else:
        status = "ready"
        note = "LLM evidence agent is ready and will join the runtime as configured."
    return {
        "enabled": enabled,
        "configured_mode": configured_mode,
        "effective_mode": effective_mode,
        "provider": provider,
        "model": str(SETTINGS.llm_model or "gpt-4o-mini"),
        "timeout_sec": float(SETTINGS.llm_timeout_sec),
        "max_output_tokens": int(SETTINGS.llm_max_output_tokens),
        "temperature": float(SETTINGS.llm_temperature),
        "key_present": key_present,
        "ready": ready,
        "status": status,
        "note": note,
    }


def _apply_llm_runtime(body: LLMRuntimeRequest) -> dict[str, Any]:
    mode = str(body.mode or "off").strip().lower()
    if mode not in {"off", "shadow", "assist"}:
        raise AppError("REQ_006", "mode must be one of: off, shadow, assist", 422)

    provider = str(body.provider or SETTINGS.llm_provider or "openai").strip().lower()
    SETTINGS.llm_provider = provider
    if body.model:
        SETTINGS.llm_model = str(body.model).strip()
    if body.timeout_sec is not None:
        SETTINGS.llm_timeout_sec = float(body.timeout_sec)
    if body.max_output_tokens is not None:
        SETTINGS.llm_max_output_tokens = int(body.max_output_tokens)
    if body.temperature is not None:
        SETTINGS.llm_temperature = float(body.temperature)

    if body.clear_api_key:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("DEEPSEEK_API_KEY", None)
    elif body.api_key and str(body.api_key).strip():
        if provider == "deepseek":
            os.environ["DEEPSEEK_API_KEY"] = str(body.api_key).strip()
        else:
            os.environ["OPENAI_API_KEY"] = str(body.api_key).strip()

    SETTINGS.llm_agent_enabled = mode != "off"
    if mode in {"shadow", "assist"}:
        SETTINGS.llm_agent_mode = mode
    return _llm_runtime_status()


@app.get("/api/v1/health")
def api_health(req: Request):
    llm = _llm_runtime_status()
    return ok_response(
        req,
        {
            "status": "ok",
            "auth_enabled": ENABLE_AUTH,
            "version": API_VERSION,
            "migrations": _MIGRATION_STATUS,
            "capabilities": {
                "window_scan": True,
                "probability_quality_factor": True,
                "hazard_prob_raw": True,
                "observation_envelope": True,
                "hazard_object_engine": True,
                "policy_engine": True,
                "truth_factory": True,
                "replay_bundle": True,
                "llm_evidence_agent": True,
                "llm_agent_enabled": bool(llm["enabled"]),
                "llm_agent_mode": str(llm["effective_mode"]),
                "llm_key_present": bool(llm["key_present"]),
                "llm_ready": bool(llm["ready"]),
                "llm_model": str(llm["model"]),
                "llm_provider": str(llm["provider"]),
            },
        },
    )


@app.get("/api/v1/runtime/llm")
def llm_runtime_get(req: Request, _: None = Depends(auth_dependency)):
    return ok_response(req, _llm_runtime_status())


@app.post("/api/v1/runtime/llm")
def llm_runtime_set(req: Request, body: LLMRuntimeRequest, _: None = Depends(auth_dependency)):
    return ok_response(req, _apply_llm_runtime(body), message="llm runtime updated")


@app.get("/api/v1/geo/city")
def geo_city(req: Request, city: str, _: None = Depends(auth_dependency)):
    try:
        geo = geocode_city(city)
    except Exception as e:
        raise AppError("DATA_002", "Failed to geocode city", 422, str(e))

    data = {
        "input_city": city,
        "name": geo.get("name", city),
        "country": geo.get("country"),
        "admin1": geo.get("admin1"),
        "latitude": geo.get("latitude"),
        "longitude": geo.get("longitude"),
        "timezone": geo.get("timezone"),
    }
    return ok_response(req, data)


@app.get("/api/v1/tiles/radar/{z}/{x}/{y}.png", include_in_schema=False)
def radar_tile_proxy(z: int, x: int, y: int, frame_path: str = ""):
    try:
        manifest = fetch_weather_maps_manifest(timeout=12)
    except Exception as e:
        raise AppError("DATA_004", "Failed to fetch radar map manifest", 503, str(e))

    host = str(manifest.get("host", "") or "")
    path = (frame_path or "").strip()
    if not path:
        frame = latest_radar_frame(manifest)
        path = str(frame.get("path", "") or "").strip()
    if not host or not path:
        raise AppError("DATA_005", "Radar tile frame unavailable", 404)

    upstream = f"{host}{path}/256/{z}/{x}/{y}/2/1_1.png"
    try:
        tile = requests.get(upstream, timeout=20)
        tile.raise_for_status()
    except Exception as e:
        raise AppError("DATA_006", "Failed to fetch upstream radar tile", 502, str(e))

    headers = {
        "Cache-Control": "public, max-age=120",
        "X-Radar-Upstream": upstream,
        "Access-Control-Allow-Origin": "*",
    }
    return Response(content=tile.content, media_type=tile.headers.get("content-type", "image/png"), headers=headers)


def _run_orchestrator(
    obs: Observation | ObservationEnvelope,
    area: str,
    min_issue_prob: float,
    window_minutes: int,
    city_for_memory: str,
    request_id: str = "",
    trace_id: str = "",
) -> tuple[dict, CycleResult, CycleResult]:
    memory = MemoryManager(min_sample_threshold=int(SETTINGS.memory_min_samples))
    legacy_obs = obs.to_observation() if isinstance(obs, ObservationEnvelope) else obs
    month = legacy_obs.timestamp.month
    learned_weights = memory.load_weights(city_for_memory, month)
    learned_calibrators = memory.load_calibrators(city_for_memory, month)

    baseline = ForecastOrchestrator(
        OrchestratorConfig(
            issue_threshold=min_issue_prob,
            clear_threshold=float(SETTINGS.clear_threshold),
            min_readiness_score=float(SETTINGS.min_readiness_score),
            min_issue_duration_minutes=int(SETTINGS.min_issue_duration_minutes),
            min_clear_duration_minutes=int(SETTINGS.min_clear_duration_minutes),
            min_area_coverage_ratio=float(SETTINGS.min_area_coverage_ratio),
            max_conflict_score_for_auto_issue=float(SETTINGS.max_conflict_score_for_auto_issue),
            stale_radar_max_minutes=float(SETTINGS.stale_radar_max_minutes),
            window_minutes=window_minutes,
            region_name=area,
            agent_weights=None,
            proxy_weight_cap=float(SETTINGS.proxy_weight_cap),
            correlation_penalty=float(SETTINGS.correlation_penalty),
            policy_version=str(SETTINGS.active_policy_version),
            feature_version=str(SETTINGS.feature_version),
            model_version=str(SETTINGS.model_version),
            enable_llm_agent=bool(SETTINGS.llm_agent_enabled),
            llm_agent_mode=str(SETTINGS.llm_agent_mode),
            llm_provider=str(SETTINGS.llm_provider),
            llm_model=str(SETTINGS.llm_model),
            llm_timeout_sec=float(SETTINGS.llm_timeout_sec),
            llm_max_output_tokens=int(SETTINGS.llm_max_output_tokens),
            llm_temperature=float(SETTINGS.llm_temperature),
        )
    )
    enhanced = ForecastOrchestrator(
        OrchestratorConfig(
            issue_threshold=min_issue_prob,
            clear_threshold=float(SETTINGS.clear_threshold),
            min_readiness_score=float(SETTINGS.min_readiness_score),
            min_issue_duration_minutes=int(SETTINGS.min_issue_duration_minutes),
            min_clear_duration_minutes=int(SETTINGS.min_clear_duration_minutes),
            min_area_coverage_ratio=float(SETTINGS.min_area_coverage_ratio),
            max_conflict_score_for_auto_issue=float(SETTINGS.max_conflict_score_for_auto_issue),
            stale_radar_max_minutes=float(SETTINGS.stale_radar_max_minutes),
            window_minutes=window_minutes,
            region_name=area,
            agent_weights=learned_weights,
            probability_calibrators=learned_calibrators,
            proxy_weight_cap=float(SETTINGS.proxy_weight_cap),
            correlation_penalty=float(SETTINGS.correlation_penalty),
            policy_version=str(SETTINGS.active_policy_version),
            feature_version=str(SETTINGS.feature_version),
            model_version=str(SETTINGS.model_version),
            enable_llm_agent=bool(SETTINGS.llm_agent_enabled),
            llm_agent_mode=str(SETTINGS.llm_agent_mode),
            llm_provider=str(SETTINGS.llm_provider),
            llm_model=str(SETTINGS.llm_model),
            llm_timeout_sec=float(SETTINGS.llm_timeout_sec),
            llm_max_output_tokens=int(SETTINGS.llm_max_output_tokens),
            llm_temperature=float(SETTINGS.llm_temperature),
        )
    )

    base_cycle = baseline.run_cycle(obs, request_id=request_id, trace_id=trace_id)
    enh_cycle = enhanced.run_cycle(obs, request_id=request_id, trace_id=trace_id)

    data = {
        "plan": build_explicit_plan(legacy_obs),
        "baseline": cycle_to_dict(base_cycle),
        "enhanced": cycle_to_dict(enh_cycle),
        "active_weights": learned_weights,
        "active_calibrators": learned_calibrators,
    }
    return data, base_cycle, enh_cycle


def _parse_optional_date(v: Optional[str]) -> Optional[datetime.date]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(v).date()
    except Exception:
        raise AppError("REQ_002", "Invalid date format, use YYYY-MM-DD", 422)


def _validate_truth_policy(v: str) -> str:
    vv = (v or "").strip().lower()
    if vv not in {"prefer", "require", "off"}:
        raise AppError("REQ_004", "truth_policy must be one of: prefer, require, off", 422)
    return vv


def _validate_headline_tiers(v: list[str]) -> list[str]:
    allowed = {"gold", "silver", "proxy"}
    out: list[str] = []
    for x in v or []:
        t = str(x).strip().lower()
        if t in allowed and t not in out:
            out.append(t)
    if not out:
        out = ["gold", "silver"]
    return out


def _validate_calibrator_method(v: str) -> str:
    method = (v or "").strip().lower()
    if method not in {"histogram", "beta"}:
        raise AppError("REQ_005", "calibrator_method must be one of: histogram, beta", 422)
    return method


@app.post("/api/v1/windows/scan")
def window_scan_api(req: Request, body: WindowScanRequest, _: None = Depends(auth_dependency)):
    try:
        start = datetime.fromisoformat(body.search_start).date()
        end = datetime.fromisoformat(body.search_end).date()
    except Exception:
        raise AppError("REQ_002", "Invalid date format, use YYYY-MM-DD", 422)
    if end < start:
        raise AppError("REQ_003", "search_end must be >= search_start", 422)

    tiers = _validate_headline_tiers(body.headline_tiers)
    cfg = WindowScanConfig(
        city=body.city,
        search_start=start,
        search_end=end,
        window_days=body.window_days,
        step_days=body.step_days,
        min_truth_coverage=body.min_truth_coverage,
        min_total_positive_labels=body.min_total_positive_labels,
        min_train_positive_labels=body.min_train_positive_labels,
        min_calibration_positive_labels=body.min_calibration_positive_labels,
        headline_tiers=tuple(tiers),
        top_k=body.top_k,
        force_rebuild_truth=body.force_rebuild_truth,
    )
    try:
        out = run_window_scan(cfg)
    except Exception as e:
        raise AppError("DATA_003", "Window scan failed", 422, str(e))
    return ok_response(req, out, message="window scan completed")


# Compatibility aliases for older frontend/client paths.
@app.post("/api/v1/window/scan")
def window_scan_api_alias_single(req: Request, body: WindowScanRequest, _: None = Depends(auth_dependency)):
    return window_scan_api(req, body, _)


@app.post("/api/v1/windows_scan")
def window_scan_api_alias_snake(req: Request, body: WindowScanRequest, _: None = Depends(auth_dependency)):
    return window_scan_api(req, body, _)


@app.post("/api/v1/scan/windows")
def window_scan_api_alias_reverse(req: Request, body: WindowScanRequest, _: None = Depends(auth_dependency)):
    return window_scan_api(req, body, _)


@app.post("/api/v1/forecast/manual")
def forecast_manual(req: Request, body: ManualForecastRequest, _: None = Depends(auth_dependency)):
    ts = body.timestamp or datetime.now()
    obs = Observation(
        timestamp=ts,
        city=body.city,
        vertical_velocity=body.observation.vertical_velocity,
        low_level_convergence=body.observation.low_level_convergence,
        cape=body.observation.cape,
        dcape=body.observation.dcape,
        shear_0_6km=body.observation.shear_0_6km,
        t850_500=body.observation.t850_500,
        wbz_km=body.observation.wbz_km,
        humidity_low=body.observation.humidity_low,
        radar_dbz_max=body.observation.radar_dbz_max,
        radar_bow_echo=body.observation.radar_bow_echo,
        storm_motion_ms=body.observation.storm_motion_ms,
        prob_guidance=body.observation.prob_guidance,
        source_meta={"mode": "manual"},
    )

    resolved_area = (body.area or "").strip() or "manual-area"
    envelope = ObservationEnvelope.from_observation(obs)
    data, base_cycle, enh_cycle = _run_orchestrator(
        envelope,
        resolved_area,
        body.min_issue_prob,
        body.window_minutes,
        body.city,
        request_id=_request_id(req),
        trace_id=_trace_id(req),
    )

    saved = None
    saved_baseline = None
    if body.save_run:
        saved_baseline = str(ReplayStore().save(base_cycle))
        saved = str(ReplayStore().save(enh_cycle))

    run_id = enh_cycle.audit.run_id if enh_cycle.audit else str(uuid.uuid4())
    reg = ExperimentRegistry()
    base_meta = reg.build_run_meta(body.model_dump(), prefix="predict")
    reg.record_predict(
        RunMeta(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            config_hash="manual_" + str(body.min_issue_prob),
            code_hash=base_meta.code_hash,
        ),
        city=body.city,
        mode="manual",
        decision=data["enhanced"]["decision"],
        data_window={"start": data["enhanced"]["decision"]["start_time"], "end": data["enhanced"]["decision"]["end_time"]},
        metadata={
            "saved_path": saved,
            "baseline_saved_path": saved_baseline,
            "request_id": _request_id(req),
            "baseline_run_id": base_cycle.audit.run_id if base_cycle.audit else None,
            "readiness": float(data["enhanced"]["decision"].get("evidence_readiness_score", 0.0)),
            "max_hazard_prob": float(max(data["enhanced"]["decision"].get("hazard_prob", {}).values() or [0.0])),
        },
    )

    data["saved_path"] = saved
    data["baseline_saved_path"] = saved_baseline
    data["run_id"] = run_id
    data["baseline_run_id"] = base_cycle.audit.run_id if base_cycle.audit else None
    data["area_resolution"] = {"status": "manual", "area_text": resolved_area}
    return ok_response(req, data)


@app.post("/api/v1/forecast/live")
def forecast_live(req: Request, body: LiveForecastRequest, _: None = Depends(auth_dependency)):
    try:
        envelope = build_live_envelope(body.city)
        obs = envelope.to_observation()
    except Exception as e:
        raise AppError("DATA_001", "Failed to fetch live weather data", 503, str(e))

    user_area = (body.area or "").strip()
    seed_area = user_area or "auto-area"
    data, base_cycle, enh_cycle = _run_orchestrator(
        envelope,
        seed_area,
        body.min_issue_prob,
        body.window_minutes,
        body.city,
        request_id=_request_id(req),
        trace_id=_trace_id(req),
    )

    area_resolution = {"status": "manual", "area_text": user_area} if user_area else None
    if body.auto_area and not user_area:
        try:
            lat = float(obs.source_meta.get("lat", "nan"))
            lon = float(obs.source_meta.get("lon", "nan"))
            area_resolution = resolve_affected_area(
                lat=lat,
                lon=lon,
                hazard_prob=data["enhanced"]["decision"]["hazard_prob"],
                city_hint=obs.city,
            )
            auto_area = str(area_resolution.get("area_text", "auto-area"))
            data["baseline"]["decision"]["affected_area"] = auto_area
            data["enhanced"]["decision"]["affected_area"] = auto_area
            base_cycle.decision.affected_area = auto_area
            enh_cycle.decision.affected_area = auto_area
        except Exception as e:
            area_resolution = {"status": "fallback", "area_text": obs.city, "reason": str(e)}
            data["baseline"]["decision"]["affected_area"] = obs.city
            data["enhanced"]["decision"]["affected_area"] = obs.city
            base_cycle.decision.affected_area = obs.city
            enh_cycle.decision.affected_area = obs.city
    elif user_area:
        data["baseline"]["decision"]["affected_area"] = user_area
        data["enhanced"]["decision"]["affected_area"] = user_area
        base_cycle.decision.affected_area = user_area
        enh_cycle.decision.affected_area = user_area

    saved = None
    saved_baseline = None
    if body.save_run:
        saved_baseline = str(ReplayStore().save(base_cycle))
        saved = str(ReplayStore().save(enh_cycle))

    run_id = enh_cycle.audit.run_id if enh_cycle.audit else str(uuid.uuid4())
    reg = ExperimentRegistry()
    live_meta = reg.build_run_meta(body.model_dump(), prefix="predict")
    reg.record_predict(
        RunMeta(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            config_hash="live_" + str(body.min_issue_prob),
            code_hash=live_meta.code_hash,
        ),
        city=body.city,
        mode="live",
        decision=data["enhanced"]["decision"],
        data_window={"start": data["enhanced"]["decision"]["start_time"], "end": data["enhanced"]["decision"]["end_time"]},
        metadata={
            "saved_path": saved,
            "baseline_saved_path": saved_baseline,
            "request_id": _request_id(req),
            "baseline_run_id": base_cycle.audit.run_id if base_cycle.audit else None,
            "area_resolution": area_resolution,
            "readiness": float(data["enhanced"]["decision"].get("evidence_readiness_score", 0.0)),
            "max_hazard_prob": float(max(data["enhanced"]["decision"].get("hazard_prob", {}).values() or [0.0])),
        },
    )

    data["saved_path"] = saved
    data["baseline_saved_path"] = saved_baseline
    data["run_id"] = run_id
    data["baseline_run_id"] = base_cycle.audit.run_id if base_cycle.audit else None
    data["area_resolution"] = area_resolution
    data["source_meta"] = obs.source_meta
    return ok_response(req, data)


@app.post("/api/v1/evaluate/recent")
def evaluate_api(req: Request, body: EvaluateRequest, _: None = Depends(auth_dependency)):
    start_date = _parse_optional_date(body.start_date)
    end_date = _parse_optional_date(body.end_date)
    if start_date and end_date and end_date < start_date:
        raise AppError("REQ_003", "end_date must be >= start_date", 422)
    truth_policy = _validate_truth_policy(body.truth_policy)
    headline_tiers = _validate_headline_tiers(body.headline_tiers)

    memory = MemoryManager(min_sample_threshold=int(SETTINGS.memory_min_samples))
    month = (end_date or datetime.now().date()).month
    weights = memory.load_weights(body.city, month)
    calibrators = memory.load_calibrators(body.city, month)
    try:
        result = evaluate_recent(
            city=body.city,
            days=body.days,
            enhanced_weights=weights,
            enhanced_calibrators=calibrators,
            force_rebuild_truth=body.force_rebuild_truth,
            start_date=start_date,
            end_date=end_date,
            truth_policy=truth_policy,
            min_truth_coverage=body.min_truth_coverage,
            min_total_positive_labels=body.min_total_positive_labels,
            headline_tiers=headline_tiers,
        )
    except RuntimeError as e:
        raise AppError("TRUTH_001", "Truth/statistical requirements not met", 422, str(e))
    reg = ExperimentRegistry()
    eval_meta = reg.build_run_meta(body.model_dump(), prefix="eval")
    reg.record_eval(eval_meta, city=body.city, result=result, metadata={"request_id": _request_id(req), "truth_policy": truth_policy})
    result["registry_run_id"] = eval_meta.run_id
    return ok_response(req, result)


@app.post("/api/v1/evolve/weights")
def evolve_api(req: Request, body: EvolveRequest, _: None = Depends(auth_dependency)):
    start_date = _parse_optional_date(body.start_date)
    end_date = _parse_optional_date(body.end_date)
    if start_date and end_date and end_date < start_date:
        raise AppError("REQ_003", "end_date must be >= start_date", 422)
    truth_policy = _validate_truth_policy(body.truth_policy)
    headline_tiers = _validate_headline_tiers(body.headline_tiers)
    calibrator_method = _validate_calibrator_method(body.calibrator_method)

    try:
        learned = optimize_agent_weights(
            city=body.city,
            days=body.days,
            start_date=start_date,
            end_date=end_date,
            truth_policy=truth_policy,
            min_truth_coverage=body.min_truth_coverage,
            min_total_positive_labels=body.min_total_positive_labels,
            force_rebuild_truth=body.force_rebuild_truth,
            min_train_samples=body.min_train_samples,
            min_calibration_samples=body.min_calibration_samples,
            calibrator_method=calibrator_method,
            headline_tiers=tuple(headline_tiers),
        )
    except RuntimeError as e:
        raise AppError("TRUTH_001", "Truth/statistical requirements not met", 422, str(e))
    mm = MemoryManager(min_sample_threshold=int(SETTINGS.memory_min_samples))
    mm.save_profile(
        city=body.city,
        month=learned["month"],
        agent_weights=learned["weights"],
        calibrators=learned.get("calibrators", {}),
        sample_count=int(learned.get("qualified_counts", {}).get("train", 0)),
        coverage_ratio=float(learned.get("truth_coverage_ratio_test", 0.0)),
        valid_window=learned.get("trained_period", {}),
        profile_version="memory.v2",
        code_version="nogit",
        shrinkage=1.0,
    )
    eval_end = end_date
    eval_start = start_date
    try:
        after = evaluate_recent(
            city=body.city,
            days=min(3, body.days),
            enhanced_weights=learned["weights"],
            enhanced_calibrators=learned.get("calibrators", {}),
            start_date=eval_start,
            end_date=eval_end,
            truth_policy=truth_policy,
            min_truth_coverage=body.min_truth_coverage,
            min_total_positive_labels=body.min_total_positive_labels,
            headline_tiers=headline_tiers,
        )
    except RuntimeError as e:
        raise AppError("TRUTH_001", "Truth/statistical requirements not met in post-eval", 422, str(e))
    reg = ExperimentRegistry()
    evolve_meta = reg.build_run_meta(body.model_dump(), prefix="evolve")
    reg.record_evolve(
        evolve_meta,
        city=body.city,
        result=learned,
        metadata={"request_id": _request_id(req)},
    )
    return ok_response(
        req,
        {"learned": learned, "post_eval": after, "registry_run_id": evolve_meta.run_id},
        message="weights updated",
    )


@app.get("/api/v1/audit/{run_id}")
def audit_get_api(req: Request, run_id: str, _: None = Depends(auth_dependency)):
    rs = ReplayStore()
    audit = rs.load_audit(run_id)
    if not audit:
        raise AppError("AUDIT_404", "Audit record not found", 404, {"run_id": run_id})
    return ok_response(req, audit)


@app.post("/api/v1/replay/case")
def replay_case_api(req: Request, body: ReplayRequest, _: None = Depends(auth_dependency)):
    rs = ReplayStore()
    run_id = body.run_id or rs.latest_run_id()
    if not run_id:
        raise AppError("REPLAY_404", "No replay run found", 404)
    cycle = rs.load_cycle(run_id)
    if not cycle:
        raise AppError("REPLAY_404", "Replay run not found", 404, {"run_id": run_id})
    audit = rs.load_audit(run_id)
    return ok_response(req, {"run_id": run_id, "cycle": cycle, "audit": audit})


@app.post("/api/v1/replay/compare")
def replay_compare_api(req: Request, body: CompareRunsRequest, _: None = Depends(auth_dependency)):
    rs = ReplayStore()
    try:
        cmp = rs.compare_runs(body.baseline_run_id, body.enhanced_run_id)
    except FileNotFoundError as e:
        raise AppError("REPLAY_404", "Run not found", 404, str(e))
    return ok_response(req, cmp)


@app.get("/api/v1/registry/eval/{run_id}")
def registry_eval_get(req: Request, run_id: str, _: None = Depends(auth_dependency)):
    reg = ExperimentRegistry()
    row = reg.fetch_eval_run(run_id)
    if not row:
        raise AppError("REG_404", "Eval run not found", 404, {"run_id": run_id})
    return ok_response(req, row)


@app.get("/api/v1/registry/evolve/{run_id}")
def registry_evolve_get(req: Request, run_id: str, _: None = Depends(auth_dependency)):
    reg = ExperimentRegistry()
    row = reg.fetch_evolve_run(run_id)
    if not row:
        raise AppError("REG_404", "Evolve run not found", 404, {"run_id": run_id})
    return ok_response(req, row)


@app.get("/api/v1/registry/predict/recent")
def registry_predict_recent(req: Request, limit: int = 3, city: Optional[str] = None, _: None = Depends(auth_dependency)):
    reg = ExperimentRegistry()
    rows = reg.fetch_recent_predict_runs(limit=limit, city=city)
    data = []
    for r in rows:
        meta = r.get("metadata") or {}
        data.append(
            {
                "run_id": r.get("run_id"),
                "created_at": r.get("created_at"),
                "city": r.get("city"),
                "mode": r.get("mode"),
                "action": r.get("action"),
                "level": r.get("level"),
                "issue": bool(r.get("issue")),
                "degraded_mode": bool(r.get("degraded_mode")),
                "readiness": float(meta.get("readiness", 0.0)),
                "max_hazard_prob": float(meta.get("max_hazard_prob", 0.0)),
            }
        )
    return ok_response(req, {"items": data, "count": len(data)})


@app.post("/api/v1/infer/realtime")
def infer_realtime(req: Request, body: LiveForecastRequest, _: None = Depends(auth_dependency)):
    return forecast_live(req, body, _)


@app.post("/api/v1/infer/object")
def infer_object(req: Request, body: LiveForecastRequest, _: None = Depends(auth_dependency)):
    res = forecast_live(req, body, _)
    payload = res.body.decode("utf-8")
    data = json.loads(payload)
    enhanced = data.get("data", {}).get("enhanced", {})
    return ok_response(
        req,
        {
            "request_id": _request_id(req),
            "trace_id": _trace_id(req),
            "decision": enhanced.get("decision"),
            "hazard_object": enhanced.get("hazard_object"),
            "decision_packet": enhanced.get("decision_packet"),
            "fusion_result": enhanced.get("fusion_result"),
        },
    )


@app.post("/api/v1/infer/batch")
def infer_batch(req: Request, body: BatchInferRequest, _: None = Depends(auth_dependency)):
    items = []
    for item in body.observations:
        item_data, _, _ = _run_orchestrator(
            ObservationEnvelope.from_observation(
                Observation(
                    timestamp=item.timestamp or datetime.now(),
                    city=item.city,
                    vertical_velocity=item.observation.vertical_velocity,
                    low_level_convergence=item.observation.low_level_convergence,
                    cape=item.observation.cape,
                    dcape=item.observation.dcape,
                    shear_0_6km=item.observation.shear_0_6km,
                    t850_500=item.observation.t850_500,
                    wbz_km=item.observation.wbz_km,
                    humidity_low=item.observation.humidity_low,
                    radar_dbz_max=item.observation.radar_dbz_max,
                    radar_bow_echo=item.observation.radar_bow_echo,
                    storm_motion_ms=item.observation.storm_motion_ms,
                    prob_guidance=item.observation.prob_guidance,
                    source_meta={"mode": "batch"},
                )
            ),
            (item.area or body.area or "batch-area"),
            body.min_issue_prob,
            body.window_minutes,
            item.city,
            request_id=_request_id(req),
            trace_id=_trace_id(req),
        )
        items.append(item_data["enhanced"])
    return ok_response(req, {"items": items, "count": len(items)})


@app.get("/api/v1/ingest/providers")
def ingest_providers(req: Request, _: None = Depends(auth_dependency)):
    reg = default_provider_registry()
    data = reg.list_definitions()
    ExperimentRegistry().registry.register("providers", "provider_runtime_catalog.v1", {"provider_version": "provider_runtime_catalog.v1", "items": data})
    return ok_response(req, {"items": data, "count": len(data)})


@app.get("/api/v1/ingest/health")
def ingest_health(req: Request, _: None = Depends(auth_dependency)):
    rs = ReplayStore()
    latest_run = rs.latest_run_id()
    latest = rs.load_audit(latest_run) if latest_run else None
    return ok_response(
        req,
        {
            "latest_run_id": latest_run,
            "source_health": (latest or {}).get("source_health", {}),
            "providers": default_provider_registry().list_definitions(),
        },
    )


@app.get("/api/v1/data/snapshots")
def data_snapshots(req: Request, _: None = Depends(auth_dependency)):
    store = ManifestObjectStore()
    truth_versions = ExperimentRegistry().list_truth()
    bundles = [str(p.parent.name) for p in (Path("runs") / "replay_bundles").rglob("bundle.json")]
    return ok_response(req, {"truth_versions": truth_versions, "replay_bundles": bundles, "object_store_root": str(store.root)})


@app.post("/api/v1/truth/build")
def truth_build_api(req: Request, body: TruthBuildRequest, _: None = Depends(auth_dependency)):
    try:
        start = datetime.fromisoformat(body.start_date).date()
        end = datetime.fromisoformat(body.end_date).date()
    except Exception:
        raise AppError("REQ_002", "Invalid date format, use YYYY-MM-DD", 422)
    if end < start:
        raise AppError("REQ_003", "end_date must be >= start_date", 422)

    art = TruthFactory().publish_truth_snapshot(
        city=body.city,
        start_date=start,
        end_date=end,
        cfg=TruthConfig(),
        force_rebuild=body.force_rebuild,
    )
    return ok_response(req, art, message="truth labels built")


@app.get("/api/v1/truth/versions")
def truth_versions_api(req: Request, _: None = Depends(auth_dependency)):
    data = TruthFactory().list_versions()
    return ok_response(req, {"items": data, "count": len(data)})


@app.get("/api/v1/truth/versions/{truth_version}")
def truth_version_get_api(req: Request, truth_version: str, _: None = Depends(auth_dependency)):
    data = TruthFactory().get_version(truth_version)
    if not data:
        raise AppError("TRUTH_404", "Truth version not found", 404, {"truth_version": truth_version})
    return ok_response(req, data)


@app.post("/api/v1/truth/validate")
def truth_validate_api(req: Request, body: TruthValidateRequest, _: None = Depends(auth_dependency)):
    out = TruthFactory().validate_truth_version(body.truth_version, headline_tier=body.headline_tier)
    if not out["ok"]:
        raise AppError("TRUTH_002", "Truth validation failed", 422, out)
    return ok_response(req, out)


@app.post("/api/v1/truth/compare")
def truth_compare_api(req: Request, body: TruthCompareRequest, _: None = Depends(auth_dependency)):
    try:
        out = TruthFactory().compare_versions(body.left_truth_version, body.right_truth_version)
    except FileNotFoundError as e:
        raise AppError("TRUTH_404", "Truth version not found", 404, str(e))
    return ok_response(req, out)


@app.post("/api/v1/eval/run")
def eval_run_api(req: Request, body: EvaluateRequest, _: None = Depends(auth_dependency)):
    return evaluate_api(req, body, _)


@app.get("/api/v1/eval/report/{run_id}")
def eval_report_api(req: Request, run_id: str, _: None = Depends(auth_dependency)):
    return registry_eval_get(req, run_id, _)


@app.post("/api/v1/evolve/run")
def evolve_run_api(req: Request, body: EvolveRequest, _: None = Depends(auth_dependency)):
    return evolve_api(req, body, _)


@app.post("/api/v1/policy/validate")
def policy_validate_api(req: Request, body: PolicyValidateRequest, _: None = Depends(auth_dependency)):
    snapshot = PolicySnapshot(
        policy_version=body.policy_version,
        issue_threshold=body.issue_threshold,
        clear_threshold=body.clear_threshold,
        lower_bound_threshold=body.lower_bound_threshold,
        min_duration=body.min_duration,
        min_area=body.min_area,
        max_proxy_share=body.max_proxy_share,
        required_independent_families=body.required_independent_families,
    )
    mgr = PolicyManager()
    issues = mgr.validate(snapshot)
    if issues:
        raise AppError("POLICY_001", "Policy validation failed", 422, {"issues": issues})
    mgr.register(snapshot)
    return ok_response(req, {"policy_version": snapshot.policy_version, "issues": []}, message="policy validated")


@app.post("/api/v1/policy/activate")
def policy_activate_api(req: Request, body: PolicyActivateRequest, _: None = Depends(auth_dependency)):
    mgr = PolicyManager()
    if not any(x.get("name") == body.policy_version for x in mgr.list()):
        raise AppError("POLICY_404", "Policy version not found", 404, {"policy_version": body.policy_version})
    mgr.activate(body.policy_version)
    return ok_response(req, {"policy_version": body.policy_version, "status": "active"})


@app.get("/api/v1/audit/trace/{request_id}")
def audit_trace_api(req: Request, request_id: str, _: None = Depends(auth_dependency)):
    trace = ReplayStore().trace(request_id)
    if not trace:
        raise AppError("AUDIT_404", "Audit trace not found", 404, {"request_id": request_id})
    return ok_response(req, trace)


@app.post("/api/v1/replay/run")
def replay_run_api(req: Request, body: ReplayBundleRequest, _: None = Depends(auth_dependency)):
    rs = ReplayStore()
    try:
        bundle = rs.replay_bundle(body.bundle_id)
    except FileNotFoundError:
        raise AppError("REPLAY_404", "Replay bundle not found", 404, {"bundle_id": body.bundle_id})
    result = rs.compare_bundle(body.bundle_id, bundle.get("cycle", {}))
    return ok_response(req, {"bundle": bundle, "verification": result})


@app.get("/api/v1/replay/bundles")
def replay_bundles_api(req: Request, limit: int = 20, _: None = Depends(auth_dependency)):
    items = ReplayStore().list_bundles(limit=limit)
    return ok_response(req, {"items": items, "count": len(items)})


@app.post("/api/v1/replay/compare-bundles")
def replay_compare_bundles_api(req: Request, body: ReplayBundleCompareRequest, _: None = Depends(auth_dependency)):
    rs = ReplayStore()
    try:
        left = rs.replay_bundle(body.left_bundle_id)
        right = rs.replay_bundle(body.right_bundle_id)
    except FileNotFoundError as e:
        raise AppError("REPLAY_404", "Replay bundle not found", 404, str(e))
    out = rs.compare_bundle(body.left_bundle_id, right.get("cycle", {}))
    return ok_response(
        req,
        {
            "left_bundle_id": body.left_bundle_id,
            "right_bundle_id": body.right_bundle_id,
            "comparison": out,
            "left": left,
            "right": right,
        },
    )


@app.get("/api/v1/registry/models")
def registry_models(req: Request, _: None = Depends(auth_dependency)):
    return ok_response(req, {"items": ExperimentRegistry().list_models()})


@app.get("/api/v1/registry/policies")
def registry_policies(req: Request, _: None = Depends(auth_dependency)):
    return ok_response(req, {"items": ExperimentRegistry().list_policies()})


@app.get("/api/v1/registry/features")
def registry_features(req: Request, _: None = Depends(auth_dependency)):
    return ok_response(req, {"items": ExperimentRegistry().list_features()})


@app.get("/api/v1/registry/truth")
def registry_truth(req: Request, _: None = Depends(auth_dependency)):
    return ok_response(req, {"items": ExperimentRegistry().list_truth()})


@app.get("/api/v1/objects/active")
def objects_active(req: Request, _: None = Depends(auth_dependency)):
    repo = ObjectRepository()
    return ok_response(req, {"items": repo.active(), "count": len(repo.active())})


@app.get("/api/v1/objects/{object_id}/history")
def object_history(req: Request, object_id: str, _: None = Depends(auth_dependency)):
    repo = ObjectRepository()
    history = repo.history(object_id)
    if not history:
        raise AppError("OBJ_404", "Object history not found", 404, {"object_id": object_id})
    return ok_response(req, {"object_id": object_id, "history": history})


@app.get("/api/v1/objects/{object_id}")
def object_get(req: Request, object_id: str, _: None = Depends(auth_dependency)):
    repo = ObjectRepository()
    obj = repo.latest(object_id)
    if not obj:
        raise AppError("OBJ_404", "Object not found", 404, {"object_id": object_id})
    return ok_response(req, obj)
