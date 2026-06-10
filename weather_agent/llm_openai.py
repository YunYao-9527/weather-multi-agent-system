from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal

import requests
from pydantic import BaseModel, Field

from weather_agent.models import ObservationEnvelope


class HazardScorePayload(BaseModel):
    short_rain: float = Field(ge=0.0, le=1.0)
    wind: float = Field(ge=0.0, le=1.0)
    hail: float = Field(ge=0.0, le=1.0)
    tornado: float = Field(ge=0.0, le=1.0)


class LLMSituationEvidencePayload(BaseModel):
    claim: str = Field(min_length=8, max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)
    hazard_scores: HazardScorePayload
    key_factors: List[str] = Field(default_factory=list)
    caution_flags: List[str] = Field(default_factory=list)
    action_bias: Literal["monitor", "manual_review", "support_issue", "support_clear"] = "monitor"


def _extract_output_text(payload: Dict[str, Any]) -> str:
    text = payload.get("output_text")
    if isinstance(text, str) and text.strip():
        return text
    chunks: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks).strip()


def _extract_chat_message_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices", []) or []
    if not choices:
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _compact_envelope(envelope: ObservationEnvelope) -> Dict[str, Any]:
    env = dict(envelope.feature_planes.get("environment", {}))
    radar = dict(envelope.feature_planes.get("radar", {}))
    guidance = dict(envelope.feature_planes.get("guidance", {}))
    radar.pop("provider_snapshot", None)
    guidance.pop("guidance_snapshot", None)
    guidance.pop("proxy_guidance_snapshot", None)
    return {
        "analysis_time": envelope.analysis_time,
        "valid_time": envelope.valid_time,
        "lead_time": envelope.lead_time,
        "domain_id": envelope.domain_id,
        "grid_id": envelope.grid_spec.grid_id,
        "object_context": dict(envelope.object_context),
        "source_tiers_summary": dict(envelope.source_tiers_summary),
        "qc_summary": dict(envelope.qc_summary),
        "environment": env,
        "radar": radar,
        "guidance": guidance,
        "providers": [
            {
                "source_id": p.source_id,
                "source_family": p.source_family,
                "source_tier": p.source_tier,
                "provider_version": p.provider_version,
                "freshness_sec": p.freshness_sec,
                "coverage_ratio": float((p.coverage or {}).get("coverage_ratio", 0.0) or 0.0),
                "status": p.status,
                "qc_flags": list(p.qc_flags),
            }
            for p in envelope.source_registry
        ],
    }


def _build_llm_prompts(summary: Dict[str, Any], provider: str) -> tuple[str, str]:
    provider = str(provider or "openai").strip().lower()
    system_prompt = (
        "You are a conservative severe convection analysis assistant. "
        "Return exactly one JSON object and no markdown. "
        "Do not echo the envelope. "
        "The JSON object must contain exactly these top-level keys: "
        "claim, confidence, hazard_scores, key_factors, caution_flags, action_bias. "
        "hazard_scores must contain exactly these keys: short_rain, wind, hail, tornado. "
        "confidence and all hazard_scores must be numbers in [0,1]. "
        "action_bias must be one of: monitor, manual_review, support_issue, support_clear. "
        "If source quality is degraded, proxy share is elevated, or conflicts exist, lower confidence and prefer manual_review."
    )
    if provider == "deepseek":
        system_prompt += " claim should be a short Chinese sentence under 80 characters."
    user_prompt = (
        "Return JSON only using exactly this shape: "
        '{"claim":"...","confidence":0.0,"hazard_scores":{"short_rain":0.0,"wind":0.0,"hail":0.0,"tornado":0.0},"key_factors":["..."],"caution_flags":["..."],"action_bias":"monitor"}. '
        "Do not include analysis_time, valid_time, domain_id, grid_id, environment, radar, guidance, providers, qc_summary, or source_tiers_summary as output keys. "
        "Envelope="
        + json.dumps(summary, ensure_ascii=True, sort_keys=True)
    )
    return system_prompt, user_prompt


class OpenAIResponsesClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        timeout_sec: float = 20.0,
        max_output_tokens: int = 900,
        temperature: float = 0.1,
        base_url: str | None = None,
    ) -> None:
        self.provider = str(provider or "openai").strip().lower()
        env_key_name = "DEEPSEEK_API_KEY" if self.provider == "deepseek" else "OPENAI_API_KEY"
        self.api_key = api_key or os.getenv(env_key_name, "")
        self.model = model
        self.timeout_sec = float(timeout_sec)
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        default_base_url = "https://api.deepseek.com" if self.provider == "deepseek" else "https://api.openai.com/v1"
        env_base_url_name = "DEEPSEEK_BASE_URL" if self.provider == "deepseek" else "OPENAI_BASE_URL"
        self.base_url = (base_url or os.getenv(env_base_url_name, default_base_url)).rstrip("/")

    @property
    def ready(self) -> bool:
        return bool((self.api_key or "").strip())

    def generate_situation_evidence(self, envelope: ObservationEnvelope) -> LLMSituationEvidencePayload:
        if not self.ready:
            raise RuntimeError(f"{'DEEPSEEK_API_KEY' if self.provider == 'deepseek' else 'OPENAI_API_KEY'} is not configured")
        summary = _compact_envelope(envelope)
        schema = LLMSituationEvidencePayload.model_json_schema()
        system_prompt, user_prompt = _build_llm_prompts(summary, self.provider)
        if self.provider == "deepseek":
            return self._generate_deepseek_chat_evidence(system_prompt, user_prompt)
        return self._generate_openai_responses_evidence(schema, system_prompt, user_prompt)

    def _generate_openai_responses_evidence(
        self,
        schema: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
    ) -> LLMSituationEvidencePayload:
        body = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "llm_situation_evidence",
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        response = requests.post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout_sec,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"openai_responses_error:{response.status_code}:{response.text[:400]}")
        raw = response.json()
        content = _extract_output_text(raw)
        if not content:
            raise RuntimeError("openai_responses_error:empty_output")
        return LLMSituationEvidencePayload.model_validate_json(content)

    def _generate_deepseek_chat_evidence(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMSituationEvidencePayload:
        body = {
            "model": self.model or "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout_sec,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"deepseek_chat_error:{response.status_code}:{response.text[:400]}")
        raw = response.json()
        content = _extract_chat_message_text(raw)
        if not content:
            raise RuntimeError("deepseek_chat_error:empty_output")
        return LLMSituationEvidencePayload.model_validate_json(content)
