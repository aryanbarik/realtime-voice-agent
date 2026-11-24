from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROFILES_DIR = Path(__file__).parent / "profiles"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.selfhost"), extra="ignore")

    deepgram_api_key: str = ""
    openai_api_key: str = ""
    cartesia_api_key: str = ""
    runpod_api_key: str = ""
    selfhost_api_key: str = ""
    va_profile: str = "v3-hosted"
    va_overrides: str = "{}"
    va_today: date | None = None
    va_port: int = 8080
    va_db_path: str = "data/backend.db"
    va_trace_dir: str = "logs/traces"
    va_admin_token: str | None = None
    va_public_demo: bool = False
    va_daily_budget_usd: float = 3.0
    va_selfhost_llm_url: str = ""
    va_selfhost_speech_url: str = ""

    @property
    def today(self) -> date:
        return self.va_today or date.today()


class AsrConfig(BaseModel):
    provider: Literal["deepgram", "parakeet"] = "deepgram"
    endpointing_ms: int = 300
    utterance_end_ms: int = 1000
    finalize_timeout_ms: int = 600


class LlmConfig(BaseModel):
    provider: Literal["openai", "vllm"] = "openai"
    model: str = "gpt-6-luna"
    temperature: float = 0.3
    max_tokens: int = 300
    extra_body: dict[str, Any] = {}


class TtsConfig(BaseModel):
    provider: Literal["cartesia", "kokoro"] = "cartesia"
    model: str = "sonic-3.6"
    voice: str = "a0e99841-438c-4a64-b679-ae501e7d6091"
    sample_rate: int = 24000


class TurnConfig(BaseModel):
    detector: Literal["push_to_talk", "asr_endpointing", "silence", "smart_turn"] = "smart_turn"
    silence_ms: int = 700
    smart_turn_threshold: float = 0.5
    smart_turn_trigger_ms: int = 200
    smart_turn_max_silence_ms: int = 2000


class VadConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.5
    neg_threshold: float = 0.35
    onset_frames: int = 2


class BargeInConfig(BaseModel):
    enabled: bool = True
    min_speech_ms: int = 128
    confirm_speech_ms: int = 600
    resume_silence_ms: int = 300
    resume_timeout_ms: int = 1500
    cleared_ack_timeout_ms: int = 300


class SpeculativeConfig(BaseModel):
    enabled: bool = False
    trigger_silence_ms: int = 200


class ChunkerConfig(BaseModel):
    first_min_chars: int = 12
    min_chars: int = 40


class PipelineConfig(BaseModel):
    mode: Literal["batch", "streaming"] = "streaming"
    asr: AsrConfig = AsrConfig()
    llm: LlmConfig = LlmConfig()
    tts: TtsConfig = TtsConfig()
    turn: TurnConfig = TurnConfig()
    vad: VadConfig = VadConfig()
    barge_in: BargeInConfig = BargeInConfig()
    speculative: SpeculativeConfig = SpeculativeConfig()
    warmup: bool = False
    chunker: ChunkerConfig = ChunkerConfig()
    max_call_s: int = 300

    @model_validator(mode="after")
    def _detector_needs_vad(self) -> PipelineConfig:
        if self.turn.detector in ("silence", "smart_turn") and not self.vad.enabled:
            raise ValueError(f"turn.detector={self.turn.detector} requires vad.enabled")
        return self


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_pipeline_config(settings: Settings) -> PipelineConfig:
    path = PROFILES_DIR / f"{settings.va_profile}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"unknown profile {settings.va_profile!r}: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    data = deep_merge(data, json.loads(settings.va_overrides or "{}"))
    return PipelineConfig.model_validate(data)
