"""Single cost formula shared by the demo spend cap and eval reports (stdlib only)."""
from __future__ import annotations

from collections.abc import Mapping

DEFAULT_PRICES: dict = {
    "deepgram_stream_per_min": 0.0048,
    "deepgram_batch_per_min": 0.0043,
    "openai_per_1m": {"gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50}},
    "cartesia_per_char": 0.0000392,
    "gpu_per_hour": {"runpod_a40_secure": 0.59},
}


def hosted_cost_usd(
    *,
    asr_audio_s: float,
    llm_input_tokens: int,
    llm_cached_tokens: int,
    llm_output_tokens: int,
    tts_chars: int,
    llm_model: str = "gpt-6-luna",
    batch_asr: bool = False,
    prices: Mapping | None = None,
) -> float:
    """Cost in USD of the usage as-is (not scaled)."""
    p = prices or DEFAULT_PRICES
    m = p["openai_per_1m"][llm_model]
    asr_rate = p["deepgram_batch_per_min" if batch_asr else "deepgram_stream_per_min"]
    uncached = max(llm_input_tokens - llm_cached_tokens, 0)
    return (
        asr_audio_s / 60.0 * asr_rate
        + (uncached * m["input"] + llm_cached_tokens * m["cached_input"] + llm_output_tokens * m["output"]) / 1e6
        + tts_chars * p["cartesia_per_char"]
    )


def cost_from_totals(
    asr_audio_s: float, usage: Mapping[str, int], *, llm_model: str = "gpt-6-luna",
    batch_asr: bool = False, prices: Mapping | None = None,
) -> float:
    """`usage` uses TurnTrace.usage keys (llm_input_tokens, ..., tts_chars)."""
    return hosted_cost_usd(
        asr_audio_s=asr_audio_s,
        llm_input_tokens=usage.get("llm_input_tokens", 0),
        llm_cached_tokens=usage.get("llm_cached_tokens", 0),
        llm_output_tokens=usage.get("llm_output_tokens", 0),
        tts_chars=usage.get("tts_chars", 0),
        llm_model=llm_model, batch_asr=batch_asr, prices=prices,
    )


def selfhosted_cost_usd(
    call_s: float, max_concurrent_calls: int = 1, *, gpu: str = "runpod_a40_secure",
    prices: Mapping | None = None,
) -> float:
    """GPU-hour share of one call; `max_concurrent_calls` is N from the load test."""
    p = prices or DEFAULT_PRICES
    return p["gpu_per_hour"][gpu] * (call_s / 3600.0) / max(max_concurrent_calls, 1)


def scale_to_3min(cost_usd: float, call_s: float) -> float:
    return cost_usd * 180.0 / call_s if call_s > 0 else 0.0
