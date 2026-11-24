from __future__ import annotations

from openai import AsyncOpenAI

from server.config import PipelineConfig, Settings
from server.stages.llm.openai_compat import OpenAICompatLLM

_clients: dict[tuple[str, str], AsyncOpenAI] = {}


def _client(base_url: str | None, api_key: str) -> AsyncOpenAI:
    key = (base_url or "", api_key)
    if key not in _clients:
        _clients[key] = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _clients[key]


def build_llm(cfg: PipelineConfig, settings: Settings) -> OpenAICompatLLM:
    if cfg.llm.provider == "vllm":
        if not settings.va_selfhost_llm_url:
            raise RuntimeError("VA_SELFHOST_LLM_URL is not set")
        if not settings.selfhost_api_key:
            raise RuntimeError("SELFHOST_API_KEY is not set")
        client = _client(f"{settings.va_selfhost_llm_url}/v1", settings.selfhost_api_key)
    else:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = _client(None, settings.openai_api_key)
    l = cfg.llm
    return OpenAICompatLLM(client, l.model, l.temperature, l.max_tokens, l.extra_body)


def build_asr(cfg: PipelineConfig, settings: Settings):
    if cfg.asr.provider == "parakeet":
        if not settings.va_selfhost_speech_url:
            raise RuntimeError("VA_SELFHOST_SPEECH_URL is not set")
        if not settings.selfhost_api_key:
            raise RuntimeError("SELFHOST_API_KEY is not set")
        from server.stages.asr.parakeet import ParakeetASR

        return ParakeetASR(settings.va_selfhost_speech_url, settings.selfhost_api_key)
    if not settings.deepgram_api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")
    from server.stages.asr.deepgram import DeepgramStreamingASR

    return DeepgramStreamingASR(
        settings.deepgram_api_key, cfg.asr.endpointing_ms, cfg.asr.utterance_end_ms
    )


def build_tts(cfg: PipelineConfig, settings: Settings):
    t = cfg.tts
    if t.provider == "kokoro":
        if not settings.va_selfhost_speech_url:
            raise RuntimeError("VA_SELFHOST_SPEECH_URL is not set")
        if not settings.selfhost_api_key:
            raise RuntimeError("SELFHOST_API_KEY is not set")
        from server.stages.tts.kokoro import KokoroTTS

        return KokoroTTS(settings.va_selfhost_speech_url, settings.selfhost_api_key, t.voice)
    if not settings.cartesia_api_key:
        raise RuntimeError("CARTESIA_API_KEY is not set")
    from server.stages.tts.cartesia import CartesiaTTS

    return CartesiaTTS(settings.cartesia_api_key, t.model, t.voice, t.sample_rate)


_silero_session = None


def build_vad(cfg: PipelineConfig, settings: Settings):
    global _silero_session
    if not cfg.vad.enabled:
        return None
    from server.stages.vad.silero import SileroVAD, load_silero

    if _silero_session is None:
        _silero_session = load_silero()
    return SileroVAD(_silero_session)


def build_turn(cfg: PipelineConfig, settings: Settings):
    d = cfg.turn.detector
    if d == "silence":
        from server.stages.turn.silence import SilenceTurnDetector

        return SilenceTurnDetector(cfg.turn.silence_ms)
    if d == "smart_turn":
        from server.stages.turn.smart_turn import SmartTurnDetector

        return SmartTurnDetector(cfg.turn.smart_turn_threshold)
    return None
