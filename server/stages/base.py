from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AsrResult:
    text: str
    is_final: bool
    speech_final: bool
    from_finalize: bool
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class AsrUtteranceEnd:
    last_word_end_s: float


class StreamingASR(Protocol):
    async def connect(self) -> None: ...
    async def send_audio(self, pcm16: bytes) -> None: ...
    async def finalize(self) -> None: ...
    def events(self) -> AsyncIterator[AsrResult | AsrUtteranceEnd]: ...
    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    cached_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class LlmDone:
    finish_reason: str
    usage: Usage | None


class LLM(Protocol):
    def stream(
        self, messages: list[dict], tools: list[dict]
    ) -> AsyncIterator[TextDelta | ToolCall | LlmDone]: ...
    async def complete(
        self, messages: list[dict], tools: list[dict]
    ) -> tuple[str, list[ToolCall], Usage]: ...
    async def warmup(self, system_prompt: str) -> None: ...


@dataclass(frozen=True, slots=True)
class TtsAudio:
    pcm16: bytes


@dataclass(frozen=True, slots=True)
class TtsWords:
    words: list[str]
    start_s: list[float]
    end_s: list[float]


class TTSContext(Protocol):
    context_id: str

    async def push_text(self, text: str) -> None: ...
    async def end_input(self) -> None: ...
    def events(self) -> AsyncIterator[TtsAudio | TtsWords]: ...
    async def cancel(self) -> None: ...


class StreamingTTS(Protocol):
    sample_rate: int

    async def connect(self) -> None: ...
    def new_context(self, context_id: str) -> TTSContext: ...
    async def close(self) -> None: ...
