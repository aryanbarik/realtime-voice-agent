from __future__ import annotations


class SilenceTurnDetector:
    """Fixed-silence end-of-turn. Holds no model; the pipeline feeds it silence durations."""

    def __init__(self, silence_ms: int) -> None:
        self.silence_ms = silence_ms

    def is_complete(self, silence_ms: float) -> bool:
        return silence_ms >= self.silence_ms
