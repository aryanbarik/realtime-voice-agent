from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

USAGE_KEYS = ("llm_input_tokens", "llm_cached_tokens", "llm_output_tokens", "tts_chars")


class TurnTrace:
    def __init__(self, session_id: str, turn_id: int, profile: str):
        self.session_id = session_id
        self.turn_id = turn_id
        self.profile = profile
        self.marks: dict[str, float] = {}
        self.events: list[tuple[str, float, dict]] = []
        self.usage: dict[str, int] = {k: 0 for k in USAGE_KEYS}
        self.user_text = ""
        self.agent_text = ""
        self.interrupted = False

    def mark(self, name: str, t: float | None = None) -> None:
        """First write wins."""
        if name not in self.marks:
            self.marks[name] = time.monotonic() if t is None else t

    def event(self, event_name: str, **data: Any) -> None:
        self.events.append((event_name, time.monotonic(), data))

    def _diff(self, end: str, start: str) -> float | None:
        a, b = self.marks.get(end), self.marks.get(start)
        if a is None or b is None:
            return None
        return round((a - b) * 1000, 1)

    def _tool_ms(self) -> float | None:
        starts: dict[str, list[float]] = {}
        total = 0.0
        n = 0
        for name, t, data in self.events:
            key = data.get("name", "")
            if name == "tool_call_start":
                starts.setdefault(key, []).append(t)
            elif name == "tool_call_end" and starts.get(key):
                total += t - starts[key].pop(0)
                n += 1
        return round(total * 1000, 1) if n else None

    def summary(self) -> dict:
        if "user_speech_start" in self.marks:
            base = self.marks["user_speech_start"]
        elif self.marks:
            base = min(self.marks.values())
        else:
            base = 0.0
        marks_ms = {k: round((v - base) * 1000, 1) for k, v in self.marks.items()}
        derived = {
            "v2v_ms": self._diff("first_audio_sent", "user_speech_end"),
            "endpoint_wait_ms": self._diff("eot_decided", "user_speech_end"),
            "asr_finalize_ms": self._diff("asr_final", "eot_decided"),
            "llm_ttft_ms": self._diff("llm_first_token", "llm_request"),
            "tool_ms": self._tool_ms(),
            "llm_first_clause_ms": self._diff("llm_first_clause", "llm_request"),
            "tts_ttfa_ms": self._diff("tts_first_audio", "tts_request"),
            "send_ms": self._diff("first_audio_sent", "tts_first_audio"),
            "barge_in_ms": self._diff("pause_sent", "barge_in_speech_start"),
        }
        return {
            "turn_id": self.turn_id,
            "profile": self.profile,
            "marks_ms": marks_ms,
            "derived": derived,
            "usage": dict(self.usage),
            "user_text": self.user_text,
            "agent_text": self.agent_text,
            "interrupted": self.interrupted,
        }


class TraceWriter:
    def __init__(self, trace_dir: str | Path):
        self.dir = Path(trace_dir)

    def _append(self, session_id: str, obj: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{session_id}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def write_turn(self, session_id: str, summary: dict) -> None:
        self._append(session_id, summary)

    def write_session_end(
        self,
        session_id: str,
        profile: str,
        duration_s: float,
        asr_audio_s: float,
        usage_totals: dict,
    ) -> None:
        self._append(
            session_id,
            {
                "type": "session_end",
                "session_id": session_id,
                "profile": profile,
                "duration_s": round(duration_s, 2),
                "asr_audio_s": round(asr_audio_s, 2),
                "usage_totals": usage_totals,
            },
        )
