"""Streaming voice session: ASR -> turn detection -> LLM (tools) -> TTS, with barge-in.

Concurrency model
- recv_loop reads the transport, re-frames mic audio, feeds ASR/VAD and handles control messages.
- asr_loop consumes ASR events.
- send_loop is the only writer to the transport; everything goes through `_Outbox`
  (control messages in a high-priority queue that is always served first).
- Agent work (finalize wait + LLM/tools + TTS) runs in one cancellable `agent_task`.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import re
import time
import uuid
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from server.chunker import ClauseChunker
from server.prompts import GREETING, build_system_prompt
from server.stages.base import (
    AsrResult,
    AsrUtteranceEnd,
    LlmDone,
    TextDelta,
    ToolCall,
    TtsAudio,
    TtsWords,
)
from server.tools.schemas import TOOLS
from server.tracing import USAGE_KEYS, TurnTrace
from server.transport import TransportClosed

IN_RATE = 16000
FRAME = 512
FRAME_BYTES = FRAME * 2
FRAME_MS = FRAME * 1000 / IN_RATE
RING_SAMPLES = 8 * IN_RATE
MAX_TOOL_ROUNDS = 4
APOLOGY = "Sorry, I had a technical problem. Could you say that again?"
BACKCHANNELS = {
    "yeah", "yes", "yep", "ok", "okay", "uh-huh", "uh", "huh", "mhm", "mm", "hmm",
    "um", "right", "sure", "alright", "cool", "got", "it",
}


@dataclass
class Stages:
    asr: Any
    llm: Any
    tts: Any
    vad: Any | None = None
    turn_detector: Any | None = None


class _Stop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class AgentTurnRecord:
    turn_id: int
    trace: TurnTrace
    greeting: bool = False
    emit_metrics: bool = True
    words: list[tuple[str, float, float]] = field(default_factory=list)
    clauses: list[str] = field(default_factory=list)
    sent_samples: int = 0
    first_audio_t: float | None = None
    history_index: int | None = None  # index of the user message
    final_index: int | None = None  # index of the final assistant message
    word_offset: int = 0  # words already committed to history by earlier tool rounds
    clause_offset: int = 0
    cleared_future: asyncio.Future | None = None
    ctx: Any = None
    pump: asyncio.Task | None = None
    pump_error: str | None = None
    playback_done: bool = False
    closed: bool = False


class _Outbox:
    """Two FIFO queues; `hi` (pause/resume/clear) is always served first."""

    def __init__(self) -> None:
        self.hi: deque = deque()
        self.lo: deque = deque()
        self._evt = asyncio.Event()
        self.busy = False

    def put(self, payload, on_sent: Callable[[], None] | None = None, hi: bool = False) -> None:
        (self.hi if hi else self.lo).append((payload, on_sent))
        self._evt.set()

    async def get(self):
        while True:
            q = self.hi or self.lo
            if q:
                item = q.popleft()
                self.busy = True
                return item
            self._evt.clear()
            await self._evt.wait()

    def pop_nowait(self):
        q = self.hi or self.lo
        return q.popleft() if q else None

    def drain_audio(self, turn_id: int) -> None:
        def keep(item) -> bool:
            p = item[0]
            if isinstance(p, bytes):
                return False
            return not (p.get("type") in ("agent_audio_start", "agent_audio_end") and p.get("turn_id") == turn_id)

        kept = [i for i in self.lo if keep(i)]
        self.lo.clear()
        self.lo.extend(kept)

    @property
    def idle(self) -> bool:
        return not self.hi and not self.lo and not self.busy


_END = object()


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", s.lower()).split())


class _Speculation:
    """Round-1 LLM generation started during a pause. No TTS, no tool execution."""

    def __init__(self, stream, snapshot: str):
        self.norm = _norm(snapshot)
        self.start_t = time.monotonic()
        self.first_t: float | None = None
        self.usage = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._stream = stream
        self.task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        try:
            async with aclosing(self._stream) as it:
                async for ev in it:
                    if self.first_t is None and isinstance(ev, (TextDelta, ToolCall)):
                        self.first_t = time.monotonic()
                    if isinstance(ev, LlmDone):
                        self.usage = ev.usage
                    self._queue.put_nowait(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._queue.put_nowait(exc)
        finally:
            self._queue.put_nowait(_END)

    async def events(self):
        while True:
            ev = await self._queue.get()
            if ev is _END:
                return
            if isinstance(ev, Exception):
                raise ev
            yield ev

    async def cancel(self) -> None:
        if not self.task.done():
            self.task.cancel()
            await asyncio.wait({self.task})


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    await asyncio.wait({task})


class StreamingSession:
    def __init__(self, transport, cfg, settings, stages: Stages, tools, trace_writer):
        self.transport, self.cfg, self.settings = transport, cfg, settings
        self.stages, self.tools, self.trace_writer = stages, tools, trace_writer
        self.session_id = uuid.uuid4().hex  # app may overwrite before run()
        self.outbox = _Outbox()

        self.state = "listening"
        self.history: list[dict] = [
            {"role": "system", "content": build_system_prompt(settings.today)},
            {"role": "assistant", "content": GREETING},
        ]
        self.turn_counter = 0
        self.pending_finals: list[AsrResult] = []
        self.consumed = 0  # prefix of pending_finals owned by the in-flight agent turn
        self.interim = ""
        self.agent_task: asyncio.Task | None = None
        self.agent_record: AgentTurnRecord | None = None
        self.interrupt_task: asyncio.Task | None = None
        self.user_trace: TurnTrace | None = None
        self.queued_eot: tuple[float, bool] | None = None
        self.spec: _Speculation | None = None
        self.finalize_future: asyncio.Future | None = None
        self._bg: set[asyncio.Task] = set()

        self.ring = np.zeros(RING_SAMPLES, dtype=np.float32)
        self.ring_filled = 0
        self.frame_clock: deque[tuple[int, float]] = deque(maxlen=30 * IN_RATE // FRAME)
        self.samples_total = 0

        self.usage_totals = {k: 0 for k in USAGE_KEYS}

        # VAD / turn state (sample-count based so logic is independent of pacing)
        self.fidx = 0
        self._voiced = False
        self.voiced_run = 0
        self.run_start_t = 0.0
        self.last_voiced_t = 0.0
        self.last_voiced_idx: int | None = None
        self.silence_ms = 0.0
        self.pause_gen = 0
        self.pause_checked = False
        self.spec_tried = False
        self.pause_idx = 0
        self.voiced_since = 0
        self.barge_start_t = 0.0

    # ------------------------------------------------------------------ public

    def snapshot(self) -> tuple[float, dict]:
        """(asr_audio_s, cumulative usage totals) for spend metering."""
        return self.samples_total / IN_RATE, dict(self.usage_totals)

    async def run(self) -> None:
        started = time.monotonic()
        reason = "disconnect"
        cfg = self.cfg
        try:
            await self.stages.asr.connect()
            if cfg.warmup:
                await asyncio.gather(self.stages.tts.connect(), self.stages.llm.warmup(self.history[0]["content"]))
            self._send({
                "type": "session", "session_id": self.session_id, "profile": self.settings.va_profile,
                "turn_mode": "auto", "input_sample_rate": IN_RATE, "output_sample_rate": cfg.tts.sample_rate,
            })
            rec = AgentTurnRecord(0, TurnTrace(self.session_id, 0, self.settings.va_profile), greeting=True,
                                  emit_metrics=False)
            rec.final_index = 1
            self.agent_record = rec
            self.state = "agent_thinking"
            self.agent_task = asyncio.create_task(self.run_agent_turn(rec, GREETING, rec.trace))
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._recv_loop())
                    tg.create_task(self._asr_loop())
                    tg.create_task(self._send_loop())
                    tg.create_task(self._call_timer())
            except* _Stop as eg:
                reason = eg.exceptions[0].reason
        finally:
            await self._shutdown(reason, started)

    # ------------------------------------------------------------------ loops

    async def _call_timer(self) -> None:
        await asyncio.sleep(self.cfg.max_call_s)
        raise _Stop("max_call_s")

    async def _send_loop(self) -> None:
        while True:
            payload, on_sent = await self.outbox.get()
            try:
                if isinstance(payload, bytes):
                    await self.transport.send_bytes(payload)
                else:
                    await self.transport.send_json(payload)
            except Exception as exc:  # noqa: BLE001
                raise _Stop("disconnect") from exc
            finally:
                self.outbox.busy = False
            if on_sent:
                on_sent()

    async def _recv_loop(self) -> None:
        buf = bytearray()
        while True:
            try:
                msg = await self.transport.recv()
            except TransportClosed:
                raise _Stop("disconnect") from None
            if isinstance(msg, bytes):
                t = time.monotonic()
                buf.extend(msg)
                while len(buf) >= FRAME_BYTES:
                    frame = bytes(buf[:FRAME_BYTES])
                    del buf[:FRAME_BYTES]
                    await self._on_audio_frame(frame, t)
                continue
            try:
                data = json.loads(msg)
            except (ValueError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            kind = data.get("type")
            rec = self.agent_record
            if kind == "playback_cleared":
                if rec and data.get("turn_id") == rec.turn_id and rec.cleared_future and not rec.cleared_future.done():
                    rec.cleared_future.set_result(int(data.get("played_samples", 0)))
            elif kind == "playback_done":
                if rec and data.get("turn_id") == rec.turn_id:
                    rec.playback_done = True
                    if self.state == "agent_speaking":
                        self._idle()
            elif kind == "hangup":
                raise _Stop("hangup")

    async def _on_audio_frame(self, frame: bytes, t: float) -> None:
        try:
            await self.stages.asr.send_audio(frame)
        except Exception as exc:  # noqa: BLE001
            self._send({"type": "error", "message": f"ASR failed: {exc}"})
            raise _Stop("asr_failed") from exc
        f32 = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        self.ring[:-FRAME] = self.ring[FRAME:]
        self.ring[-FRAME:] = f32
        self.ring_filled = min(RING_SAMPLES, self.ring_filled + FRAME)
        self.samples_total += FRAME
        self.frame_clock.append((self.samples_total, t))
        if self.stages.vad is not None:
            await self._on_frame(f32, t)

    def _clock_at(self, audio_s: float) -> float:
        target = int(audio_s * IN_RATE)
        for end, t in self.frame_clock:
            if end >= target:
                return t
        return time.monotonic()

    async def _asr_loop(self) -> None:
        try:
            async for ev in self.stages.asr.events():
                if isinstance(ev, AsrResult):
                    self._on_asr_result(ev)
                elif isinstance(ev, AsrUtteranceEnd):
                    self._on_utterance_end(ev)
        except Exception as exc:  # noqa: BLE001
            self._send({"type": "error", "message": f"ASR failed: {exc}"})
            raise _Stop("asr_failed") from exc
        raise _Stop("asr_failed")

    # ------------------------------------------------------------------ ASR events

    def _unconsumed_text(self) -> str:
        return " ".join(r.text.strip() for r in self.pending_finals[self.consumed:] if r.text.strip()).strip()

    def _peek_turn_id(self) -> int:
        return self.user_trace.turn_id if self.user_trace else self.turn_counter + 1

    def _new_user_trace(self, start_t: float | None) -> TurnTrace:
        self.turn_counter += 1
        tr = TurnTrace(self.session_id, self.turn_counter, self.settings.va_profile)
        if start_t is not None:
            tr.mark("user_speech_start", start_t)
        self.user_trace = tr
        return tr

    def _on_asr_result(self, res: AsrResult) -> None:
        text = res.text.strip()
        state = self.state
        if self.stages.vad is None and text and self.user_trace is None and state in ("listening", "agent_thinking"):
            self._new_user_trace(self._clock_at(res.start_s))
        if not res.is_final:
            if state == "agent_speaking":
                return
            self.interim = text
            if text:
                self._send({"type": "transcript", "role": "user", "turn_id": self._peek_turn_id(),
                            "text": text, "final": False})
            if state == "paused":
                self._check_paused()
            return
        if state != "agent_speaking":
            if text:
                self.pending_finals.append(res)
            self.interim = ""
        fut = self.finalize_future
        if res.from_finalize and fut is not None and not fut.done():
            fut.set_result(None)
        if state == "paused":
            self._check_paused()
        elif state != "agent_speaking" and res.speech_final and self.cfg.turn.detector == "asr_endpointing":
            if self._unconsumed_text():
                self.end_of_turn(self._clock_at(res.end_s), already_final=True)

    def _on_utterance_end(self, ev: AsrUtteranceEnd) -> None:
        if self.cfg.turn.detector != "asr_endpointing" or self.state == "agent_speaking":
            return
        if self._unconsumed_text():
            self.end_of_turn(self._clock_at(ev.last_word_end_s), already_final=False)

    # ------------------------------------------------------------------ VAD / turn state machine

    async def _on_frame(self, frame: np.ndarray, t: float) -> None:
        vcfg = self.cfg.vad
        p = self.stages.vad.prob(frame)
        if p >= vcfg.threshold:
            self._voiced = True
        elif p < vcfg.neg_threshold:
            self._voiced = False
        self.fidx += 1
        onset = False
        if self._voiced:
            self.voiced_run += 1
            if self.voiced_run == 1:
                self.run_start_t = t
            self.last_voiced_t = t
            self.last_voiced_idx = self.fidx
            onset = self.voiced_run == vcfg.onset_frames
            self.silence_ms = 0.0
        else:
            self.voiced_run = 0
            self.silence_ms = (
                (self.fidx - self.last_voiced_idx) * FRAME_MS if self.last_voiced_idx is not None else 0.0
            )

        state = self.state
        if state in ("listening", "user_speaking", "agent_thinking"):
            if onset:
                await self._on_onset()
            elif self.state == "user_speaking" and not self._voiced:
                await self._on_pause_frame()
        elif state == "agent_speaking":
            be = self.cfg.barge_in
            if be.enabled and self.voiced_run == max(1, math.ceil(be.min_speech_ms / FRAME_MS)):
                self._enter_pause()
        elif state == "paused":
            if self._voiced:
                self.voiced_since += 1
            self._check_paused()

    async def _on_onset(self) -> None:
        rec = self.agent_record
        if self.state == "agent_thinking" and rec is not None and rec.greeting:
            return
        self.pause_gen += 1
        self.pause_checked = False
        self.spec_tried = False
        await self._cancel_spec()
        if self.state == "agent_thinking":
            await self._resume_from_thinking()
        if self.user_trace is None:
            self._new_user_trace(self.run_start_t)
        self.state = "user_speaking"

    async def _on_pause_frame(self) -> None:
        cfg = self.cfg
        if (
            cfg.speculative.enabled
            and self.spec is None
            and not self.spec_tried
            and self.silence_ms >= cfg.speculative.trigger_silence_ms
        ):
            self.spec_tried = True
            snapshot = " ".join(
                x for x in [self._unconsumed_text(), self.interim.strip()] if x
            ).strip()
            if snapshot:
                msgs = self.history + [{"role": "user", "content": snapshot}]
                self.spec = _Speculation(self.stages.llm.stream(msgs, TOOLS), snapshot)

        mode = cfg.turn.detector
        td = self.stages.turn_detector
        if mode == "silence":
            done = td.is_complete(self.silence_ms) if td is not None else self.silence_ms >= cfg.turn.silence_ms
            if done:
                self.end_of_turn(self.last_voiced_t)
        elif mode == "smart_turn":
            if self.silence_ms >= cfg.turn.smart_turn_max_silence_ms:
                self.end_of_turn(self.last_voiced_t)
            elif self.silence_ms >= cfg.turn.smart_turn_trigger_ms and not self.pause_checked:
                self.pause_checked = True
                audio = self.ring[RING_SAMPLES - self.ring_filled:].copy()
                self._spawn(self._smart_turn_check(audio, self.pause_gen, self.last_voiced_t))

    async def _smart_turn_check(self, audio: np.ndarray, gen: int, voiced_t: float) -> None:
        prob = await self.stages.turn_detector.predict(audio)
        if gen == self.pause_gen and self.state == "user_speaking" and prob >= self.cfg.turn.smart_turn_threshold:
            self.end_of_turn(voiced_t)

    def _spawn(self, coro) -> None:
        async def guarded():
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._send({"type": "error", "message": str(exc)})

        task = asyncio.create_task(guarded())
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _cancel_spec(self) -> None:
        spec, self.spec = self.spec, None
        if spec is None:
            return
        await spec.cancel()
        if spec.usage is not None:  # cost of a discarded speculation still counts
            self.usage_totals["llm_input_tokens"] += spec.usage.input_tokens
            self.usage_totals["llm_cached_tokens"] += spec.usage.cached_tokens
            self.usage_totals["llm_output_tokens"] += spec.usage.output_tokens

    async def _resume_from_thinking(self) -> None:
        """User spoke again before any agent audio: abandon the turn, merge later."""
        rec = self.agent_record
        self.state = "user_speaking"
        await _cancel_task(self.agent_task)
        self.agent_record = None
        self.consumed = 0
        self.queued_eot = None
        if rec is None:
            return
        rec.closed = True
        idx = rec.history_index
        if idx is not None:
            del self.history[idx]
            self.history[idx:] = [m for m in self.history[idx:] if m["role"] == "tool" or m.get("tool_calls")]

    # ------------------------------------------------------------------ barge-in

    def _paused_words(self) -> list[str]:
        text = " ".join([r.text for r in self.pending_finals] + [self.interim]).lower()
        return re.sub(r"[^\w\s'-]", "", text).split()

    def _enter_pause(self) -> None:
        rec = self.agent_record
        if rec is None or rec.playback_done:
            return
        rec.trace.mark("barge_in_speech_start", self.run_start_t)
        self.barge_start_t = self.run_start_t
        self.outbox.put({"type": "pause"}, lambda: rec.trace.mark("pause_sent"), hi=True)
        self.state = "paused"
        self.pending_finals.clear()
        self.consumed = 0
        self.interim = ""
        self.pause_idx = self.fidx
        self.voiced_since = 1

    def _check_paused(self) -> None:
        if self.state != "paused":
            return
        be = self.cfg.barge_in
        words = self._paused_words()
        if (
            self.voiced_since * FRAME_MS >= be.confirm_speech_ms
            or any(w not in BACKCHANNELS for w in words)
            or len(words) >= 3
        ):
            self._begin_confirm()
            return
        elapsed = (self.fidx - self.pause_idx) * FRAME_MS
        if (not self._voiced and self.silence_ms >= be.resume_silence_ms) or elapsed >= be.resume_timeout_ms:
            self._resume_playback()

    def _resume_playback(self) -> None:
        rec = self.agent_record
        self.pending_finals.clear()
        self.interim = ""
        self.voiced_run = 0
        if rec is None or rec.playback_done:
            self._idle()
            return
        self.outbox.put({"type": "resume"}, hi=True)
        self.state = "agent_speaking"

    def _begin_confirm(self) -> None:
        rec = self.agent_record
        self.state = "user_speaking"
        self.pause_gen += 1
        self.pause_checked = False
        self.spec_tried = False
        if self.user_trace is None:
            self._new_user_trace(self.barge_start_t)
        if rec is None:
            return
        self.interrupt_task = asyncio.create_task(self._confirm_barge_in(rec))

    async def _confirm_barge_in(self, rec: AgentTurnRecord) -> None:
        try:
            rec.trace.interrupted = True
            await _cancel_task(self.agent_task)
            if rec.ctx is not None:
                with contextlib.suppress(Exception):
                    await rec.ctx.cancel()
            self.outbox.drain_audio(rec.turn_id)
            loop = asyncio.get_running_loop()
            rec.cleared_future = loop.create_future()
            written = loop.create_future()

            def on_sent() -> None:
                rec.trace.mark("clear_sent")
                if not written.done():
                    written.set_result(None)

            self.outbox.put({"type": "clear", "turn_id": rec.turn_id}, on_sent, hi=True)
            try:
                played = await asyncio.wait_for(rec.cleared_future, self.cfg.barge_in.cleared_ack_timeout_ms / 1000)
            except TimeoutError:
                est = (time.monotonic() - rec.first_audio_t) * self.cfg.tts.sample_rate if rec.first_audio_t else 0
                played = int(min(rec.sent_samples, est))
            heard = self._patch_history(rec, played)
            self._send({"type": "transcript", "role": "agent", "turn_id": rec.turn_id,
                        "text": heard + "…", "final": True})
            rec.trace.agent_text = heard
            with contextlib.suppress(Exception):
                await asyncio.wait_for(written, 1.0)
            if self.agent_record is rec:
                self.agent_record = None
            self._close_turn(rec)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._send({"type": "error", "message": str(exc)})

    def _patch_history(self, rec: AgentTurnRecord, played_samples: int) -> str:
        rate = self.cfg.tts.sample_rate
        played_s = played_samples / rate
        if rec.words:
            heard_words = [w for w, _s, e in rec.words if e <= played_s][rec.word_offset:]
            heard = " ".join(heard_words)
        else:
            text = " ".join(rec.clauses[rec.clause_offset:])
            frac = min(1.0, played_samples / rec.sent_samples) if rec.sent_samples else 0.0
            heard = text[: int(len(text) * frac)]
            if frac < 1.0 and " " in heard:
                heard = heard.rsplit(" ", 1)[0]
            heard = heard.strip()
        content = f"{heard} [interrupted]" if heard else "[interrupted]"
        if rec.final_index is not None:
            self.history[rec.final_index]["content"] = content
        else:
            self.history.append({"role": "assistant", "content": content})
            rec.final_index = len(self.history) - 1
        return heard

    # ------------------------------------------------------------------ end of turn

    def end_of_turn(self, user_speech_end: float, already_final: bool = False) -> None:
        if self.state not in ("listening", "user_speaking"):
            if self.cfg.turn.detector == "asr_endpointing" and self.state == "agent_thinking":
                self.queued_eot = (user_speech_end, already_final)
            return
        trace = self.user_trace or self._new_user_trace(None)
        self.user_trace = None
        trace.mark("user_speech_end", user_speech_end)
        trace.mark("eot_decided")
        self.state = "agent_thinking"
        rec = AgentTurnRecord(trace.turn_id, trace)
        self.agent_record = rec
        self.agent_task = asyncio.create_task(self._turn_task(rec, already_final))

    async def _finalize(self) -> None:
        fut = asyncio.get_running_loop().create_future()
        self.finalize_future = fut
        try:
            await self.stages.asr.finalize()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(fut, self.cfg.asr.finalize_timeout_ms / 1000)
        finally:
            self.finalize_future = None

    async def _turn_task(self, rec: AgentTurnRecord, already_final: bool) -> None:
        trace = rec.trace
        spec = None
        try:
            it = self.interrupt_task
            if it is not None and not it.done():
                await asyncio.wait({it})
            if not already_final:
                await self._finalize()
            text = self._unconsumed_text()
            trace.mark("asr_final")
            spec, self.spec = self.spec, None
            if not text:
                if spec:
                    self.spec = spec
                    await self._cancel_spec()
                rec.closed = True
                self._idle()
                return
            if spec is not None:
                if spec.norm == _norm(text):
                    trace.event("spec_hit")
                else:
                    trace.event("spec_miss")
                    self.spec = spec
                    await self._cancel_spec()
                    spec = None
            self.consumed = len(self.pending_finals)
            trace.user_text = text
            self._send({"type": "transcript", "role": "user", "turn_id": rec.turn_id, "text": text, "final": True})
            await self.run_agent_turn(rec, text, trace, spec)
        except asyncio.CancelledError:
            if spec is not None:
                self.spec = spec
                await self._cancel_spec()
            raise
        except Exception as exc:  # noqa: BLE001
            self._send({"type": "error", "message": str(exc)})
            if self.agent_record is rec and self.state == "agent_thinking":
                rec.closed = True
                self._idle()

    # ------------------------------------------------------------------ agent turn

    def _add_usage(self, trace: TurnTrace, key: str, n: int) -> None:
        trace.usage[key] += n
        self.usage_totals[key] += n

    def _on_first_audio(self, rec: AgentTurnRecord) -> None:
        if self.agent_record is rec and self.state == "agent_thinking":
            self.state = "agent_speaking"
            del self.pending_finals[: self.consumed]
            self.consumed = 0

    async def _pump(self, rec: AgentTurnRecord, ctx) -> None:
        trace = rec.trace
        started = False
        try:
            async for ev in ctx.events():
                if isinstance(ev, TtsAudio):
                    if not started:
                        started = True
                        now = time.monotonic()
                        rec.first_audio_t = now
                        trace.mark("tts_first_audio", now)
                        self._on_first_audio(rec)
                        self._send({"type": "agent_audio_start", "turn_id": rec.turn_id})
                        self._send(ev.pcm16, lambda: trace.mark("first_audio_sent"))
                    else:
                        self._send(ev.pcm16)
                    rec.sent_samples += len(ev.pcm16) // 2
                elif isinstance(ev, TtsWords):
                    rec.words.extend(zip(ev.words, ev.start_s, ev.end_s))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            rec.pump_error = str(exc)
            self._send({"type": "error", "message": f"TTS failed: {exc}"})
        if started:
            self._send({"type": "agent_audio_end", "turn_id": rec.turn_id},
                       lambda: trace.mark("agent_audio_end_sent"))

    async def _push_chunk(self, rec: AgentTurnRecord, chunk: str) -> None:
        trace = rec.trace
        trace.mark("llm_first_clause")
        trace.mark("tts_request")
        self._add_usage(trace, "tts_chars", len(chunk))
        await rec.ctx.push_text(chunk)
        rec.clauses.append(chunk)

    async def _run_tool(self, rec: AgentTurnRecord, call: ToolCall) -> None:
        trace = rec.trace
        trace.event("tool_call_start", name=call.name)
        task = asyncio.ensure_future(self.tools.execute(call.name, call.arguments))
        cancelled = False
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            try:  # let a started side effect (booking) finish so history stays truthful
                result = await task
            except BaseException:  # noqa: BLE001
                result = json.dumps({"error": "interrupted"})
        trace.event("tool_call_end", name=call.name)
        self.history.append({"role": "tool", "tool_call_id": call.id, "content": result})
        if cancelled:
            raise asyncio.CancelledError
        self._send({"type": "tool", "turn_id": rec.turn_id, "name": call.name,
                    "args": call.arguments, "result": result})

    async def run_agent_turn(self, rec: AgentTurnRecord, user_text: str, trace: TurnTrace,
                             spec: _Speculation | None = None) -> None:
        cfg = self.cfg
        cancelled = False
        failed = False
        if not rec.greeting:
            self.history.append({"role": "user", "content": user_text})
            rec.history_index = len(self.history) - 1
        ctx = self.stages.tts.new_context(f"{self.session_id}-{rec.turn_id}")
        rec.ctx = ctx
        rec.pump = asyncio.create_task(self._pump(rec, ctx))
        try:
            if rec.greeting:
                await self._push_chunk(rec, GREETING)
            else:
                chunker = ClauseChunker(cfg.chunker.first_min_chars, cfg.chunker.min_chars)
                final_text = ""
                for rnd in range(MAX_TOOL_ROUNDS):
                    from_spec = rnd == 0 and spec is not None
                    if from_spec:
                        trace.mark("llm_request", spec.start_t)
                        events = spec.events()
                    else:
                        trace.mark("llm_request")
                        events = self.stages.llm.stream(self.history, TOOLS)
                    round_text = ""
                    calls: list[ToolCall] = []
                    async with aclosing(events) as it:
                        async for ev in it:
                            if isinstance(ev, TextDelta):
                                trace.mark("llm_first_token", spec.first_t if from_spec else None)
                                round_text += ev.text
                                for chunk in chunker.push(ev.text):
                                    await self._push_chunk(rec, chunk)
                            elif isinstance(ev, ToolCall):
                                trace.mark("llm_first_token", spec.first_t if from_spec else None)
                                calls.append(ev)
                            elif isinstance(ev, LlmDone) and ev.usage is not None:
                                self._add_usage(trace, "llm_input_tokens", ev.usage.input_tokens)
                                self._add_usage(trace, "llm_cached_tokens", ev.usage.cached_tokens)
                                self._add_usage(trace, "llm_output_tokens", ev.usage.output_tokens)
                    tail = chunker.flush()
                    if tail:
                        await self._push_chunk(rec, tail)
                    if not calls:
                        final_text = round_text
                        break
                    self.history.append({
                        "role": "assistant", "content": round_text or None,
                        "tool_calls": [{"id": c.id, "type": "function",
                                        "function": {"name": c.name, "arguments": c.arguments}} for c in calls],
                    })
                    rec.word_offset = sum(len(c.split()) for c in rec.clauses)
                    rec.clause_offset = len(rec.clauses)
                    for i, call in enumerate(calls):
                        try:
                            await self._run_tool(rec, call)
                        except asyncio.CancelledError:
                            for rest in calls[i + 1:]:
                                self.history.append({"role": "tool", "tool_call_id": rest.id,
                                                     "content": json.dumps({"error": "interrupted"})})
                            raise
                else:
                    final_text = ""
                if final_text.strip():
                    self.history.append({"role": "assistant", "content": final_text})
                    rec.final_index = len(self.history) - 1
            await ctx.end_input()
            await rec.pump
            spoken = " ".join(rec.clauses)
            trace.agent_text = spoken
            if spoken:
                self._send({"type": "transcript", "role": "agent", "turn_id": rec.turn_id,
                            "text": spoken, "final": True})
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:  # noqa: BLE001
            failed = True
            self._send({"type": "error", "message": str(exc)})
            with contextlib.suppress(Exception):
                await ctx.cancel()
            await asyncio.wait({rec.pump})
            if not rec.greeting:
                self.history.append({"role": "assistant", "content": APOLOGY})
        finally:
            if cancelled:
                with contextlib.suppress(Exception):
                    await ctx.cancel()
                await _cancel_task(rec.pump)
            if spec is not None:
                await spec.cancel()
        if (failed or not cancelled) and rec.first_audio_t is None and self.agent_record is rec \
                and self.state == "agent_thinking":
            self._idle()

    # ------------------------------------------------------------------ bookkeeping

    def _send(self, payload, on_sent: Callable[[], None] | None = None, hi: bool = False) -> None:
        self.outbox.put(payload, on_sent, hi)

    def _close_turn(self, rec: AgentTurnRecord) -> None:
        if rec.closed:
            return
        rec.closed = True
        rec.trace.mark("turn_end")
        if not rec.emit_metrics:
            return
        summary = rec.trace.summary()
        self._send({"type": "turn_metrics", **summary})
        self.trace_writer.write_turn(self.session_id, summary)

    def _idle(self) -> None:
        """The agent turn is over (played out, silent, or empty): back to listening."""
        rec, self.agent_record = self.agent_record, None
        if rec is not None:
            self._close_turn(rec)
        if self.consumed:
            del self.pending_finals[: self.consumed]
            self.consumed = 0
        self.state = "listening"
        q, self.queued_eot = self.queued_eot, None
        if q is not None:
            self.end_of_turn(*q)

    async def _shutdown(self, reason: str, started: float) -> None:
        await _cancel_task(self.agent_task)
        await _cancel_task(self.interrupt_task)
        await self._cancel_spec()
        for t in list(self._bg):
            await _cancel_task(t)
        rec = self.agent_record
        if rec is not None:
            self._close_turn(rec)
        audio_s = self.samples_total / IN_RATE
        usage = {**self.usage_totals, "asr_audio_s": round(audio_s, 2)}
        self._send({"type": "session_end", "reason": reason, "usage": usage})
        await self._flush(1.0)
        self.trace_writer.write_session_end(
            self.session_id, self.settings.va_profile, time.monotonic() - started, audio_s, dict(self.usage_totals)
        )
        for stage in (self.stages.asr, self.stages.tts):
            with contextlib.suppress(Exception):
                await stage.close()

    async def _flush(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            item = self.outbox.pop_nowait()
            if item is None:
                return
            payload, on_sent = item
            if isinstance(payload, bytes):
                continue
            try:
                await asyncio.wait_for(self.transport.send_json(payload), max(0.05, deadline - time.monotonic()))
            except Exception:  # noqa: BLE001
                return
            if on_sent:
                on_sent()
