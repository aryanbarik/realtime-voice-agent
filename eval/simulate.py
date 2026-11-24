"""Simulated caller: LLM-written lines, Cartesia TTS, real-time 32 ms frame clock,
virtual 24 kHz player, and client-side latency metrics (plan 5.3)."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

import httpx
import numpy as np
import websockets
from openai import AsyncOpenAI

from eval.noise import (
    AGENT_VOICE,
    CACHE_DIR,
    FRAME,
    SAMPLE_RATE,
    NoiseSource,
    TTS_MODEL,
    ensure_babble,
    load_voices,
    pcm16_to_f32,
    rms,
)
from eval.scenario import Scenario
from server.config import Settings
from server.stages.tts.cartesia import synthesize_bytes

__all__ = ["CallResult", "SimulatedCaller", "load_voices"]

CALLER_MODEL = "gpt-6-luna"
FRAME_S = FRAME / SAMPLE_RATE  # 0.032
OUT_RATE = 24000
PLAYER_START_SAMPLES = 1440
REPLY_DELAY_S = 0.4
CALL_TIMEOUT_S = 300.0
SPEECH_RMS = 0.1  # -20 dBFS
SPEECH_END_THRESHOLD = 10 ** (-40 / 20)
TTS_CACHE = CACHE_DIR / "tts"

CALLER_RULES = """\
You are a person phoning a restaurant (Bella Vista) to talk to the host. Stay in character.
Rules:
- Say only what you would say out loud on the phone, in one or two short sentences.
- Use spoken numbers (say "seven thirty", "four people"), never digits or lists.
- Answer the host's questions; do not invent details that contradict your goal.
- If spell_name is set below, give your name and then spell it letter by letter.
- When your goal is done and confirmed, or clearly impossible, say goodbye and end that message with [HANGUP].
"""


@dataclass
class CallResult:
    scenario_id: str
    end_reason: str = ""
    duration_s: float = 0.0
    caller_lines: list[str] = field(default_factory=list)
    user_finals: list[str] = field(default_factory=list)
    agent_finals: list[str] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)
    turn_metrics: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    usage: dict | None = None
    error: str | None = None


@dataclass
class _Line:
    text: str
    frames: np.ndarray  # [n, 512] float32 speech (normalized), padded
    end_idx: int  # index of the frame holding the speech end
    after_turn: int | None
    is_interrupt: bool = False
    hangup: bool = False
    release_t: float = 0.0
    # filled while sending
    idx: int = 0
    first_frame_t: float | None = None
    speech_end_t: float | None = None
    turn_ref: int = -1
    turn: dict = field(default_factory=dict)


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


class SimulatedCaller:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        http: httpx.AsyncClient,
        voices: list[str],
        cartesia_api_key: str | None = None,
    ) -> None:
        self.openai = openai_client
        self.http = http
        self.voices = [v for v in voices if v != AGENT_VOICE] or voices
        self.cartesia_key = cartesia_api_key or Settings().cartesia_api_key

    # ------------------------------------------------------------------ TTS
    async def _tts(self, voice: str, text: str) -> np.ndarray:
        path = TTS_CACHE / f"{_sha1(voice + text)}.pcm"
        if path.exists():
            pcm = path.read_bytes()
        else:
            pcm = await synthesize_bytes(
                self.http, self.cartesia_key, TTS_MODEL, voice, text, SAMPLE_RATE
            )
            TTS_CACHE.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pcm)
        x = pcm16_to_f32(pcm)
        r = rms(x)
        return x * (SPEECH_RMS / r) if r > 1e-6 else x

    async def _speech(self, voice: str, text: str, pause_ms: int) -> np.ndarray:
        words = text.split()
        if pause_ms > 0 and len(words) >= 2:
            mid = len(text) / 2
            best, pos = 1, 0
            bestd = 1e9
            for i, w in enumerate(words[:-1], start=1):
                pos += len(w) + (1 if i > 1 else 0)
                if abs(pos - mid) < bestd:
                    bestd, best = abs(pos - mid), i
            a, b = " ".join(words[:best]), " ".join(words[best:])
            xa, xb = await asyncio.gather(self._tts(voice, a), self._tts(voice, b))
            gap = np.zeros(int(pause_ms * SAMPLE_RATE / 1000), dtype=np.float32)
            return np.concatenate([xa, gap, xb])
        return await self._tts(voice, text)

    @staticmethod
    def _frames(x: np.ndarray) -> tuple[np.ndarray, int]:
        n = len(x) // FRAME
        end_idx = 0
        for i in range(n + (1 if len(x) % FRAME else 0)):
            seg = x[i * FRAME : (i + 1) * FRAME]
            if len(seg) == FRAME and rms(seg) > SPEECH_END_THRESHOLD:
                end_idx = i
        total = -(-len(x) // FRAME)
        padded = np.zeros(total * FRAME, dtype=np.float32)
        padded[: len(x)] = x
        return padded.reshape(total, FRAME), min(end_idx, max(total - 1, 0))

    # ------------------------------------------------------------------ run
    async def run(
        self,
        ws_url: str,
        scenario: Scenario,
        tenant: str,
        extra_query: dict[str, str] | None = None,
    ) -> CallResult:
        call = _Call(self, scenario, tenant, ws_url, extra_query or {})
        return await call.run()


class _Call:
    def __init__(self, owner, scenario, tenant, ws_url, extra_query) -> None:
        self.o = owner
        self.sc: Scenario = scenario
        self.tenant = tenant
        u = urlparse(ws_url)
        q = urlencode({"tenant": tenant, **extra_query})
        q = f"{u.query}&{q}" if u.query else q
        self.url = urlunparse(u._replace(query=q))
        self.res = CallResult(scenario_id=scenario.id)
        self.voice = owner.voices[int(_sha1(scenario.id), 16) % len(owner.voices)]
        self.ws = None
        self.turn_mode = "auto"
        self.done = asyncio.Event()
        self.history: list[dict] = []
        sys_prompt = CALLER_RULES + (
            f"\nYour name is {scenario.caller.name}. spell_name: {scenario.caller.spell_name}.\n"
            f"Your goal: {scenario.caller.goal}\nPersona: {scenario.caller.persona}\n"
        )
        self.system = sys_prompt
        # sending
        self.send_q: deque[_Line] = deque()
        self.cur: _Line | None = None
        self.noise: NoiseSource | None = None
        self.lines_made = 0
        self.first_line = True
        self.gen_task: asyncio.Task | None = None
        self.gen_busy = False
        self.ready: _Line | None = None  # generated, waiting for playback_done
        self.last_agent_turn = -1
        self.audio_start_t: dict[int, float] = {}
        self.awaiting: list[_Line] = []  # sent lines awaiting first agent audio
        self.done_t: dict[int, float] = {}
        self.agent_text_seen: set[int] = set()
        self.agent_turns_seen: set[int] = set()
        # interrupt
        self.intr = scenario.behaviors.interrupt
        self.int_task: asyncio.Task | None = None
        self.int_line: _Line | None = None
        self.int_released = False
        self.int_started = False
        self.int_hist_idx: int | None = None
        # virtual player
        self.p_queue = 0.0
        self.p_started = False
        self.p_paused = False
        self.p_ended = False
        self.p_turn = -1
        self.p_last = time.monotonic()
        self.played: dict[int, float] = {}
        self.cleared: set[int] = set()
        self.done_sent: set[int] = set()
        self.t0 = 0.0

    # ----- player
    def _advance(self, now: float) -> None:
        dt = now - self.p_last
        self.p_last = now
        if self.p_started and not self.p_paused and self.p_queue > 0:
            used = min(self.p_queue, dt * OUT_RATE)
            self.p_queue -= used
            self.played[self.p_turn] = self.played.get(self.p_turn, 0.0) + used
        if self.p_queue <= 0:
            self.p_queue = 0.0
            if not self.p_ended:
                self.p_started = False
        elif not self.p_started and (self.p_queue >= PLAYER_START_SAMPLES or self.p_ended):
            self.p_started = True

    async def _player_loop(self) -> None:
        while True:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            self._advance(now)
            t = self.p_turn
            if (
                self.p_ended
                and self.p_queue <= 0
                and not self.p_paused
                and t >= 0
                and t not in self.done_sent
                and t not in self.cleared
            ):
                self.done_sent.add(t)
                self.done_t[t] = now
                await self._send({"type": "playback_done", "turn_id": t})
                await self._on_turn_done(t)
            if (
                self.intr
                and self.int_line is not None
                and not self.int_released
                and t == self.intr.agent_turn
                and self.played.get(t, 0.0) * 1000 / OUT_RATE >= self.intr.after_ms
            ):
                self.int_released = True
                self.int_line.release_t = 0.0
                self.send_q.append(self.int_line)

    async def _on_turn_done(self, turn: int) -> None:
        self._try_release(turn)

    def _busy(self) -> bool:
        return bool(
            self.cur
            or self.send_q
            or self.ready
            or self.gen_busy
            or (self.int_task and not self.int_task.done())
            or (self.int_line is not None and not self.int_released)
        )

    # ----- send helpers
    async def _send(self, obj_or_bytes) -> None:
        try:
            if isinstance(obj_or_bytes, bytes):
                await self.ws.send(obj_or_bytes)
            else:
                await self.ws.send(json.dumps(obj_or_bytes))
        except websockets.ConnectionClosed:
            self._finish("closed")

    def _finish(self, reason: str) -> None:
        if not self.done.is_set():
            self.res.end_reason = reason
            self.done.set()

    async def _frame_loop(self) -> None:
        silence = np.zeros(FRAME, dtype=np.float32)
        self.t0 = time.monotonic()
        k = 0
        while not self.done.is_set():
            target = self.t0 + k * FRAME_S
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            k += 1
            now = time.monotonic()
            if self.cur is None and self.send_q and self.send_q[0].release_t <= now:
                self.cur = self.send_q.popleft()
                line = self.cur
                line.turn_ref = self.last_agent_turn
                line.turn = {
                    "client_v2v_ms": None,
                    "false_cut_in": False,
                    "is_interrupt": line.is_interrupt,
                    "client_barge_in_ms": None,
                }
                self.res.turns.append(line.turn)
                if self.turn_mode == "push_to_talk":
                    await self._send({"type": "ptt_start"})
            line = self.cur
            frame = silence
            if line is not None:
                frame = line.frames[line.idx]
            if self.noise is not None:
                frame = frame + self.noise.next_frame()
            pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            await self._send(pcm)
            if line is None:
                continue
            t_sent = time.monotonic()
            if line.idx == 0:
                line.first_frame_t = t_sent
            if line.idx == line.end_idx:
                line.speech_end_t = t_sent
                self.awaiting.append(line)
                if self.turn_mode == "push_to_talk":
                    await self._send({"type": "ptt_end"})
            last = (
                line.end_idx
                if self.turn_mode == "push_to_talk"
                else len(line.frames) - 1
            )
            line.idx += 1
            if line.idx > last:
                self.cur = None
                if line.hangup:
                    await self._send({"type": "hangup"})
                    self._finish("hangup")

    # ----- caller line generation
    async def _llm_line(self, extra: str = "", synthetic_user: str | None = None) -> str:
        msgs = [{"role": "system", "content": self.system + extra}] + self.history
        if not self.history or synthetic_user and self.history[-1]["role"] != "user":
            msgs.append({"role": "user", "content": synthetic_user or "(The line is connected.)"})
        r = await self.o.openai.chat.completions.create(
            model=CALLER_MODEL, temperature=0.4, messages=msgs, max_tokens=120
        )
        return (r.choices[0].message.content or "").strip()

    async def _build_line(self, text: str, after_turn: int | None, interrupt: bool) -> _Line:
        hangup = "[HANGUP]" in text
        spoken = text.replace("[HANGUP]", "").strip()
        pause = self.sc.behaviors.pause_ms if self.first_line else 0
        self.first_line = False
        x = await self.o._speech(self.voice, spoken, pause)
        frames, end_idx = self.o._frames(x)
        self.res.caller_lines.append(spoken)
        return _Line(spoken, frames, end_idx, after_turn, interrupt, hangup)

    def _start_gen(self, after_turn: int) -> None:
        if self.lines_made >= self.sc.max_caller_turns:
            self._finish("max_turns")
            return
        self.gen_busy = True
        self.gen_task = asyncio.create_task(self._gen(after_turn))

    async def _gen(self, after_turn: int) -> None:
        try:
            text = await self._llm_line()
            if not text:
                self._finish("caller_empty")
                return
            self.lines_made += 1
            self.history.append({"role": "assistant", "content": text})
            line = await self._build_line(text, after_turn, False)
            self.ready = line
            self._try_release(after_turn)
        except Exception as e:  # noqa: BLE001
            self.res.error = f"caller: {e!r}"
            self._finish("caller_error")
        finally:
            self.gen_busy = False

    def _try_release(self, turn: int) -> None:
        line = self.ready
        if line is None or line.after_turn != turn or turn not in self.done_t:
            return
        self.ready = None
        line.release_t = self.done_t[turn] + REPLY_DELAY_S
        self.send_q.append(line)

    async def _gen_interrupt(self, turn: int) -> None:
        try:
            extra = (
                f"\nYou are cutting the agent off mid-sentence: {self.intr.instruction}."
            )
            text = await self._llm_line(extra, "(The host is mid-sentence. Interrupt now.)")
            if not text:
                return
            self.lines_made += 1
            line = await self._build_line(text, turn, True)
            self.history.append({"role": "assistant", "content": text})
            if turn not in self.agent_text_seen:
                self.int_hist_idx = len(self.history) - 1
            self.int_line = line
        except Exception as e:  # noqa: BLE001
            self.res.error = f"caller: {e!r}"
            self._finish("caller_error")

    def _add_agent_text(self, turn: int, text: str) -> None:
        msg = {"role": "user", "content": text}
        if self.int_hist_idx is not None and turn == (self.intr.agent_turn if self.intr else -1):
            self.history.insert(self.int_hist_idx, msg)
            self.int_hist_idx = None
        else:
            self.history.append(msg)
        self.agent_text_seen.add(turn)

    # ----- receive
    async def _on_audio(self, nbytes: int, now: float) -> None:
        t = self.p_turn
        if t in self.cleared:
            return
        self._advance(now)
        self.p_queue += nbytes // 2
        if not self.p_started and self.p_queue >= PLAYER_START_SAMPLES:
            self.p_started = True
        for line in list(self.awaiting):
            if (
                line.speech_end_t is not None
                and now >= line.speech_end_t
                and t > line.turn_ref
                and self.audio_start_t.get(t, 0.0) >= line.speech_end_t
            ):
                line.turn["client_v2v_ms"] = (now - line.speech_end_t) * 1000
                line.turn["agent_turn_id"] = t
                self.awaiting.remove(line)

    async def _on_json(self, msg: dict, now: float) -> None:
        typ = msg.get("type")
        if typ == "session":
            self.turn_mode = msg.get("turn_mode", "auto")
            self._session.set()
        elif typ == "agent_audio_start":
            t = msg["turn_id"]
            self._advance(now)
            self.p_turn, self.p_ended, self.p_queue = t, False, 0.0
            self.p_started = False
            self.played.setdefault(t, 0.0)
            self.last_agent_turn = max(self.last_agent_turn, t)
            self.audio_start_t[t] = now
            self.agent_turns_seen.add(t)
            cur = self.cur
            if (
                cur is not None
                and not cur.is_interrupt
                and cur.idx <= cur.end_idx
            ):
                cur.turn["false_cut_in"] = True
            if self.intr and t == self.intr.agent_turn and not self.int_started:
                self.int_started = True
                self.int_task = asyncio.create_task(self._gen_interrupt(t))
        elif typ == "agent_audio_end":
            if msg["turn_id"] == self.p_turn:
                self._advance(now)
                self.p_ended = True
                if not self.p_started and self.p_queue > 0:
                    self.p_started = True
        elif typ == "pause":
            self._advance(now)
            self.p_paused = True
            line = self.int_line
            if (
                line is not None
                and line.first_frame_t is not None
                and line.turn.get("client_barge_in_ms") is None
                and now >= line.first_frame_t
            ):
                line.turn["client_barge_in_ms"] = (now - line.first_frame_t) * 1000
        elif typ == "resume":
            self._advance(now)
            self.p_paused = False
        elif typ == "clear":
            t = msg["turn_id"]
            self._advance(now)
            played = int(self.played.get(t, 0.0))
            self.p_queue = 0.0
            self.p_started = False
            self.p_paused = False
            self.cleared.add(t)
            self.done_t[t] = now
            await self._send({"type": "playback_cleared", "turn_id": t, "played_samples": played})
            await self._on_turn_done(t)
        elif typ == "transcript":
            if not msg.get("final"):
                return
            if msg["role"] == "user":
                self.res.user_finals.append(msg["text"])
            else:
                t = msg["turn_id"]
                self.res.agent_finals.append(msg["text"])
                self._add_agent_text(t, msg["text"])
                interrupt_turn = bool(self.intr and t == self.intr.agent_turn)
                if not self._busy() and not (interrupt_turn and self.int_started):
                    self._start_gen(t)
                    if t in self.cleared or t in self.done_sent:
                        self._try_release(t)
        elif typ == "tool":
            self.res.tools.append(msg)
        elif typ == "turn_metrics":
            self.res.turn_metrics.append(msg)
        elif typ == "error":
            self.res.error = self.res.error or str(msg.get("message"))
        elif typ == "session_end":
            self.res.usage = msg.get("usage")
            self._finish(msg.get("reason") or "session_end")

    async def _recv_loop(self) -> None:
        try:
            async for m in self.ws:
                now = time.monotonic()
                if isinstance(m, bytes):
                    await self._on_audio(len(m), now)
                else:
                    await self._on_json(json.loads(m), now)
        except websockets.ConnectionClosed:
            pass
        self._finish("closed")

    # ----- main
    async def run(self) -> CallResult:
        start = time.monotonic()
        self._session = asyncio.Event()
        tasks: list[asyncio.Task] = []
        try:
            if self.sc.noise is not None:
                await ensure_babble(self.o.http, self.o.cartesia_key)
                off = int(_sha1(self.sc.id)[:8], 16)
                self.noise = NoiseSource.from_file(self.sc.noise.snr_db, off)
            async with websockets.connect(self.url, max_size=None) as ws:
                self.ws = ws
                tasks.append(asyncio.create_task(self._recv_loop()))
                await asyncio.wait_for(self._session.wait(), 15)
                tasks.append(asyncio.create_task(self._frame_loop()))
                tasks.append(asyncio.create_task(self._player_loop()))
                try:
                    await asyncio.wait_for(self.done.wait(), CALL_TIMEOUT_S)
                except asyncio.TimeoutError:
                    self._finish("timeout")
                if self.res.end_reason != "hangup" and not ws.close_code:
                    with contextlib.suppress(Exception):
                        await ws.send(json.dumps({"type": "hangup"}))
                    # allow session_end (usage) to arrive
                    for _ in range(30):
                        if self.res.usage is not None or ws.close_code:
                            break
                        await asyncio.sleep(0.1)
                elif self.res.end_reason == "hangup":
                    for _ in range(30):
                        if self.res.usage is not None or ws.close_code:
                            break
                        await asyncio.sleep(0.1)
        except Exception as e:  # noqa: BLE001
            self.res.error = self.res.error or f"{type(e).__name__}: {e}"
            self.res.end_reason = self.res.end_reason or "error"
        finally:
            for t in tasks + [x for x in (self.gen_task, self.int_task) if x]:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.res.duration_s = time.monotonic() - start
        if not self.res.end_reason:
            self.res.end_reason = "closed"
        return self.res
