"""Self-hosted speech server (Parakeet ASR + Kokoro TTS) for the GPU pod.

Runs in the GPU venv and imports nothing from ``server/``.

Auth: every route requires ``Authorization: Bearer $SELFHOST_API_KEY``.
WebSocket connections without it are accepted and immediately closed with 1008.

WS /asr   binary PCM16LE 16 kHz in; JSON partial/final out.
WS /tts   JSON speak/cancel in; JSON audio/words/done out.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import inspect
import json
import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect

log = logging.getLogger("speech_server")

ASR_RATE = 16000
TTS_RATE = 24000
FRAME = 512  # samples per Silero frame (32 ms)
FRAME_BYTES = FRAME * 2

PREROLL_S = 0.3
PARTIAL_EVERY_S = 0.4
PARTIAL_MAX_S = 20.0
SILENCE_FINAL_S = 3.0
MIN_ASR_SAMPLES = int(0.3 * ASR_RATE)
VAD_ON = 0.5
VAD_OFF = 0.35
VAD_ONSET_FRAMES = 2
DEFAULT_VOICE = "af_heart"


class Models:
    """Loaded models plus one lock and one single-thread executor per GPU model."""

    def __init__(self) -> None:
        self.asr = None
        self.kokoro = None
        self.asr_lock = asyncio.Lock()
        self.tts_lock = asyncio.Lock()
        self.asr_pool = ThreadPoolExecutor(1, thread_name_prefix="asr")
        self.tts_pool = ThreadPoolExecutor(1, thread_name_prefix="tts")

    # ---- blocking implementations (run only inside the model's executor) ----
    def _transcribe(self, audio: np.ndarray) -> str:
        if audio.shape[0] < MIN_ASR_SAMPLES:
            audio = np.pad(audio, (0, MIN_ASR_SAMPLES - audio.shape[0]))
        with torch.inference_mode():
            out = self.asr.transcribe([audio], batch_size=1, verbose=False)
        if isinstance(out, tuple):  # older NeMo: (best, all)
            out = out[0]
        hyp = out[0]
        return (hyp if isinstance(hyp, str) else hyp.text).strip()

    async def transcribe(self, audio: np.ndarray) -> str:
        loop = asyncio.get_running_loop()
        async with self.asr_lock:
            return await loop.run_in_executor(self.asr_pool, self._transcribe, audio)


models = Models()


def _load_vad():
    from silero_vad import load_silero_vad

    return load_silero_vad()


def _make_kpipeline():
    from kokoro import KPipeline

    kwargs = {"lang_code": "a"}
    if "device" in inspect.signature(KPipeline.__init__).parameters:
        kwargs["device"] = "cuda"
    return KPipeline(**kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("SELFHOST_API_KEY"):
        raise RuntimeError("SELFHOST_API_KEY is not set")
    torch.set_num_threads(1)
    from nemo.collections.asr.models import ASRModel

    log.info("loading parakeet")
    models.asr = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").cuda().eval()
    log.info("loading kokoro")
    models.kokoro = _make_kpipeline()
    log.info("loading silero")
    vad = _load_vad()

    log.info("warming models")
    await models.transcribe(np.zeros(ASR_RATE, dtype=np.float32))
    loop = asyncio.get_running_loop()

    def warm_tts() -> None:
        for _ in models.kokoro("Hello there.", voice=DEFAULT_VOICE):
            pass

    await loop.run_in_executor(models.tts_pool, warm_tts)
    with torch.inference_mode():
        for _ in range(4):
            vad(torch.zeros(FRAME), ASR_RATE)
        vad.reset_states()
    log.info("ready")
    yield
    models.asr_pool.shutdown(wait=False, cancel_futures=True)
    models.tts_pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------- auth

def _key_ok(header: str | None) -> bool:
    expected = os.environ.get("SELFHOST_API_KEY", "")
    if not expected or not header:
        return False
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(token.strip().encode(), expected.encode())


async def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not _key_ok(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")


async def ws_authorized(ws: WebSocket) -> bool:
    """Accept then close with 1008 if the bearer key is missing or wrong."""
    await ws.accept()
    if not _key_ok(ws.headers.get("authorization")):
        await ws.close(code=1008)
        return False
    return True


@app.get("/health", dependencies=[Depends(require_auth)])
async def health() -> dict:
    return {"ok": True}


# ---------------------------------------------------------------- ASR

class AsrConn:
    """Per-connection VAD gate, buffer, and transcription scheduling."""

    def __init__(self, ws: WebSocket, partials: bool) -> None:
        self.ws = ws
        self.partials = partials
        self.vad = _load_vad()  # silero model is stateful: one per connection
        self.send_lock = asyncio.Lock()
        self.tasks: set[asyncio.Task] = set()
        self.pending = bytearray()  # incomplete input frame
        self.total = 0  # samples received
        self.preroll: deque[bytes] = deque(maxlen=int(PREROLL_S * ASR_RATE / FRAME) + 1)
        self.preroll_start = 0  # sample index of the first preroll frame
        self.generation = 0
        self.voiced = False
        self.voiced_run = 0
        self.reset_utterance()

    def reset_utterance(self) -> None:
        self.generation += 1
        self.active = False
        self.buf = bytearray()
        self.buf_start = 0  # sample index where buf begins
        self.last_voiced_end = 0
        self.last_partial_at = 0
        self.partial_busy = False

    async def send(self, obj: dict) -> None:
        async with self.send_lock:
            await self.ws.send_text(json.dumps(obj))

    def spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    def _prob(self, frame: bytes) -> float:
        x = torch.from_numpy(np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0)
        with torch.inference_mode():
            return float(self.vad(x, ASR_RATE).item())

    def feed(self, data: bytes) -> None:
        self.pending += data
        while len(self.pending) >= FRAME_BYTES:
            frame = bytes(self.pending[:FRAME_BYTES])
            del self.pending[:FRAME_BYTES]
            self._on_frame(frame)

    def _on_frame(self, frame: bytes) -> None:
        start = self.total
        self.total += FRAME
        p = self._prob(frame)
        if p >= VAD_ON:
            self.voiced = True
        elif p < VAD_OFF:
            self.voiced = False
        self.voiced_run = self.voiced_run + 1 if self.voiced else 0

        if not self.active:
            if not self.preroll:
                self.preroll_start = start
            elif len(self.preroll) == self.preroll.maxlen:
                self.preroll_start += FRAME
            self.preroll.append(frame)
            if self.voiced_run >= VAD_ONSET_FRAMES:
                self.active = True
                self.buf = bytearray(b"".join(self.preroll))
                self.buf_start = self.preroll_start
                self.preroll.clear()
                self.last_voiced_end = self.total
                self.last_partial_at = self.total
            return

        self.buf += frame
        if self.voiced:
            self.last_voiced_end = self.total
        if (
            self.partials
            and not self.partial_busy
            and self.total - self.last_partial_at >= PARTIAL_EVERY_S * ASR_RATE
        ):
            self.last_partial_at = self.total
            self.partial_busy = True
            tail = int(PARTIAL_MAX_S * ASR_RATE) * 2
            snap = bytes(self.buf[-tail:])
            snap_start = self.buf_start + (len(self.buf) - len(snap)) // 2
            self.spawn(self._partial(self.generation, snap, snap_start, self.last_voiced_end))
        if self.total - self.last_voiced_end >= SILENCE_FINAL_S * ASR_RATE:
            self.spawn(self._final(self._take(), from_finalize=False))

    def _take(self) -> tuple[bytes, int, int]:
        """Snapshot the utterance and reset immediately so new audio starts a fresh one."""
        snap = (bytes(self.buf), self.buf_start, self.last_voiced_end if self.active else self.total)
        self.reset_utterance()
        self.preroll.clear()
        return snap

    async def _partial(self, gen: int, pcm: bytes, start: int, end: int) -> None:
        try:
            text = await models.transcribe(_f32(pcm))
            if gen != self.generation or not text:
                return
            await self.send(
                {"type": "partial", "text": text, "start_s": start / ASR_RATE, "end_s": end / ASR_RATE}
            )
        finally:
            if gen == self.generation:
                self.partial_busy = False

    async def _final(self, snap: tuple[bytes, int, int], from_finalize: bool) -> None:
        pcm, start, end = snap
        text = await models.transcribe(_f32(pcm)) if pcm else ""
        await self.send(
            {
                "type": "final",
                "text": text,
                "from_finalize": from_finalize,
                "start_s": start / ASR_RATE,
                "end_s": max(end, start) / ASR_RATE,
            }
        )

    def finalize(self) -> None:
        self.spawn(self._final(self._take(), from_finalize=True))


def _f32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


@app.websocket("/asr")
async def asr_ws(ws: WebSocket) -> None:
    if not await ws_authorized(ws):
        return
    conn = AsrConn(ws, partials=ws.query_params.get("partials", "1") != "0")
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                conn.feed(msg["bytes"])
            elif msg.get("text") is not None:
                try:
                    obj = json.loads(msg["text"])
                except ValueError:
                    continue
                if obj.get("type") == "finalize":
                    conn.finalize()
    except WebSocketDisconnect:
        pass
    finally:
        for t in list(conn.tasks):
            t.cancel()


# ---------------------------------------------------------------- TTS

@dataclass
class TtsContext:
    cancelled: bool = False
    audio_s: float = 0.0
    queue: deque = field(default_factory=deque)  # of (text, final)


class TtsConn:
    def __init__(self, ws: WebSocket, voice: str) -> None:
        self.ws = ws
        self.voice = voice
        self.contexts: dict[str, TtsContext] = {}
        self.cancelled_ids: deque[str] = deque(maxlen=256)
        self.order: asyncio.Queue[str | None] = asyncio.Queue()  # one token per queued item

    async def send(self, obj: dict) -> None:
        await self.ws.send_text(json.dumps(obj))

    def speak(self, cid: str, text: str, final: bool) -> None:
        if cid in self.cancelled_ids:
            return
        ctx = self.contexts.setdefault(cid, TtsContext())
        ctx.queue.append((text, final))
        self.order.put_nowait(cid)

    def cancel(self, cid: str) -> None:
        if cid not in self.cancelled_ids:
            self.cancelled_ids.append(cid)
        ctx = self.contexts.pop(cid, None)
        if ctx:
            ctx.cancelled = True
            ctx.queue.clear()

    async def worker(self) -> None:
        while True:
            cid = await self.order.get()
            if cid is None:
                return
            ctx = self.contexts.get(cid)
            if ctx is None or ctx.cancelled or not ctx.queue:
                continue
            text, final = ctx.queue.popleft()
            if text.strip():
                await self._synthesize(cid, ctx, text)
            if final and not ctx.cancelled:
                self.contexts.pop(cid, None)
                await self.send({"type": "done", "context_id": cid})

    async def _synthesize(self, cid: str, ctx: TtsContext, text: str) -> None:
        loop = asyncio.get_running_loop()
        done = object()
        it = iter(models.kokoro(text, voice=self.voice))

        def step():
            return next(it, done)

        while not ctx.cancelled:
            async with models.tts_lock:
                if ctx.cancelled:
                    break
                res = await loop.run_in_executor(models.tts_pool, step)
            if res is done:
                break
            audio = getattr(res, "audio", None)
            if audio is None:
                continue
            if torch.is_tensor(audio):
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                continue
            pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            dur = audio.size / TTS_RATE
            chunk_text = getattr(res, "graphemes", None) or text
            words = chunk_text.split()
            if ctx.cancelled:
                break
            await self.send(
                {"type": "audio", "context_id": cid, "data": base64.b64encode(pcm).decode("ascii")}
            )
            if words:
                total_chars = sum(len(w) for w in words)
                starts, ends, t = [], [], ctx.audio_s
                for w in words:
                    span = dur * len(w) / total_chars
                    starts.append(round(t, 4))
                    t += span
                    ends.append(round(t, 4))
                await self.send(
                    {"type": "words", "context_id": cid, "words": words, "start": starts, "end": ends}
                )
            ctx.audio_s += dur


@app.websocket("/tts")
async def tts_ws(ws: WebSocket) -> None:
    if not await ws_authorized(ws):
        return
    conn = TtsConn(ws, ws.query_params.get("voice") or DEFAULT_VOICE)
    worker = asyncio.create_task(conn.worker())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is None:
                continue
            try:
                obj = json.loads(msg["text"])
            except ValueError:
                continue
            kind, cid = obj.get("type"), obj.get("context_id")
            if not isinstance(cid, str):
                continue
            if kind == "speak":
                conn.speak(cid, str(obj.get("text", "")), bool(obj.get("final", False)))
            elif kind == "cancel":
                conn.cancel(cid)
    except WebSocketDisconnect:
        pass
    finally:
        for c in list(conn.contexts):
            conn.cancel(c)
        conn.order.put_nowait(None)
        worker.cancel()
