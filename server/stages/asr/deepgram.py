from __future__ import annotations

import asyncio
import contextlib
import io
import json
import time
import wave
from collections.abc import AsyncIterator

import httpx
import websockets

from server.stages.base import AsrResult, AsrUtteranceEnd

LISTEN_URL = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&punctuate=true"


def pcm_to_wav(pcm16: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


async def transcribe_wav(http: httpx.AsyncClient, api_key: str, wav: bytes) -> str:
    resp = await http.post(
        LISTEN_URL,
        content=wav,
        headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["results"]["channels"][0]["alternatives"][0]["transcript"]


class AsrError(RuntimeError):
    pass


_END = object()


class DeepgramStreamingASR:
    def __init__(
        self, api_key: str, endpointing_ms: int = 300, utterance_end_ms: int = 1000
    ) -> None:
        self.api_key = api_key
        self.url = (
            "wss://api.deepgram.com/v1/listen?model=nova-3&encoding=linear16"
            "&sample_rate=16000&channels=1&interim_results=true&punctuate=true"
            f"&smart_format=true&endpointing={endpointing_ms}"
            f"&utterance_end_ms={utterance_end_ms}&vad_events=true"
        )
        self.audio_seconds_sent = 0.0
        self._ws = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._recv_task: asyncio.Task | None = None
        self._closing = False
        self._last_audio_t = 0.0

    async def _open(self):
        return await websockets.connect(
            self.url, additional_headers={"Authorization": f"Token {self.api_key}"}
        )

    async def connect(self) -> None:
        self._closing = False
        self._ws = await self._open()
        self._last_audio_t = time.monotonic()
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def send_audio(self, pcm16: bytes) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(pcm16)
        except websockets.ConnectionClosed:
            # The receive loop owns reconnection; audio during the gap is dropped.
            return
        self.audio_seconds_sent += len(pcm16) / 32000.0
        self._last_audio_t = time.monotonic()

    async def finalize(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            if time.monotonic() - self._last_audio_t > 4.0:
                await ws.send(json.dumps({"type": "KeepAlive"}))
            await ws.send(json.dumps({"type": "Finalize"}))
        except websockets.ConnectionClosed:
            pass

    async def events(self) -> AsyncIterator[AsrResult | AsrUtteranceEnd]:
        while True:
            item = await self._queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _parse(self, raw) -> AsrResult | AsrUtteranceEnd | None:
        if isinstance(raw, bytes):
            return None
        msg = json.loads(raw)
        kind = msg.get("type")
        if kind == "Results":
            alts = msg.get("channel", {}).get("alternatives") or [{}]
            start = float(msg.get("start", 0.0))
            return AsrResult(
                text=alts[0].get("transcript", ""),
                is_final=bool(msg.get("is_final", False)),
                speech_final=bool(msg.get("speech_final", False)),
                from_finalize=bool(msg.get("from_finalize", False)),
                start_s=start,
                end_s=start + float(msg.get("duration", 0.0)),
            )
        if kind == "UtteranceEnd":
            return AsrUtteranceEnd(float(msg.get("last_word_end", 0.0)))
        return None

    async def _recv_loop(self) -> None:
        reconnected = False
        while True:
            try:
                async for raw in self._ws:
                    ev = self._parse(raw)
                    if ev is not None:
                        reconnected = False
                        self._queue.put_nowait(ev)
                if self._closing:
                    break
                err: Exception = AsrError("Deepgram connection closed unexpectedly")
            except websockets.ConnectionClosed as e:
                if self._closing:
                    break
                err = e
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                if self._closing:
                    break
                err = e
            if reconnected:
                self._queue.put_nowait(AsrError(f"Deepgram connection failed: {err}"))
                return
            reconnected = True
            try:
                self._ws = await self._open()
            except Exception as e:  # noqa: BLE001
                self._queue.put_nowait(AsrError(f"Deepgram reconnect failed: {e}"))
                return
        self._queue.put_nowait(_END)

    async def close(self) -> None:
        self._closing = True
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "CloseStream"}))
            with contextlib.suppress(Exception):
                await ws.close()
        task = self._recv_task
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
        self._queue.put_nowait(_END)
