from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator

import httpx
import websockets

from server.stages.base import TtsAudio, TtsWords

CARTESIA_VERSION = "2026-08-14"
BYTES_URL = "https://api.cartesia.ai/tts/bytes"


async def synthesize_bytes(
    http: httpx.AsyncClient,
    api_key: str,
    model: str,
    voice: str,
    text: str,
    sample_rate: int,
) -> bytes:
    resp = await http.post(
        BYTES_URL,
        json={
            "model_id": model,
            "transcript": text,
            "voice": {"id": voice},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": sample_rate,
            },
            "language": "en",
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Cartesia-Version": CARTESIA_VERSION,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


class TtsError(RuntimeError):
    pass


WS_URL = f"wss://api.cartesia.ai/tts/websocket?cartesia_version={CARTESIA_VERSION}"
_END = object()


class CartesiaContext:
    def __init__(self, tts: CartesiaTTS, context_id: str) -> None:
        self.context_id = context_id
        self._tts = tts
        self._queue: asyncio.Queue = asyncio.Queue()
        self._done = False

    def _payload(self, text: str, cont: bool) -> dict:
        t = self._tts
        return {
            "model_id": t.model,
            "transcript": text,
            "voice": {"id": t.voice},
            "language": "en",
            "context_id": self.context_id,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": t.sample_rate,
            },
            "continue": cont,
            "add_timestamps": True,
            "max_buffer_delay_ms": 0,
        }

    async def push_text(self, text: str) -> None:
        if self._done:
            return
        self._tts.chars_sent += len(text)
        await self._tts._send(self._payload(text + " ", True))

    async def end_input(self) -> None:
        if self._done:
            return
        await self._tts._send(self._payload("", False))

    async def events(self) -> AsyncIterator[TtsAudio | TtsWords]:
        while True:
            item = await self._queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def cancel(self) -> None:
        if self._done:
            return
        self._done = True
        self._tts._contexts.pop(self.context_id, None)
        self._queue.put_nowait(_END)
        with contextlib.suppress(Exception):
            await self._tts._send({"context_id": self.context_id, "cancel": True}, connect=False)


class CartesiaTTS:
    def __init__(self, api_key: str, model: str, voice: str, sample_rate: int = 24000) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self.chars_sent = 0
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._contexts: dict[str, CartesiaContext] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._lock:
            await self._connect_locked()

    async def _connect_locked(self) -> None:
        if self._ws is not None and self._ws.state.name == "OPEN":
            return
        self._ws = await websockets.connect(
            WS_URL, additional_headers={"Authorization": f"Bearer {self.api_key}"}
        )
        self._recv_task = asyncio.create_task(self._recv_loop(self._ws))

    async def _send(self, payload: dict, connect: bool = True) -> None:
        if connect:
            async with self._lock:
                await self._connect_locked()
        ws = self._ws
        if ws is None:
            return
        await ws.send(json.dumps(payload))

    def new_context(self, context_id: str) -> CartesiaContext:
        ctx = CartesiaContext(self, context_id)
        self._contexts[context_id] = ctx
        return ctx

    async def _recv_loop(self, ws) -> None:
        failure: BaseException | None = None
        try:
            async for raw in ws:
                msg = json.loads(raw)
                ctx = self._contexts.get(msg.get("context_id", ""))
                if ctx is None:
                    continue
                kind = msg.get("type")
                if kind == "chunk":
                    ctx._queue.put_nowait(TtsAudio(base64.b64decode(msg["data"])))
                elif kind == "timestamps":
                    wt = msg.get("word_timestamps") or {}
                    ctx._queue.put_nowait(
                        TtsWords(
                            list(wt.get("words", [])),
                            [float(x) for x in wt.get("start", [])],
                            [float(x) for x in wt.get("end", [])],
                        )
                    )
                elif kind == "done":
                    ctx._done = True
                    ctx._queue.put_nowait(_END)
                    self._contexts.pop(ctx.context_id, None)
                elif kind == "error":
                    ctx._done = True
                    ctx._queue.put_nowait(
                        TtsError(str(msg.get("error") or msg.get("message") or msg))
                    )
                    self._contexts.pop(ctx.context_id, None)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            failure = e
        if self._ws is ws:
            self._ws = None
        for ctx in list(self._contexts.values()):
            if not ctx._done:
                ctx._done = True
                ctx._queue.put_nowait(TtsError(f"Cartesia connection lost: {failure or 'closed'}"))
        self._contexts.clear()

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        task = self._recv_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        for ctx in list(self._contexts.values()):
            ctx._done = True
            ctx._queue.put_nowait(_END)
        self._contexts.clear()
