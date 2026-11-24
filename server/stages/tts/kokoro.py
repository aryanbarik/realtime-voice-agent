from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator
from urllib.parse import quote

import websockets

from server.stages.base import TtsAudio, TtsWords
from server.stages.tts.cartesia import TtsError

_END = object()


class KokoroContext:
    def __init__(self, tts: KokoroTTS, context_id: str) -> None:
        self.context_id = context_id
        self._tts = tts
        self._queue: asyncio.Queue = asyncio.Queue()
        self._done = False

    def _speak(self, text: str, final: bool) -> dict:
        return {"type": "speak", "context_id": self.context_id, "text": text, "final": final}

    async def push_text(self, text: str) -> None:
        if self._done:
            return
        self._tts.chars_sent += len(text)
        await self._tts._send(self._speak(text + " ", False))

    async def end_input(self) -> None:
        if self._done:
            return
        await self._tts._send(self._speak("", True))

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
            await self._tts._send(
                {"type": "cancel", "context_id": self.context_id}, connect=False
            )


class KokoroTTS:
    """Client for the self-hosted speech server's /tts websocket."""

    sample_rate = 24000

    def __init__(self, base_url: str, api_key: str, voice: str = "af_heart") -> None:
        self.url = f"{base_url.rstrip('/')}/tts?voice={quote(voice)}"
        self.api_key = api_key
        self.chars_sent = 0
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._contexts: dict[str, KokoroContext] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._lock:
            await self._connect_locked()

    async def _connect_locked(self) -> None:
        if self._ws is not None and self._ws.state.name == "OPEN":
            return
        self._ws = await websockets.connect(
            self.url, additional_headers={"Authorization": f"Bearer {self.api_key}"}
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

    def new_context(self, context_id: str) -> KokoroContext:
        ctx = KokoroContext(self, context_id)
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
                if kind == "audio":
                    ctx._queue.put_nowait(TtsAudio(base64.b64decode(msg["data"])))
                elif kind == "words":
                    ctx._queue.put_nowait(
                        TtsWords(
                            list(msg.get("words", [])),
                            [float(x) for x in msg.get("start", [])],
                            [float(x) for x in msg.get("end", [])],
                        )
                    )
                elif kind == "done":
                    ctx._done = True
                    ctx._queue.put_nowait(_END)
                    self._contexts.pop(ctx.context_id, None)
                elif kind == "error":
                    ctx._done = True
                    ctx._queue.put_nowait(TtsError(str(msg.get("message") or msg)))
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
                ctx._queue.put_nowait(TtsError(f"Kokoro connection lost: {failure or 'closed'}"))
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
