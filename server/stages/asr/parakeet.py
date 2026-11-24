from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator

import websockets

from server.stages.asr.deepgram import AsrError
from server.stages.base import AsrResult, AsrUtteranceEnd

_END = object()


class ParakeetASR:
    """Client for the self-hosted speech server's /asr websocket."""

    def __init__(self, base_url: str, api_key: str, partials: bool = True) -> None:
        self.url = base_url.rstrip("/") + "/asr" + ("" if partials else "?partials=0")
        self.api_key = api_key
        self.audio_seconds_sent = 0.0
        self._ws = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._recv_task: asyncio.Task | None = None
        self._closing = False

    async def _open(self):
        return await websockets.connect(
            self.url, additional_headers={"Authorization": f"Bearer {self.api_key}"}
        )

    async def connect(self) -> None:
        self._closing = False
        self._ws = await self._open()
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def send_audio(self, pcm16: bytes) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(pcm16)
        except websockets.ConnectionClosed:
            return
        self.audio_seconds_sent += len(pcm16) / 32000.0

    async def finalize(self) -> None:
        ws = self._ws
        if ws is None:
            return
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(json.dumps({"type": "finalize"}))

    async def events(self) -> AsyncIterator[AsrResult | AsrUtteranceEnd]:
        while True:
            item = await self._queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    @staticmethod
    def _parse(raw) -> AsrResult | None:
        if isinstance(raw, bytes):
            return None
        msg = json.loads(raw)
        kind = msg.get("type")
        if kind not in ("partial", "final"):
            return None
        final = kind == "final"
        return AsrResult(
            text=msg.get("text", ""),
            is_final=final,
            speech_final=final,
            from_finalize=bool(msg.get("from_finalize", False)) if final else False,
            start_s=float(msg.get("start_s", 0.0)),
            end_s=float(msg.get("end_s", 0.0)),
        )

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
                err: Exception = AsrError("Parakeet connection closed unexpectedly")
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                if self._closing:
                    break
                err = e
            if reconnected:
                self._queue.put_nowait(AsrError(f"Parakeet connection failed: {err}"))
                return
            reconnected = True
            try:
                self._ws = await self._open()
            except Exception as e:  # noqa: BLE001
                self._queue.put_nowait(AsrError(f"Parakeet reconnect failed: {e}"))
                return
        self._queue.put_nowait(_END)

    async def close(self) -> None:
        self._closing = True
        ws, self._ws = self._ws, None
        if ws is not None:
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
