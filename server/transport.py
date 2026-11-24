from typing import Protocol

from fastapi import WebSocket, WebSocketDisconnect


class TransportClosed(Exception):
    pass


class Transport(Protocol):
    async def recv(self) -> bytes | str: ...
    async def send_bytes(self, b: bytes) -> None: ...
    async def send_json(self, obj: dict) -> None: ...
    async def close(self, code: int = 1000) -> None: ...


class WebSocketTransport:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def recv(self) -> bytes | str:
        try:
            message = await self.websocket.receive()
            if message['type'] == 'websocket.disconnect':
                raise TransportClosed()
            if message.get('bytes') is not None:
                return message['bytes']
            if message.get('text') is not None:
                return message['text']
            raise TransportClosed()
        except WebSocketDisconnect as exc:
            raise TransportClosed() from exc

    async def send_bytes(self, b: bytes) -> None:
        await self.websocket.send_bytes(b)

    async def send_json(self, obj: dict) -> None:
        await self.websocket.send_json(obj)

    async def close(self, code: int = 1000) -> None:
        await self.websocket.close(code=code)
