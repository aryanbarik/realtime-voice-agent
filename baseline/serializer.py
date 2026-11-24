"""Wire-protocol serializer: our browser/simulator protocol <-> Pipecat frames.

Only frames that Pipecat's FastAPI WebSocket output transport actually hands to
``serialize()`` appear here: audio, interruptions and transport messages.
``TTSStartedFrame`` / ``TTSStoppedFrame`` / ``TranscriptionFrame`` never reach
a serializer, so ``bot.py`` converts them into ``OutputTransportMessageFrame``
dicts (see ``ProtocolEmitter``) which this serializer sends as JSON text.
"""

from __future__ import annotations

import json

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class ClientProtocolSerializer(FrameSerializer):
    class InputParams(FrameSerializer.InputParams):
        audio_in_sample_rate: int | None = 16000
        audio_out_sample_rate: int | None = 24000

    def __init__(self, params: "ClientProtocolSerializer.InputParams | None" = None, **kwargs):
        super().__init__(params or ClientProtocolSerializer.InputParams(), **kwargs)
        # Updated by ProtocolEmitter so `clear` carries the current turn id.
        self.turn_id = 0

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return await self._resample_audio_out(frame)
        if isinstance(frame, InterruptionFrame):
            await self._reset_audio_out()
            return json.dumps({"type": "clear", "turn_id": self.turn_id})
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame) or not isinstance(frame.message, dict):
                return None
            return json.dumps(frame.message)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, (bytes, bytearray)):
            audio = await self._resample_audio_in(bytes(data))
            if not audio:
                return None
            return InputAudioRawFrame(audio=audio, sample_rate=self._sample_rate, num_channels=1)
        # JSON control messages (ptt_*, playback_*, hangup) need no pipeline
        # action: Pipecat plays out and detects turns itself, and a hangup is
        # followed by the client closing the socket.
        return None
