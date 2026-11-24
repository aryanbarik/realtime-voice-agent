"""Pipecat baseline for the voice agent: same stack as v3-hosted, framework-built.

Run: uv run python bot.py --port 8090
Speaks the same /ws protocol as the main server (see serializer.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import date
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from serializer import ClientProtocolSerializer  # noqa: E402

from server.prompts import GREETING, build_system_prompt  # noqa: E402
from server.tools.schemas import TOOLS  # noqa: E402

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema  # noqa: E402
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema  # noqa: E402
from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    Frame,
    InterruptionFrame,
    LLMRunFrame,
    OutputTransportMessageFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.worker import PipelineParams, PipelineWorker  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.services.cartesia.tts import CartesiaTTSService  # noqa: E402
from pipecat.services.deepgram.stt import DeepgramSTTService  # noqa: E402
from pipecat.services.llm_service import FunctionCallParams  # noqa: E402
from pipecat.services.openai.llm import OpenAILLMService  # noqa: E402
from pipecat.transports.websocket.fastapi import (  # noqa: E402
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

TODAY = date(2026, 11, 23)  # matches VA_TODAY used by eval.run_eval
LLM_MODEL = "gpt-6-luna"
TTS_MODEL = "sonic-3.6"
VOICE = "a0e99841-438c-4a64-b679-ae501e7d6091"
TENANT_RE = r"^[a-z0-9-]{1,64}$"

app = FastAPI()


class ProtocolEmitter(FrameProcessor):
    """Turns Pipecat frames into protocol messages that the output transport
    serializes in order with the audio.

    ``user_side=True`` (placed after STT): user final transcripts.
    ``user_side=False`` (placed after TTS): agent_audio_start/end and clear
    bookkeeping. A turn id advances when the user has spoken since the last
    agent audio, so a tool-call filler phrase and the answer share one turn.
    """

    def __init__(self, serializer: ClientProtocolSerializer, state: dict, *, user_side: bool):
        super().__init__()
        self._ser = serializer
        self._state = state
        self._user_side = user_side

    async def _emit(self, message: dict):
        await self.push_frame(OutputTransportMessageFrame(message=message))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if self._user_side:
            await self.push_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame) and frame.text.strip():
                self._state["user_spoke"] = True
                await self._emit(
                    {
                        "type": "transcript",
                        "role": "user",
                        "turn_id": self._state["turn_id"] + 1,
                        "text": frame.text.strip(),
                        "final": True,
                    }
                )
            return

        if isinstance(frame, TTSStartedFrame):
            if self._state["user_spoke"]:
                self._state["turn_id"] += 1
                self._state["user_spoke"] = False
            self._ser.turn_id = self._state["turn_id"]
            await self.push_frame(frame, direction)
            await self._emit({"type": "agent_audio_start", "turn_id": self._state["turn_id"]})
        elif isinstance(frame, TTSStoppedFrame):
            await self.push_frame(frame, direction)
            await self._emit({"type": "agent_audio_end", "turn_id": self._state["turn_id"]})
        else:
            await self.push_frame(frame, direction)


def _backend_call(backend: str, tenant: str, name: str, args: dict):
    base = f"{backend.rstrip('/')}/t/{tenant}"
    if name == "check_availability":
        return "GET", f"{base}/availability", {"params": args}
    if name == "book_reservation":
        return "POST", f"{base}/reservations", {"json": args}
    if name == "cancel_reservation":
        return "POST", f"{base}/reservations/cancel", {"json": args}
    return None


def make_tool_handler(name: str, http: httpx.AsyncClient, backend: str, tenant: str):
    async def handler(params: FunctionCallParams):
        call = _backend_call(backend, tenant, name, dict(params.arguments))
        try:
            assert call is not None
            method, url, kwargs = call
            resp = await http.request(method, url, **kwargs)
            try:
                body = resp.json()
            except ValueError:
                body = {"error": f"http_{resp.status_code}"}
        except (httpx.HTTPError, AssertionError):
            body = {"error": "backend_unavailable"}
        await params.result_callback(body)

    return handler


def build_tools(http: httpx.AsyncClient, backend: str, tenant: str) -> ToolsSchema:
    schemas = []
    for tool in TOOLS:
        fn = tool["function"]
        schemas.append(
            FunctionSchema(
                name=fn["name"],
                description=fn["description"],
                properties=fn["parameters"]["properties"],
                required=fn["parameters"]["required"],
                handler=make_tool_handler(fn["name"], http, backend, tenant),
            )
        )
    return ToolsSchema(standard_tools=schemas)


async def run_bot(websocket: WebSocket, tenant: str, backend: str):
    serializer = ClientProtocolSerializer()
    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTService.Settings(model="nova-3"),
    )
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(model=TTS_MODEL, voice=VOICE),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(model=LLM_MODEL, temperature=0.3, max_tokens=300),
    )

    async with httpx.AsyncClient(timeout=2.0) as http:
        context = LLMContext(
            messages=[
                {"role": "system", "content": build_system_prompt(TODAY)},
                {"role": "assistant", "content": GREETING},
            ],
            tools=build_tools(http, backend, tenant),
        )
        user_agg, asst_agg = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        state = {"turn_id": 0, "user_spoke": False}
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                ProtocolEmitter(serializer, state, user_side=True),
                user_agg,
                llm,
                tts,
                ProtocolEmitter(serializer, state, user_side=False),
                transport.output(),
                asst_agg,
            ]
        )
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        )
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)

        @asst_agg.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message):
            text = message.content.strip()
            if not text:
                return
            if message.interrupted:
                text += "…"
            await worker.queue_frames(
                [
                    OutputTransportMessageFrame(
                        message={
                            "type": "transcript",
                            "role": "agent",
                            "turn_id": state["turn_id"],
                            "text": text,
                            "final": True,
                        }
                    )
                ]
            )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            # Greeting is already in `context`; speak it without re-adding it.
            await worker.queue_frames([TTSSpeakFrame(GREETING, append_to_context=False)])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            await runner.cancel()

        await runner.run()


@app.get("/healthz")
async def healthz():
    return {"ok": True, "stack": "pipecat"}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, tenant: str = "demo", backend: str = ""):
    import re

    await websocket.accept()
    if not re.match(TENANT_RE, tenant) or not backend:
        await websocket.close(code=1008)
        return
    await websocket.send_json(
        {
            "type": "session",
            "session_id": uuid.uuid4().hex,
            "profile": "baseline-pipecat",
            "turn_mode": "auto",
            "input_sample_rate": 16000,
            "output_sample_rate": 24000,
        }
    )
    try:
        await run_bot(websocket, tenant, backend)
    except Exception as exc:  # report to the client instead of dropping silently
        logger.exception("bot failed")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    for key in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "CARTESIA_API_KEY"):
        if not os.environ.get(key):
            sys.exit(f"{key} is not set (put it in the repo-root .env)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
