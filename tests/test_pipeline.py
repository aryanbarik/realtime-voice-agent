import asyncio
import json
from datetime import date

import numpy as np
import pytest

from server.config import PipelineConfig, Settings
from server.pipeline import StreamingSession, Stages
from server.prompts import GREETING
from server.stages.base import AsrResult, LlmDone, TextDelta, ToolCall, TtsAudio, TtsWords, Usage
from server.tracing import TraceWriter
from server.transport import TransportClosed

VOICED = np.full(512, 10000, dtype="<i2").tobytes()
SILENT = bytes(1024)
BLOCK = object()


class FakeTransport:
    def __init__(self):
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list = []

    async def recv(self):
        item = await self.incoming.get()
        if item is None:
            raise TransportClosed()
        return item

    async def send_bytes(self, b):
        self.sent.append(b)

    async def send_json(self, obj):
        self.sent.append(obj)

    async def close(self, code=1000):
        pass

    def types(self):
        return [m["type"] for m in self.sent if isinstance(m, dict)]

    def json(self, kind):
        return [m for m in self.sent if isinstance(m, dict) and m["type"] == kind]

    def say(self, msg: dict):
        self.incoming.put_nowait(json.dumps(msg))


class FakeVAD:
    def prob(self, frame):
        return 1.0 if frame.max() > 0.1 else 0.0


class FakeASR:
    def __init__(self):
        self.q: asyncio.Queue = asyncio.Queue()

    async def connect(self):
        pass

    async def send_audio(self, pcm16):
        pass

    async def finalize(self):
        self.q.put_nowait(AsrResult("", True, False, True, 0.0, 0.0))

    async def events(self):
        while True:
            ev = await self.q.get()
            if ev is None:
                return
            yield ev

    async def close(self):
        self.q.put_nowait(None)

    def interim(self, text):
        self.q.put_nowait(AsrResult(text, False, False, False, 0.0, 0.0))

    def final(self, text):
        self.q.put_nowait(AsrResult(text, True, False, False, 0.0, 0.0))


class FakeLLM:
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls: list[list[dict]] = []
        self.closed = 0

    async def stream(self, messages, tools):
        self.calls.append([dict(m) for m in messages])
        script = self.scripts.pop(0)
        try:
            for ev in script:
                if ev is BLOCK:
                    await asyncio.Event().wait()
                yield ev
        finally:
            self.closed += 1

    async def warmup(self, system_prompt):
        pass


class FakeCtx:
    def __init__(self, context_id):
        self.context_id = context_id
        self.q: asyncio.Queue = asyncio.Queue()
        self.cancelled = False
        self.t = 0.0

    async def push_text(self, text):
        self.q.put_nowait(text)

    async def end_input(self):
        self.q.put_nowait(None)

    async def cancel(self):
        self.cancelled = True
        self.q.put_nowait(None)

    async def events(self):
        while True:
            text = await self.q.get()
            if text is None or self.cancelled:
                return
            for w in text.split():
                yield TtsWords([w], [self.t], [self.t + 0.2])
                self.t += 0.2
                yield TtsAudio(b"\x01\x00" * 4800)
                await asyncio.sleep(0)


class FakeTTS:
    sample_rate = 24000

    def __init__(self):
        self.contexts: list[FakeCtx] = []

    async def connect(self):
        pass

    def new_context(self, context_id):
        ctx = FakeCtx(context_id)
        self.contexts.append(ctx)
        return ctx

    async def close(self):
        pass


class FakeTools:
    def __init__(self):
        self.calls = []

    async def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return json.dumps({"available": True})


class Harness:
    def __init__(self, tmp_path, scripts):
        self.transport = FakeTransport()
        self.asr, self.llm, self.tts, self.tools = FakeASR(), FakeLLM(scripts), FakeTTS(), FakeTools()
        cfg = PipelineConfig.model_validate({"turn": {"detector": "silence", "silence_ms": 96}})
        settings = Settings(_env_file=None, va_today=date(2026, 11, 23))
        self.session = StreamingSession(
            self.transport, cfg, settings,
            Stages(self.asr, self.llm, self.tts, FakeVAD(), None), self.tools, TraceWriter(tmp_path),
        )
        self.task = asyncio.create_task(self.session.run())

    async def until(self, pred, timeout=3.0):
        end = asyncio.get_running_loop().time() + timeout
        while not pred():
            if asyncio.get_running_loop().time() > end:
                raise AssertionError("condition not reached")
            await asyncio.sleep(0.005)

    async def frames(self, data, n):
        for _ in range(n):
            self.transport.incoming.put_nowait(data)
            await asyncio.sleep(0.002)
        await asyncio.sleep(0.01)

    async def finish_greeting(self):
        await self.until(lambda: "agent_audio_end" in self.transport.types())
        self.transport.say({"type": "playback_done", "turn_id": 0})
        await self.until(lambda: self.session.state == "listening")

    async def close(self):
        self.transport.say({"type": "hangup"})
        await asyncio.wait_for(self.task, 3)


@pytest.fixture
async def h(tmp_path, request):
    harness = Harness(tmp_path, getattr(request, "param", []))
    yield harness
    if not harness.task.done():
        await harness.close()


def done():
    return LlmDone("stop", Usage(10, 4, 5))


async def interrupt(h, turn_id, played_samples):
    """Talk over the agent until barge-in is confirmed, then ack the clear."""
    await h.until(lambda: h.session.state == "agent_speaking")
    await h.frames(VOICED, 4)
    await h.until(lambda: "pause" in h.transport.types())
    h.asr.interim("no wait stop that")
    await h.until(lambda: "clear" in h.transport.types())
    h.transport.say({"type": "playback_cleared", "turn_id": turn_id, "played_samples": played_samples})


async def test_confirmed_barge_in(h):
    await interrupt(h, 0, 9600)  # two words of the greeting
    await h.until(lambda: h.session.history[1]["content"].endswith("[interrupted]"))
    types = h.transport.types()
    assert types.index("pause") < types.index("clear")
    assert h.tts.contexts[0].cancelled
    assert h.session.history[1]["content"] == "Thanks for [interrupted]"
    await h.until(lambda: any(m["text"] == "Thanks for…" for m in h.transport.json("transcript")))
    assert h.session.state == "user_speaking"


async def test_backchannel_resumes(h):
    await h.until(lambda: h.session.state == "agent_speaking")
    before = [dict(m) for m in h.session.history]
    await h.frames(VOICED, 7)
    await h.until(lambda: "pause" in h.transport.types())
    h.asr.interim("okay")
    await asyncio.sleep(0.02)
    await h.frames(SILENT, 10)
    await h.until(lambda: "resume" in h.transport.types())
    assert "clear" not in h.transport.types()
    assert not h.tts.contexts[0].cancelled
    assert h.session.history == before
    assert h.session.state == "agent_speaking"


@pytest.mark.parametrize(
    "h",
    [[[
        TextDelta("One moment. "),
        ToolCall("c1", "check_availability", "{}"),
        done(),
    ], [
        TextDelta("Seven thirty is free, would that work for you?"),
        done(),
    ]]],
    indirect=True,
)
async def test_tool_survives_interrupt(h):
    await h.finish_greeting()
    await h.frames(VOICED, 3)
    h.asr.final("book me a table")
    await asyncio.sleep(0.02)
    await h.frames(SILENT, 4)
    await h.until(lambda: h.session.state == "agent_speaking")
    await interrupt(h, 1, 4 * 4800)  # "One moment." + "Seven thirty"
    await h.until(lambda: h.session.history[-1]["content"].endswith("[interrupted]"))
    hist = h.session.history
    assert any(m.get("tool_calls") for m in hist)
    assert any(m["role"] == "tool" for m in hist)
    assert h.tools.calls == [("check_availability", "{}")]
    assert hist[-1] == {"role": "assistant", "content": "Seven thirty [interrupted]"}
    assert h.transport.json("tool")[0]["name"] == "check_availability"
    await h.until(lambda: h.transport.json("turn_metrics"))
    assert h.transport.json("turn_metrics")[0]["interrupted"] is True


@pytest.mark.parametrize(
    "h",
    [[[BLOCK], [TextDelta("Four it is.  "), done()]]],
    indirect=True,
)
async def test_user_resumes_while_thinking(h):
    await h.finish_greeting()
    await h.frames(VOICED, 3)
    h.asr.final("I would like a table")
    await asyncio.sleep(0.02)
    await h.frames(SILENT, 4)
    await h.until(lambda: h.session.state == "agent_thinking" and len(h.llm.calls) == 1)
    await h.frames(VOICED, 3)
    await h.until(lambda: h.session.state == "user_speaking")
    assert h.llm.closed == 1
    h.asr.final("for four")
    await asyncio.sleep(0.02)
    await h.frames(SILENT, 4)
    await h.until(lambda: len(h.llm.calls) == 2)
    users = [m for m in h.llm.calls[1] if m["role"] == "user"]
    assert users == [{"role": "user", "content": "I would like a table for four"}]
    await h.until(lambda: h.session.state == "agent_speaking")
    assert h.session.history[-1] == {"role": "assistant", "content": "Four it is.  "}
    asr_s, usage = h.session.snapshot()
    assert asr_s > 0 and usage["llm_input_tokens"] == 10
