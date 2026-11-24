import asyncio
import json
from types import SimpleNamespace

import pytest

from server.config import PipelineConfig, Settings
from server.pipeline_batch import TurnBasedSession
from server.stages.base import ToolCall, Usage
from server.tracing import TraceWriter
from server.transport import TransportClosed


@pytest.mark.asyncio
async def test_batch_booking_tool_flow(tmp_path, monkeypatch):
    import server.pipeline_batch as batch

    async def transcribe(*args):
        return 'A table for two'

    async def synthesize(*args):
        return b'\x00\x00' * 100

    monkeypatch.setattr(batch, 'transcribe_wav', transcribe)
    monkeypatch.setattr(batch, 'synthesize_bytes', synthesize)

    class Transport:
        def __init__(self):
            self.incoming = asyncio.Queue()
            self.sent = []
        async def recv(self):
            item = await self.incoming.get()
            if item is None:
                raise TransportClosed()
            return item
        async def send_json(self, item):
            self.sent.append(item)
        async def send_bytes(self, item):
            self.sent.append(item)

    class LLM:
        def __init__(self):
            self.calls = 0
        async def complete(self, history, tools):
            self.calls += 1
            if self.calls == 1:
                return 'One moment.', [ToolCall('1', 'check_availability', '{}')], Usage(1, 0, 1)
            return 'A table is available.', [], Usage(1, 0, 1)

    class Tools:
        async def execute(self, name, args):
            return '{"available": true}'

    t = Transport()
    session = TurnBasedSession(t, PipelineConfig(), Settings(va_profile='v1-batch'), LLM(), Tools(), TraceWriter(tmp_path), None)
    task = asyncio.create_task(session.run())
    await t.incoming.put(json.dumps({'type': 'ptt_start'}))
    await t.incoming.put(b'\x00\x00' * 512)
    await t.incoming.put(json.dumps({'type': 'ptt_end'}))
    for _ in range(100):
        if any(isinstance(x, dict) and x.get('type') == 'turn_metrics' for x in t.sent):
            break
        await asyncio.sleep(0.01)
    await t.incoming.put(None)
    await task
    assert any(isinstance(x, dict) and x.get('type') == 'tool' and x['name'] == 'check_availability' for x in t.sent)
    assert any(isinstance(x, dict) and x.get('type') == 'transcript' and x.get('text') == 'A table is available.' for x in t.sent)
    lines = (tmp_path / f'{session.session_id}.jsonl').read_text().splitlines()
    assert json.loads(lines[0])['derived']['v2v_ms'] is not None
    assert json.loads(lines[-1])['type'] == 'session_end'
