import asyncio
import io
import json
import time
import uuid
import wave
from datetime import date

from server.prompts import GREETING, build_system_prompt
from server.stages.asr.deepgram import transcribe_wav
from server.stages.tts.cartesia import synthesize_bytes
from server.tools.schemas import TOOLS
from server.tracing import TurnTrace
from server.transport import TransportClosed


class TurnBasedSession:
    def __init__(self, transport, cfg, settings, llm, tools, trace_writer, http):
        self.transport, self.cfg, self.settings = transport, cfg, settings
        self.llm, self.tools, self.trace_writer, self.http = llm, tools, trace_writer, http
        self.session_id = uuid.uuid4().hex
        self.out_q = asyncio.Queue()
        self.history = [{'role': 'system', 'content': build_system_prompt(settings.va_today or date.today())},
                        {'role': 'assistant', 'content': GREETING}]
        self.turn_id = 0
        self.asr_audio_s = 0.0
        self.usage = dict.fromkeys(('llm_input_tokens', 'llm_cached_tokens', 'llm_output_tokens', 'tts_chars'), 0)

    def snapshot(self):
        return self.asr_audio_s, self.usage.copy()

    async def _send_loop(self):
        while True:
            item, trace, mark = await self.out_q.get()
            try:
                if isinstance(item, bytes):
                    await self.transport.send_bytes(item)
                else:
                    await self.transport.send_json(item)
                if trace and mark:
                    trace.mark(mark)
            finally:
                self.out_q.task_done()

    async def _enqueue(self, item, trace=None, mark=None):
        await self.out_q.put((item, trace, mark))

    async def _audio(self, turn_id, text, trace=None):
        if trace:
            trace.mark('tts_request')
            trace.usage['tts_chars'] += len(text)
        self.usage['tts_chars'] += len(text)
        audio = await synthesize_bytes(self.http, self.settings.cartesia_api_key, self.cfg.tts.model,
                                       self.cfg.tts.voice, text, self.cfg.tts.sample_rate)
        if trace:
            trace.mark('tts_first_audio')
        await self._enqueue({'type': 'agent_audio_start', 'turn_id': turn_id})
        for i in range(0, len(audio), 9600):
            await self._enqueue(audio[i:i + 9600], trace, 'first_audio_sent')
        await self._enqueue({'type': 'agent_audio_end', 'turn_id': turn_id}, trace, 'agent_audio_end_sent')
        await self._enqueue({'type': 'transcript', 'role': 'agent', 'turn_id': turn_id, 'text': text, 'final': True})
        await self.out_q.join()

    async def _turn(self, pcm, trace):
        wav = io.BytesIO()
        with wave.open(wav, 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(pcm)
        self.asr_audio_s += len(pcm) / 32000
        text = await transcribe_wav(self.http, self.settings.deepgram_api_key, wav.getvalue())
        trace.mark('asr_final')
        trace.user_text = text
        if not text.strip():
            return
        await self._enqueue({'type': 'transcript', 'role': 'user', 'turn_id': self.turn_id, 'text': text, 'final': True})
        self.history.append({'role': 'user', 'content': text})
        answer = ''
        for _ in range(4):
            trace.mark('llm_request')
            answer, calls, usage = await self.llm.complete(self.history, TOOLS)
            trace.mark('llm_first_token')
            if usage:
                trace.usage['llm_input_tokens'] += usage.input_tokens
                trace.usage['llm_cached_tokens'] += usage.cached_tokens
                trace.usage['llm_output_tokens'] += usage.output_tokens
                self.usage['llm_input_tokens'] += usage.input_tokens
                self.usage['llm_cached_tokens'] += usage.cached_tokens
                self.usage['llm_output_tokens'] += usage.output_tokens
            if not calls:
                self.history.append({'role': 'assistant', 'content': answer})
                break
            self.history.append({'role': 'assistant', 'content': answer or None, 'tool_calls': [
                {'id': c.id, 'type': 'function', 'function': {'name': c.name, 'arguments': c.arguments}} for c in calls]})
            for call in calls:
                trace.event('tool_call_start', name=call.name)
                result = await self.tools.execute(call.name, call.arguments)
                trace.event('tool_call_end', name=call.name)
                self.history.append({'role': 'tool', 'tool_call_id': call.id, 'content': result})
                await self._enqueue({'type': 'tool', 'turn_id': self.turn_id, 'name': call.name,
                                     'args': call.arguments, 'result': result})
        if answer:
            trace.agent_text = answer
            trace.mark('llm_first_clause')
            await self._audio(self.turn_id, answer, trace)
        trace.mark('turn_end')
        summary = trace.summary()
        await self._enqueue({'type': 'turn_metrics', **summary})
        self.trace_writer.write_turn(self.session_id, summary)

    async def run(self):
        sender = asyncio.create_task(self._send_loop())
        started = time.monotonic()
        reason = 'disconnect'
        try:
            await self._enqueue({'type': 'session', 'session_id': self.session_id, 'profile': self.settings.va_profile,
                                 'turn_mode': 'push_to_talk', 'input_sample_rate': 16000, 'output_sample_rate': 24000})
            await self._audio(0, GREETING)
            recording = False
            buf = bytearray()
            trace = None
            last_frame = None
            while True:
                try:
                    message = await asyncio.wait_for(self.transport.recv(), timeout=max(0.001, started + self.cfg.max_call_s - time.monotonic()))
                except TimeoutError:
                    reason = 'timeout'
                    break
                if isinstance(message, bytes):
                    if recording:
                        buf.extend(message)
                        last_frame = time.monotonic()
                    continue
                try:
                    msg = json.loads(message)
                except (ValueError, TypeError):
                    continue
                if msg.get('type') == 'hangup':
                    reason = 'hangup'
                    break
                if msg.get('type') == 'ptt_start':
                    recording, buf, last_frame = True, bytearray(), None
                    self.turn_id += 1
                    trace = TurnTrace(self.session_id, self.turn_id, self.settings.va_profile)
                    trace.mark('user_speech_start')
                elif msg.get('type') == 'ptt_end' and recording and trace:
                    recording = False
                    trace.mark('user_speech_end', last_frame)
                    trace.mark('eot_decided')
                    try:
                        await self._turn(bytes(buf), trace)
                    except Exception as exc:
                        await self._enqueue({'type': 'error', 'message': str(exc)})
        except TransportClosed:
            pass
        finally:
            self.trace_writer.write_session_end(self.session_id, self.settings.va_profile, time.monotonic() - started, self.asr_audio_s, self.usage)
            if reason != 'disconnect':
                await self._enqueue({'type': 'session_end', 'reason': reason, 'usage': {**self.usage, 'asr_audio_s': self.asr_audio_s}})
                await self.out_q.join()
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
