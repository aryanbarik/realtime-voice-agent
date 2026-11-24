import asyncio
import contextlib
import uuid
import re
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app import create_backend_app
from server.config import Settings, load_pipeline_config
from server.limits import DemoLimiter, Refused, SpendMeter, apply_demo_config, client_ip, demo_tenant
from server.models import ensure_models
from server.pipeline_batch import TurnBasedSession
from server.pipeline import Stages, StreamingSession
from server.stages.factory import build_asr, build_llm, build_tts, build_turn, build_vad
from server.tools.reservations import ReservationClient
from server.tracing import TraceWriter
from server.transport import WebSocketTransport

settings = Settings()
cfg = load_pipeline_config(settings)
backend = create_backend_app(settings.va_db_path, settings.va_today or date.today(), settings.va_admin_token, settings.va_public_demo)
limiter = DemoLimiter(backend.state.db, settings.va_daily_budget_usd) if settings.va_public_demo else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_models()
    app.state.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=backend), base_url='http://backend')
    app.state.provider_http = httpx.AsyncClient(timeout=30)
    app.state.traces = TraceWriter(settings.va_trace_dir)
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.provider_http.aclose()


app = FastAPI(lifespan=lifespan)
app.mount('/backend', backend)
app.mount('/static', StaticFiles(directory=Path(__file__).resolve().parent.parent / 'client'), name='static')


@app.get('/')
def index():
    return FileResponse(Path(__file__).resolve().parent.parent / 'client' / 'index.html')


@app.get('/healthz')
def health():
    return {'ok': True, 'profile': settings.va_profile}


@app.websocket('/ws')
async def websocket(ws: WebSocket):
    tenant = ws.query_params.get('tenant', 'demo')
    if not re.fullmatch(r'[a-z0-9-]{1,64}', tenant):
        await ws.close(code=1008)
        return
    await ws.accept()
    transport = WebSocketTransport(ws)
    lease = None
    meter = None
    meter_task = None
    try:
        session_id = uuid.uuid4().hex
        run_cfg = cfg
        if limiter:
            try:
                lease = limiter.acquire(client_ip(ws))
            except Refused as exc:
                await transport.send_json({'type': 'error', 'message': exc.reason})
                await transport.close(code=1013)
                return
            tenant = demo_tenant(session_id)
            run_cfg = apply_demo_config(cfg)
        llm = build_llm(run_cfg, settings)
        tools = ReservationClient(app.state.http, tenant)
        if run_cfg.mode == 'batch':
            if not settings.deepgram_api_key:
                raise RuntimeError('DEEPGRAM_API_KEY is not set')
            if not settings.cartesia_api_key:
                raise RuntimeError('CARTESIA_API_KEY is not set')
            session = TurnBasedSession(transport, run_cfg, settings, llm, tools, app.state.traces, app.state.provider_http)
        else:
            stages = Stages(build_asr(run_cfg, settings), llm, build_tts(run_cfg, settings),
                            build_vad(run_cfg, settings), build_turn(run_cfg, settings))
            session = StreamingSession(transport, run_cfg, settings, stages, tools, app.state.traces)
        session.session_id = session_id
        if limiter:
            meter = SpendMeter(limiter, session.snapshot, run_cfg.llm.model)
            meter_task = asyncio.create_task(meter.run())
            call_task = asyncio.create_task(session.run())
            done, _ = await asyncio.wait((meter_task, call_task), return_when=asyncio.FIRST_COMPLETED)
            if meter_task in done and not call_task.done():
                call_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await call_task
        else:
            await session.run()
    except Exception as exc:
        with contextlib.suppress(Exception):
            await transport.send_json({'type': 'error', 'message': str(exc)})
    finally:
        if meter_task:
            meter_task.cancel()
            await asyncio.gather(meter_task, return_exceptions=True)
        if meter:
            await meter.final()
        if lease:
            lease.release()
        with contextlib.suppress(Exception):
            await transport.close()
