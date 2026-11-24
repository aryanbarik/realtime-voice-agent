FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY server ./server
COPY backend ./backend
COPY client ./client
RUN uv sync --frozen --no-dev \
 && uv run --no-sync python -c "from server.models import ensure_models; ensure_models()"
EXPOSE 8080
CMD ["uv","run","--no-sync","uvicorn","server.app:app","--host","0.0.0.0","--port","8080"]
