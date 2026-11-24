from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from server.stages.base import LlmDone, TextDelta, ToolCall, Usage


def _usage(u) -> Usage:
    if u is None:
        return Usage(0, 0, 0)
    details = getattr(u, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    return Usage(u.prompt_tokens or 0, cached or 0, u.completion_tokens or 0)


class OpenAICompatLLM:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        temperature: float,
        max_tokens: int,
        extra_body: dict | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}

    async def complete(
        self, messages: list[dict], tools: list[dict]
    ) -> tuple[str, list[ToolCall], Usage]:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body or None,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(tc.id, tc.function.name, tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        return msg.content or "", calls, _usage(resp.usage)

    async def stream(
        self, messages: list[dict], tools: list[dict]
    ) -> AsyncIterator[TextDelta | ToolCall | LlmDone]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=self.extra_body or None,
        )
        pending: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        usage: Usage | None = None
        try:
            async for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = _usage(chunk.usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is not None:
                    if delta.content:
                        yield TextDelta(delta.content)
                    for tc in delta.tool_calls or []:
                        slot = pending.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id and not slot["id"]:
                            slot["id"] = tc.id
                        fn = tc.function
                        if fn is not None:
                            if fn.name and not slot["name"]:
                                slot["name"] = fn.name
                            if fn.arguments:
                                slot["arguments"] += fn.arguments
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    for idx in sorted(pending):
                        p = pending[idx]
                        yield ToolCall(p["id"], p["name"], p["arguments"])
                    pending.clear()
            yield LlmDone(finish_reason, usage)
        finally:
            await stream.close()

    async def warmup(self, system_prompt: str) -> None:
        try:
            await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "hi"},
                ],
                max_tokens=1,
                extra_body=self.extra_body or None,
            )
        except Exception:
            pass
