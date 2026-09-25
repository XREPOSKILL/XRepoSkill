"""One LLM entry point for every stage, plus retry and JSON helpers."""
from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Type, TypeVar

T = TypeVar("T")


def resolve_model(configured: str) -> str:
    """``PIPELINE_MODEL`` in the environment overrides every configured model."""
    return os.environ.get("PIPELINE_MODEL") or configured


def _transient_errors() -> tuple[Type[BaseException], ...]:
    import litellm
    return (litellm.exceptions.InternalServerError,
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.Timeout,
            litellm.exceptions.RateLimitError)


def call_with_retry(fn: Callable[[], T], *, max_attempts: int = 4,
                    retry_on: tuple[Type[BaseException], ...] | None = None,
                    wait_seconds: float = 2.0) -> T:
    """Retry ``fn`` on transient provider errors with linear backoff."""
    retry_on = retry_on if retry_on is not None else _transient_errors()
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retry_on as e:
            last = e
            if attempt + 1 < max_attempts:
                time.sleep(wait_seconds * (attempt + 1))
    assert last is not None
    raise last


@dataclass
class Reply:
    text: str
    in_tokens: int
    out_tokens: int
    cost: float
    finish_reason: str


def chat(*, model: str, system: str, user: str, max_tokens: int = 24000,
         temperature: float | None = 0.2, timeout: int = 180,
         reasoning_effort: str | None = None,
         drop_params: bool = True) -> Reply:
    """One chat completion. Tests replace this function.

    ``max_tokens`` is generous because reasoning models spend their thinking
    tokens inside the same budget and return empty content when it is small.
    ``drop_params`` lets litellm drop arguments a provider rejects (for
    example ``temperature`` for gpt-5.x or ``reasoning_effort`` for MiMo).
    """
    import litellm
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens, timeout=timeout, drop_params=drop_params)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    resp = call_with_retry(lambda: litellm.completion(**kwargs))
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    return Reply(
        text=choice.message.content or "",
        in_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        out_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cost=float(resp.get("response_cost", 0.0) or 0.0)
        if hasattr(resp, "get") else 0.0,
        finish_reason=str(getattr(choice, "finish_reason", "") or ""),
    )


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> Any:
    """Parse the JSON value in an LLM reply; ``None`` when there is none.

    Accepts a bare JSON document, a fenced code block, or a document with
    prose around it (the outermost ``{...}`` or ``[...]`` is taken).
    """
    if not text:
        return None
    candidates = [text.strip()]
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    for start, end in (("{", "}"), ("[", "]")):
        i, j = text.find(start), text.rfind(end)
        if 0 <= i < j:
            candidates.append(text[i:j + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def parse_json_obj(text: str) -> dict:
    obj = parse_json(text)
    return obj if isinstance(obj, dict) else {}
