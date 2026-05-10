"""Temporary Codex SDK compatibility helpers.

The Codex app-server can return ``serviceTier = "priority"`` from thread
lifecycle RPCs before the generated Python SDK's ``ServiceTier`` enum accepts
that value (openai/codex#21871). Keep the workaround centralized here so the
eventual upstream fix is easy to remove: callers can go back to the typed SDK
``thread_start`` / ``thread_resume`` methods and delete this module.

This module preserves the repo's Codex lazy-import contract: importing it does
not import ``codex_app_server``. The SDK import happens only after a compatible
thread response is received and an ``AsyncThread`` wrapper is needed.
"""

from __future__ import annotations

import inspect
import importlib.util
import os
import shutil
from typing import Any, cast

from pydantic import BaseModel, ConfigDict


def resolve_codex_bin_for_app_server() -> str | None:
    """Return the explicit Codex binary path to give ``AppServerConfig``.

    The Python app-server SDK only resolves its bundled ``codex_cli_bin``
    package when ``codex_bin`` is ``None``; it does not search ``PATH`` on
    its own. Mala supports a normal on-PATH ``codex`` install too, so pass
    that resolved path explicitly. ``CODEX_BINARY`` remains the highest
    precedence operator override and is returned even if stale so the SDK
    can report the precise invalid-path error.
    """
    override = os.environ.get("CODEX_BINARY")
    if override:
        return override
    return shutil.which("codex")


def codex_runtime_resolvable() -> bool:
    """Return True iff the SDK app-server can resolve a Codex runtime."""
    if resolve_codex_bin_for_app_server() is not None:
        return True
    try:
        spec = importlib.util.find_spec("codex_cli_bin")
    except (ImportError, ValueError):
        spec = None
    return spec is not None


class _ThreadCompatResponse(BaseModel):
    """Permissive subset of thread lifecycle responses.

    We only need ``thread.id``; every other response field, including the stale
    ``serviceTier`` enum, is deliberately ignored.
    """

    model_config = ConfigDict(extra="allow")

    thread: dict[str, object]


async def _maybe_await(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value


async def _request_thread_id(
    codex: object, method: str, params: dict[str, object]
) -> str | None:
    raw_client = getattr(codex, "_client", None)
    raw_request = getattr(raw_client, "_request_raw", None)
    sdk_request = getattr(raw_client, "request", None)
    if callable(sdk_request):
        # Prefer the SDK's public-ish generic request API so response parsing
        # still flows through a pydantic model; keep the private raw hook only
        # as a compatibility fallback for older sync-backed test/client shapes.
        response = await _maybe_await(
            sdk_request(method, params, response_model=_ThreadCompatResponse)
        )
    elif callable(raw_request):
        response = await _maybe_await(raw_request(method, params))
    else:
        return None

    if isinstance(response, BaseModel):
        response = response.model_dump()
    if not isinstance(response, dict):
        raise TypeError(f"{method} raw response must be an object")

    response_dict: dict[Any, Any] = response
    thread = response_dict.get("thread")
    thread_id = thread.get("id") if isinstance(thread, dict) else None
    if not isinstance(thread_id, str) or not thread_id:
        raise TypeError(f"{method} raw response missing thread.id")
    return thread_id


async def start_thread_compat(
    codex: object,
    params: dict[str, object],
    *,
    typed_kwargs: dict[str, object],
) -> Any:  # noqa: ANN401 - codex_app_server.AsyncThread, avoid lazy-import
    """Start a Codex thread, bypassing stale typed response validation when possible."""
    thread_id = await _request_thread_id(codex, "thread/start", params)
    if thread_id is None:
        typed_thread_start = getattr(codex, "thread_start")
        return await typed_thread_start(**typed_kwargs)

    from codex_app_server import (  # type: ignore[import-not-found]
        AsyncThread,
    )

    return AsyncThread(cast("Any", codex), thread_id)


async def resume_thread_compat(codex: object, thread_id: str) -> Any:  # noqa: ANN401 - codex_app_server.AsyncThread, avoid lazy-import
    """Resume a Codex thread, bypassing stale typed response validation when possible."""
    resumed_thread_id = await _request_thread_id(
        codex,
        "thread/resume",
        {"threadId": thread_id},
    )
    if resumed_thread_id is None:
        typed_thread_resume = getattr(codex, "thread_resume")
        return await typed_thread_resume(thread_id)

    from codex_app_server import (  # type: ignore[import-not-found]
        AsyncThread,
    )

    return AsyncThread(cast("Any", codex), resumed_thread_id)
