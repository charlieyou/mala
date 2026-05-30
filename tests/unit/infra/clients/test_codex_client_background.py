"""Codex client background-task capability surface.

Codex is request/response only, so :meth:`CodexClient.supports_background_tasks`
must report False — that is the signal the pipeline gates the keep-connected
wait/resume path on (Claude returns True). The check touches no SDK state, so
a lightweight runtime stand-in is sufficient.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from src.infra.clients.codex_client import CodexClient

if TYPE_CHECKING:
    from src.infra.clients.codex_runtime import CodexRuntime


def test_codex_does_not_support_background_tasks() -> None:
    runtime = cast("CodexRuntime", SimpleNamespace(resume_thread_id=None))
    client = CodexClient(runtime)
    assert client.supports_background_tasks() is False
