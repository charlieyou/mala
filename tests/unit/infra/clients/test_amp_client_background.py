"""Amp client background-task capability surface.

Amp is request/response only, so :meth:`AmpClient.supports_background_tasks`
must report False — that is the signal the pipeline gates the keep-connected
wait/resume path on (Claude returns True). The check touches no subprocess
state, so a lightweight options stand-in is sufficient.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from src.infra.clients.amp_client import AmpClient

if TYPE_CHECKING:
    from src.infra.clients.amp_client import AmpClientOptions


def test_amp_does_not_support_background_tasks() -> None:
    options = cast("AmpClientOptions", SimpleNamespace(thread_id=None))
    client = AmpClient(options)
    assert client.supports_background_tasks() is False
