"""End-to-end fake-Amp integration test (T007 RED -> T013 GREEN).

This is the **integration-path** test for the Amp coder feature
(``mala-eymhx.7``). It exercises the per-issue lifecycle the orchestrator
runs when ``coder=amp`` is selected: factory selection picks
:class:`AmpAgentProvider`, ``install_prerequisites`` runs once, and the
pipeline subsequently tries to spawn an Amp client. In T007 the stub
``AmpAgentProvider`` raises :class:`NotImplementedError` from its client
factory before any subprocess starts; the test is therefore expected to
fail at this task with that error.

The ``xfail(strict=True)`` marker enforces the RED state: if T007 lands
without the rest of the impl and a future change accidentally makes the
test pass, ``strict=True`` flips it into a regression, surfacing the
silent-pass before T013 has actually shipped the real Amp client.

When T013 finishes (real ``AmpClient`` + ``install_prerequisites`` self-
test), the marker should be removed (or replaced with the regular
integration-suite expectation that this test passes).
"""

from __future__ import annotations

import os
import stat
from typing import TYPE_CHECKING

import pytest

from src.infra.clients.amp_provider import AmpAgentProvider
from src.infra.io.config import MalaConfig
from src.orchestration.factory import (
    OrchestratorConfig,
    OrchestratorDependencies,
    create_orchestrator,
)
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

if TYPE_CHECKING:
    from pathlib import Path


_FAKE_AMP_BINARY = """#!/usr/bin/env bash
# Fake amp binary used by the T007 integration test. Real Amp would emit
# stream-json on stdout; this fake just exits 0 so PATH-resolution works.
# T013 replaces this with a richer canned-output fake that drives the full
# stream-json pipeline.
exit 0
"""


def _install_fake_amp(tmp_path: Path) -> Path:
    """Install a fake ``amp`` executable into ``tmp_path/bin`` and return
    the directory so callers can prepend it to PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_amp = bin_dir / "amp"
    fake_amp.write_text(_FAKE_AMP_BINARY)
    st = fake_amp.stat()
    fake_amp.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


@pytest.fixture
def fake_amp_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Place a fake ``amp`` binary on PATH for the duration of the test."""
    bin_dir = _install_fake_amp(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("AMP_API_KEY", "fake-test-key")
    return bin_dir


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T007: Amp provider is wired but not implemented. "
        "AmpAgentProvider.client_factory.create() raises NotImplementedError "
        "until T013 ships the real AmpClient. The strict marker flips this "
        "into a regression if a future change makes it pass before then."
    ),
)
@pytest.mark.asyncio
async def test_run_amp_implementer_against_fake_amp_binary(
    tmp_path: Path,
    fake_amp_on_path: Path,
) -> None:
    """Per-issue lifecycle with ``coder=amp`` against a fake amp binary.

    This is the failing-by-design integration test. T013 makes it green by
    shipping the real ``AmpClient`` (T008), real plugin self-test (T013),
    and supporting pieces. Until then, ``run_implementer`` raises
    :class:`NotImplementedError` from the stub Amp client factory.
    """
    del fake_amp_on_path  # PATH is configured by the fixture

    issue = FakeIssue(
        id="bd-amp-fake.1",
        title="Fake-amp integration smoke",
        description="Should exercise the full Amp pipeline.",
    )
    issue_provider = FakeIssueProvider(issues=[issue])  # ty:ignore[invalid-argument-type]

    config = OrchestratorConfig(
        repo_path=tmp_path,
        max_agents=1,
        max_issues=1,
        timeout_minutes=1,
    )
    mala_config = MalaConfig(
        runs_dir=tmp_path / "runs",
        lock_dir=tmp_path / "locks",
        coder="amp",
    )
    deps = OrchestratorDependencies(
        issue_provider=issue_provider,
    )
    orchestrator = create_orchestrator(config, mala_config=mala_config, deps=deps)
    assert isinstance(orchestrator._agent_provider, AmpAgentProvider)

    # Drive the per-issue lifecycle. The stub client factory raises
    # NotImplementedError when AgentSessionRunner tries to open an SDK
    # client; the run() loop surfaces that as a failed issue, but for the
    # purpose of this RED test we want the underlying NotImplementedError
    # to escape so the test fails clearly. Calling run_implementer directly
    # bypasses run()'s exception swallowing.
    result = await orchestrator.run_implementer(issue.id)
    # If the test ever reaches here it means the Amp impl shipped without
    # the test being un-xfailed. ``xfail(strict=True)`` will flip this into
    # a regression to force the maintainer to remove the marker.
    assert result.success
