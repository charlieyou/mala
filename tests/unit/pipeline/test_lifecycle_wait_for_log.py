"""Unit tests for ``Effect.WAIT_FOR_LOG`` handling.

Phase A7 (issue ``mala-b18dd.5.3``) decoupled the lifecycle wait step from
filesystem assumptions: ``AgentSessionRunner._handle_log_waiting`` now
delegates readiness to ``EvidenceProvider.wait_for_session_ready``. These
tests use a stub provider to verify both the fast-path (provider returns
immediately → lifecycle transitions to ``RUNNING_GATE``) and the
timeout-path (provider raises ``TimeoutError`` → lifecycle transitions to
``FAILED``) without touching the disk.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest

from src.domain.lifecycle import (
    Effect,
    ImplementerLifecycle,
    LifecycleConfig,
    LifecycleContext,
    LifecycleState,
)
from src.pipeline.agent_session_runner import (
    AgentSessionConfig,
    AgentSessionRunner,
    SessionPrompts,
)
from tests.fakes.agent_provider import FakeAgentProvider
from tests.fakes.sdk_client import FakeSDKClientFactory
from tests.helpers.protocol_stubs import (
    StubGateRunner,
    StubReviewRunner,
    StubSessionLifecycle,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.evidence import EvidenceProvider, JsonlEntryProtocol


def _make_prompts() -> SessionPrompts:
    return SessionPrompts(
        gate_followup="gate-followup",
        review_followup="review-followup",
        idle_resume="idle-resume",
    )


@dataclass
class StubEvidenceProvider:
    """Minimal :class:`EvidenceProvider` for readiness-gate tests.

    Only :meth:`wait_for_session_ready` is exercised; the remaining
    protocol methods raise so accidental use during these tests is loud.
    """

    log_path: Path
    delay: float = 0.0
    raise_timeout: bool = False
    calls: list[tuple[Path, str, float, float]] = field(default_factory=list)

    def get_log_path(self, repo_path: Path, session_id: str) -> Path:
        del repo_path, session_id
        return self.log_path

    def iter_session_events(self, log_path: Path, offset: int = 0) -> object:
        del log_path, offset
        raise AssertionError("iter_session_events not used in these tests")

    def iter_thread_evidence(self, log_path: Path, offset: int = 0) -> object:
        del log_path, offset
        raise AssertionError("iter_thread_evidence not used in these tests")

    def get_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        del log_path, start_offset
        raise AssertionError("get_end_offset not used in these tests")

    async def wait_for_session_ready(
        self,
        repo_path: Path,
        session_id: str,
        *,
        timeout: float,
        poll_interval: float = 0.5,
    ) -> None:
        self.calls.append((repo_path, session_id, timeout, poll_interval))
        if self.raise_timeout:
            raise TimeoutError(f"Session log missing after timeout: {self.log_path}")
        if self.delay > 0:
            await asyncio.sleep(self.delay)

    def extract_bash_commands(self, entry: JsonlEntryProtocol) -> list[tuple[str, str]]:
        del entry
        return []

    def extract_tool_results(self, entry: JsonlEntryProtocol) -> list[tuple[str, bool]]:
        del entry
        return []

    def extract_assistant_text_blocks(self, entry: JsonlEntryProtocol) -> list[str]:
        del entry
        return []

    def extract_tool_result_content(
        self, entry: JsonlEntryProtocol
    ) -> list[tuple[str, str]]:
        del entry
        return []


def _make_runner(
    *,
    repo_path: Path,
    log_path: Path,
    evidence_provider: EvidenceProvider,
) -> AgentSessionRunner:
    config = AgentSessionConfig(
        repo_path=repo_path,
        timeout_seconds=300,
        prompts=_make_prompts(),
        review_enabled=False,
    )
    return AgentSessionRunner(
        config=config,
        agent_provider=FakeAgentProvider(
            FakeSDKClientFactory(),
            evidence_provider=evidence_provider,
        ),
        gate_runner=StubGateRunner(),
        review_runner=StubReviewRunner(),
        session_lifecycle=StubSessionLifecycle(log_path=log_path),
    )


def _lifecycle_in_awaiting_log(ctx: LifecycleContext) -> ImplementerLifecycle:
    lifecycle = ImplementerLifecycle(LifecycleConfig())
    lifecycle.start()
    lifecycle.on_messages_complete(ctx, has_session_id=True)
    assert lifecycle.state == LifecycleState.AWAITING_LOG
    return lifecycle


@pytest.mark.unit
class TestHandleLogWaiting:
    """Behavior of ``_handle_log_waiting`` after Phase A7 rewiring."""

    @pytest.mark.asyncio
    async def test_fast_provider_transitions_to_run_gate(self, tmp_path: Path) -> None:
        """Provider returns immediately → lifecycle moves to RUNNING_GATE."""
        log_path = tmp_path / "session.jsonl"
        provider = StubEvidenceProvider(log_path=log_path)
        runner = _make_runner(
            repo_path=tmp_path,
            log_path=log_path,
            evidence_provider=cast("EvidenceProvider", provider),
        )

        ctx = LifecycleContext(session_id="sess-1")
        lifecycle = _lifecycle_in_awaiting_log(ctx)

        new_log_path, result = await runner._handle_log_waiting(
            "sess-1",
            "issue-1",
            None,
            lifecycle,
            ctx,
            log_file_wait_timeout=10.0,
            log_file_poll_interval=0.5,
        )

        assert new_log_path == log_path
        assert result.effect == Effect.RUN_GATE
        assert lifecycle.state == LifecycleState.RUNNING_GATE
        assert provider.calls == [(tmp_path, "sess-1", 10.0, 0.5)]

    @pytest.mark.asyncio
    async def test_timeout_transitions_to_failed(self, tmp_path: Path) -> None:
        """Provider raises TimeoutError → lifecycle moves to FAILED."""
        log_path = tmp_path / "missing.jsonl"
        provider = StubEvidenceProvider(log_path=log_path, raise_timeout=True)
        runner = _make_runner(
            repo_path=tmp_path,
            log_path=log_path,
            evidence_provider=cast("EvidenceProvider", provider),
        )

        ctx = LifecycleContext(session_id="sess-1")
        lifecycle = _lifecycle_in_awaiting_log(ctx)

        new_log_path, result = await runner._handle_log_waiting(
            "sess-1",
            "issue-1",
            None,
            lifecycle,
            ctx,
            log_file_wait_timeout=0.0,
            log_file_poll_interval=0.5,
        )

        assert new_log_path == log_path
        assert result.effect == Effect.COMPLETE_FAILURE
        assert lifecycle.state == LifecycleState.FAILED
        assert "Session log missing after timeout" in (ctx.final_result or "")

    @pytest.mark.asyncio
    async def test_slow_provider_blocks_runner(self, tmp_path: Path) -> None:
        """Slow readiness signal causes the runner to block until ready."""
        log_path = tmp_path / "session.jsonl"
        provider = StubEvidenceProvider(log_path=log_path, delay=0.05)
        runner = _make_runner(
            repo_path=tmp_path,
            log_path=log_path,
            evidence_provider=cast("EvidenceProvider", provider),
        )

        ctx = LifecycleContext(session_id="sess-1")
        lifecycle = _lifecycle_in_awaiting_log(ctx)

        loop = asyncio.get_running_loop()
        start = loop.time()
        new_log_path, result = await runner._handle_log_waiting(
            "sess-1",
            "issue-1",
            None,
            lifecycle,
            ctx,
            log_file_wait_timeout=1.0,
            log_file_poll_interval=0.5,
        )
        elapsed = loop.time() - start

        assert new_log_path == log_path
        assert result.effect == Effect.RUN_GATE
        # Provider blocked for ~50ms; allow generous slack for slow CI.
        assert elapsed >= 0.04

    @pytest.mark.asyncio
    async def test_explicit_evidence_provider_overrides_agent_provider(
        self, tmp_path: Path
    ) -> None:
        """Injected ``evidence_provider`` is used over ``agent_provider``'s.

        Mirrors the orchestrator wiring where
        ``OrchestratorDependencies.evidence_provider`` is supplied without
        overriding ``agent_provider``. Readiness must use the injected
        provider (which the session lifecycle also uses for log-path
        resolution) rather than the agent provider's default.
        """
        log_path = tmp_path / "session.jsonl"
        injected = StubEvidenceProvider(log_path=log_path)
        agent_default = StubEvidenceProvider(log_path=log_path, raise_timeout=True)

        config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=300,
            prompts=_make_prompts(),
            review_enabled=False,
        )
        runner = AgentSessionRunner(
            config=config,
            agent_provider=FakeAgentProvider(
                FakeSDKClientFactory(),
                evidence_provider=cast("EvidenceProvider", agent_default),
            ),
            gate_runner=StubGateRunner(),
            review_runner=StubReviewRunner(),
            session_lifecycle=StubSessionLifecycle(log_path=log_path),
            evidence_provider=cast("EvidenceProvider", injected),
        )

        ctx = LifecycleContext(session_id="sess-1")
        lifecycle = _lifecycle_in_awaiting_log(ctx)

        new_log_path, result = await runner._handle_log_waiting(
            "sess-1",
            "issue-1",
            None,
            lifecycle,
            ctx,
            log_file_wait_timeout=10.0,
            log_file_poll_interval=0.5,
        )

        assert new_log_path == log_path
        assert result.effect == Effect.RUN_GATE
        assert injected.calls == [(tmp_path, "sess-1", 10.0, 0.5)]
        assert agent_default.calls == []
