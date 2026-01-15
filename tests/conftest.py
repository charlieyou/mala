"""Pytest configuration for mala tests."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

# Type alias for lock releaser function - defined at runtime to support get_type_hints()
LockReleaserFunc = Callable[[list[str]], int]

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.issue import IssueProvider
    from src.core.protocols.log import LogProvider
    from src.core.protocols.review import CodeReviewer
    from src.core.protocols.validation import GateChecker
    from src.infra.io.config import MalaConfig
    from src.infra.telemetry import TelemetryProvider
    from src.orchestration.orchestrator import MalaOrchestrator

# Ignore fixture templates under tests/fixtures
collect_ignore_glob = ["fixtures/e2e-fixture/**"]


def pytest_configure(config: pytest.Config) -> None:
    """Configure test environment before collection.

    Sets up environment variables to:
    - Redirect run metadata to /tmp to avoid polluting ~/.config/mala/runs/
    - Redirect Claude SDK logs to /tmp to avoid polluting ~/.claude/projects/
    """
    # Disable debug logging in tests for better test isolation
    # (avoids disk I/O and global logging state changes in each test)
    os.environ["MALA_DISABLE_DEBUG_LOG"] = "1"

    # Redirect run metadata to /tmp to avoid polluting user config
    os.environ["MALA_RUNS_DIR"] = "/tmp/mala-test-runs"

    # Redirect lock directory to /tmp to avoid cross-test interference
    os.environ["MALA_LOCK_DIR"] = "/tmp/mala-test-locks"

    # Redirect Claude SDK logs to /tmp to avoid polluting user Claude config
    os.environ["CLAUDE_CONFIG_DIR"] = "/tmp/mala-test-claude"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Clean up test artifacts after test session completes.

    This runs even when tests are interrupted (Ctrl+C), ensuring lock files
    don't accumulate between test runs.
    """
    import shutil
    from pathlib import Path

    lock_dir = Path("/tmp/mala-test-locks")
    if lock_dir.exists():
        shutil.rmtree(lock_dir, ignore_errors=True)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Apply markers based on test file path."""
    for item in items:
        # Check if test already has an explicit marker (not just package name in keywords)
        if any(item.get_closest_marker(m) for m in ("unit", "integration", "e2e")):
            continue
        # Path-based auto-marking (use item.path for pytest 7+ compatibility)
        # Normalize path separators for cross-platform compatibility (Windows uses backslashes)
        path = str(item.path).replace("\\", "/")
        if "/e2e/" in path:
            item.add_marker(pytest.mark.e2e)
        elif "/integration/" in path:
            item.add_marker(pytest.mark.integration)
        elif "/unit/" in path:
            item.add_marker(pytest.mark.unit)
        else:
            pytest.fail(
                f"Test {item.nodeid} is not in a recognized test category "
                f"(unit/, integration/, e2e/). Move it to the appropriate directory.",
                pytrace=False,
            )


@pytest.fixture
def make_orchestrator() -> Callable[..., MalaOrchestrator]:
    """Factory fixture for creating MalaOrchestrator instances.

    Returns a callable that creates orchestrators using the factory pattern.
    This replaces direct MalaOrchestrator() constructor calls in tests.

    Usage:
        def test_something(make_orchestrator, tmp_path):
            orchestrator = make_orchestrator(
                repo_path=tmp_path,
                max_agents=2,
            )
            ...
    """
    from src.orchestration.factory import OrchestratorConfig, create_orchestrator

    def _make_orchestrator(
        repo_path: Path,
        max_agents: int | None = None,
        timeout_minutes: int | None = None,
        max_issues: int | None = None,
        epic_id: str | None = None,
        only_ids: list[str] | None = None,
        max_gate_retries: int = 3,
        max_review_retries: int = 3,
        disable_validations: set[str] | None = None,
        include_wip: bool = False,
        strict_resume: bool = False,
        focus: bool = True,
        orphans_only: bool = False,
        cli_args: dict[str, Any] | None = None,
        epic_override_ids: set[str] | None = None,
        issue_provider: IssueProvider | None = None,
        code_reviewer: CodeReviewer | None = None,
        gate_checker: GateChecker | None = None,
        log_provider: LogProvider | None = None,
        telemetry_provider: TelemetryProvider | None = None,
        event_sink: MalaEventSink | None = None,
        config: MalaConfig | None = None,
        runs_dir: Path | None = None,
        lock_releaser: LockReleaserFunc | None = None,
    ) -> MalaOrchestrator:
        """Create an orchestrator using the factory pattern."""
        from src.orchestration.factory import OrchestratorDependencies

        orch_config = OrchestratorConfig(
            repo_path=repo_path,
            max_agents=max_agents,
            timeout_minutes=timeout_minutes,
            max_issues=max_issues,
            epic_id=epic_id,
            only_ids=only_ids,
            max_gate_retries=max_gate_retries,
            max_review_retries=max_review_retries,
            disable_validations=disable_validations,
            include_wip=include_wip,
            strict_resume=strict_resume,
            focus=focus,
            orphans_only=orphans_only,
            cli_args=cli_args,
            epic_override_ids=epic_override_ids or set(),
        )

        deps = OrchestratorDependencies(
            issue_provider=issue_provider,
            code_reviewer=code_reviewer,
            gate_checker=gate_checker,
            log_provider=log_provider,
            telemetry_provider=telemetry_provider,
            event_sink=event_sink,
            runs_dir=runs_dir,
            lock_releaser=lock_releaser,
        )

        return create_orchestrator(orch_config, mala_config=config, deps=deps)

    return _make_orchestrator


@pytest.fixture
def log_provider() -> LogProvider:
    """Provide a FileSystemLogProvider for tests that need log parsing.

    Returns:
        A LogProvider instance for reading session logs from filesystem.
    """
    from src.infra.io.session_log_parser import FileSystemLogProvider

    return FileSystemLogProvider()
