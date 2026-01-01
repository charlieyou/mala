"""Pytest configuration for mala tests."""

import os
import shutil
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure test environment before collection.

    Sets up environment variables to:
    - Disable Braintrust tracing
    - Redirect run metadata to /tmp to avoid polluting ~/.config/mala/runs/
    - Redirect Claude SDK logs to /tmp to avoid polluting ~/.claude/projects/
    - Copy OAuth credentials to test config dir for SDK integration tests
    """
    # Remove BRAINTRUST_API_KEY to disable Braintrust tracing
    os.environ.pop("BRAINTRUST_API_KEY", None)

    # Redirect run metadata to /tmp to avoid polluting user config
    os.environ["MALA_RUNS_DIR"] = "/tmp/mala-test-runs"

    # Redirect Claude SDK logs to /tmp to avoid polluting user Claude config
    test_claude_dir = Path("/tmp/mala-test-claude")
    test_claude_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CLAUDE_CONFIG_DIR"] = str(test_claude_dir)

    # Copy OAuth credentials to test config dir so SDK integration tests work
    real_credentials = Path.home() / ".claude" / ".credentials.json"
    if real_credentials.exists():
        shutil.copy2(real_credentials, test_claude_dir / ".credentials.json")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Default unmarked tests to unit category."""
    for item in items:
        if any(marker in item.keywords for marker in ("unit", "integration", "e2e")):
            continue
        item.add_marker(pytest.mark.unit)


@pytest.fixture
def make_orchestrator():
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
        only_ids: set | None = None,
        braintrust_enabled: bool | None = None,
        max_gate_retries: int = 3,
        max_review_retries: int = 3,
        disable_validations: set | None = None,
        coverage_threshold: float | None = None,
        morph_enabled: bool | None = None,
        prioritize_wip: bool = False,
        focus: bool = True,
        cli_args: dict | None = None,
        epic_override_ids: set | None = None,
        issue_provider=None,
        code_reviewer=None,
        gate_checker=None,
        log_provider=None,
        telemetry_provider=None,
        event_sink=None,
        config=None,
    ):
        """Create an orchestrator using the factory pattern."""
        from src.orchestration.factory import OrchestratorDependencies

        orch_config = OrchestratorConfig(
            repo_path=repo_path,
            max_agents=max_agents,
            timeout_minutes=timeout_minutes,
            max_issues=max_issues,
            epic_id=epic_id,
            only_ids=only_ids,
            braintrust_enabled=braintrust_enabled,
            max_gate_retries=max_gate_retries,
            max_review_retries=max_review_retries,
            disable_validations=disable_validations,
            coverage_threshold=coverage_threshold,
            morph_enabled=morph_enabled,
            prioritize_wip=prioritize_wip,
            focus=focus,
            cli_args=cli_args,
            epic_override_ids=epic_override_ids,
        )

        deps = OrchestratorDependencies(
            issue_provider=issue_provider,
            code_reviewer=code_reviewer,
            gate_checker=gate_checker,
            log_provider=log_provider,
            telemetry_provider=telemetry_provider,
            event_sink=event_sink,
        )

        return create_orchestrator(orch_config, mala_config=config, deps=deps)

    return _make_orchestrator
