"""Unit tests for ``src.orchestration.run_command``.

Exercise the run-command workflow helper without Typer. The helper builds
the orchestrator from CLI inputs, runs it under ``asyncio.run``, and
returns a typed outcome carrying the exit code and optional error message
the CLI renders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.core.models import OrderPreference
from src.orchestration import factory
from src.orchestration.cli_options import OrderResolution, ScopeConfig
from src.orchestration.cli_overrides import CLIOverrideOptions
from src.orchestration.run_command import RunCommandOutcome, execute_run_command

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


class FakeOrchestrator:
    """In-memory orchestrator stand-in capturing run-time inputs."""

    def __init__(
        self,
        *,
        success_count: int = 1,
        total: int = 1,
        exit_code: int = 0,
    ) -> None:
        self._success_count = success_count
        self._total = total
        self._exit_code = exit_code
        self.last_watch_config: Any = None
        self.last_orch_config: Any = None
        self.last_mala_config: Any = None

    async def run(
        self,
        *,
        watch_config: object = None,
        validation_config: object = None,
    ) -> tuple[int, int]:
        self.last_watch_config = watch_config
        return (self._success_count, self._total)

    @property
    def exit_code(self) -> int:
        return self._exit_code


def _install_fake_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator: FakeOrchestrator,
) -> list[dict[str, object]]:
    """Patch factory.create_orchestrator to return the given fake and capture args."""
    captured: list[dict[str, object]] = []

    def fake_create_orchestrator(
        config: object,
        *,
        mala_config: object = None,
        deps: object = None,
    ) -> FakeOrchestrator:
        orchestrator.last_orch_config = config
        orchestrator.last_mala_config = mala_config
        captured.append({"config": config, "mala_config": mala_config})
        return orchestrator

    monkeypatch.setattr(factory, "create_orchestrator", fake_create_orchestrator)
    return captured


def _build_resolution(focus: bool = True) -> OrderResolution:
    return OrderResolution(
        preference=(
            OrderPreference.EPIC_PRIORITY if focus else OrderPreference.ISSUE_PRIORITY
        ),
        focus=focus,
    )


class TestRunCommandOutcomeShape:
    def test_default_no_error_message(self) -> None:
        outcome = RunCommandOutcome(exit_code=0)
        assert outcome.exit_code == 0
        assert outcome.error_message is None


class TestSuccessfulRun:
    def test_run_passes_scope_to_orchestrator_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        orchestrator = FakeOrchestrator(success_count=1, total=1, exit_code=0)
        _install_fake_orchestrator(monkeypatch, orchestrator)
        scope = ScopeConfig(scope_type="ids", ids=["T-1", "T-2"])

        outcome = execute_run_command(
            repo_path=tmp_path,
            max_agents=2,
            timeout=7,
            max_issues=3,
            scope_config=scope,
            order_resolution=_build_resolution(focus=True),
            resume=False,
            strict=False,
            fresh=False,
            watch=False,
            override_options=CLIOverrideOptions(),
        )

        assert outcome.exit_code == 0
        assert outcome.error_message is None
        orch_config = orchestrator.last_orch_config
        assert orch_config is not None
        assert orch_config.repo_path == tmp_path
        assert orch_config.max_agents == 2
        assert orch_config.timeout_minutes == 7
        assert orch_config.max_issues == 3
        assert orch_config.only_ids == ["T-1", "T-2"]
        assert orch_config.epic_id is None
        assert orch_config.orphans_only is False
        assert orch_config.focus is True
        assert orch_config.include_wip is False

    def test_run_forwards_epic_scope(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        orchestrator = FakeOrchestrator()
        _install_fake_orchestrator(monkeypatch, orchestrator)
        scope = ScopeConfig(scope_type="epic", epic_id="epic-42")

        execute_run_command(
            repo_path=tmp_path,
            max_agents=None,
            timeout=None,
            max_issues=None,
            scope_config=scope,
            order_resolution=_build_resolution(focus=True),
            resume=False,
            strict=False,
            fresh=False,
            watch=False,
            override_options=CLIOverrideOptions(),
        )

        assert orchestrator.last_orch_config.epic_id == "epic-42"
        assert orchestrator.last_orch_config.only_ids is None
        assert orchestrator.last_orch_config.orphans_only is False

    def test_run_forwards_orphans_scope(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        orchestrator = FakeOrchestrator()
        _install_fake_orchestrator(monkeypatch, orchestrator)
        scope = ScopeConfig(scope_type="orphans")

        execute_run_command(
            repo_path=tmp_path,
            max_agents=None,
            timeout=None,
            max_issues=None,
            scope_config=scope,
            order_resolution=_build_resolution(focus=False),
            resume=False,
            strict=False,
            fresh=False,
            watch=False,
            override_options=CLIOverrideOptions(),
        )

        assert orchestrator.last_orch_config.orphans_only is True
        assert orchestrator.last_orch_config.epic_id is None
        assert orchestrator.last_orch_config.only_ids is None

    def test_run_forwards_resume_strict_fresh(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        orchestrator = FakeOrchestrator()
        _install_fake_orchestrator(monkeypatch, orchestrator)

        execute_run_command(
            repo_path=tmp_path,
            max_agents=None,
            timeout=None,
            max_issues=None,
            scope_config=None,
            order_resolution=_build_resolution(focus=True),
            resume=True,
            strict=True,
            fresh=False,
            watch=True,
            override_options=CLIOverrideOptions(),
        )

        cfg = orchestrator.last_orch_config
        assert cfg.include_wip is True
        assert cfg.strict_resume is True
        assert cfg.fresh_session is False
        assert cfg.cli_args == {"wip": True, "max_issues": None, "watch": True}
        assert orchestrator.last_watch_config is not None
        assert orchestrator.last_watch_config.enabled is True


class TestExitCodeComputation:
    """Verify the exit code mapping from orchestrator results."""

    def test_orchestrator_exit_code_propagates_when_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        orchestrator = FakeOrchestrator(success_count=0, total=1, exit_code=130)
        _install_fake_orchestrator(monkeypatch, orchestrator)

        outcome = execute_run_command(
            repo_path=tmp_path,
            max_agents=None,
            timeout=None,
            max_issues=None,
            scope_config=None,
            order_resolution=_build_resolution(focus=True),
            resume=False,
            strict=False,
            fresh=False,
            watch=False,
            override_options=CLIOverrideOptions(),
        )

        assert outcome.exit_code == 130
        assert outcome.error_message is None

    def test_exit_zero_when_no_issues(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        orchestrator = FakeOrchestrator(success_count=0, total=0, exit_code=0)
        _install_fake_orchestrator(monkeypatch, orchestrator)

        outcome = execute_run_command(
            repo_path=tmp_path,
            max_agents=None,
            timeout=None,
            max_issues=None,
            scope_config=None,
            order_resolution=_build_resolution(focus=True),
            resume=False,
            strict=False,
            fresh=False,
            watch=False,
            override_options=CLIOverrideOptions(),
        )

        assert outcome.exit_code == 0

    def test_exit_one_when_all_failed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        orchestrator = FakeOrchestrator(success_count=0, total=3, exit_code=0)
        _install_fake_orchestrator(monkeypatch, orchestrator)

        outcome = execute_run_command(
            repo_path=tmp_path,
            max_agents=None,
            timeout=None,
            max_issues=None,
            scope_config=None,
            order_resolution=_build_resolution(focus=True),
            resume=False,
            strict=False,
            fresh=False,
            watch=False,
            override_options=CLIOverrideOptions(),
        )

        assert outcome.exit_code == 1


class TestConfigResolutionErrors:
    def test_value_error_surfaces_as_error_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A ValueError from build_resolved_mala_config becomes an outcome error."""
        import src.orchestration.run_command as module

        def boom(repo_path: object, options: object) -> None:
            raise ValueError("invalid override")

        monkeypatch.setattr(module, "build_resolved_mala_config", boom)

        outcome = execute_run_command(
            repo_path=tmp_path,
            max_agents=None,
            timeout=None,
            max_issues=None,
            scope_config=None,
            order_resolution=_build_resolution(focus=True),
            resume=False,
            strict=False,
            fresh=False,
            watch=False,
            override_options=CLIOverrideOptions(),
        )

        assert outcome.exit_code == 1
        assert outcome.error_message == "invalid override"
