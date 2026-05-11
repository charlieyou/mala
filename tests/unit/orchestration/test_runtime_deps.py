"""Unit tests for RuntimeDeps named dependency assembly.

Validates the dataclass construction surface and verifies that the factory's
``_build_dependencies`` returns a RuntimeDeps with attribute access matching
the orchestrator's expectations.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.infra.io.config import MalaConfig
from src.orchestration.config_resolution import _ReviewerConfig
from src.orchestration.factory import _build_dependencies
from src.orchestration.runtime_deps import RuntimeDeps
from src.orchestration.types import (
    OrchestratorConfig,
    OrchestratorDependencies,
    _DerivedConfig,
)
from tests.fakes import (
    FakeAgentProvider,
    FakeCommandRunner,
    FakeEnvConfig,
    FakeEventSink,
    FakeIssueProvider,
    FakeLockManager,
    FakeSDKClientFactory,
)


def _make_runtime_deps() -> RuntimeDeps:
    """Construct a RuntimeDeps with fakes for shape tests."""
    return RuntimeDeps(
        evidence_check=MagicMock(),
        code_reviewer=MagicMock(),
        beads=FakeIssueProvider(),
        event_sink=FakeEventSink(),
        agent_provider=FakeAgentProvider(FakeSDKClientFactory()),  # type: ignore[arg-type]
        command_runner=FakeCommandRunner(allow_unregistered=True),
        env_config=FakeEnvConfig(),
        lock_manager=FakeLockManager(),
        mala_config=MalaConfig(runs_dir=Path("/tmp/runs"), lock_dir=Path("/tmp/locks")),
        evidence_provider=MagicMock(),
        telemetry_provider=MagicMock(),
    )


class TestRuntimeDepsConstruction:
    """Construction-level contracts for RuntimeDeps."""

    def test_is_frozen_dataclass(self) -> None:
        """RuntimeDeps must be a frozen dataclass to prevent mutation."""
        assert dataclasses.is_dataclass(RuntimeDeps)
        params = RuntimeDeps.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen is True

    def test_epic_verifier_defaults_to_none(self) -> None:
        """epic_verifier is optional; callers may omit it."""
        runtime = _make_runtime_deps()
        assert runtime.epic_verifier is None

    def test_attribute_access_returns_constructed_values(self) -> None:
        """Each field is reachable by name (no tuple indexing required)."""
        gate = MagicMock()
        beads = FakeIssueProvider()
        runtime = RuntimeDeps(
            evidence_check=gate,
            code_reviewer=MagicMock(),
            beads=beads,
            event_sink=FakeEventSink(),
            agent_provider=FakeAgentProvider(FakeSDKClientFactory()),  # type: ignore[arg-type]
            command_runner=FakeCommandRunner(allow_unregistered=True),
            env_config=FakeEnvConfig(),
            lock_manager=FakeLockManager(),
            mala_config=MalaConfig(
                runs_dir=Path("/tmp/runs"), lock_dir=Path("/tmp/locks")
            ),
            evidence_provider=MagicMock(),
            telemetry_provider=MagicMock(),
            epic_verifier=None,
        )
        assert runtime.evidence_check is gate
        assert runtime.beads is beads

    def test_cannot_mutate_after_construction(self) -> None:
        """Frozen dataclass rejects attribute writes."""
        runtime = _make_runtime_deps()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(runtime, "beads", FakeIssueProvider())


class TestFactoryReturnsRuntimeDeps:
    """The factory boundary returns a RuntimeDeps, not a positional tuple."""

    def test_build_dependencies_returns_runtime_deps_instance(
        self, tmp_path: Path
    ) -> None:
        """_build_dependencies returns a RuntimeDeps, callers use attributes."""
        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        derived = _DerivedConfig(
            timeout_seconds=600,
            disabled_validations=set(),
            max_idle_retries=3,
            idle_timeout_seconds=None,
        )

        result = _build_dependencies(
            config, mala_config, derived, None, _ReviewerConfig()
        )

        assert isinstance(result, RuntimeDeps)

    def test_build_dependencies_propagates_overrides_via_attributes(
        self, tmp_path: Path
    ) -> None:
        """Caller-supplied protocol overrides appear as named attributes."""
        fake_runner = FakeCommandRunner(allow_unregistered=True)
        fake_env = FakeEnvConfig()
        fake_lock = FakeLockManager()
        deps = OrchestratorDependencies(
            command_runner=fake_runner,
            env_config=fake_env,
            lock_manager=fake_lock,
        )
        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        derived = _DerivedConfig(
            timeout_seconds=600,
            disabled_validations=set(),
            max_idle_retries=3,
            idle_timeout_seconds=None,
        )

        result = _build_dependencies(
            config, mala_config, derived, deps, _ReviewerConfig()
        )

        assert result.command_runner is fake_runner
        assert result.env_config is fake_env
        assert result.lock_manager is fake_lock
