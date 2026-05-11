"""Unit tests for PipelineConfig narrow views (T_C5).

These tests assert each ``@cached_property`` view exposes the same data as the
canonical PipelineConfig fields it derives from, so a future T_C6 runner
migration can swap to the views without semantic drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.domain.prompts import PromptProvider
from src.domain.validation.config_types import PromptValidationCommands
from src.orchestration.types import (
    AgentSessionView,
    CumulativeReviewView,
    FixerServiceView,
    GateRunnerView,
    PipelineConfig,
    ReviewRunnerView,
    RunCoordinatorView,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def prompts() -> PromptProvider:
    return PromptProvider(
        implementer_prompt="implementer",
        review_followup_prompt="review",
        gate_followup_prompt="gate",
        fixer_prompt="fixer-template",
        idle_resume_prompt="idle",
        checkpoint_request_prompt="checkpoint",
        continuation_prompt="continuation",
    )


@pytest.fixture
def prompt_commands() -> PromptValidationCommands:
    return PromptValidationCommands(
        lint="echo lint",
        format="echo format",
        typecheck="echo typecheck",
        test="echo test",
        custom_commands=(),
    )


@pytest.fixture
def pipeline(
    prompts: PromptProvider,
    prompt_commands: PromptValidationCommands,
    tmp_path: Path,
) -> PipelineConfig:
    return PipelineConfig(
        repo_path=tmp_path,
        timeout_seconds=3600,
        max_gate_retries=5,
        max_review_retries=7,
        disabled_validations={"lint", "typecheck"},
        max_idle_retries=4,
        idle_timeout_seconds=120.0,
        prompts=prompts,
        prompt_validation_commands=prompt_commands,
        validation_config=None,
        validation_config_missing=True,
        deadlock_monitor=None,
        review_timeout_seconds=300,
    )


@pytest.mark.unit
class TestPipelineConfigViews:
    def test_gate_runner_view_matches_source(self, pipeline: PipelineConfig) -> None:
        view = pipeline.gate_runner_view
        assert isinstance(view, GateRunnerView)
        assert view.repo_path == pipeline.repo_path
        assert view.max_gate_retries == pipeline.max_gate_retries
        assert view.disabled_validations == pipeline.disabled_validations
        assert view.validation_config is pipeline.validation_config
        assert view.validation_config_missing == pipeline.validation_config_missing

    def test_review_runner_view_matches_source(self, pipeline: PipelineConfig) -> None:
        view = pipeline.review_runner_view
        assert isinstance(view, ReviewRunnerView)
        assert view.max_review_retries == pipeline.max_review_retries
        assert view.review_timeout_seconds == pipeline.review_timeout_seconds

    def test_cumulative_review_view_matches_source(
        self, pipeline: PipelineConfig
    ) -> None:
        view = pipeline.cumulative_review_view
        assert isinstance(view, CumulativeReviewView)
        assert view.repo_path == pipeline.repo_path
        assert view.review_timeout_seconds == pipeline.review_timeout_seconds

    def test_fixer_service_view_matches_source(self, pipeline: PipelineConfig) -> None:
        view = pipeline.fixer_service_view
        assert isinstance(view, FixerServiceView)
        assert view.repo_path == pipeline.repo_path
        assert view.timeout_seconds == pipeline.timeout_seconds
        assert view.fixer_prompt == pipeline.prompts.fixer_prompt

    def test_run_coordinator_view_matches_source(
        self, pipeline: PipelineConfig
    ) -> None:
        view = pipeline.run_coordinator_view
        assert isinstance(view, RunCoordinatorView)
        assert view.repo_path == pipeline.repo_path
        assert view.timeout_seconds == pipeline.timeout_seconds
        assert view.max_gate_retries == pipeline.max_gate_retries
        assert view.disabled_validations == pipeline.disabled_validations
        assert view.fixer_prompt == pipeline.prompts.fixer_prompt
        assert view.validation_config is pipeline.validation_config
        assert view.validation_config_missing == pipeline.validation_config_missing

    def test_agent_session_view_matches_source(self, pipeline: PipelineConfig) -> None:
        view = pipeline.agent_session_view
        assert isinstance(view, AgentSessionView)
        assert view.repo_path == pipeline.repo_path
        assert view.timeout_seconds == pipeline.timeout_seconds
        assert view.prompts is pipeline.prompts
        assert view.max_gate_retries == pipeline.max_gate_retries
        assert view.max_review_retries == pipeline.max_review_retries
        assert view.prompt_validation_commands is pipeline.prompt_validation_commands
        assert view.max_idle_retries == pipeline.max_idle_retries
        assert view.idle_timeout_seconds == pipeline.idle_timeout_seconds
        assert view.deadlock_monitor is pipeline.deadlock_monitor

    def test_views_are_cached(self, pipeline: PipelineConfig) -> None:
        """Cached views return the same object on repeat access (no recompute)."""
        assert pipeline.gate_runner_view is pipeline.gate_runner_view
        assert pipeline.review_runner_view is pipeline.review_runner_view
        assert pipeline.cumulative_review_view is pipeline.cumulative_review_view
        assert pipeline.fixer_service_view is pipeline.fixer_service_view
        assert pipeline.run_coordinator_view is pipeline.run_coordinator_view
        assert pipeline.agent_session_view is pipeline.agent_session_view
