"""Narrow PipelineConfig views consumed by pipeline runners.

Each runner receives a frozen view dataclass that exposes exactly the
PipelineConfig fields it needs. Views are produced by ``@cached_property``
methods on :class:`src.orchestration.types.PipelineConfig` so the source
remains canonical; pipeline runners never construct views directly outside
those getters.

These types live in the pipeline layer (not orchestration) so that runner
modules can reference them without violating the layered-architecture
contract (pipeline cannot import from orchestration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.prompts import PromptProvider
    from src.domain.validation.config_types import (
        PromptValidationCommands,
        ValidationConfig,
    )
    from src.domain.deadlock import DeadlockMonitor


@dataclass(frozen=True)
class GateRunnerView:
    """Narrow view of PipelineConfig for gate runner wiring."""

    repo_path: Path
    max_gate_retries: int
    disabled_validations: set[str] | None
    validation_config: ValidationConfig | None
    validation_config_missing: bool


@dataclass(frozen=True)
class ReviewRunnerView:
    """Narrow view of PipelineConfig for review runner wiring."""

    max_review_retries: int
    review_timeout_seconds: int | None


@dataclass(frozen=True)
class CumulativeReviewView:
    """Narrow view of PipelineConfig for cumulative review runner wiring."""

    repo_path: Path
    review_timeout_seconds: int | None


@dataclass(frozen=True)
class FixerServiceView:
    """Narrow view of PipelineConfig for fixer service wiring."""

    repo_path: Path
    timeout_seconds: int
    fixer_prompt: str


@dataclass(frozen=True)
class RunCoordinatorView:
    """Narrow view of PipelineConfig for run coordinator wiring."""

    repo_path: Path
    timeout_seconds: int
    max_gate_retries: int
    disabled_validations: set[str] | None
    fixer_prompt: str
    validation_config: ValidationConfig | None
    validation_config_missing: bool


@dataclass(frozen=True)
class AgentSessionView:
    """Narrow view of PipelineConfig for agent session runner wiring."""

    repo_path: Path
    timeout_seconds: int
    prompts: PromptProvider
    max_gate_retries: int
    max_review_retries: int
    prompt_validation_commands: PromptValidationCommands
    max_idle_retries: int
    idle_timeout_seconds: float | None
    deadlock_monitor: DeadlockMonitor | None
