"""MalaOrchestrator: Orchestrates parallel issue processing using Claude Agent SDK."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast, overload

from .beads_client import BeadsClient
from .config import MalaConfig
from .event_sink_console import ConsoleEventSink
from .telemetry import BraintrustProvider, NullTelemetryProvider
from .git_utils import (
    get_git_commit_async,
    get_git_branch_async,
    get_baseline_for_issue,
    get_issue_commits_async,
)
from .mcp import MORPH_DISALLOWED_TOOLS
from .log_output.console import (
    truncate_text,
    is_verbose_enabled,
)
from .log_output.run_metadata import (
    RunMetadata,
    RunConfig,
    IssueRun,
    QualityGateResult,
    ValidationResult as MetaValidationResult,
    remove_run_marker,
    write_run_marker,
)
from .pipeline.agent_session_runner import (
    AgentSessionConfig,
    AgentSessionInput,
    AgentSessionRunner,
    SessionCallbacks,
)
from .cerberus_review import DefaultReviewer
from .pipeline.gate_runner import (
    GateRunner,
    GateRunnerConfig,
    PerIssueGateInput,
)
from .pipeline.review_runner import (
    NoProgressInput,
    ReviewInput,
    ReviewRunner,
    ReviewRunnerConfig,
)
from .pipeline.run_coordinator import (
    RunCoordinator,
    RunCoordinatorConfig,
    RunLevelValidationInput,
)
from .quality_gate import QUALITY_GATE_IGNORED_COMMANDS, QualityGate
from .session_log_parser import FileSystemLogProvider
from .tools.env import (
    USER_CONFIG_DIR,
    SCRIPTS_DIR,
    get_lock_dir,
    get_runs_dir,
)
from .tools.locking import (
    release_run_locks,
    cleanup_agent_locks,
)
from .validation.spec import (
    ValidationScope,
    build_validation_spec,
)
from .epic_verifier import ClaudeEpicVerificationModel, EpicVerifier
from .models import RetryConfig
from .orchestrator_types import DEFAULT_AGENT_TIMEOUT_MINUTES

if TYPE_CHECKING:
    from .cerberus_review import ReviewResult
    from .event_sink import EventRunConfig, MalaEventSink
    from .lifecycle import RetryState
    from .models import IssueResolution
    from .orchestrator_types import OrchestratorConfig, _DerivedConfig
    from .protocols import (
        CodeReviewer,
        EpicVerificationModel,
        GateChecker,
        GateResultProtocol,
        IssueProvider,
        LogProvider,
        ReviewIssueProtocol,
        ReviewResultProtocol,
        ValidationSpecProtocol,
    )
    from .quality_gate import GateResult
    from .telemetry import TelemetryProvider
    from .validation.spec import ValidationSpec


# Version (from package metadata)
from importlib.metadata import version as pkg_version

__version__ = pkg_version("mala")

# Prompt file paths (actual file reads deferred to first use)
_PROMPT_DIR = Path(__file__).parent / "prompts"
PROMPT_FILE = _PROMPT_DIR / "implementer_prompt.md"
REVIEW_FOLLOWUP_FILE = _PROMPT_DIR / "review_followup.md"
FIXER_PROMPT_FILE = _PROMPT_DIR / "fixer.md"

# Bounded wait for log file (seconds) - used by AgentSessionRunner
# Re-exported here for backwards compatibility with tests.
# 60s allows time for Claude SDK to flush logs, especially under load.
LOG_FILE_WAIT_TIMEOUT = 60
LOG_FILE_POLL_INTERVAL = 0.5


@functools.cache
def _get_implementer_prompt() -> str:
    """Load implementer prompt template (cached on first use)."""
    return PROMPT_FILE.read_text()


@functools.cache
def _get_review_followup_prompt() -> str:
    """Load review follow-up prompt (cached on first use)."""
    return REVIEW_FOLLOWUP_FILE.read_text()


@functools.cache
def _get_fixer_prompt() -> str:
    """Load fixer prompt (cached on first use)."""
    return FIXER_PROMPT_FILE.read_text()


@dataclass
class IssueResult:
    """Result from a single issue implementation."""

    issue_id: str
    agent_id: str
    success: bool
    summary: str
    duration_seconds: float = 0.0
    session_id: str | None = None  # Claude SDK session ID
    gate_attempts: int = 1  # Number of gate retry attempts
    review_attempts: int = 0  # Number of Codex review attempts
    resolution: IssueResolution | None = None  # Resolution outcome if using markers
    low_priority_review_issues: list[ReviewIssueProtocol] | None = None  # P2/P3 issues


@dataclass
class GateMetadata:
    """Metadata extracted from quality gate results for finalization.

    This dataclass holds the processed results from a quality gate check,
    separating the extraction logic from the finalization flow.
    """

    quality_gate_result: QualityGateResult | None = None
    validation_result: MetaValidationResult | None = None


def _build_gate_metadata(
    gate_result: GateResult | GateResultProtocol | None,
    passed: bool,
) -> GateMetadata:
    """Build GateMetadata from a stored gate result.

    Extracts evidence from the stored gate result without re-running validation.
    This is the primary path used when gate results are available from
    _run_quality_gate_sync.

    Args:
        gate_result: The stored gate result (may be None if no gate ran).
        passed: Whether the overall run passed (affects quality_gate_result.passed).

    Returns:
        GateMetadata with extracted quality gate and validation results.
    """
    if gate_result is None:
        return GateMetadata()

    evidence = gate_result.validation_evidence
    commit_hash = gate_result.commit_hash

    # Build evidence dict from stored evidence
    evidence_dict: dict[str, bool] = {}
    if evidence is not None:
        evidence_dict = evidence.to_evidence_dict()
    evidence_dict["commit_found"] = commit_hash is not None

    quality_gate_result = QualityGateResult(
        passed=passed if passed else gate_result.passed,
        evidence=evidence_dict,
        failure_reasons=[] if passed else list(gate_result.failure_reasons),
    )

    validation_result: MetaValidationResult | None = None
    if evidence is not None:
        # Use KIND_TO_NAME for consistent command names with commands_failed
        commands_run = [
            QualityGate.KIND_TO_NAME.get(kind, kind.value)
            for kind, ran in evidence.commands_ran.items()
            if ran
        ]
        # Filter to only show commands that affected the gate decision
        # (exclude ignored commands like 'uv sync')
        gate_failed_commands = [
            cmd
            for cmd in evidence.failed_commands
            if cmd not in QUALITY_GATE_IGNORED_COMMANDS
        ]
        validation_result = MetaValidationResult(
            passed=passed if passed else gate_result.passed,
            commands_run=commands_run,
            commands_failed=gate_failed_commands,
        )

    return GateMetadata(
        quality_gate_result=quality_gate_result,
        validation_result=validation_result,
    )


def _build_gate_metadata_from_logs(
    log_path: Path,
    result_summary: str,
    result_success: bool,
    quality_gate: GateChecker,
    per_issue_spec: ValidationSpec | ValidationSpecProtocol | None,
) -> GateMetadata:
    """Build GateMetadata by parsing logs directly (fallback path).

    This is a fallback path used when no stored gate result is available.
    It parses the log file directly to extract validation evidence.

    Args:
        log_path: Path to the session log file.
        result_summary: Summary from the issue result (for extracting failure reasons).
        result_success: Whether the run succeeded (determines passed status).
        quality_gate: The GateChecker instance for parsing.
        per_issue_spec: ValidationSpec for parsing evidence (if None, returns empty).

    Returns:
        GateMetadata with extracted results, or empty if spec is None.
    """
    if per_issue_spec is None:
        return GateMetadata()

    evidence = quality_gate.parse_validation_evidence_with_spec(
        log_path, cast("ValidationSpecProtocol", per_issue_spec)
    )

    # Extract failure reasons from result summary
    failure_reasons: list[str] = []
    if "Quality gate failed:" in result_summary:
        reasons_part = result_summary.replace("Quality gate failed: ", "")
        failure_reasons = [r.strip() for r in reasons_part.split(";")]

    # Build spec-driven evidence dict
    evidence_dict = evidence.to_evidence_dict()

    # Check commit exists (we don't have stored result, so check now)
    # Note: We don't call check_commit_exists here because it's expensive
    # and this is a fallback path - just mark as unknown
    evidence_dict["commit_found"] = False

    quality_gate_result = QualityGateResult(
        passed=result_success,
        evidence=evidence_dict,
        failure_reasons=failure_reasons,
    )

    return GateMetadata(
        quality_gate_result=quality_gate_result,
        validation_result=None,  # No validation metadata in fallback path
    )


class MalaOrchestrator:
    """Orchestrates parallel issue processing using Claude Agent SDK.

    This class can be instantiated in two ways:
    1. Legacy: Direct constructor with individual parameters (backward compatible)
    2. Factory: Via create_orchestrator() with OrchestratorConfig and deps

    The factory pattern is preferred for new code as it provides cleaner
    separation of concerns and easier testing.
    """

    @overload
    def __init__(
        self,
        repo_path: Path,
        max_agents: int | None = ...,
        timeout_minutes: int | None = ...,
        max_issues: int | None = ...,
        epic_id: str | None = ...,
        only_ids: set[str] | None = ...,
        braintrust_enabled: bool | None = ...,
        max_gate_retries: int = ...,
        max_review_retries: int = ...,
        disable_validations: set[str] | None = ...,
        coverage_threshold: float | None = ...,
        morph_enabled: bool | None = ...,
        prioritize_wip: bool = ...,
        focus: bool = ...,
        cli_args: dict[str, object] | None = ...,
        issue_provider: IssueProvider | None = ...,
        code_reviewer: CodeReviewer | None = ...,
        gate_checker: GateChecker | None = ...,
        log_provider: LogProvider | None = ...,
        telemetry_provider: TelemetryProvider | None = ...,
        event_sink: MalaEventSink | None = ...,
        config: MalaConfig | None = ...,
        epic_override_ids: set[str] | None = ...,
    ) -> None:
        """Legacy constructor signature for backward compatibility."""
        ...

    @overload
    def __init__(
        self,
        *,
        _config: OrchestratorConfig,
        _mala_config: MalaConfig,
        _derived: _DerivedConfig,
        _issue_provider: IssueProvider,
        _code_reviewer: CodeReviewer,
        _gate_checker: GateChecker,
        _log_provider: LogProvider,
        _telemetry_provider: TelemetryProvider,
        _event_sink: MalaEventSink,
        _epic_verifier: EpicVerifier | None,
    ) -> None:
        """Internal constructor used by create_orchestrator()."""
        ...

    def __init__(
        self,
        repo_path: Path | None = None,
        max_agents: int | None = None,
        timeout_minutes: int | None = None,
        max_issues: int | None = None,
        epic_id: str | None = None,
        only_ids: set[str] | None = None,
        braintrust_enabled: bool | None = None,
        max_gate_retries: int = 3,
        max_review_retries: int = 3,
        disable_validations: set[str] | None = None,
        coverage_threshold: float | None = None,
        morph_enabled: bool | None = None,
        prioritize_wip: bool = False,
        focus: bool = True,
        cli_args: dict[str, object] | None = None,
        # Protocol-based dependency injection for testability
        issue_provider: IssueProvider | None = None,
        code_reviewer: CodeReviewer | None = None,
        gate_checker: GateChecker | None = None,
        log_provider: LogProvider | None = None,
        telemetry_provider: TelemetryProvider | None = None,
        event_sink: MalaEventSink | None = None,
        # Centralized configuration (optional, uses from_env() if not provided)
        config: MalaConfig | None = None,
        # Epic verification override (bypasses verification for listed epic IDs)
        epic_override_ids: set[str] | None = None,
        # Internal parameters used by create_orchestrator() - prefixed with underscore
        _config: OrchestratorConfig | None = None,
        _mala_config: MalaConfig | None = None,
        _derived: _DerivedConfig | None = None,
        _issue_provider: IssueProvider | None = None,
        _code_reviewer: CodeReviewer | None = None,
        _gate_checker: GateChecker | None = None,
        _log_provider: LogProvider | None = None,
        _telemetry_provider: TelemetryProvider | None = None,
        _event_sink: MalaEventSink | None = None,
        _epic_verifier: EpicVerifier | None = None,
    ):
        # Detect which initialization path to use
        if _config is not None:
            # Factory path: use pre-computed config and dependencies
            self._init_from_factory(
                _config,
                _mala_config,
                _derived,
                _issue_provider,
                _code_reviewer,
                _gate_checker,
                _log_provider,
                _telemetry_provider,
                _event_sink,
                _epic_verifier,
            )
        else:
            # Legacy path: compute everything from individual parameters
            if repo_path is None:
                raise ValueError("repo_path is required for legacy initialization")
            self._init_legacy(
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
                issue_provider=issue_provider,
                code_reviewer=code_reviewer,
                gate_checker=gate_checker,
                log_provider=log_provider,
                telemetry_provider=telemetry_provider,
                event_sink=event_sink,
                config=config,
                epic_override_ids=epic_override_ids,
            )

    def _init_from_factory(
        self,
        orch_config: OrchestratorConfig,
        mala_config: MalaConfig | None,
        derived: _DerivedConfig | None,
        issue_provider: IssueProvider | None,
        code_reviewer: CodeReviewer | None,
        gate_checker: GateChecker | None,
        log_provider: LogProvider | None,
        telemetry_provider: TelemetryProvider | None,
        event_sink: MalaEventSink | None,
        epic_verifier: EpicVerifier | None,
    ) -> None:
        """Initialize from factory-provided config and dependencies."""
        if mala_config is None:
            raise ValueError("_mala_config is required when using factory path")
        if derived is None:
            raise ValueError("_derived is required when using factory path")
        if issue_provider is None:
            raise ValueError("_issue_provider is required when using factory path")
        if code_reviewer is None:
            raise ValueError("_code_reviewer is required when using factory path")
        if gate_checker is None:
            raise ValueError("_gate_checker is required when using factory path")
        if log_provider is None:
            raise ValueError("_log_provider is required when using factory path")
        if telemetry_provider is None:
            raise ValueError("_telemetry_provider is required when using factory path")
        if event_sink is None:
            raise ValueError("_event_sink is required when using factory path")

        # Store configs
        self._mala_config = mala_config

        # Set all attributes from config
        self.repo_path = orch_config.repo_path.resolve()
        self.max_agents = orch_config.max_agents
        self.timeout_seconds = derived.timeout_seconds
        self.max_issues = orch_config.max_issues
        self.epic_id = orch_config.epic_id
        self.only_ids = orch_config.only_ids
        self.braintrust_enabled = derived.braintrust_enabled
        self.morph_enabled = derived.morph_enabled
        self.max_gate_retries = orch_config.max_gate_retries
        self.max_review_retries = orch_config.max_review_retries
        self.disable_validations = orch_config.disable_validations
        self._disabled_validations = derived.disabled_validations
        self.coverage_threshold = orch_config.coverage_threshold
        self.prioritize_wip = orch_config.prioritize_wip
        self.focus = orch_config.focus
        self.cli_args = orch_config.cli_args
        self.epic_override_ids = orch_config.epic_override_ids or set()

        # Review disabled reason (if any)
        self.review_disabled_reason = derived.review_disabled_reason

        # Initialize runtime state
        self._init_runtime_state()

        # Set dependencies
        self.log_provider = log_provider
        self.quality_gate = gate_checker
        self.event_sink = event_sink
        self.beads = issue_provider
        self.epic_verifier = epic_verifier
        self.code_reviewer = code_reviewer
        self.telemetry_provider = telemetry_provider

        # Build pipeline runners
        self._init_pipeline_runners()

    def _init_legacy(
        self,
        repo_path: Path,
        max_agents: int | None,
        timeout_minutes: int | None,
        max_issues: int | None,
        epic_id: str | None,
        only_ids: set[str] | None,
        braintrust_enabled: bool | None,
        max_gate_retries: int,
        max_review_retries: int,
        disable_validations: set[str] | None,
        coverage_threshold: float | None,
        morph_enabled: bool | None,
        prioritize_wip: bool,
        focus: bool,
        cli_args: dict[str, object] | None,
        issue_provider: IssueProvider | None,
        code_reviewer: CodeReviewer | None,
        gate_checker: GateChecker | None,
        log_provider: LogProvider | None,
        telemetry_provider: TelemetryProvider | None,
        event_sink: MalaEventSink | None,
        config: MalaConfig | None,
        epic_override_ids: set[str] | None,
    ) -> None:
        """Initialize using legacy individual parameters."""
        import os
        import shutil

        self.repo_path = repo_path.resolve()
        self.max_agents = max_agents
        # Use default timeout if not specified
        effective_timeout = (
            timeout_minutes if timeout_minutes else DEFAULT_AGENT_TIMEOUT_MINUTES
        )
        self.timeout_seconds = effective_timeout * 60
        self.max_issues = max_issues
        self.epic_id = epic_id
        self.only_ids = only_ids

        # Use config if provided, otherwise load from environment
        self._mala_config = (
            config if config is not None else MalaConfig.from_env(validate=False)
        )

        # Derive feature flags from config if not explicitly provided
        if braintrust_enabled is not None:
            self.braintrust_enabled = braintrust_enabled
        else:
            self.braintrust_enabled = self._mala_config.braintrust_enabled

        if morph_enabled is not None:
            self.morph_enabled = morph_enabled
        else:
            self.morph_enabled = self._mala_config.morph_enabled

        self.max_gate_retries = max_gate_retries
        self.max_review_retries = max_review_retries
        self.disable_validations = disable_validations
        self._disabled_validations = (
            set(disable_validations) if disable_validations else set()
        )
        self.coverage_threshold = coverage_threshold
        self.prioritize_wip = prioritize_wip
        self.focus = focus
        self.cli_args = cli_args
        self.epic_override_ids = epic_override_ids or set()

        # Initialize runtime state
        self._init_runtime_state()

        # Initialize dependencies
        self.log_provider: LogProvider = (
            FileSystemLogProvider() if log_provider is None else log_provider
        )

        self.quality_gate: GateChecker = (
            QualityGate(self.repo_path, log_provider=self.log_provider)
            if gate_checker is None
            else gate_checker
        )

        self.event_sink: MalaEventSink = (
            ConsoleEventSink() if event_sink is None else event_sink
        )

        def log_warning(msg: str) -> None:
            self.event_sink.on_warning(msg)

        self.beads: IssueProvider = (
            BeadsClient(self.repo_path, log_warning=log_warning)
            if issue_provider is None
            else issue_provider
        )

        # EpicVerifier setup
        if isinstance(self.beads, BeadsClient):
            verification_model = ClaudeEpicVerificationModel(
                api_key=self._mala_config.llm_api_key,
                base_url=self._mala_config.llm_base_url,
                timeout_ms=self.timeout_seconds * 1000,
                retry_config=RetryConfig(),
            )
            self.epic_verifier: EpicVerifier | None = EpicVerifier(
                beads=self.beads,
                model=cast("EpicVerificationModel", verification_model),
                repo_path=self.repo_path,
                max_diff_size_kb=self._mala_config.max_diff_size_kb,
                event_sink=self.event_sink,
                lock_manager=True,
            )
        else:
            self.epic_verifier = None

        # Check review availability
        self.review_disabled_reason: str | None = None
        if code_reviewer is None and "review" not in self._disabled_validations:
            review_gate_path = (
                self._mala_config.cerberus_bin_path / "review-gate"
                if self._mala_config.cerberus_bin_path
                else None
            )
            if review_gate_path is None:
                cerberus_env_dict = dict(self._mala_config.cerberus_env)
                if "PATH" in cerberus_env_dict:
                    effective_path = (
                        cerberus_env_dict["PATH"]
                        + os.pathsep
                        + os.environ.get("PATH", "")
                    )
                else:
                    effective_path = os.environ.get("PATH", "")
                if shutil.which("review-gate", path=effective_path) is None:
                    self.review_disabled_reason = (
                        "cerberus plugin not detected (review-gate unavailable)"
                    )
            elif not review_gate_path.exists():
                self.review_disabled_reason = (
                    f"review-gate missing at {review_gate_path}"
                )
            elif not review_gate_path.is_file():
                self.review_disabled_reason = (
                    f"review-gate path is not a file: {review_gate_path}"
                )
            elif not os.access(review_gate_path, os.X_OK):
                self.review_disabled_reason = (
                    f"review-gate not executable at {review_gate_path}"
                )

            if self.review_disabled_reason:
                self._disabled_validations.add("review")

        self.code_reviewer: CodeReviewer = (
            DefaultReviewer(
                repo_path=self.repo_path,
                bin_path=self._mala_config.cerberus_bin_path,
                spawn_args=self._mala_config.cerberus_spawn_args,
                wait_args=self._mala_config.cerberus_wait_args,
                env=dict(self._mala_config.cerberus_env),
            )
            if code_reviewer is None
            else code_reviewer
        )

        if telemetry_provider is not None:
            self.telemetry_provider: TelemetryProvider = telemetry_provider
        elif self.braintrust_enabled:
            self.telemetry_provider = BraintrustProvider()
        else:
            self.telemetry_provider = NullTelemetryProvider()

        # Build pipeline runners
        self._init_pipeline_runners()

    def _init_runtime_state(self) -> None:
        """Initialize runtime state that's common to both init paths."""
        self.active_tasks: dict[str, asyncio.Task[IssueResult]] = {}
        self.agent_ids: dict[str, str] = {}
        self.completed: list[IssueResult] = []
        self.failed_issues: set[str] = set()
        self.abort_run: bool = False
        self.abort_reason: str | None = None
        self.session_log_paths: dict[str, Path] = {}
        self.review_log_paths: dict[str, str] = {}
        self.last_gate_results: dict[str, GateResult | GateResultProtocol] = {}
        self.verified_epics: set[str] = set()
        self.per_issue_spec: ValidationSpec | None = None

    def _init_pipeline_runners(self) -> None:
        """Initialize pipeline runner components."""
        # GateRunner
        gate_runner_config = GateRunnerConfig(
            max_gate_retries=self.max_gate_retries,
            disable_validations=self._disabled_validations,
            coverage_threshold=self.coverage_threshold,
        )
        self.gate_runner = GateRunner(
            gate_checker=self.quality_gate,
            repo_path=self.repo_path,
            config=gate_runner_config,
        )

        # ReviewRunner
        review_runner_config = ReviewRunnerConfig(
            max_review_retries=self.max_review_retries,
            capture_session_log=False,
            review_timeout=self._mala_config.review_timeout,
        )
        self.review_runner = ReviewRunner(
            code_reviewer=self.code_reviewer,
            config=review_runner_config,
            gate_checker=self.quality_gate,
        )

        # RunCoordinator
        run_coordinator_config = RunCoordinatorConfig(
            repo_path=self.repo_path,
            timeout_seconds=self.timeout_seconds,
            max_gate_retries=self.max_gate_retries,
            disable_validations=self._disabled_validations,
            coverage_threshold=self.coverage_threshold,
            morph_enabled=self.morph_enabled,
            morph_api_key=self._mala_config.morph_api_key,
        )
        self.run_coordinator = RunCoordinator(
            config=run_coordinator_config,
            gate_checker=self.quality_gate,
            event_sink=self.event_sink,
        )

    # Backward compatibility: expose _config as alias for _mala_config
    @property
    def _config(self) -> MalaConfig:
        """Backward compatibility property for accessing MalaConfig."""
        return self._mala_config

    @_config.setter
    def _config(self, value: MalaConfig) -> None:
        """Backward compatibility setter for MalaConfig."""
        self._mala_config = value

    def _cleanup_agent_locks(self, agent_id: str) -> None:
        """Remove locks held by a specific agent (crash/timeout cleanup)."""
        cleaned = cleanup_agent_locks(agent_id)
        if cleaned:
            self.event_sink.on_locks_cleaned(agent_id, cleaned)

    def _request_abort(self, reason: str) -> None:
        """Signal that the current run should stop due to a fatal error."""
        if self.abort_run:
            return
        self.abort_run = True
        self.abort_reason = reason
        self.event_sink.on_abort_requested(reason)

    def _is_review_enabled(self) -> bool:
        """Return whether review should run for this orchestrator instance."""
        if "review" not in self._disabled_validations:
            return True
        if self.review_disabled_reason and not isinstance(
            self.review_runner.code_reviewer, DefaultReviewer
        ):
            return True
        return False

    def _run_quality_gate_sync(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryState,
    ) -> tuple[GateResult | GateResultProtocol, int]:
        """Synchronous quality gate check (blocking I/O).

        This is the blocking implementation that gets run via asyncio.to_thread.
        Delegates to GateRunner for the actual gate checking logic.
        """
        # Sync per_issue_spec with gate_runner (orchestrator may have built it at run start)
        if self.per_issue_spec is not None:
            self.gate_runner.set_cached_spec(self.per_issue_spec)

        gate_input = PerIssueGateInput(
            issue_id=issue_id,
            log_path=log_path,
            retry_state=retry_state,
            spec=self.per_issue_spec,
        )
        output = self.gate_runner.run_per_issue_gate(gate_input)

        # Sync cached spec back to orchestrator (gate_runner may have built it)
        if self.per_issue_spec is None:
            self.per_issue_spec = self.gate_runner.get_cached_spec()

        # Store gate result to avoid duplicate validation in _finalize_issue_result
        # The gate result includes validation_evidence parsed from logs
        self.last_gate_results[issue_id] = output.gate_result

        return (output.gate_result, output.new_log_offset)

    async def _run_quality_gate_async(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryState,
    ) -> tuple[GateResult | GateResultProtocol, int]:
        """Run quality gate check asynchronously via to_thread.

        Wraps the blocking _run_quality_gate_sync to avoid stalling the event loop.
        This allows the orchestrator to service other agents while a gate runs.

        Args:
            issue_id: The issue being checked.
            log_path: Path to the session log file.
            retry_state: Current retry state for this issue.

        Returns:
            Tuple of (GateResult, new_log_offset).
        """
        return await asyncio.to_thread(
            self._run_quality_gate_sync, issue_id, log_path, retry_state
        )

    async def _check_epic_closure(self, issue_id: str) -> None:
        """Check if closing this issue should also close its parent epic.

        Args:
            issue_id: The issue that was just closed.
        """
        parent_epic = await self.beads.get_parent_epic_async(issue_id)
        if parent_epic is None or parent_epic in self.verified_epics:
            return

        if self.epic_verifier is not None:
            # Verify and close only the affected epic
            human_override = parent_epic in self.epic_override_ids
            verification_result = await self.epic_verifier.verify_and_close_epic(
                parent_epic, human_override=human_override
            )
            # Mark as verified to prevent duplicate verifications
            if verification_result.verified_count > 0:
                self.verified_epics.add(parent_epic)
        elif await self.beads.close_eligible_epics_async():
            # Fallback for mock providers without EpicVerifier
            self.event_sink.on_epic_closed(issue_id)

    def _record_issue_run(
        self,
        issue_id: str,
        result: IssueResult,
        log_path: Path | None,
        gate_metadata: GateMetadata,
        run_metadata: RunMetadata,
    ) -> None:
        """Record an issue run to the run metadata.

        Args:
            issue_id: The issue ID.
            result: The issue result.
            log_path: Path to the session log (if any).
            gate_metadata: Extracted gate metadata.
            run_metadata: The run metadata to record to.
        """
        issue_run = IssueRun(
            issue_id=result.issue_id,
            agent_id=result.agent_id,
            status="success" if result.success else "failed",
            duration_seconds=result.duration_seconds,
            session_id=result.session_id,
            log_path=str(log_path) if log_path else None,
            quality_gate=gate_metadata.quality_gate_result,
            error=result.summary if not result.success else None,
            gate_attempts=result.gate_attempts,
            review_attempts=result.review_attempts,
            validation=gate_metadata.validation_result,
            resolution=result.resolution,
            review_log_path=self.review_log_paths.get(issue_id),
        )
        run_metadata.record_issue(issue_run)

    def _cleanup_session_paths(self, issue_id: str) -> None:
        """Remove stored session and review log paths for an issue.

        Args:
            issue_id: The issue ID to clean up paths for.
        """
        self.session_log_paths.pop(issue_id, None)
        self.review_log_paths.pop(issue_id, None)

    async def _emit_completion(
        self,
        issue_id: str,
        result: IssueResult,
        log_path: Path | None,
    ) -> None:
        """Emit completion event and handle failure tracking.

        Args:
            issue_id: The issue ID.
            result: The issue result.
            log_path: Path to the session log (if any).
        """
        self.event_sink.on_issue_completed(
            issue_id,
            issue_id,
            result.success,
            result.duration_seconds,
            truncate_text(result.summary, 50) if result.success else result.summary,
        )
        if not result.success:
            self.failed_issues.add(issue_id)
            await self.beads.mark_needs_followup_async(
                issue_id, result.summary, log_path=log_path
            )

    async def _create_review_tracking_issues(
        self,
        source_issue_id: str,
        review_issues: list[ReviewIssueProtocol],
    ) -> None:
        """Create beads issues from P2/P3 review findings.

        These are low-priority issues that didn't block the review but should
        be tracked for later resolution. Each issue is created with appropriate
        priority and linked to the source issue via tags.

        Args:
            source_issue_id: The issue ID that triggered the review.
            review_issues: List of ReviewIssueProtocol objects from the review.
        """
        for issue in review_issues:
            # Access protocol-defined attributes directly
            file_path = issue.file
            line_start = issue.line_start
            line_end = issue.line_end
            priority = issue.priority
            title = issue.title
            body = issue.body
            reviewer = issue.reviewer

            # Map priority to beads priority string (P2, P3, etc.)
            priority_str = f"P{priority}" if priority is not None else "P3"

            # Build location string
            if line_start == line_end or line_end == 0:
                location = f"{file_path}:{line_start}" if file_path else ""
            else:
                location = f"{file_path}:{line_start}-{line_end}" if file_path else ""

            # Build dedup tag from content hash to prevent duplicate issues
            # Hash key: file + line + title (not body, which may vary)
            hash_key = f"{file_path}:{line_start}:{line_end}:{title}"
            content_hash = hashlib.sha256(hash_key.encode()).hexdigest()[:12]
            dedup_tag = f"review_finding:{content_hash}"

            # Check for existing issue with this dedup tag
            existing_id = await self.beads.find_issue_by_tag_async(dedup_tag)
            if existing_id:
                # Already exists, skip creation
                continue

            # Build issue title with location
            issue_title = f"[Review] {title}"
            if location:
                issue_title = f"[Review] {location}: {title}"

            # Build description
            description_parts = [
                "## Review Finding",
                "",
                f"This issue was auto-created from a {priority_str} review finding.",
                "",
                f"**Source issue:** {source_issue_id}",
                f"**Reviewer:** {reviewer}",
            ]
            if location:
                description_parts.append(f"**Location:** {location}")
            if body:
                description_parts.extend(["", "## Details", "", body])

            description = "\n".join(description_parts)

            # Tags for tracking and deduplication
            tags = [
                "auto_generated",
                "review_finding",
                f"source:{source_issue_id}",
                dedup_tag,
            ]

            new_issue_id = await self.beads.create_issue_async(
                title=issue_title,
                description=description,
                priority=priority_str,
                tags=tags,
            )
            if new_issue_id:
                self.event_sink.on_warning(
                    f"Created tracking issue {new_issue_id} for {priority_str} review finding",
                    agent_id=source_issue_id,
                )

    async def _finalize_issue_result(
        self,
        issue_id: str,
        result: IssueResult,
        run_metadata: RunMetadata,
    ) -> None:
        """Record an issue result, update metadata, and emit logs.

        Uses stored gate result to derive metadata, avoiding duplicate
        validation parsing. Gate result includes validation_evidence
        already parsed during quality gate check.
        """
        log_path = self.session_log_paths.get(issue_id)
        stored_gate_result = self.last_gate_results.get(issue_id)

        # Build gate metadata from stored result or logs
        if stored_gate_result is not None:
            gate_metadata = _build_gate_metadata(stored_gate_result, result.success)
        elif not result.success and log_path and log_path.exists():
            gate_metadata = _build_gate_metadata_from_logs(
                log_path,
                result.summary,
                result.success,
                self.quality_gate,
                self.per_issue_spec,
            )
        else:
            gate_metadata = GateMetadata()

        # Handle successful issue closure and epic verification
        if result.success and await self.beads.close_async(issue_id):
            self.event_sink.on_issue_closed(issue_id, issue_id)
            await self._check_epic_closure(issue_id)

            # Create tracking issues for P2/P3 review findings (if enabled)
            if self._config.track_review_issues and result.low_priority_review_issues:
                await self._create_review_tracking_issues(
                    issue_id, result.low_priority_review_issues
                )

        # Update tracking state
        self.completed.append(result)
        self.active_tasks.pop(issue_id, None)

        # Record to run metadata and cleanup
        self._record_issue_run(issue_id, result, log_path, gate_metadata, run_metadata)
        self._cleanup_session_paths(issue_id)
        await self._emit_completion(issue_id, result, log_path)

    async def _abort_active_tasks(self, run_metadata: RunMetadata) -> None:
        """Cancel active tasks and mark them as failed.

        Tasks that have already completed are finalized with their real results
        rather than being marked as aborted.
        """
        if not self.active_tasks:
            return
        reason = self.abort_reason or "Unrecoverable error"
        self.event_sink.on_tasks_aborting(len(self.active_tasks), reason)
        # Cancel tasks that are still running
        for task in self.active_tasks.values():
            if not task.done():
                task.cancel()

        # Finalize each remaining issue - use real result if already done
        for issue_id, task in list(self.active_tasks.items()):
            if task.done():
                # Task completed before we could cancel - use real result
                try:
                    result = task.result()
                except asyncio.CancelledError:
                    result = IssueResult(
                        issue_id=issue_id,
                        agent_id=self.agent_ids.get(issue_id, "unknown"),
                        success=False,
                        summary=f"Aborted due to unrecoverable error: {reason}",
                    )
                except Exception as e:
                    result = IssueResult(
                        issue_id=issue_id,
                        agent_id=self.agent_ids.get(issue_id, "unknown"),
                        success=False,
                        summary=str(e),
                    )
            else:
                # Task was still running - mark as aborted
                result = IssueResult(
                    issue_id=issue_id,
                    agent_id=self.agent_ids.get(issue_id, "unknown"),
                    success=False,
                    summary=f"Aborted due to unrecoverable error: {reason}",
                )
            await self._finalize_issue_result(issue_id, result, run_metadata)

    def _build_session_callbacks(self, issue_id: str) -> SessionCallbacks:
        """Build callbacks for session operations.

        Args:
            issue_id: The issue ID for tracking state.

        Returns:
            SessionCallbacks with gate, review, and logging callbacks.
        """

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult | GateResultProtocol, int]:
            return await self._run_quality_gate_async(issue_id, log_path, retry_state)

        async def on_review_check(
            issue_id: str,
            issue_desc: str | None,
            baseline: str | None,
            session_id: str | None,
            retry_state: RetryState,
        ) -> ReviewResult | ReviewResultProtocol:
            current_head = await get_git_commit_async(self.repo_path)
            self.review_runner.config.capture_session_log = is_verbose_enabled()
            commit_shas = await get_issue_commits_async(
                self.repo_path,
                issue_id,
                since_timestamp=retry_state.baseline_timestamp,
            )
            review_input = ReviewInput(
                issue_id=issue_id,
                repo_path=self.repo_path,
                commit_sha=current_head,
                issue_description=issue_desc,
                baseline_commit=baseline,
                commit_shas=commit_shas or None,
                claude_session_id=session_id,
            )
            output = await self.review_runner.run_review(review_input)
            if output.session_log_path:
                self.review_log_paths[issue_id] = output.session_log_path
            return output.result

        def on_review_no_progress(
            log_path: Path,
            log_offset: int,
            prev_commit: str | None,
            curr_commit: str | None,
        ) -> bool:
            no_progress_input = NoProgressInput(
                log_path=log_path,
                log_offset=log_offset,
                previous_commit_hash=prev_commit,
                current_commit_hash=curr_commit,
                spec=self.per_issue_spec,
            )
            return self.review_runner.check_no_progress(no_progress_input)

        def get_log_path(session_id: str) -> Path:
            log_path = self.log_provider.get_log_path(self.repo_path, session_id)
            self.session_log_paths[issue_id] = log_path
            return log_path

        def get_log_offset(log_path: Path, start_offset: int) -> int:
            return self.quality_gate.get_log_end_offset(log_path, start_offset)

        def on_tool_use(agent_id: str, tool_name: str, arguments: dict | None) -> None:
            self.event_sink.on_tool_use(agent_id, tool_name, arguments=arguments)

        def on_agent_text(agent_id: str, text: str) -> None:
            self.event_sink.on_agent_text(agent_id, text)

        return SessionCallbacks(
            on_gate_check=on_gate_check,
            on_review_check=on_review_check,
            on_review_no_progress=on_review_no_progress,
            get_log_path=get_log_path,
            get_log_offset=get_log_offset,
            on_abort=self._request_abort,
            on_tool_use=on_tool_use,
            on_agent_text=on_agent_text,
        )

    async def run_implementer(self, issue_id: str) -> IssueResult:
        """Run implementer agent for a single issue with gate retry support.

        Delegates to AgentSessionRunner for SDK-specific session handling.
        """
        temp_agent_id = f"{issue_id}-{uuid.uuid4().hex[:8]}"
        self.agent_ids[issue_id] = temp_agent_id

        # Prepare session inputs
        issue_description = await self.beads.get_issue_description_async(issue_id)
        baseline_commit = await get_baseline_for_issue(self.repo_path, issue_id)
        if baseline_commit is None:
            baseline_commit = await get_git_commit_async(self.repo_path)

        prompt = _get_implementer_prompt().format(
            issue_id=issue_id,
            repo_path=self.repo_path,
            lock_dir=get_lock_dir(),
            scripts_dir=SCRIPTS_DIR,
            agent_id=temp_agent_id,
        )

        review_enabled = self._is_review_enabled()
        session_config = AgentSessionConfig(
            repo_path=self.repo_path,
            timeout_seconds=self.timeout_seconds,
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            morph_enabled=self.morph_enabled,
            morph_api_key=self._config.morph_api_key,
            review_enabled=review_enabled,
        )
        session_input = AgentSessionInput(
            issue_id=issue_id,
            prompt=prompt,
            baseline_commit=baseline_commit,
            issue_description=issue_description,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=self._build_session_callbacks(issue_id),
            event_sink=self.event_sink,
        )
        tracer = self.telemetry_provider.create_span(
            issue_id,
            metadata={
                "agent_id": temp_agent_id,
                "mala_version": __version__,
                "project_dir": self.repo_path.name,
                "git_branch": await get_git_branch_async(self.repo_path),
                "git_commit": await get_git_commit_async(self.repo_path),
                "timeout_seconds": self.timeout_seconds,
            },
        )

        try:
            with tracer:
                tracer.log_input(prompt)
                output = await runner.run_session(session_input, tracer=tracer)
                self.agent_ids[issue_id] = output.agent_id
                if output.success:
                    tracer.set_success(True)
                else:
                    tracer.set_error(output.summary)
        finally:
            agent_id = self.agent_ids.pop(issue_id, temp_agent_id)
            self._cleanup_agent_locks(agent_id)

        return IssueResult(
            issue_id=issue_id,
            agent_id=output.agent_id,
            success=output.success,
            summary=output.summary,
            duration_seconds=output.duration_seconds,
            session_id=output.session_id,
            gate_attempts=output.gate_attempts,
            review_attempts=output.review_attempts,
            resolution=output.resolution,
            low_priority_review_issues=output.low_priority_review_issues,
        )

    async def spawn_agent(self, issue_id: str) -> bool:
        """Spawn a new agent task for an issue. Returns True if spawned."""
        if not await self.beads.claim_async(issue_id):
            self.failed_issues.add(issue_id)
            self.event_sink.on_claim_failed(issue_id, issue_id)
            return False

        task = asyncio.create_task(self.run_implementer(issue_id))
        self.active_tasks[issue_id] = task
        self.event_sink.on_agent_started(issue_id, issue_id)
        return True

    def _build_run_config(self) -> EventRunConfig:
        """Build EventRunConfig for on_run_started event."""
        from .event_sink import EventRunConfig

        review_enabled = self._is_review_enabled()

        # Compute morph disabled reason
        morph_disabled_reason: str | None = None
        if not self.morph_enabled:
            if self.cli_args and self.cli_args.get("no_morph"):
                morph_disabled_reason = "--no-morph"
            elif not self._config.morph_api_key:
                morph_disabled_reason = "MORPH_API_KEY not set"
            else:
                morph_disabled_reason = "disabled by config"

        # Compute braintrust disabled reason
        braintrust_disabled_reason: str | None = None
        if not self.braintrust_enabled:
            braintrust_disabled_reason = (
                f"add BRAINTRUST_API_KEY to {USER_CONFIG_DIR}/.env"
            )

        return EventRunConfig(
            repo_path=str(self.repo_path),
            max_agents=self.max_agents,
            timeout_minutes=self.timeout_seconds // 60
            if self.timeout_seconds
            else None,
            max_issues=self.max_issues,
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            epic_id=self.epic_id,
            only_ids=list(self.only_ids) if self.only_ids else None,
            braintrust_enabled=self.braintrust_enabled,
            braintrust_disabled_reason=braintrust_disabled_reason,
            review_enabled=review_enabled,
            review_disabled_reason=self.review_disabled_reason,
            morph_enabled=self.morph_enabled,
            morph_disallowed_tools=list(MORPH_DISALLOWED_TOOLS)
            if self.morph_enabled
            else None,
            morph_disabled_reason=morph_disabled_reason,
            prioritize_wip=self.prioritize_wip,
            cli_args=self.cli_args,
        )

    def _create_run_metadata(self) -> RunMetadata:
        """Create run metadata tracker with current configuration."""
        run_config = RunConfig(
            max_agents=self.max_agents,
            timeout_minutes=self.timeout_seconds // 60
            if self.timeout_seconds
            else None,
            max_issues=self.max_issues,
            epic_id=self.epic_id,
            only_ids=list(self.only_ids) if self.only_ids else None,
            braintrust_enabled=self.braintrust_enabled,
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            review_enabled=self._is_review_enabled(),
            cli_args=self.cli_args,
        )
        return RunMetadata(self.repo_path, run_config, __version__)

    async def _run_main_loop(self, run_metadata: RunMetadata) -> int:
        """Run the main agent spawning and completion loop.

        Returns the number of issues spawned.
        """
        issues_spawned = 0
        while True:
            if self.abort_run:
                await self._abort_active_tasks(run_metadata)
                break

            limit_reached = (
                self.max_issues is not None and issues_spawned >= self.max_issues
            )

            suppress_warn_ids = None
            if self.only_ids:
                suppress_warn_ids = (
                    self.failed_issues
                    | set(self.active_tasks.keys())
                    | {result.issue_id for result in self.completed}
                )
            ready = (
                await self.beads.get_ready_async(
                    self.failed_issues,
                    epic_id=self.epic_id,
                    only_ids=self.only_ids,
                    suppress_warn_ids=suppress_warn_ids,
                    prioritize_wip=self.prioritize_wip,
                    focus=self.focus,
                )
                if not limit_reached
                else []
            )

            if ready:
                self.event_sink.on_ready_issues(list(ready))

            while (
                self.max_agents is None or len(self.active_tasks) < self.max_agents
            ) and ready:
                issue_id = ready.pop(0)
                if issue_id not in self.active_tasks:
                    if await self.spawn_agent(issue_id):
                        issues_spawned += 1
                        if (
                            self.max_issues is not None
                            and issues_spawned >= self.max_issues
                        ):
                            break

            if not self.active_tasks:
                if limit_reached:
                    self.event_sink.on_no_more_issues(
                        f"limit_reached ({self.max_issues})"
                    )
                elif not ready:
                    self.event_sink.on_no_more_issues("none_ready")
                break

            self.event_sink.on_waiting_for_agents(len(self.active_tasks))
            done, _ = await asyncio.wait(
                self.active_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                for issue_id, t in list(self.active_tasks.items()):
                    if t is task:
                        try:
                            result = task.result()
                        except asyncio.CancelledError:
                            result = IssueResult(
                                issue_id=issue_id,
                                agent_id=self.agent_ids.get(issue_id, "unknown"),
                                success=False,
                                summary=(
                                    f"Aborted due to unrecoverable error: {self.abort_reason}"
                                    if self.abort_reason
                                    else "Aborted due to unrecoverable error"
                                ),
                            )
                        except Exception as e:
                            result = IssueResult(
                                issue_id=issue_id,
                                agent_id=self.agent_ids.get(issue_id, "unknown"),
                                success=False,
                                summary=str(e),
                            )
                        await self._finalize_issue_result(
                            issue_id, result, run_metadata
                        )
                        break

            if self.abort_run:
                await self._abort_active_tasks(run_metadata)
                break

        return issues_spawned

    async def _finalize_run(
        self,
        run_metadata: RunMetadata,
        run_validation_passed: bool,
    ) -> tuple[int, int]:
        """Log summary and return final results."""
        success_count = sum(1 for r in self.completed if r.success)
        total = len(self.completed)

        # Emit run completed event
        abort_reason = (
            self.abort_reason or "Unrecoverable error" if self.abort_run else None
        )
        self.event_sink.on_run_completed(
            success_count, total, run_validation_passed, abort_reason
        )

        if success_count > 0:
            if await self.beads.commit_issues_async():
                self.event_sink.on_issues_committed()

        if total > 0:
            metadata_path = run_metadata.save()
            self.event_sink.on_run_metadata_saved(str(metadata_path))

        print()
        if self.abort_run:
            return (0, total)
        if not run_validation_passed:
            return (0, total)
        return (success_count, total)

    async def run(self) -> tuple[int, int]:
        """Main orchestration loop. Returns (success_count, total_count)."""
        self.event_sink.on_run_started(self._build_run_config())

        get_lock_dir().mkdir(parents=True, exist_ok=True)
        get_runs_dir().mkdir(parents=True, exist_ok=True)

        run_metadata = self._create_run_metadata()
        self.per_issue_spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
            disable_validations=self._disabled_validations,
            coverage_threshold=self.coverage_threshold,
            repo_path=self.repo_path,
        )
        write_run_marker(
            run_id=run_metadata.run_id,
            repo_path=self.repo_path,
            max_agents=self.max_agents,
        )

        try:
            await self._run_main_loop(run_metadata)
        finally:
            released = release_run_locks(list(self.agent_ids.values()))
            # Only emit event if released is a positive integer
            # (release_run_locks returns int, but tests may mock it)
            if isinstance(released, int) and released > 0:
                self.event_sink.on_locks_released(released)
            remove_run_marker(run_metadata.run_id)

        success_count = sum(1 for r in self.completed if r.success)
        run_validation_passed = True
        if success_count > 0 and not self.abort_run:
            validation_input = RunLevelValidationInput(run_metadata=run_metadata)
            validation_output = await self.run_coordinator.run_validation(
                validation_input
            )
            run_validation_passed = validation_output.passed

        return await self._finalize_run(run_metadata, run_validation_passed)

    def run_sync(self) -> tuple[int, int]:
        """Synchronous wrapper for run(). Returns (success_count, total_count).

        Use this method when calling from synchronous code. It handles event loop
        creation and cleanup automatically.

        Raises:
            RuntimeError: If called from within an async context (e.g., inside an
                async function or when an event loop is already running). In that
                case, use `await orchestrator.run()` instead.

        Example usage::

            # From sync code (scripts, CLI, tests)
            orchestrator = MalaOrchestrator(repo_path=Path("."))
            success_count, total = orchestrator.run_sync()

            # From async code
            async def main():
                orchestrator = MalaOrchestrator(repo_path=Path("."))
                success_count, total = await orchestrator.run()

        Returns:
            Tuple of (success_count, total_count).
        """
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # If we get here, there's a running event loop - can't use run_sync()
            raise RuntimeError(
                "run_sync() cannot be called from within an async context. "
                "Use 'await orchestrator.run()' instead."
            )
        except RuntimeError as e:
            # Check if it's our error or the "no running event loop" error
            if "run_sync() cannot be called" in str(e):
                raise
            # No running event loop - this is the expected case for sync usage

        # Create a new event loop and run
        return asyncio.run(self.run())
