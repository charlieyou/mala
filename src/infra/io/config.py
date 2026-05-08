"""Configuration dataclass for mala.

Provides MalaConfig for centralized configuration management. This allows
programmatic users to construct configuration without relying on environment
variables, while CLI users can continue using env vars via from_env().

Environment Variables:
    MALA_RUNS_DIR: Directory for run metadata files (default: ~/.config/mala/runs)
    MALA_LOCK_DIR: Directory for file locks (default: /tmp/mala-locks)
    CLAUDE_CONFIG_DIR: Claude SDK config directory (default: ~/.claude)
    MALA_CLAUDE_SETTINGS_SOURCES: Comma-separated Claude settings sources
    LLM_API_KEY: API key for LLM calls (fallback to ANTHROPIC_API_KEY)
    LLM_BASE_URL: Base URL for LLM API

Note: The following env vars are deprecated and will be removed in a future release.
Configure these in mala.yaml instead:
    - MALA_REVIEW_TIMEOUT → validation_triggers.<trigger>.code_review.cerberus.timeout
    - MALA_CERBERUS_SPAWN_ARGS → validation_triggers.<trigger>.code_review.cerberus.spawn_args
    - MALA_CERBERUS_WAIT_ARGS → validation_triggers.<trigger>.code_review.cerberus.wait_args
    - MALA_CERBERUS_ENV → validation_triggers.<trigger>.code_review.cerberus.env
    - MALA_MAX_EPIC_VERIFICATION_RETRIES → validation_triggers.epic_completion.max_epic_verification_retries
    - MALA_MAX_DIFF_SIZE_KB → max_diff_size_kb (root level in mala.yaml)
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import warnings
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Literal

from src.core.constants import (
    DEFAULT_CLAUDE_SETTINGS_SOURCES,
    DEFAULT_CODEX_APPROVAL_POLICY,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_SANDBOX,
    VALID_CLAUDE_SETTINGS_SOURCES,
    VALID_CODEX_APPROVAL_POLICIES,
    VALID_CODEX_SANDBOXES,
    VALID_EFFORTS,
    default_effort_for,
    validate_amp_effort_for_mode,
    validate_codex_effort,
)

_logger = logging.getLogger(__name__)

VALID_CODERS: frozenset[str] = frozenset({"claude", "amp", "codex"})
VALID_AMP_MODES: frozenset[str] = frozenset({"smart", "rush", "deep"})
DEFAULT_CODER: Literal["claude", "amp", "codex"] = "amp"
DEFAULT_AMP_MODE: Literal["smart", "rush", "deep"] = "deep"


def parse_cerberus_args(raw: str | None, *, source: str) -> list[str]:
    if not raw or not raw.strip():
        return []
    try:
        return shlex.split(raw)
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from exc


def parse_cerberus_env(raw: str | None, *, source: str) -> dict[str, str]:
    if not raw or not raw.strip():
        return {}

    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: invalid JSON ({exc})") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{source}: JSON must be an object")
        return {str(key): str(value) for key, value in data.items()}

    env: dict[str, str] = {}
    for part in [item.strip() for item in raw.split(",") if item.strip()]:
        if "=" not in part:
            raise ValueError(f"{source}: invalid entry '{part}' (expected KEY=VALUE)")
        key, value = part.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{source}: invalid entry '{part}' (empty key)")
        env[key] = value
    return env


def _normalize_cerberus_env(env: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Normalize env map into a stable, hashable tuple of key/value pairs."""
    return tuple(sorted(env.items()))


def _safe_int(value: str | None, default: int) -> int:
    """Safely parse an integer with fallback to default."""
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def parse_claude_settings_sources(
    raw: str | None, *, source: str
) -> tuple[str, ...] | None:
    """Parse comma-separated claude settings sources.

    Args:
        raw: Raw comma-separated string like "local,project,user"
        source: Source name for error messages (e.g., "MALA_CLAUDE_SETTINGS_SOURCES")

    Returns:
        Tuple of source strings, or None if raw is empty/None.

    Raises:
        ValueError: If any source is not in VALID_CLAUDE_SETTINGS_SOURCES.
    """
    if not raw or not raw.strip():
        return None

    # Split by comma, strip whitespace, filter empty parts
    sources = [s.strip() for s in raw.split(",") if s.strip()]
    if not sources:
        return None

    # Validate each source
    for src in sources:
        if src not in VALID_CLAUDE_SETTINGS_SOURCES:
            valid = ", ".join(sorted(VALID_CLAUDE_SETTINGS_SOURCES))
            raise ValueError(
                f"{source}: Invalid source '{src}'. Valid sources: {valid}"
            )

    return tuple(sources)


def parse_coder(
    raw: str | None, *, source: str
) -> Literal["claude", "amp", "codex"] | None:
    """Parse a coder selection string.

    Args:
        raw: Raw value, "claude", "amp", or "codex" (after strip; case-sensitive
            to match the strict-enum YAML schema in
            src/domain/validation/config.py).
        source: Source name for error messages (e.g., "MALA_CODER", "CLI").

    Returns:
        Validated coder string, or None if raw is empty/None.

    Raises:
        ValueError: If raw is not in VALID_CODERS.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped not in VALID_CODERS:
        valid = ", ".join(sorted(VALID_CODERS))
        raise ValueError(f"{source}: Invalid coder '{stripped}'. Valid coders: {valid}")
    # Narrow Literal type for the type checker.
    if stripped == "amp":
        return "amp"
    if stripped == "codex":
        return "codex"
    return "claude"


def parse_amp_mode(
    raw: str | None, *, source: str
) -> Literal["smart", "rush", "deep"] | None:
    """Parse an Amp mode string.

    Args:
        raw: Raw value like "smart", "rush", or "deep" (after strip).
        source: Source name for error messages (e.g., "MALA_AMP_MODE", "CLI").

    Returns:
        Validated mode string, or None if raw is empty/None.

    Raises:
        ValueError: If raw is not in VALID_AMP_MODES.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped not in VALID_AMP_MODES:
        valid = ", ".join(sorted(VALID_AMP_MODES))
        raise ValueError(
            f"{source}: Invalid amp mode '{stripped}'. Valid modes: {valid}"
        )
    if stripped == "rush":
        return "rush"
    if stripped == "deep":
        return "deep"
    return "smart"


def parse_codex_model(raw: str | None, *, source: str) -> str | None:
    """Parse a Codex model string.

    Codex accepts arbitrary model identifiers (the SDK does not enforce a
    fixed set), so validation is shape-only: empty/whitespace -> None,
    otherwise pass through the trimmed value.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    del source
    return stripped


def parse_codex_effort(raw: str | None, *, source: str) -> str | None:
    """Parse a Codex effort string with strict-enum validation.

    Validates against the Codex SDK's ``ReasoningEffort`` enum (or the
    documented fallback when the SDK is not installed). Empty/whitespace ->
    None.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    validate_codex_effort(stripped, source=source)
    return stripped


def parse_codex_approval_policy(
    raw: str | None, *, source: str
) -> Literal["never", "on-request", "on-failure", "untrusted"] | None:
    """Parse a Codex ``approval_policy`` string with strict-enum validation."""
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped not in VALID_CODEX_APPROVAL_POLICIES:
        valid = ", ".join(sorted(VALID_CODEX_APPROVAL_POLICIES))
        raise ValueError(
            f"{source}: Invalid Codex approval_policy '{stripped}'. "
            f"Valid policies: {valid}"
        )
    if stripped == "on-request":
        return "on-request"
    if stripped == "on-failure":
        return "on-failure"
    if stripped == "untrusted":
        return "untrusted"
    return "never"


def parse_codex_sandbox(
    raw: str | None, *, source: str
) -> Literal["read-only", "workspace-write", "danger-full-access"] | None:
    """Parse a Codex ``sandbox`` string with strict-enum validation."""
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped not in VALID_CODEX_SANDBOXES:
        valid = ", ".join(sorted(VALID_CODEX_SANDBOXES))
        raise ValueError(
            f"{source}: Invalid Codex sandbox '{stripped}'. Valid sandboxes: {valid}"
        )
    if stripped == "read-only":
        return "read-only"
    if stripped == "workspace-write":
        return "workspace-write"
    return "danger-full-access"


def parse_effort(raw: str | None, *, source: str) -> str | None:
    """Parse a reasoning-effort string.

    Args:
        raw: Raw value like "low", "medium", "high", "xhigh", or "max"
            (after strip).
        source: Source name for error messages (e.g., "MALA_EFFORT", "CLI",
            "--effort").

    Returns:
        Validated effort string, or None if raw is empty/None.

    Raises:
        ValueError: If raw is not in VALID_EFFORTS.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped not in VALID_EFFORTS:
        valid = ", ".join(sorted(VALID_EFFORTS))
        raise ValueError(
            f"{source}: Invalid effort '{stripped}'. Valid efforts: {valid}"
        )
    return stripped


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message)


# Deprecated env vars and their replacements in mala.yaml
_DEPRECATED_ENV_VARS = {
    "MALA_REVIEW_TIMEOUT": "validation_triggers.<trigger>.code_review.cerberus.timeout",
    "MALA_CERBERUS_SPAWN_ARGS": "validation_triggers.<trigger>.code_review.cerberus.spawn_args",
    "MALA_CERBERUS_WAIT_ARGS": "validation_triggers.<trigger>.code_review.cerberus.wait_args",
    "MALA_CERBERUS_ENV": "validation_triggers.<trigger>.code_review.cerberus.env",
    "MALA_MAX_EPIC_VERIFICATION_RETRIES": "validation_triggers.epic_completion.max_epic_verification_retries",
    "MALA_MAX_DIFF_SIZE_KB": "max_diff_size_kb (root level)",
}


def _warn_deprecated_env_vars() -> None:
    """Emit warnings for deprecated env vars that are set but no longer read."""
    for env_var, yaml_path in _DEPRECATED_ENV_VARS.items():
        if os.environ.get(env_var):
            warnings.warn(
                f"{env_var} is deprecated and no longer read. "
                f"Use {yaml_path} in mala.yaml instead.",
                DeprecationWarning,
                stacklevel=3,
            )


@dataclass(frozen=True)
class AmpOptions:
    """Amp-coder specific options.

    Attributes:
        mode: Amp execution mode. One of "smart", "rush", or "deep" (default).
            Only consulted when MalaConfig.coder == "amp".
    """

    mode: Literal["smart", "rush", "deep"] = DEFAULT_AMP_MODE


@dataclass(frozen=True)
class CodexOptions:
    """Codex-coder specific options (Phase B).

    Attributes:
        model: Codex model identifier. Default ``gpt-5.5`` (decision #9).
        effort: Optional ``ReasoningEffort`` value passed through to Codex.
            ``None`` means "use SDK default"; non-None values are validated
            against the Codex SDK's ``ReasoningEffort`` enum at parse time
            (decision #13).
        approval_policy: Codex turn approval policy. Default ``never``
            (decision #3) — combined with ``sandbox=danger-full-access``,
            the bundled hook is the only safety gate for unattended runs.
        sandbox: Codex sandbox mode. Default ``danger-full-access``
            (decision #2) — Mala's hook + locking MCP enforce safety;
            relying on the Codex sandbox would block normal turn behavior.
        mcp_servers: Optional pre-merged MCP server map. Phase G3 merges
            user-supplied servers with the bundled ``mala-locking`` server;
            Phase B passes the raw map through unchanged.
    """

    model: str = DEFAULT_CODEX_MODEL
    effort: str | None = None
    approval_policy: Literal["never", "on-request", "on-failure", "untrusted"] = (
        DEFAULT_CODEX_APPROVAL_POLICY
    )
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = (
        DEFAULT_CODEX_SANDBOX
    )
    mcp_servers: tuple[tuple[str, object], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CoderOptions:
    """Per-coder option container."""

    amp: AmpOptions = field(default_factory=AmpOptions)
    codex: CodexOptions = field(default_factory=CodexOptions)


@dataclass(frozen=True)
class MalaConfig:
    """Centralized configuration for mala orchestrator.

    This dataclass consolidates all configuration that was previously scattered
    across environment variable accesses. It can be constructed programmatically
    or loaded from environment variables using from_env().

    Attributes:
        runs_dir: Directory where run metadata files are stored.
            Env: MALA_RUNS_DIR (default: ~/.config/mala/runs)
        lock_dir: Directory for file locks during parallel processing.
            Env: MALA_LOCK_DIR (default: /tmp/mala-locks)
        claude_config_dir: Claude SDK configuration directory.
            Env: CLAUDE_CONFIG_DIR (default: ~/.claude)
        review_enabled: Whether automated code review is enabled.
            Defaults to True.
        review_timeout: Timeout in seconds for review operations.
            Defaults to 600. Configure via mala.yaml code_review settings.
        cerberus_spawn_args: Extra args for `review-gate spawn-code-review`.
            Defaults to empty. Configure via mala.yaml code_review.cerberus.spawn_args.
        cerberus_wait_args: Extra args for `review-gate wait`.
            Defaults to empty. Configure via mala.yaml code_review.cerberus.wait_args.
        cerberus_env: Extra environment variables for review-gate.
            Defaults to empty. Configure via mala.yaml code_review.cerberus.env.
        track_review_issues: Whether to create beads issues for P2/P3 review findings.
            Env: MALA_TRACK_REVIEW_ISSUES (default: True). Deprecated: use
            validation_triggers.<trigger>.code_review.track_review_issues in mala.yaml.
        llm_api_key: API key for LLM calls (epic verification).
            Env: LLM_API_KEY (falls back to ANTHROPIC_API_KEY if not set)
        llm_base_url: Base URL for LLM API requests.
            Env: LLM_BASE_URL (for proxy/routing)
        max_epic_verification_retries: Maximum retries for epic verification loop.
            Defaults to 3. Configure via mala.yaml epic_completion settings.

    Example:
        # Programmatic construction (no env vars needed):
        config = MalaConfig(
            runs_dir=Path("/custom/runs"),
            lock_dir=Path("/custom/locks"),
            claude_config_dir=Path("/custom/claude"),
        )

        # Load from environment:
        config = MalaConfig.from_env()
    """

    # Paths
    runs_dir: Path = field(
        default_factory=lambda: Path.home() / ".config" / "mala" / "runs"
    )
    lock_dir: Path = field(default_factory=lambda: Path("/tmp/mala-locks"))
    claude_config_dir: Path = field(default_factory=lambda: Path.home() / ".claude")

    # Review settings
    review_enabled: bool = field(default=True)
    review_timeout: int = field(default=600)
    cerberus_bin_path: Path | None = None  # Path to cerberus bin/ directory
    cerberus_spawn_args: tuple[str, ...] = field(default_factory=tuple)
    cerberus_wait_args: tuple[str, ...] = field(default_factory=tuple)
    cerberus_env: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    track_review_issues: bool = field(default=True)  # Create beads issues for P2/P3

    # LLM configuration (for epic verification and other direct API calls)
    llm_api_key: str | None = (
        None  # API key for LLM calls (falls back to ANTHROPIC_API_KEY)
    )
    llm_base_url: str | None = None  # Base URL for LLM API (for proxy/routing)

    # Epic verification retry configuration
    max_epic_verification_retries: int = field(default=3)

    # Claude settings sources (which Claude configuration files to use)
    # InitVar accepts None for init, normalized to non-None in __post_init__
    claude_settings_sources_init: InitVar[tuple[str, ...] | None] = (
        DEFAULT_CLAUDE_SETTINGS_SOURCES
    )
    # Stored field is always non-None after __post_init__
    _claude_settings_sources: tuple[str, ...] = field(init=False)

    # Coder selection (which agent backend to drive issue execution)
    # Mirrors claude_settings_sources resolver pattern: env > yaml > default here;
    # CLI override is layered on top in build_resolved_config().
    coder: Literal["claude", "amp", "codex"] = DEFAULT_CODER
    coder_options: CoderOptions = field(default_factory=CoderOptions)

    # Mala-level reasoning effort. Forwarded to ``ClaudeAgentOptions.effort``
    # for the Claude coder and to ``--effort <value>`` for supported Amp modes.
    # ``None`` means "use mala's backend/mode default" during config resolution.
    effort: str | None = None

    def __post_init__(
        self, claude_settings_sources_init: tuple[str, ...] | None
    ) -> None:
        """Normalize mutable fields to immutable types.

        Since the dataclass is frozen, we use object.__setattr__ to set
        derived fields after initialization.
        """
        # Normalize Cerberus overrides for immutability/consistency
        if isinstance(self.cerberus_spawn_args, list):
            object.__setattr__(
                self, "cerberus_spawn_args", tuple(self.cerberus_spawn_args)
            )
        if isinstance(self.cerberus_wait_args, list):
            object.__setattr__(
                self, "cerberus_wait_args", tuple(self.cerberus_wait_args)
            )
        if isinstance(self.cerberus_env, dict):
            object.__setattr__(
                self, "cerberus_env", _normalize_cerberus_env(self.cerberus_env)
            )
        elif isinstance(self.cerberus_env, list):
            object.__setattr__(self, "cerberus_env", tuple(self.cerberus_env))
        # Normalize claude_settings_sources: convert list to tuple, None to default
        if claude_settings_sources_init is None:
            object.__setattr__(
                self, "_claude_settings_sources", DEFAULT_CLAUDE_SETTINGS_SOURCES
            )
        elif isinstance(claude_settings_sources_init, list):
            object.__setattr__(
                self, "_claude_settings_sources", tuple(claude_settings_sources_init)
            )
        else:
            object.__setattr__(
                self, "_claude_settings_sources", claude_settings_sources_init
            )
        if self.effort is None:
            object.__setattr__(
                self,
                "effort",
                default_effort_for(
                    coder=self.coder,
                    mode=self.coder_options.amp.mode,
                ),
            )

    @property
    def claude_settings_sources(self) -> tuple[str, ...]:
        """Claude settings sources (always non-None after construction)."""
        return self._claude_settings_sources

    @classmethod
    def from_env(
        cls,
        *,
        validate: bool = True,
        yaml_claude_settings_sources: tuple[str, ...] | None = None,
        yaml_coder: Literal["claude", "amp", "codex"] | None = None,
        yaml_amp_mode: Literal["smart", "rush", "deep"] | None = None,
        yaml_effort: str | None = None,
        yaml_codex_options: CodexOptions | None = None,
    ) -> MalaConfig:
        """Create MalaConfig by loading from environment variables with validation.

        Reads the following environment variables:
            - MALA_RUNS_DIR: Run metadata directory (optional)
            - MALA_LOCK_DIR: Lock files directory (optional)
            - CLAUDE_CONFIG_DIR: Claude SDK config directory (optional)
            - MALA_TRACK_REVIEW_ISSUES: Create beads issues for P2/P3 findings (deprecated)
            - MALA_CLAUDE_SETTINGS_SOURCES: Comma-separated Claude settings sources (optional)
            - MALA_CODER: Coder selection ("claude", "amp", or "codex") (optional)
            - MALA_AMP_MODE: Amp execution mode ("smart", "rush", "deep") (optional)
            - LLM_API_KEY: API key for LLM calls (optional)
            - LLM_BASE_URL: Base URL for LLM API (optional)

        Deprecated environment variables (configure in mala.yaml instead):
            - MALA_REVIEW_TIMEOUT: Use validation_triggers.<trigger>.code_review.cerberus.timeout
            - MALA_CERBERUS_SPAWN_ARGS: Use validation_triggers.<trigger>.code_review.cerberus.spawn_args
            - MALA_CERBERUS_WAIT_ARGS: Use validation_triggers.<trigger>.code_review.cerberus.wait_args
            - MALA_CERBERUS_ENV: Use validation_triggers.<trigger>.code_review.cerberus.env
            - MALA_MAX_EPIC_VERIFICATION_RETRIES: Use validation_triggers.epic_completion.max_epic_verification_retries
            - MALA_MAX_DIFF_SIZE_KB: Use max_diff_size_kb (root level)

        Args:
            validate: If True (default), run validation and raise ConfigurationError
                on any errors. Set to False to skip validation.
            yaml_claude_settings_sources: Claude settings sources from mala.yaml.
                Used as fallback when env var is not set.
                Precedence: env var > yaml > default.
            yaml_coder: Coder selection from mala.yaml. Used as fallback when
                MALA_CODER is not set. Precedence: env > yaml > default.
            yaml_amp_mode: Amp mode from mala.yaml (top-level `amp_mode`).
                Used as fallback when MALA_AMP_MODE is not set.
                Precedence: env > yaml > default.

        Returns:
            MalaConfig instance with values from environment or defaults.

        Raises:
            ConfigurationError: If validate=True and configuration is invalid.

        Example:
            # Load configuration (validates by default)
            config = MalaConfig.from_env()

            # Skip validation if needed
            config = MalaConfig.from_env(validate=False)

            # With yaml fallback
            config = MalaConfig.from_env(yaml_claude_settings_sources=("local",))
        """
        # Get path values from environment with defaults
        runs_dir = Path(
            os.environ.get(
                "MALA_RUNS_DIR", str(Path.home() / ".config" / "mala" / "runs")
            )
        )
        lock_dir = Path(os.environ.get("MALA_LOCK_DIR", "/tmp/mala-locks"))
        claude_config_dir = Path(
            os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))
        )

        parse_errors: list[str] = []

        # Emit warnings for deprecated env vars (no longer read - use mala.yaml)
        _warn_deprecated_env_vars()

        # Auto-detect cerberus bin path from Claude plugins
        from src.infra.tools.cerberus import find_cerberus_bin_path

        cerberus_bin_path = find_cerberus_bin_path(claude_config_dir)

        # Parse track_review_issues flag (defaults to True)
        track_review_issues_raw = os.environ.get("MALA_TRACK_REVIEW_ISSUES", "").lower()
        track_review_issues = track_review_issues_raw not in ("0", "false", "no", "off")

        # Get LLM configuration (for epic verification and other direct API calls)
        # Falls back to ANTHROPIC_API_KEY if LLM_API_KEY is not set
        llm_api_key = (
            os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or None
        )
        llm_base_url = os.environ.get("LLM_BASE_URL") or None

        # Parse claude_settings_sources
        try:
            claude_settings_sources = parse_claude_settings_sources(
                os.environ.get("MALA_CLAUDE_SETTINGS_SOURCES"),
                source="MALA_CLAUDE_SETTINGS_SOURCES",
            )
        except ValueError as exc:
            parse_errors.append(str(exc))
            claude_settings_sources = None

        # Parse MALA_CODER (env > yaml > default)
        try:
            env_coder = parse_coder(os.environ.get("MALA_CODER"), source="MALA_CODER")
        except ValueError as exc:
            parse_errors.append(str(exc))
            env_coder = None
        resolved_coder: Literal["claude", "amp", "codex"] = (
            env_coder
            if env_coder is not None
            else (yaml_coder if yaml_coder is not None else DEFAULT_CODER)
        )

        # Parse MALA_AMP_MODE (env > yaml > default)
        try:
            env_amp_mode = parse_amp_mode(
                os.environ.get("MALA_AMP_MODE"), source="MALA_AMP_MODE"
            )
        except ValueError as exc:
            parse_errors.append(str(exc))
            env_amp_mode = None
        resolved_amp_mode: Literal["smart", "rush", "deep"] = (
            env_amp_mode
            if env_amp_mode is not None
            else (yaml_amp_mode if yaml_amp_mode is not None else DEFAULT_AMP_MODE)
        )

        # Parse MALA_EFFORT (env > yaml > backend/mode default)
        try:
            env_effort = parse_effort(
                os.environ.get("MALA_EFFORT"), source="MALA_EFFORT"
            )
        except ValueError as exc:
            parse_errors.append(str(exc))
            env_effort = None
        resolved_effort: str | None = (
            env_effort if env_effort is not None else yaml_effort
        )

        # Parse MALA_CODEX_MODEL (env > yaml > default).
        try:
            env_codex_model = parse_codex_model(
                os.environ.get("MALA_CODEX_MODEL"), source="MALA_CODEX_MODEL"
            )
        except ValueError as exc:
            parse_errors.append(str(exc))
            env_codex_model = None
        # Parse MALA_CODEX_EFFORT (env > yaml > default; None == SDK default).
        try:
            env_codex_effort = parse_codex_effort(
                os.environ.get("MALA_CODEX_EFFORT"), source="MALA_CODEX_EFFORT"
            )
        except ValueError as exc:
            parse_errors.append(str(exc))
            env_codex_effort = None
        # Parse MALA_CODEX_APPROVAL_POLICY (env > yaml > default).
        try:
            env_codex_approval_policy = parse_codex_approval_policy(
                os.environ.get("MALA_CODEX_APPROVAL_POLICY"),
                source="MALA_CODEX_APPROVAL_POLICY",
            )
        except ValueError as exc:
            parse_errors.append(str(exc))
            env_codex_approval_policy = None
        # Parse MALA_CODEX_SANDBOX (env > yaml > default).
        try:
            env_codex_sandbox = parse_codex_sandbox(
                os.environ.get("MALA_CODEX_SANDBOX"), source="MALA_CODEX_SANDBOX"
            )
        except ValueError as exc:
            parse_errors.append(str(exc))
            env_codex_sandbox = None
        yaml_codex = (
            yaml_codex_options if yaml_codex_options is not None else CodexOptions()
        )
        resolved_codex_options = CodexOptions(
            model=(
                env_codex_model if env_codex_model is not None else yaml_codex.model
            ),
            effort=(
                env_codex_effort if env_codex_effort is not None else yaml_codex.effort
            ),
            approval_policy=(
                env_codex_approval_policy
                if env_codex_approval_policy is not None
                else yaml_codex.approval_policy
            ),
            sandbox=(
                env_codex_sandbox
                if env_codex_sandbox is not None
                else yaml_codex.sandbox
            ),
            mcp_servers=yaml_codex.mcp_servers,
        )
        if resolved_effort is None:
            resolved_effort = default_effort_for(
                coder=resolved_coder,
                mode=resolved_amp_mode,
            )
        try:
            validate_amp_effort_for_mode(
                coder=resolved_coder,
                mode=resolved_amp_mode,
                effort=resolved_effort,
                source="configuration",
            )
        except ValueError as exc:
            parse_errors.append(str(exc))

        config = cls(
            runs_dir=runs_dir,
            lock_dir=lock_dir,
            claude_config_dir=claude_config_dir,
            review_timeout=600,  # Default; use mala.yaml for custom values
            cerberus_bin_path=cerberus_bin_path,
            cerberus_spawn_args=(),  # Default; use mala.yaml for custom values
            cerberus_wait_args=(),  # Default; use mala.yaml for custom values
            cerberus_env=(),  # Default; use mala.yaml for custom values
            track_review_issues=track_review_issues,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            max_epic_verification_retries=3,  # Default; use mala.yaml for custom values
            # Precedence: env var > yaml > default
            claude_settings_sources_init=(
                claude_settings_sources
                if claude_settings_sources is not None
                else (
                    yaml_claude_settings_sources
                    if yaml_claude_settings_sources is not None
                    else DEFAULT_CLAUDE_SETTINGS_SOURCES
                )
            ),
            coder=resolved_coder,
            coder_options=CoderOptions(
                amp=AmpOptions(mode=resolved_amp_mode),
                codex=resolved_codex_options,
            ),
            effort=resolved_effort,
        )

        if validate:
            errors = config.validate()
            errors.extend(parse_errors)
            if errors:
                raise ConfigurationError(errors)
        elif parse_errors:
            raise ConfigurationError(parse_errors)

        return config

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Checks:
            - Paths are absolute

        Note: Parent directories are not checked since ensure_directories()
        creates them with parents=True. This allows first-run on fresh machines.

        Returns:
            List of error messages. Empty list if configuration is valid.

        Example:
            config = MalaConfig()
            errors = config.validate()  # Returns [] if paths are valid
        """
        errors: list[str] = []

        # Validate paths are absolute (recommended for deterministic behavior)
        if not self.runs_dir.is_absolute():
            errors.append(f"runs_dir should be an absolute path, got: {self.runs_dir}")
        if not self.lock_dir.is_absolute():
            errors.append(f"lock_dir should be an absolute path, got: {self.lock_dir}")
        if not self.claude_config_dir.is_absolute():
            errors.append(
                f"claude_config_dir should be an absolute path, got: {self.claude_config_dir}"
            )
        if self.review_timeout < 0:
            errors.append(f"review_timeout must be >= 0, got: {self.review_timeout}")

        return errors

    def ensure_directories(self) -> None:
        """Create configuration directories if they don't exist.

        Creates runs_dir and lock_dir with parents=True.
        Does not create claude_config_dir (managed by Claude SDK).
        """
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CLIOverrides:
    """CLI override values that modify MalaConfig.

    This represents the raw string values from CLI arguments that will be
    parsed and merged with MalaConfig to produce a ResolvedConfig.

    Attributes:
        cerberus_spawn_args: Raw string of extra args for review-gate spawn.
        cerberus_wait_args: Raw string of extra args for review-gate wait.
        cerberus_env: Raw string of extra env vars (JSON or KEY=VALUE,KEY=VALUE).
        review_timeout: Override for review timeout in seconds.
        max_epic_verification_retries: Override for max epic verification retries.
        disable_review: Whether review is disabled via overrides.
        claude_settings_sources: Raw comma-separated list of settings sources.
        coder: Raw coder selection ("claude" or "amp").
        amp_mode: Raw Amp mode ("smart", "rush", or "deep").
    """

    cerberus_spawn_args: str | None = None
    cerberus_wait_args: str | None = None
    cerberus_env: str | None = None
    review_timeout: int | None = None
    max_epic_verification_retries: int | None = None
    disable_review: bool = False
    claude_settings_sources: str | None = None
    coder: str | None = None
    amp_mode: str | None = None
    effort: str | None = None
    codex_model: str | None = None
    codex_effort: str | None = None
    codex_approval_policy: str | None = None
    codex_sandbox: str | None = None


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully resolved configuration combining MalaConfig and CLI overrides.

    This is the final configuration object used by the orchestrator. It contains
    all fields from MalaConfig plus derived fields computed from the combination
    of base config and CLI overrides.

    Attributes:
        runs_dir: Directory where run metadata files are stored.
        lock_dir: Directory for file locks during parallel processing.
        claude_config_dir: Claude SDK configuration directory.
        review_enabled: Whether automated code review is enabled.
        review_timeout: Timeout in seconds for review operations.
        cerberus_bin_path: Path to cerberus bin/ directory.
        cerberus_spawn_args: Parsed extra args for review-gate spawn.
        cerberus_wait_args: Parsed extra args for review-gate wait.
        cerberus_env: Parsed extra environment variables for review-gate.
        track_review_issues: Whether to create beads issues for P2/P3 (deprecated fallback).
        llm_api_key: API key for LLM calls.
        llm_base_url: Base URL for LLM API.
        max_epic_verification_retries: Maximum retries for epic verification loop.
        claude_settings_sources: Tuple of settings sources for Claude SDK.
        coder: Selected coder backend ("claude" or "amp").
        coder_options: Per-coder options container.
    """

    # Paths
    runs_dir: Path
    lock_dir: Path
    claude_config_dir: Path

    # Review settings
    review_enabled: bool
    review_timeout: int
    cerberus_bin_path: Path | None
    cerberus_spawn_args: tuple[str, ...]
    cerberus_wait_args: tuple[str, ...]
    cerberus_env: tuple[tuple[str, str], ...]
    track_review_issues: bool

    # LLM configuration
    llm_api_key: str | None
    llm_base_url: str | None

    # Epic verification
    max_epic_verification_retries: int

    # Claude SDK settings
    claude_settings_sources: tuple[str, ...]

    # Coder selection
    coder: Literal["claude", "amp", "codex"]
    coder_options: CoderOptions
    effort: str | None


def build_resolved_config(
    base_config: MalaConfig,
    cli_overrides: CLIOverrides | None = None,
) -> ResolvedConfig:
    """Build a ResolvedConfig by merging MalaConfig with CLI overrides.

    Takes a base MalaConfig (typically from environment) and applies CLI
    overrides, parsing string values and computing derived fields.

    Args:
        base_config: Base configuration from MalaConfig.from_env() or constructed.
        cli_overrides: Optional CLI overrides to apply on top of base config.

    Returns:
        A frozen ResolvedConfig with all values resolved and derived fields computed.

    Raises:
        ValueError: If CLI override values cannot be parsed.

    Example:
        config = MalaConfig.from_env()
        overrides = CLIOverrides(
            cerberus_spawn_args="--mode fast",
        )
        resolved = build_resolved_config(config, overrides)
    """
    overrides = cli_overrides or CLIOverrides()

    # Parse CLI override strings, falling back to base config values
    if overrides.cerberus_spawn_args is not None:
        spawn_args = tuple(
            parse_cerberus_args(overrides.cerberus_spawn_args, source="CLI")
        )
    else:
        spawn_args = base_config.cerberus_spawn_args

    if overrides.cerberus_wait_args is not None:
        wait_args = tuple(
            parse_cerberus_args(overrides.cerberus_wait_args, source="CLI")
        )
    else:
        wait_args = base_config.cerberus_wait_args

    if overrides.cerberus_env is not None:
        env = _normalize_cerberus_env(
            parse_cerberus_env(overrides.cerberus_env, source="CLI")
        )
    else:
        env = base_config.cerberus_env

    # Apply timeout override
    review_timeout = (
        overrides.review_timeout
        if overrides.review_timeout is not None
        else base_config.review_timeout
    )

    # Apply max_epic_verification_retries override
    max_epic_verification_retries = (
        overrides.max_epic_verification_retries
        if overrides.max_epic_verification_retries is not None
        else base_config.max_epic_verification_retries
    )

    # Determine if review is enabled after CLI overrides
    review_enabled = base_config.review_enabled and not overrides.disable_review

    # Apply claude_settings_sources override (CLI > Env > YAML > default)
    # base_config already has Env > YAML > default applied
    if overrides.claude_settings_sources is not None:
        claude_settings = parse_claude_settings_sources(
            overrides.claude_settings_sources, source="CLI"
        )
        # parse returns None for empty string, use base_config in that case
        if claude_settings is None:
            claude_settings = base_config.claude_settings_sources
    else:
        claude_settings = base_config.claude_settings_sources

    # Apply coder override (CLI > env > yaml > default)
    # base_config already has env > yaml > default applied
    cli_coder = parse_coder(overrides.coder, source="CLI")
    coder: Literal["claude", "amp", "codex"] = (
        cli_coder if cli_coder is not None else base_config.coder
    )

    # Apply amp mode override (CLI > env > yaml > default)
    cli_amp_mode = parse_amp_mode(overrides.amp_mode, source="CLI")
    amp_mode: Literal["smart", "rush", "deep"] = (
        cli_amp_mode if cli_amp_mode is not None else base_config.coder_options.amp.mode
    )

    # Apply codex options overrides (CLI > env > yaml > default).
    cli_codex_model = parse_codex_model(overrides.codex_model, source="--codex-model")
    cli_codex_effort = parse_codex_effort(
        overrides.codex_effort, source="--codex-effort"
    )
    cli_codex_approval_policy = parse_codex_approval_policy(
        overrides.codex_approval_policy, source="--codex-approval-policy"
    )
    cli_codex_sandbox = parse_codex_sandbox(
        overrides.codex_sandbox, source="--codex-sandbox"
    )
    base_codex = base_config.coder_options.codex
    codex_options = CodexOptions(
        model=cli_codex_model if cli_codex_model is not None else base_codex.model,
        effort=cli_codex_effort if cli_codex_effort is not None else base_codex.effort,
        approval_policy=(
            cli_codex_approval_policy
            if cli_codex_approval_policy is not None
            else base_codex.approval_policy
        ),
        sandbox=(
            cli_codex_sandbox if cli_codex_sandbox is not None else base_codex.sandbox
        ),
        mcp_servers=base_codex.mcp_servers,
    )
    coder_options = CoderOptions(amp=AmpOptions(mode=amp_mode), codex=codex_options)

    # Apply effort override (CLI > env > yaml > backend/mode default).
    # base_config already has env > yaml > default applied; if a CLI override
    # changes the selected coder or Amp mode without setting effort, recompute
    # the default for the final selection.
    cli_effort = parse_effort(overrides.effort, source="CLI")
    effort: str | None = cli_effort if cli_effort is not None else base_config.effort
    base_default_effort = default_effort_for(
        coder=base_config.coder,
        mode=base_config.coder_options.amp.mode,
    )
    if (
        cli_effort is None
        and (overrides.coder is not None or overrides.amp_mode is not None)
        and effort == base_default_effort
    ):
        effort = default_effort_for(coder=coder, mode=amp_mode)
    validate_amp_effort_for_mode(
        coder=coder,
        mode=amp_mode,
        effort=effort,
        source="CLI/configuration",
    )

    # Info-level log when options are set against the inactive coder.
    # Heuristic: explicit CLI override OR resolved value differs from default.
    amp_mode_explicit = overrides.amp_mode is not None or amp_mode != DEFAULT_AMP_MODE
    claude_sources_explicit = (
        overrides.claude_settings_sources is not None
        or claude_settings != DEFAULT_CLAUDE_SETTINGS_SOURCES
    )
    if coder == "claude" and amp_mode_explicit:
        _logger.info(
            "amp mode '%s' ignored — coder is claude",
            amp_mode,
        )
    if coder == "amp" and claude_sources_explicit:
        _logger.info(
            "claude_settings_sources %s ignored — coder is amp",
            claude_settings,
        )

    return ResolvedConfig(
        runs_dir=base_config.runs_dir,
        lock_dir=base_config.lock_dir,
        claude_config_dir=base_config.claude_config_dir,
        review_enabled=review_enabled,
        review_timeout=review_timeout,
        cerberus_bin_path=base_config.cerberus_bin_path,
        cerberus_spawn_args=spawn_args,
        cerberus_wait_args=wait_args,
        cerberus_env=env,
        track_review_issues=base_config.track_review_issues,
        llm_api_key=base_config.llm_api_key,
        llm_base_url=base_config.llm_base_url,
        max_epic_verification_retries=max_epic_verification_retries,
        claude_settings_sources=claude_settings,
        coder=coder,
        coder_options=coder_options,
        effort=effort,
    )
