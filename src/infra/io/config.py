"""Configuration dataclass for mala.

Provides MalaConfig for centralized configuration management. This allows
programmatic users to construct configuration without relying on environment
variables, while CLI users can continue using env vars via from_env().

Environment Variables:
    MALA_RUNS_DIR: Directory for run metadata files (default: ~/.config/mala/runs)
    MALA_LOCK_DIR: Directory for file locks (default: /tmp/mala-locks)
    CLAUDE_CONFIG_DIR: Claude SDK config directory (default: ~/.claude)
    MALA_REVIEW_TIMEOUT: Timeout in seconds for review-gate wait
    MALA_CERBERUS_SPAWN_ARGS: Extra args for `review-gate spawn-code-review`
    MALA_CERBERUS_WAIT_ARGS: Extra args for `review-gate wait`
    MALA_CERBERUS_ENV: Extra env for review-gate (JSON dict or comma KEY=VALUE list)
    MALA_MAX_DIFF_SIZE_KB: Max diff size for epic verification (KB)
    MALA_MAX_EPIC_VERIFICATION_RETRIES: Max retries for epic verification loop
    LLM_API_KEY: API key for LLM calls (fallback to ANTHROPIC_API_KEY)
    LLM_BASE_URL: Base URL for LLM API
"""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path


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


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message)


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
            Defaults to 300.
        cerberus_spawn_args: Extra args for `review-gate spawn-code-review`.
            Defaults to empty (no extra args).
        cerberus_wait_args: Extra args for `review-gate wait`.
            Defaults to empty (no extra args).
        cerberus_env: Extra environment variables for review-gate.
            Defaults to empty (no extra env).
        track_review_issues: Whether to create beads issues for P2/P3 review findings.
            Env: MALA_TRACK_REVIEW_ISSUES (default: True)
        llm_api_key: API key for LLM calls (epic verification).
            Env: LLM_API_KEY (falls back to ANTHROPIC_API_KEY if not set)
        llm_base_url: Base URL for LLM API requests.
            Env: LLM_BASE_URL (for proxy/routing)
        max_epic_verification_retries: Maximum retries for epic verification loop.
            Env: MALA_MAX_EPIC_VERIFICATION_RETRIES (default: 3)

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
    review_timeout: int = field(default=1200)
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

    def __post_init__(self) -> None:
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

    @classmethod
    def from_env(cls, *, validate: bool = True) -> MalaConfig:
        """Create MalaConfig by loading from environment variables with validation.

        Reads the following environment variables:
            - MALA_RUNS_DIR: Run metadata directory (optional)
            - MALA_LOCK_DIR: Lock files directory (optional)
            - CLAUDE_CONFIG_DIR: Claude SDK config directory (optional)
            - MALA_REVIEW_TIMEOUT: Review timeout in seconds (optional)
            - MALA_TRACK_REVIEW_ISSUES: Create beads issues for P2/P3 findings (optional)
            - MALA_CERBERUS_SPAWN_ARGS: Extra args for review-gate spawn (optional)
            - MALA_CERBERUS_WAIT_ARGS: Extra args for review-gate wait (optional)
            - MALA_CERBERUS_ENV: Extra env for review-gate (optional)
            - MALA_MAX_DIFF_SIZE_KB: Max diff size for epic verification (optional)
            - MALA_MAX_EPIC_VERIFICATION_RETRIES: Max epic verification retries (optional)
            - LLM_API_KEY: API key for LLM calls (optional)
            - LLM_BASE_URL: Base URL for LLM API (optional)

        Args:
            validate: If True (default), run validation and raise ConfigurationError
                on any errors. Set to False to skip validation.

        Returns:
            MalaConfig instance with values from environment or defaults.

        Raises:
            ConfigurationError: If validate=True and configuration is invalid.

        Example:
            # Load configuration (validates by default)
            config = MalaConfig.from_env()

            # Skip validation if needed
            config = MalaConfig.from_env(validate=False)
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

        review_timeout = None
        review_timeout_raw = os.environ.get("MALA_REVIEW_TIMEOUT")
        parse_errors: list[str] = []
        if review_timeout_raw:
            try:
                review_timeout = int(review_timeout_raw)
            except ValueError:
                parse_errors.append(
                    f"MALA_REVIEW_TIMEOUT: invalid integer '{review_timeout_raw}'"
                )
                review_timeout = None

        # Parse Cerberus override settings
        try:
            cerberus_spawn_args = parse_cerberus_args(
                os.environ.get("MALA_CERBERUS_SPAWN_ARGS"),
                source="MALA_CERBERUS_SPAWN_ARGS",
            )
        except ValueError as exc:
            parse_errors.append(str(exc))
            cerberus_spawn_args = []

        try:
            cerberus_wait_args = parse_cerberus_args(
                os.environ.get("MALA_CERBERUS_WAIT_ARGS"),
                source="MALA_CERBERUS_WAIT_ARGS",
            )
        except ValueError as exc:
            parse_errors.append(str(exc))
            cerberus_wait_args = []

        try:
            cerberus_env = parse_cerberus_env(
                os.environ.get("MALA_CERBERUS_ENV"),
                source="MALA_CERBERUS_ENV",
            )
        except ValueError as exc:
            parse_errors.append(str(exc))
            cerberus_env = {}

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

        # Parse max_epic_verification_retries
        max_epic_verification_retries = _safe_int(
            os.environ.get("MALA_MAX_EPIC_VERIFICATION_RETRIES"), 3
        )

        config = cls(
            runs_dir=runs_dir,
            lock_dir=lock_dir,
            claude_config_dir=claude_config_dir,
            review_timeout=review_timeout if review_timeout is not None else 1200,
            cerberus_bin_path=cerberus_bin_path,
            cerberus_spawn_args=tuple(cerberus_spawn_args),
            cerberus_wait_args=tuple(cerberus_wait_args),
            cerberus_env=_normalize_cerberus_env(cerberus_env),
            track_review_issues=track_review_issues,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            max_epic_verification_retries=max_epic_verification_retries,
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
        disable_review: Whether 'review' is in --disable.
    """

    cerberus_spawn_args: str | None = None
    cerberus_wait_args: str | None = None
    cerberus_env: str | None = None
    review_timeout: int | None = None
    max_epic_verification_retries: int | None = None
    disable_review: bool = False


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
        track_review_issues: Whether to create beads issues for P2/P3.
        llm_api_key: API key for LLM calls.
        llm_base_url: Base URL for LLM API.
        max_epic_verification_retries: Maximum retries for epic verification loop.
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
    )
