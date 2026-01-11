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
import os
import shlex
import warnings
from dataclasses import InitVar, dataclass, field
from pathlib import Path

from src.core.constants import (
    DEFAULT_CLAUDE_SETTINGS_SOURCES,
    VALID_CLAUDE_SETTINGS_SOURCES,
)


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
            Defaults to 1200. Configure via mala.yaml code_review settings.
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

    # Claude settings sources (which Claude configuration files to use)
    # InitVar accepts None for init, normalized to non-None in __post_init__
    claude_settings_sources_init: InitVar[tuple[str, ...] | None] = (
        DEFAULT_CLAUDE_SETTINGS_SOURCES
    )
    # Stored field is always non-None after __post_init__
    _claude_settings_sources: tuple[str, ...] = field(init=False)

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
    ) -> MalaConfig:
        """Create MalaConfig by loading from environment variables with validation.

        Reads the following environment variables:
            - MALA_RUNS_DIR: Run metadata directory (optional)
            - MALA_LOCK_DIR: Lock files directory (optional)
            - CLAUDE_CONFIG_DIR: Claude SDK config directory (optional)
            - MALA_TRACK_REVIEW_ISSUES: Create beads issues for P2/P3 findings (deprecated)
            - MALA_CLAUDE_SETTINGS_SOURCES: Comma-separated Claude settings sources (optional)
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

        config = cls(
            runs_dir=runs_dir,
            lock_dir=lock_dir,
            claude_config_dir=claude_config_dir,
            review_timeout=1200,  # Default; use mala.yaml for custom values
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
    """

    cerberus_spawn_args: str | None = None
    cerberus_wait_args: str | None = None
    cerberus_env: str | None = None
    review_timeout: int | None = None
    max_epic_verification_retries: int | None = None
    disable_review: bool = False
    claude_settings_sources: str | None = None


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
    )
