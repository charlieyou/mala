"""Configuration dataclass for mala.

Provides MalaConfig for centralized configuration management. This allows
programmatic users to construct configuration without relying on environment
variables, while CLI users can continue using env vars via from_env().

Environment Variables:
    MALA_RUNS_DIR: Directory for run metadata files (default: ~/.config/mala/runs)
    MALA_LOCK_DIR: Directory for file locks (default: /tmp/mala-locks)
    CLAUDE_CONFIG_DIR: Claude SDK config directory (default: ~/.claude)
    BRAINTRUST_API_KEY: Braintrust API key (required when braintrust_enabled=True)
    MORPH_API_KEY: Morph API key (required when morph_enabled=True)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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
        braintrust_api_key: Braintrust API key for tracing.
            Env: BRAINTRUST_API_KEY (required when braintrust_enabled=True)
        morph_api_key: Morph API key for MCP features.
            Env: MORPH_API_KEY (required when morph_enabled=True)
        braintrust_enabled: Whether Braintrust tracing is enabled.
            Derived from braintrust_api_key presence.
        morph_enabled: Whether Morph MCP features are enabled.
            Derived from morph_api_key presence.

    Example:
        # Programmatic construction (no env vars needed):
        config = MalaConfig(
            runs_dir=Path("/custom/runs"),
            lock_dir=Path("/custom/locks"),
            claude_config_dir=Path("/custom/claude"),
            morph_api_key="my-api-key",
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

    # API keys (optional)
    braintrust_api_key: str | None = None
    morph_api_key: str | None = None

    # Feature flags (derived from API key presence)
    braintrust_enabled: bool = field(default=False)
    morph_enabled: bool = field(default=False)

    def __post_init__(self) -> None:
        """Derive feature flags from API key presence.

        Since the dataclass is frozen, we use object.__setattr__ to set
        derived fields after initialization.
        """
        # Derive braintrust_enabled from api key presence if not explicitly set
        if not self.braintrust_enabled and self.braintrust_api_key:
            object.__setattr__(self, "braintrust_enabled", True)
        # Derive morph_enabled from api key presence if not explicitly set
        if not self.morph_enabled and self.morph_api_key:
            object.__setattr__(self, "morph_enabled", True)

    @classmethod
    def from_env(cls, *, validate: bool = True) -> MalaConfig:
        """Create MalaConfig by loading from environment variables with validation.

        Reads the following environment variables:
            - MALA_RUNS_DIR: Run metadata directory (optional)
            - MALA_LOCK_DIR: Lock files directory (optional)
            - CLAUDE_CONFIG_DIR: Claude SDK config directory (optional)
            - BRAINTRUST_API_KEY: Braintrust API key (optional)
            - MORPH_API_KEY: Morph API key (optional)

        Args:
            validate: If True (default), run validation and raise ConfigurationError
                on any errors. Set to False to skip validation.

        Returns:
            MalaConfig instance with values from environment or defaults.

        Raises:
            ConfigurationError: If validate=True and configuration is invalid.

        Example:
            # Set environment variables first
            os.environ["MORPH_API_KEY"] = "my-key"

            # Load configuration (validates by default)
            config = MalaConfig.from_env()
            assert config.morph_enabled is True

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

        # Get optional API keys (treat empty strings as None)
        braintrust_api_key = os.environ.get("BRAINTRUST_API_KEY") or None
        morph_api_key = os.environ.get("MORPH_API_KEY") or None

        config = cls(
            runs_dir=runs_dir,
            lock_dir=lock_dir,
            claude_config_dir=claude_config_dir,
            braintrust_api_key=braintrust_api_key,
            morph_api_key=morph_api_key,
        )

        if validate:
            errors = config.validate()
            if errors:
                raise ConfigurationError(errors)

        return config

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Checks:
            - Feature flags have required API keys
            - Path directories can be created (or already exist)
            - Paths are absolute

        Returns:
            List of error messages. Empty list if configuration is valid.

        Example:
            config = MalaConfig(braintrust_enabled=True)  # Missing API key
            errors = config.validate()
            # errors = ["braintrust_enabled=True requires BRAINTRUST_API_KEY"]
        """
        errors: list[str] = []

        # Check required API keys for enabled features
        if self.braintrust_enabled and not self.braintrust_api_key:
            errors.append(
                "braintrust_enabled=True requires BRAINTRUST_API_KEY to be set"
            )
        if self.morph_enabled and not self.morph_api_key:
            errors.append("morph_enabled=True requires MORPH_API_KEY to be set")

        # Validate paths are absolute (recommended for deterministic behavior)
        if not self.runs_dir.is_absolute():
            errors.append(f"runs_dir should be an absolute path, got: {self.runs_dir}")
        if not self.lock_dir.is_absolute():
            errors.append(f"lock_dir should be an absolute path, got: {self.lock_dir}")
        if not self.claude_config_dir.is_absolute():
            errors.append(
                f"claude_config_dir should be an absolute path, got: {self.claude_config_dir}"
            )

        # Check parent directories exist (paths will be created at runtime)
        for name, path in [
            ("runs_dir", self.runs_dir),
            ("lock_dir", self.lock_dir),
        ]:
            parent = path.parent
            if not parent.exists():
                errors.append(f"{name} parent directory does not exist: {parent}")

        return errors

    def ensure_directories(self) -> None:
        """Create configuration directories if they don't exist.

        Creates runs_dir and lock_dir with parents=True.
        Does not create claude_config_dir (managed by Claude SDK).
        """
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
