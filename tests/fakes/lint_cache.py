"""Fake lint cache for testing."""

from dataclasses import dataclass, field


@dataclass
class FakeLintCache:
    """In-memory lint cache implementing LintCacheProtocol.

    Tracks all operations for test assertions.

    Usage:
        cache = FakeLintCache()
        cache.configure_detect("ruff check", "ruff")
        result = cache.detect_lint_command("ruff check")  # returns "ruff"
        cache.mark_success("ruff", "ruff check")
        assert cache.marked_successes == [("ruff", "ruff check")]
    """

    lint_commands: dict[str, str] = field(default_factory=dict)
    detected_commands: list[tuple[str, str]] = field(default_factory=list)
    marked_successes: list[tuple[str, str]] = field(default_factory=list)

    def configure_detect(self, command: str, lint_type: str) -> None:
        """Configure detect_lint_command to return lint_type for command."""
        self.lint_commands[command] = lint_type

    def detect_lint_command(self, command: str) -> str | None:
        """Return lint type if command matches a configured pattern.

        Records all detection calls in detected_commands for assertions.
        """
        self.detected_commands.append(("detect", command))
        return self.lint_commands.get(command)

    def mark_success(self, lint_type: str, command: str) -> None:
        """Record a successful lint command."""
        self.marked_successes.append((lint_type, command))
