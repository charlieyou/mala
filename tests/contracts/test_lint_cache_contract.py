"""Contract tests for LintCacheProtocol implementations.

Ensures FakeLintCache implements all methods of LintCacheProtocol.
"""

import pytest

from src.pipeline.message_stream_processor import LintCacheProtocol
from tests.contracts import get_protocol_members
from tests.fakes.lint_cache import FakeLintCache


@pytest.mark.unit
def test_fake_lint_cache_implements_all_protocol_methods() -> None:
    """FakeLintCache must implement all public methods of LintCacheProtocol."""
    protocol_methods = get_protocol_members(LintCacheProtocol)
    fake_methods = {name for name in dir(FakeLintCache) if not name.startswith("_")}

    missing = protocol_methods - fake_methods
    assert not missing, f"FakeLintCache missing protocol methods: {sorted(missing)}"


@pytest.mark.unit
def test_fake_lint_cache_has_required_methods() -> None:
    """FakeLintCache has all methods with correct signatures."""
    cache = FakeLintCache()
    # Verify method signatures by calling them
    result = cache.detect_lint_command("test")
    assert result is None or isinstance(result, str)
    cache.mark_success("type", "cmd")  # Should not raise


class TestFakeLintCacheBehavior:
    """Behavioral tests for FakeLintCache."""

    @pytest.mark.unit
    def test_detect_lint_command_returns_none_when_not_configured(self) -> None:
        """detect_lint_command() returns None for unknown commands."""
        cache = FakeLintCache()
        result = cache.detect_lint_command("unknown command")
        assert result is None

    @pytest.mark.unit
    def test_configure_detect_enables_detection(self) -> None:
        """configure_detect() makes detect_lint_command return configured type."""
        cache = FakeLintCache()
        cache.configure_detect("ruff check .", "ruff")
        result = cache.detect_lint_command("ruff check .")
        assert result == "ruff"

    @pytest.mark.unit
    def test_detect_lint_command_records_calls(self) -> None:
        """detect_lint_command() records all detection attempts."""
        cache = FakeLintCache()
        cache.detect_lint_command("ruff check")
        cache.detect_lint_command("pytest")
        assert len(cache.detected_commands) == 2
        assert cache.detected_commands[0] == ("detect", "ruff check")
        assert cache.detected_commands[1] == ("detect", "pytest")

    @pytest.mark.unit
    def test_mark_success_records_calls(self) -> None:
        """mark_success() records successful lint commands."""
        cache = FakeLintCache()
        cache.mark_success("ruff", "ruff check .")
        cache.mark_success("mypy", "mypy src/")
        assert len(cache.marked_successes) == 2
        assert cache.marked_successes[0] == ("ruff", "ruff check .")
        assert cache.marked_successes[1] == ("mypy", "mypy src/")

    @pytest.mark.unit
    def test_lint_commands_dict_for_direct_configuration(self) -> None:
        """lint_commands dict can be used for direct configuration."""
        cache = FakeLintCache()
        cache.lint_commands["pytest tests/"] = "pytest"
        result = cache.detect_lint_command("pytest tests/")
        assert result == "pytest"
