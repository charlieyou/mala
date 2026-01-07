"""Unit tests for CommandRunner.kill_active_process_groups()."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from src.infra.tools import command_runner
from src.infra.tools.command_runner import CommandRunner

# Skip Unix-specific tests on Windows
unix_only = pytest.mark.skipif(sys.platform == "win32", reason="Unix-only test")


class TestKillActiveProcessGroups:
    """Tests for CommandRunner.kill_active_process_groups()."""

    def setup_method(self) -> None:
        """Clear the pgid set before each test."""
        command_runner._SIGINT_FORWARD_PGIDS.clear()

    def teardown_method(self) -> None:
        """Clean up pgid set after each test."""
        command_runner._SIGINT_FORWARD_PGIDS.clear()

    @unix_only
    def test_sends_sigkill_to_tracked_pgids(self) -> None:
        """SIGKILL is sent to all tracked process groups."""
        import signal

        command_runner._SIGINT_FORWARD_PGIDS.update({1001, 1002, 1003})

        with patch("os.killpg") as mock_killpg:
            CommandRunner.kill_active_process_groups()

        assert mock_killpg.call_count == 3
        called_pgids = {call.args[0] for call in mock_killpg.call_args_list}
        assert called_pgids == {1001, 1002, 1003}
        for call in mock_killpg.call_args_list:
            assert call.args[1] == signal.SIGKILL

    @unix_only
    def test_preserves_concurrent_additions(self) -> None:
        """Pgids added between copy() and difference_update() are preserved.

        This test would fail if the implementation used clear() instead of
        difference_update(), catching regressions to the race-prone pattern.
        """

        class SetThatAddsDuringCopy(set[int]):
            """Set subclass that simulates concurrent add during copy()."""

            def copy(self) -> set[int]:
                snapshot = set(self)  # Take snapshot first
                self.add(9999)  # Simulate concurrent registration
                return snapshot

        # Replace the global set with our instrumented version
        original_set = command_runner._SIGINT_FORWARD_PGIDS
        instrumented_set: set[int] = SetThatAddsDuringCopy({1001, 1002})
        command_runner._SIGINT_FORWARD_PGIDS = instrumented_set

        try:
            with patch("os.killpg"):
                CommandRunner.kill_active_process_groups()

            # 9999 was added after copy() but before difference_update()
            # With difference_update(): 9999 is preserved (only 1001, 1002 removed)
            # With clear(): 9999 would be cleared (test would fail)
            assert 9999 in command_runner._SIGINT_FORWARD_PGIDS
            # Original pgids should be removed
            assert 1001 not in command_runner._SIGINT_FORWARD_PGIDS
            assert 1002 not in command_runner._SIGINT_FORWARD_PGIDS
        finally:
            command_runner._SIGINT_FORWARD_PGIDS = original_set

    @unix_only
    def test_handles_empty_pgid_set(self) -> None:
        """No error when pgid set is empty."""
        assert len(command_runner._SIGINT_FORWARD_PGIDS) == 0

        with patch("os.killpg") as mock_killpg:
            CommandRunner.kill_active_process_groups()

        mock_killpg.assert_not_called()

    def test_register_unregister_sigint_pgid(self) -> None:
        """Register/unregister should update the pgid tracking set."""
        command_runner._SIGINT_FORWARD_PGIDS.clear()

        CommandRunner.register_sigint_pgid(4242)
        assert command_runner._SIGINT_FORWARD_PGIDS == {4242}

        CommandRunner.unregister_sigint_pgid(4242)
        assert command_runner._SIGINT_FORWARD_PGIDS == set()

    def test_noop_on_windows(self) -> None:
        """No-op on Windows platform."""
        command_runner._SIGINT_FORWARD_PGIDS.update({1001, 1002})

        with patch("src.infra.tools.command_runner.sys.platform", "win32"):
            CommandRunner.kill_active_process_groups()

        # pgids should NOT be cleared on Windows (early return)
        assert command_runner._SIGINT_FORWARD_PGIDS == {1001, 1002}

    @unix_only
    def test_handles_process_lookup_error(self) -> None:
        """ProcessLookupError is silently handled."""
        command_runner._SIGINT_FORWARD_PGIDS.update({1001, 1002})

        def raise_process_lookup_error(pgid: int, sig: int) -> None:
            if pgid == 1001:
                raise ProcessLookupError("No such process")

        with patch("os.killpg", side_effect=raise_process_lookup_error):
            CommandRunner.kill_active_process_groups()

        assert len(command_runner._SIGINT_FORWARD_PGIDS) == 0

    @unix_only
    def test_handles_permission_error(self) -> None:
        """PermissionError is silently handled."""
        command_runner._SIGINT_FORWARD_PGIDS.update({1001, 1002})

        def raise_permission_error(pgid: int, sig: int) -> None:
            if pgid == 1002:
                raise PermissionError("Operation not permitted")

        with patch("os.killpg", side_effect=raise_permission_error):
            CommandRunner.kill_active_process_groups()

        assert len(command_runner._SIGINT_FORWARD_PGIDS) == 0

    @unix_only
    def test_safe_to_call_multiple_times(self) -> None:
        """Calling multiple times is safe (idempotent after first call)."""
        command_runner._SIGINT_FORWARD_PGIDS.update({1001})

        with patch("os.killpg") as mock_killpg:
            CommandRunner.kill_active_process_groups()
            CommandRunner.kill_active_process_groups()
            CommandRunner.kill_active_process_groups()

        # Only called once (first call), subsequent calls find empty set
        assert mock_killpg.call_count == 1
