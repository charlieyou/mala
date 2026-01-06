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
    def test_removes_only_killed_pgids(self) -> None:
        """Only killed pgids are removed, preserving concurrent additions."""
        command_runner._SIGINT_FORWARD_PGIDS.update({1001, 1002})

        with patch("os.killpg"):
            CommandRunner.kill_active_process_groups()

        assert len(command_runner._SIGINT_FORWARD_PGIDS) == 0

    @unix_only
    def test_handles_empty_pgid_set(self) -> None:
        """No error when pgid set is empty."""
        assert len(command_runner._SIGINT_FORWARD_PGIDS) == 0

        with patch("os.killpg") as mock_killpg:
            CommandRunner.kill_active_process_groups()

        mock_killpg.assert_not_called()

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
