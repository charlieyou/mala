"""Contract tests for CommandRunnerPort implementations.

Ensures FakeCommandRunner implements all methods of CommandRunnerPort protocol
and exhibits correct behavioral parity with the real CommandRunner.
"""

import pytest

from src.core.protocols.infra import CommandRunnerPort
from tests.contracts import get_protocol_members
from tests.fakes.command_runner import FakeCommandRunner


@pytest.mark.unit
def test_fake_command_runner_implements_all_protocol_methods() -> None:
    """FakeCommandRunner must implement all public methods of CommandRunnerPort."""
    protocol_methods = get_protocol_members(CommandRunnerPort)
    fake_methods = {name for name in dir(FakeCommandRunner) if not name.startswith("_")}

    missing = protocol_methods - fake_methods
    assert not missing, f"FakeCommandRunner missing protocol methods: {sorted(missing)}"


@pytest.mark.unit
def test_fake_command_runner_protocol_compliance() -> None:
    """FakeCommandRunner passes runtime isinstance check for CommandRunnerPort."""
    runner = FakeCommandRunner()
    assert isinstance(runner, CommandRunnerPort)


class TestFakeCommandRunnerBehavior:
    """Behavioral tests for FakeCommandRunner."""

    @pytest.mark.unit
    def test_run_returns_registered_response(self) -> None:
        """run() returns pre-configured response for registered command."""
        from src.infra.tools.command_runner import CommandResult

        runner = FakeCommandRunner(
            responses={
                ("echo", "hello"): CommandResult(
                    command=["echo", "hello"],
                    returncode=0,
                    stdout="hello\n",
                    stderr="",
                )
            }
        )
        result = runner.run(["echo", "hello"])
        assert result.returncode == 0
        assert result.stdout == "hello\n"

    @pytest.mark.unit
    def test_run_tracks_calls(self) -> None:
        """run() records call in calls list."""
        runner = FakeCommandRunner(allow_unregistered=True)
        runner.run(["git", "status"], timeout=30.0)
        assert len(runner.calls) == 1
        cmd_tuple, kwargs = runner.calls[0]
        assert cmd_tuple == ("git", "status")
        assert kwargs["timeout"] == 30.0

    @pytest.mark.unit
    async def test_run_async_returns_registered_response(self) -> None:
        """run_async() returns pre-configured response for registered command."""
        from src.infra.tools.command_runner import CommandResult

        runner = FakeCommandRunner(
            responses={
                ("ls", "-la"): CommandResult(
                    command=["ls", "-la"],
                    returncode=0,
                    stdout="file1\nfile2\n",
                    stderr="",
                )
            }
        )
        result = await runner.run_async(["ls", "-la"])
        assert result.returncode == 0
        assert "file1" in result.stdout

    @pytest.mark.unit
    async def test_run_async_tracks_calls(self) -> None:
        """run_async() records call in calls list."""
        runner = FakeCommandRunner(allow_unregistered=True)
        await runner.run_async(["pytest", "-v"], cwd=None)
        assert len(runner.calls) == 1
        cmd_tuple, _ = runner.calls[0]
        assert cmd_tuple == ("pytest", "-v")

    @pytest.mark.unit
    def test_unregistered_command_raises_by_default(self) -> None:
        """Unregistered commands raise UnregisteredCommandError."""
        from tests.fakes.command_runner import UnregisteredCommandError

        runner = FakeCommandRunner()
        with pytest.raises(UnregisteredCommandError) as exc_info:
            runner.run(["unknown", "cmd"])
        assert exc_info.value.cmd == ("unknown", "cmd")

    @pytest.mark.unit
    def test_allow_unregistered_returns_success(self) -> None:
        """With allow_unregistered=True, unregistered commands return success."""
        runner = FakeCommandRunner(allow_unregistered=True)
        result = runner.run(["any", "command"])
        assert result.returncode == 0

    @pytest.mark.unit
    def test_has_call_with_prefix(self) -> None:
        """has_call_with_prefix() finds calls starting with prefix."""
        runner = FakeCommandRunner(allow_unregistered=True)
        runner.run(["git", "commit", "-m", "msg"])
        assert runner.has_call_with_prefix(["git"])
        assert runner.has_call_with_prefix(["git", "commit"])
        assert not runner.has_call_with_prefix(["git", "push"])

    @pytest.mark.unit
    def test_has_call_containing(self) -> None:
        """has_call_containing() finds calls containing substring."""
        runner = FakeCommandRunner(allow_unregistered=True)
        runner.run("echo 'hello world'", shell=True)
        assert runner.has_call_containing("hello")
        assert not runner.has_call_containing("goodbye")
