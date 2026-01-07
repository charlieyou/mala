"""Claude SDK transport overrides for SIGINT isolation."""

from __future__ import annotations


def ensure_sigint_isolated_cli_transport() -> None:
    """Patch Claude SDK SubprocessCLITransport to isolate SIGINT handling."""
    from claude_agent_sdk._internal.transport import subprocess_cli

    if getattr(subprocess_cli, "_MALA_SIGINT_PATCHED", False):
        return

    import os
    import sys
    from pathlib import Path
    from subprocess import PIPE

    import anyio
    from anyio.streams.text import TextReceiveStream, TextSendStream

    from claude_agent_sdk._errors import CLIConnectionError, CLINotFoundError
    from claude_agent_sdk._version import __version__
    from src.infra.tools.command_runner import CommandRunner

    class MalaSubprocessCLITransport(subprocess_cli.SubprocessCLITransport):
        """Subprocess transport that isolates SIGINT from the parent process."""

        _mala_sigint_pgid: int | None = None

        def _clear_sigint_pgid(self) -> None:
            if self._mala_sigint_pgid is None:
                return
            CommandRunner.unregister_sigint_pgid(self._mala_sigint_pgid)
            self._mala_sigint_pgid = None

        async def connect(self) -> None:
            """Start subprocess in its own session so SIGINT doesn't propagate."""
            if self._process:
                return

            if not os.environ.get("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"):
                await self._check_claude_version()

            cmd = self._build_command()
            try:
                # Merge environment variables: system -> user -> SDK required
                process_env = {
                    **os.environ,
                    **self._options.env,  # User-provided env vars
                    "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
                    "CLAUDE_AGENT_SDK_VERSION": __version__,
                }

                # Enable file checkpointing if requested
                if self._options.enable_file_checkpointing:
                    process_env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] = "true"

                if self._cwd:
                    process_env["PWD"] = self._cwd

                # Pipe stderr if we have a callback OR debug mode is enabled
                should_pipe_stderr = (
                    self._options.stderr is not None
                    or "debug-to-stderr" in self._options.extra_args
                )

                # For backward compat: use debug_stderr file object if no callback and debug is on
                stderr_dest = PIPE if should_pipe_stderr else None

                start_new_session = sys.platform != "win32"
                self._process = await anyio.open_process(
                    cmd,
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=stderr_dest,
                    cwd=self._cwd,
                    env=process_env,
                    user=self._options.user,
                    start_new_session=start_new_session,
                )

                if start_new_session and self._process is not None:
                    self._mala_sigint_pgid = self._process.pid
                    CommandRunner.register_sigint_pgid(self._mala_sigint_pgid)

                if self._process.stdout:
                    self._stdout_stream = TextReceiveStream(self._process.stdout)

                # Setup stderr stream if piped
                if should_pipe_stderr and self._process.stderr:
                    self._stderr_stream = TextReceiveStream(self._process.stderr)
                    # Start async task to read stderr
                    self._stderr_task_group = anyio.create_task_group()
                    await self._stderr_task_group.__aenter__()
                    self._stderr_task_group.start_soon(self._handle_stderr)

                # Setup stdin for streaming mode
                if self._is_streaming and self._process.stdin:
                    self._stdin_stream = TextSendStream(self._process.stdin)
                elif not self._is_streaming and self._process.stdin:
                    # String mode: close stdin immediately
                    await self._process.stdin.aclose()

                self._ready = True

            except FileNotFoundError as e:
                self._clear_sigint_pgid()
                # Check if the error comes from the working directory or the CLI
                if self._cwd and not Path(self._cwd).exists():
                    error = CLIConnectionError(
                        f"Working directory does not exist: {self._cwd}"
                    )
                    self._exit_error = error
                    raise error from e
                error = CLINotFoundError(
                    f"Claude Code not found at: {self._cli_path}"
                )
                self._exit_error = error
                raise error from e
            except Exception as e:
                self._clear_sigint_pgid()
                error = CLIConnectionError(f"Failed to start Claude Code: {e}")
                self._exit_error = error
                raise error from e

        async def close(self) -> None:
            try:
                await super().close()
            finally:
                self._clear_sigint_pgid()

    subprocess_cli.SubprocessCLITransport = MalaSubprocessCLITransport
    subprocess_cli._MALA_SIGINT_PATCHED = True
