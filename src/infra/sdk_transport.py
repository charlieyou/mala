"""SDK transport overrides for SIGINT isolation."""

from __future__ import annotations

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


def ensure_sigint_isolated_cli_transport() -> None:
    """Patch Claude SDK SubprocessCLITransport to isolate SIGINT handling."""
    from claude_agent_sdk._internal.transport import subprocess_cli

    if getattr(subprocess_cli, "_MALA_SIGINT_PATCHED", False):
        return

    import os
    import sys
    from pathlib import Path
    from subprocess import DEVNULL, PIPE

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

            if self._cli_path is None:
                self._cli_path = self._find_cli()

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
                # Use DEVNULL when not piping to suppress Claude CLI warnings (e.g., ENXIO socket watcher errors)
                stderr_dest = PIPE if should_pipe_stderr else DEVNULL

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
                    # Use MALA_SDK_FLOW env var if set, otherwise default to "implementer"
                    flow = process_env.get("MALA_SDK_FLOW", "implementer")
                    logger.info(
                        "sdk_subprocess_spawned pid=%s pgid=%s flow=%s",
                        self._process.pid,
                        self._mala_sigint_pgid,
                        flow,
                    )

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
                error = CLINotFoundError(f"Claude Code not found at: {self._cli_path}")
                self._exit_error = error
                raise error from e
            except Exception as e:
                self._clear_sigint_pgid()
                error = CLIConnectionError(f"Failed to start Claude Code: {e}")
                self._exit_error = error
                raise error from e

        async def close(self) -> None:
            try:
                # Kill entire process group before closing, not just the leader.
                # Without this, child processes (e.g., pytest spawned by Bash tool)
                # become orphans and run forever when the Claude CLI exits.
                if self._mala_sigint_pgid is not None and hasattr(os, "killpg"):
                    import signal

                    from src.infra.tools.command_runner import (
                        DEFAULT_KILL_GRACE_SECONDS,
                    )

                    try:
                        os.killpg(self._mala_sigint_pgid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                    else:
                        # Grace period, then force kill if still running
                        await anyio.sleep(DEFAULT_KILL_GRACE_SECONDS)
                        try:
                            os.killpg(self._mala_sigint_pgid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
                await super().close()
            finally:
                self._clear_sigint_pgid()

    subprocess_cli.SubprocessCLITransport = MalaSubprocessCLITransport  # type: ignore[invalid-assignment]  # ty:ignore[invalid-assignment]
    subprocess_cli._MALA_SIGINT_PATCHED = True  # type: ignore[unresolved-attribute]  # ty:ignore[unresolved-attribute]


def ensure_codex_sigint_isolated_app_server() -> None:
    """Patch Codex app-server spawning to isolate terminal SIGINT.

    Codex's Python SDK starts ``codex app-server`` with ``subprocess.Popen``.
    Without ``start_new_session=True`` that child remains in Mala's terminal
    foreground process group, so the first Ctrl-C reaches Codex directly even
    though Mala only entered drain mode. Running the app-server in its own
    process group preserves Mala's escalation contract: first Ctrl-C drains,
    second Ctrl-C is forwarded intentionally by :class:`CommandRunner`.
    """
    import sys

    try:
        import codex_app_server.client as codex_client
    except ModuleNotFoundError:
        # Unit tests often install a minimal fake ``codex_app_server`` module
        # that exposes only the API objects CodexClient consumes. There is no
        # subprocess to isolate in that fake shape.
        return
    from src.infra.tools.command_runner import CommandRunner

    codex_client_any = cast("Any", codex_client)
    if getattr(codex_client_any, "_MALA_SIGINT_PATCHED", False):
        return

    original_subprocess = codex_client_any.subprocess
    original_popen = original_subprocess.Popen

    class _SubprocessProxy:
        """Module-like proxy that overrides only ``Popen`` for Codex SDK code."""

        def __init__(self, base: object, popen: object) -> None:
            self._base = base
            self.Popen = popen

        def __getattr__(self, name: str) -> object:
            return getattr(self._base, name)

    def mala_popen(*args: object, **kwargs: object) -> object:
        pgid_from_process_group: int | None = None
        if sys.platform != "win32":
            # ``Popen`` rejects some process-group controls in combination. The
            # SDK does not set these today, but keep the wrapper conservative so
            # it does not break a future SDK that starts managing groups itself.
            has_preexec = kwargs.get("preexec_fn") is not None
            process_group = kwargs.get("process_group")
            if process_group is None and not has_preexec:
                kwargs["start_new_session"] = True
            elif isinstance(process_group, int):
                pgid_from_process_group = process_group

        proc = original_popen(*args, **kwargs)
        pgid: int | None = None
        pid = getattr(proc, "pid", None)
        if sys.platform != "win32":
            has_preexec = kwargs.get("preexec_fn") is not None
            if (
                kwargs.get("start_new_session") is True
                and not has_preexec
                and isinstance(pid, int)
            ):
                pgid = pid
            elif pgid_from_process_group == 0 and isinstance(pid, int):
                pgid = pid
            elif pgid_from_process_group is not None and pgid_from_process_group > 0:
                pgid = pgid_from_process_group
        if pgid is not None:
            CommandRunner.register_sigint_pgid(pgid)
            setattr(proc, "_mala_sigint_pgid", pgid)
            logger.info("codex_app_server_spawned pid=%s pgid=%s", pid, pgid)
        return proc

    codex_client_any.subprocess = _SubprocessProxy(original_subprocess, mala_popen)
    codex_client_any._MALA_SIGINT_PATCHED = True
