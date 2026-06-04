"""Canonical Bash wrapper for validation commands.

The wrapper produced by `build_canonical_wrapper` is the single point of
agreement between the prompt path (which renders snippets the agent should
run) and the checker path (which recognizes them line-for-line). Both sides
call this module so the wrapper text and its single-quote escaping cannot
drift.
"""

from __future__ import annotations

import os
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.validation.spec import ValidationCommand


def escape_for_bash_lc(command: str) -> str:
    """Escape `command` so it can be embedded inside `bash -lc '<escaped>'`.

    Replaces each single quote with the standard `'\\''` shell sequence. The
    caller is responsible for adding the surrounding single quotes; this
    matches the existing convention in `src/domain/prompts.py` and lets the
    checker compare wrapper text byte-for-byte.
    """
    return command.replace("'", "'\\''")


def build_canonical_wrapper(
    command: ValidationCommand,
    *,
    issue_id: str,
    validation_log_dir: Path,
) -> str:
    """Build the canonical Bash wrapper text for `command`.

    The wrapper creates the log directory, runs `command.command` under
    `timeout`, redirects stdout and stderr to an absolute log file
    `<validation_log_dir>/<issue_id>.<name>.log`, and emits one
    `MALA_EVIDENCE` summary line that the checker recognizes. The literal
    paths are written directly into the commands (instead of routing through
    shell substitutions such as `$(dirname "$__mala_log")` or `>"$__mala_log"`)
    so safety hooks can classify validation-log writes as scratch output rather
    than unresolvable dynamic write targets. The body sits inside a `(...)`
    subshell so the trailing `exit` terminates only the subshell.

    Strict commands (`allow_fail=False`) propagate the command's status with
    `exit "$__mala_status"`. Advisory commands (`allow_fail=True`) end in
    `exit 0`; the real status is still recorded in the summary line.
    """
    log_dir = os.fspath(validation_log_dir)
    log_path = os.fspath(validation_log_dir / f"{issue_id}.{command.name}.log")
    quoted_log_dir = shlex.quote(log_dir)
    quoted_log_path = shlex.quote(log_path)
    escaped = escape_for_bash_lc(command.command)
    exit_expression = "0" if command.allow_fail else '"$__mala_status"'
    return (
        f"__mala_log={quoted_log_path}\n"
        f"mkdir -p {quoted_log_dir}\n"
        "(\n"
        f"  if timeout {command.timeout} bash -lc '{escaped}'"
        f" >{quoted_log_path} 2>&1; then\n"
        "    __mala_status=0\n"
        "  else\n"
        "    __mala_status=$?\n"
        "  fi\n"
        "  printf 'MALA_EVIDENCE name=%s exit=%s log=%s\\n'"
        f" '{command.name}'"
        f" \"$__mala_status\" {quoted_log_path}\n"
        f"  exit {exit_expression}\n"
        ")"
    )
