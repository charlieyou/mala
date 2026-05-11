"""Unit tests for ``src.infra.hooks.codex_pre_tool_use``.

Covers the AC #19 heuristic-table matrix (every row denied when target is
unlocked or wrong-owner, allowed when locked by current agent), plus the
AC #9 partial criteria (apply_patch lock-mismatch, dangerous-cmd,
disallowed-tool, env-missing) and the residual obfuscation regression
(documented behavior: a base64-encoded ``python -c`` write is *not*
detected).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.infra.hooks.codex_pre_tool_use import decide
from src.infra.tools.locking import try_lock

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "locks"
    d.mkdir()
    monkeypatch.setenv("MALA_LOCK_DIR", str(d))
    return d


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    return r


@pytest.fixture
def env(lock_dir: Path, repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set the MALA_* env vars used by the hook."""
    monkeypatch.setenv("MALA_AGENT_ID", "agent-me")
    monkeypatch.setenv("MALA_LOCK_DIR", str(lock_dir))
    monkeypatch.setenv("MALA_REPO_NAMESPACE", str(repo))
    monkeypatch.delenv("MALA_DISALLOWED_TOOLS", raising=False)
    return {
        "MALA_AGENT_ID": "agent-me",
        "MALA_LOCK_DIR": str(lock_dir),
        "MALA_REPO_NAMESPACE": str(repo),
    }


def make_payload(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    cwd: Path | None = None,
    session_id: str = "thr_test",
    turn_id: str = "trn_test",
) -> dict[str, Any]:
    return {
        "cwd": str(cwd) if cwd is not None else "/tmp",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "session_id": session_id,
        "turn_id": turn_id,
    }


def is_deny(result: dict[str, Any]) -> bool:
    spec = result.get("hookSpecificOutput") or {}
    return spec.get("permissionDecision") == "deny"


def deny_reason(result: dict[str, Any]) -> str:
    spec = result.get("hookSpecificOutput") or {}
    return spec.get("permissionDecisionReason", "")


def is_allow(result: dict[str, Any]) -> bool:
    spec = result.get("hookSpecificOutput") or {}
    return (
        spec.get("hookEventName") == "PreToolUse" and "permissionDecision" not in spec
    )


# ---------------------------------------------------------------------------
# Allow / deny baseline (apply_patch branch — AC #9)
# ---------------------------------------------------------------------------


class TestApplyPatchBranch:
    def test_allows_when_agent_holds_lock(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "src.py"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))

        result = decide(make_payload("apply_patch", {"path": str(target)}))

        assert is_allow(result), result

    def test_denies_when_unlocked(self, env: dict[str, str], repo: Path) -> None:
        target = repo / "src.py"

        result = decide(make_payload("apply_patch", {"path": str(target)}))

        assert is_deny(result)
        assert "has no active lock" in deny_reason(result)

    def test_denies_when_other_agent_holds_lock(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "src.py"
        assert try_lock(str(target), "agent-other", repo_namespace=str(repo))

        result = decide(make_payload("apply_patch", {"path": str(target)}))

        assert is_deny(result)
        msg = deny_reason(result)
        assert "is locked by agent 'agent-other'" in msg
        assert "current agent is 'agent-me'" in msg


# ---------------------------------------------------------------------------
# Dangerous command detection (AC #9.b)
# ---------------------------------------------------------------------------


class TestDangerousCommands:
    @pytest.mark.parametrize(
        "command,pattern",
        [
            ("rm -rf /", "rm -rf /"),
            ("git stash", "git stash"),
            ("git reset --hard HEAD~1", "git reset --hard"),
            ("git rebase main", "git rebase"),
        ],
    )
    def test_blocks_dangerous_command(
        self, env: dict[str, str], command: str, pattern: str
    ) -> None:
        result = decide(make_payload("bash", {"command": command}))

        assert is_deny(result)
        assert pattern in deny_reason(result)


# ---------------------------------------------------------------------------
# Disallowed-tool gate (AC #9.c)
# ---------------------------------------------------------------------------


class TestDisallowedTools:
    def test_denies_disallowed_tool(
        self, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MALA_DISALLOWED_TOOLS", "TodoWrite,SomeOther")

        result = decide(make_payload("TodoWrite", {}))

        assert is_deny(result)
        assert "TodoWrite" in deny_reason(result)
        assert "MALA_DISALLOWED_TOOLS" in deny_reason(result)

    def test_allows_non_disallowed_tool(
        self, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MALA_DISALLOWED_TOOLS", "TodoWrite")

        result = decide(make_payload("mala-locking.lock_acquire", {}))

        assert is_allow(result)


# ---------------------------------------------------------------------------
# Env-missing fail-closed (plan L820, deny-msg L860)
# ---------------------------------------------------------------------------


class TestEnvMissing:
    def test_denies_apply_patch_when_env_missing(
        self,
        repo: Path,
        lock_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No MALA_AGENT_ID etc. set.
        monkeypatch.delenv("MALA_AGENT_ID", raising=False)
        monkeypatch.delenv("MALA_REPO_NAMESPACE", raising=False)

        result = decide(make_payload("apply_patch", {"path": str(repo / "f.py")}))

        assert is_deny(result)
        assert "MALA_AGENT_ID/MALA_LOCK_DIR/MALA_REPO_NAMESPACE missing" in deny_reason(
            result
        )

    def test_denies_bash_with_writes_when_env_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MALA_AGENT_ID", raising=False)
        monkeypatch.delenv("MALA_LOCK_DIR", raising=False)
        monkeypatch.delenv("MALA_REPO_NAMESPACE", raising=False)

        result = decide(make_payload("bash", {"command": "echo hi > /tmp/somefile"}))

        assert is_deny(result)
        assert "MALA_AGENT_ID/MALA_LOCK_DIR/MALA_REPO_NAMESPACE missing" in deny_reason(
            result
        )

    def test_allows_bash_without_writes_when_env_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Read-only commands need no lock, so env-missing is irrelevant.
        monkeypatch.delenv("MALA_AGENT_ID", raising=False)
        monkeypatch.delenv("MALA_REPO_NAMESPACE", raising=False)

        result = decide(make_payload("bash", {"command": "ls -la"}))

        assert is_allow(result)


# ---------------------------------------------------------------------------
# Shell-write heuristic table (AC #19)
# ---------------------------------------------------------------------------


def _command_for_target(template: str, target: str) -> str:
    return template.format(target=target)


# (description, command-template) — the {target} placeholder is replaced
# with the absolute test path so the heuristic resolves to a real lock key.
HEURISTIC_ROWS: list[tuple[str, str]] = [
    ("redirect-overwrite", "echo hi > {target}"),
    ("redirect-append", "echo hi >> {target}"),
    ("redirect-stderr", "ls 2> {target}"),
    ("redirect-amp", "cmd &> {target}"),
    ("tee", "echo x | tee {target}"),
    ("tee-append", "echo x | tee -a {target}"),
    ("sed-i", "sed -i 's/foo/bar/' {target}"),
    ("sed-i-bak", "sed -i.bak 's/foo/bar/' {target}"),
    ("perl-i", "perl -i -pe 's/foo/bar/' {target}"),
    ("perl-pi", "perl -pi -e 's/foo/bar/' {target}"),
    ("awk-inplace", "awk -i inplace '{{print}}' {target}"),
    ("gawk-inplace", "gawk -i inplace '{{print}}' {target}"),
    ("python-c-open", "python -c \"open('{target}','w').write('x')\""),
    ("python3-c-open", "python3 -c \"open('{target}','a').write('x')\""),
    ("node-e-write", "node -e \"fs.writeFileSync('{target}','x')\""),
    ("ruby-e-write", "ruby -e \"File.write('{target}','x')\""),
    ("bash-c-redirect", "bash -c 'echo x > {target}'"),
    ("sh-c-redirect", "sh -c 'echo x > {target}'"),
    ("touch", "touch {target}"),
    ("cp", "cp /etc/hosts {target}"),
    ("mv", "mv /tmp/somefile {target}"),
    ("rm", "rm {target}"),
    ("mkdir", "mkdir {target}"),
    ("ln", "ln -s /tmp/x {target}"),
    ("chmod", "chmod 644 {target}"),
    ("chown", "chown user {target}"),
    ("install", "install -m 0644 /tmp/src {target}"),
    # ``dd if=...`` is caught earlier by the dangerous-cmd gate; use a
    # stdin-fed form so the heuristic gate is the one exercised here.
    ("dd", "printf X | dd of={target} bs=1 count=1"),
    # ``git checkout -- f`` and ``git restore f`` are both DESTRUCTIVE_GIT_PATTERNS;
    # they short-circuit at the dangerous-cmd gate before the heuristic runs.
    # Use ``git checkout f`` (single positional, no ``--``) to exercise the
    # heuristic gate. ``git restore``/``git stash apply``/``git apply`` are
    # covered by TestPatchAndApply / TestDangerousGitCoverage below.
    ("git-checkout", "git checkout {target}"),
]


class TestShellWriteHeuristic:
    """Each heuristic-table row: deny when unlocked, allow when locked."""

    @pytest.mark.parametrize(
        "label,template", HEURISTIC_ROWS, ids=[r[0] for r in HEURISTIC_ROWS]
    )
    def test_denies_when_target_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        label: str,
        template: str,
    ) -> None:
        target = repo / f"target_{label}.txt"
        command = _command_for_target(template, str(target))

        result = decide(make_payload("bash", {"command": command}))

        assert is_deny(result), f"{label}: expected deny, got {result}"
        assert "has no active lock" in deny_reason(result), (
            f"{label}: unexpected reason: {deny_reason(result)}"
        )

    @pytest.mark.parametrize(
        "label,template", HEURISTIC_ROWS, ids=[r[0] for r in HEURISTIC_ROWS]
    )
    def test_denies_when_target_locked_by_other(
        self,
        env: dict[str, str],
        repo: Path,
        label: str,
        template: str,
    ) -> None:
        target = repo / f"target_{label}.txt"
        assert try_lock(str(target), "agent-other", repo_namespace=str(repo))
        command = _command_for_target(template, str(target))

        result = decide(make_payload("bash", {"command": command}))

        assert is_deny(result), f"{label}: expected deny"
        assert "is locked by agent 'agent-other'" in deny_reason(result)

    @pytest.mark.parametrize(
        "label,template", HEURISTIC_ROWS, ids=[r[0] for r in HEURISTIC_ROWS]
    )
    def test_allows_when_target_locked_by_current(
        self,
        env: dict[str, str],
        repo: Path,
        label: str,
        template: str,
    ) -> None:
        target = repo / f"target_{label}.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = _command_for_target(template, str(target))

        result = decide(make_payload("bash", {"command": command}))

        assert is_allow(result), f"{label}: expected allow, got {result}"


class TestDangerousGitCoverage:
    """Heuristic-table rows that are *also* in DESTRUCTIVE_GIT_PATTERNS.

    AC #19 requires denial when the target is unlocked. These commands are
    short-circuited at the dangerous-cmd gate, which still produces a deny;
    that satisfies AC #19 even though the lock-check branch never runs.
    Pinning the deny-via-dangerous behavior here documents the dual coverage.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "git checkout -- file.py",
            "git restore file.py",
            "git stash apply",
            "git stash pop",
        ],
    )
    def test_destructive_git_always_denied(
        self, env: dict[str, str], command: str
    ) -> None:
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)


class TestPatchAndApply:
    """git apply / git stash apply / patch are denied via unresolved sentinel."""

    def test_git_apply_denies(self, env: dict[str, str]) -> None:
        result = decide(make_payload("bash", {"command": "git apply some.patch"}))
        assert is_deny(result)

    def test_git_stash_apply_denies(self, env: dict[str, str]) -> None:
        # Note: `git stash` is also caught earlier by the dangerous-command
        # gate, so the deny reason will refer to that pattern. Either deny
        # path is acceptable per AC #19.
        result = decide(make_payload("bash", {"command": "git stash apply"}))
        assert is_deny(result)

    def test_patch_denies(self, env: dict[str, str]) -> None:
        result = decide(make_payload("bash", {"command": "patch -p1 < some.patch"}))
        assert is_deny(result)


# ---------------------------------------------------------------------------
# Residual obfuscation regression (AC #19 documented limitation)
# ---------------------------------------------------------------------------


class TestResidualObfuscation:
    """Documented behavior: base64-decoded ``python -c`` is NOT detected.

    The hook is best-effort defense-in-depth, not a sandbox. A turn that
    decodes its write call from base64 (or constructs the filename from
    env var indirection / ``eval`` / etc.) bypasses the heuristic. This
    test pins the documented behavior so a future tightening or
    accidental loosening shows up as a test diff.
    """

    def test_base64_python_write_is_not_detected_and_is_allowed(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # The literal command never mentions a write call; ``open`` is
        # encoded inside the base64 payload. The heuristic sees no write
        # hint, so the command is allowed even though the runtime
        # behavior writes to ``target``. Tighter enforcement (e.g.,
        # ptrace/eBPF write monitoring) is follow-up work.
        target = repo / "evil.txt"
        del target  # not referenced in the command — that's the point
        command = 'python -c "$(echo b3BlbignaScsJ3cnKS53cml0ZSgneCcpCg== | base64 -d)"'

        result = decide(make_payload("bash", {"command": command}))

        # Documented gap: this is allowed. If/when this changes, update
        # the docstring + plan L851 first.
        assert is_allow(result), (
            "Residual obfuscation case must remain documented as not-detected; "
            "see plan L851. If you tightened the heuristic, update both."
        )

    def test_python_c_with_unresolved_write_is_denied(
        self, env: dict[str, str]
    ) -> None:
        # Conservative deny: write hint is visible but path is dynamic.
        # Per plan L846: deny on parseable-but-unresolved write expressions.
        command = "python -c \"import os; open(os.environ['F'], 'w').write('x')\""

        result = decide(make_payload("bash", {"command": command}))

        assert is_deny(result)


# ---------------------------------------------------------------------------
# Env transport (plan L815, AC-3 — env injection is the sole transport)
# ---------------------------------------------------------------------------


class TestEnvTransport:
    """Env injection is the sole supported transport.

    The Phase B/C spike confirmed ``codex_app_server.AppServerConfig.env``
    accepts an explicit per-process env dict, and
    :class:`CodexRuntimeBuilder.build()` overlays the full
    ``MALA_AGENT_ID`` + ``MALA_LOCK_DIR`` + ``MALA_REPO_NAMESPACE``
    bundle onto that dict (plan ``L815``, AC-3). The state-file
    fallback at ``~/.config/mala/agent-state/{session_id}.env`` is no
    longer part of the hook contract; the hook reads strictly from
    ``os.environ``.
    """

    def test_env_missing_denies_apply_patch(
        self,
        repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A missing ``MALA_*`` bundle in ``os.environ`` denies fail-closed."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.delenv("MALA_AGENT_ID", raising=False)
        monkeypatch.delenv("MALA_LOCK_DIR", raising=False)
        monkeypatch.delenv("MALA_REPO_NAMESPACE", raising=False)

        result = decide(
            make_payload(
                "apply_patch",
                {"path": str(repo / "f.py")},
                session_id="thr_missing",
            )
        )

        assert is_deny(result)
        assert "missing" in deny_reason(result)

    def test_state_file_is_ignored(
        self,
        repo: Path,
        lock_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A populated state file does NOT supply the env bundle.

        Pins the AC-3 contract that env is the sole supported transport.
        Before this fix the hook read the state file as a fallback; with
        env unset and only the state file populated, the hook would
        previously have ALLOWED the lock-checked write. Now it must DENY
        because ``os.environ`` is the only source.
        """
        fake_home = tmp_path / "home"
        state_dir = fake_home / ".config" / "mala" / "agent-state"
        state_dir.mkdir(parents=True)
        session_id = "thr_state_ignored"
        (state_dir / f"{session_id}.env").write_text(
            "MALA_AGENT_ID=agent-me\n"
            f"MALA_LOCK_DIR={lock_dir}\n"
            f"MALA_REPO_NAMESPACE={repo}\n"
        )
        monkeypatch.setenv("HOME", str(fake_home))
        # MALA_LOCK_DIR is required at try_lock time so the lock store
        # is real, but the *hook bundle* env vars are unset so only the
        # state file could provide them.
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_dir))
        target = repo / "src.py"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        # Now also unset MALA_AGENT_ID / MALA_REPO_NAMESPACE so the only
        # populated source for those keys is the state file.
        monkeypatch.delenv("MALA_AGENT_ID", raising=False)
        monkeypatch.delenv("MALA_REPO_NAMESPACE", raising=False)

        result = decide(
            make_payload("apply_patch", {"path": str(target)}, session_id=session_id)
        )

        assert is_deny(result), result
        assert "missing" in deny_reason(result)


# ---------------------------------------------------------------------------
# Output-shape contract
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_allow_shape(self, env: dict[str, str]) -> None:
        result = decide(make_payload("bash", {"command": "echo hi"}))
        spec = result["hookSpecificOutput"]
        assert spec == {"hookEventName": "PreToolUse"}

    def test_deny_shape(self, env: dict[str, str]) -> None:
        result = decide(make_payload("bash", {"command": "rm -rf /"}))
        spec = result["hookSpecificOutput"]
        assert spec["hookEventName"] == "PreToolUse"
        assert spec["permissionDecision"] == "deny"
        assert spec["permissionDecisionReason"]


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 2 — P1/P2 findings)
# ---------------------------------------------------------------------------


class TestRedirectionDigitPreservation:
    """Regression: ``>2026-log.txt`` (no space) must keep the leading digits.

    Before the fix, ``tok.lstrip("&012>")`` ate any leading 0/1/2/&/> chars
    from the *filename*, so ``>2026-log.txt`` was lock-checked against
    ``6-log.txt`` — a different path, allowing the real write to slip past
    the gate. Pin the digit-preserving behavior.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "echo hi >{target}",
            "echo hi >>{target}",
            "ls 2>{target}",
            "cmd &>{target}",
        ],
    )
    def test_digit_filenames_locked_correctly(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "2026-log.txt"
        # Lock the *correct* path (with leading digits intact).
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = command_template.format(target=str(target))

        result = decide(make_payload("bash", {"command": command}))

        assert is_allow(result), (
            f"Lock on the digit-leading filename should match the "
            f"redirection target, got: {result}"
        )

    def test_digit_filename_unlocked_denies_correct_path(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # No lock on /repo/2026-log.txt; locking only the digit-stripped
        # path 6-log.txt would have made the buggy implementation allow.
        wrong_path = repo / "6-log.txt"
        assert try_lock(str(wrong_path), "agent-me", repo_namespace=str(repo))
        target = repo / "2026-log.txt"
        command = f"echo hi >{target}"

        result = decide(make_payload("bash", {"command": command}))

        assert is_deny(result), (
            f"Lock on the digit-stripped path must NOT satisfy the "
            f"actual write target's lock requirement: {result}"
        )
        assert "2026-log.txt" in deny_reason(result)


class TestCwdResolution:
    """Regression: shell relative paths resolve against the payload cwd.

    Before the fix, ``cwd=/repo/pkg`` + ``command='touch generated.py'``
    lock-checked ``/repo/generated.py`` (namespace root) instead of
    ``/repo/pkg/generated.py``, so a root-level lock allowed an unowned
    write inside the subdirectory. Pin the cwd-resolved behavior.
    """

    def test_relative_path_resolves_against_cwd(
        self, env: dict[str, str], repo: Path
    ) -> None:
        sub = repo / "pkg"
        sub.mkdir()
        target = sub / "generated.py"
        # Lock only the cwd-resolved path; the namespace-root path stays
        # unlocked. The hook should consult the cwd-resolved path.
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))

        result = decide(
            make_payload(
                "bash",
                {"command": "touch generated.py"},
                cwd=sub,
            )
        )

        assert is_allow(result), result

    def test_root_lock_does_not_satisfy_subdir_write(
        self, env: dict[str, str], repo: Path
    ) -> None:
        sub = repo / "pkg"
        sub.mkdir()
        # Lock /repo/generated.py (namespace root) but the command runs
        # in /repo/pkg, so the actual write is /repo/pkg/generated.py —
        # which is *not* locked.
        assert try_lock(
            str(repo / "generated.py"), "agent-me", repo_namespace=str(repo)
        )

        result = decide(
            make_payload(
                "bash",
                {"command": "touch generated.py"},
                cwd=sub,
            )
        )

        assert is_deny(result), (
            f"Root-level lock must not satisfy the cwd-resolved write target: {result}"
        )


class TestSedLongInplace:
    """Regression: ``sed --in-place`` long form must be detected."""

    def test_sed_long_inplace_denied_when_unlocked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "long.txt"
        result = decide(
            make_payload(
                "bash",
                {"command": f"sed --in-place s/a/b/ {target}"},
            )
        )
        assert is_deny(result)

    def test_sed_long_inplace_with_eq_value_denied(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "long.txt"
        result = decide(
            make_payload(
                "bash",
                {"command": f"sed --in-place=.bak s/a/b/ {target}"},
            )
        )
        assert is_deny(result)

    def test_sed_long_inplace_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "long.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(
            make_payload(
                "bash",
                {"command": f"sed --in-place s/a/b/ {target}"},
            )
        )
        assert is_allow(result)


class TestAwkInplaceProgramFlag:
    """Regression: ``awk -i inplace -f script.awk file`` must lock-check ``file``.

    Before the fix, the heuristic always dropped the first positional as
    "the inline awk program", so the only positional (the actual file
    target) was discarded and the heuristic returned an empty list,
    silently allowing the in-place edit.
    """

    def test_awk_with_program_file_denies_when_unlocked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "data.txt"
        result = decide(
            make_payload(
                "bash",
                {"command": f"awk -i inplace -f script.awk {target}"},
            )
        )
        assert is_deny(result)
        assert "has no active lock" in deny_reason(result)

    def test_awk_with_program_file_allows_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "data.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(
            make_payload(
                "bash",
                {"command": f"awk -i inplace -f script.awk {target}"},
            )
        )
        assert is_allow(result)


class TestExtendedDangerousGitGuards:
    """Regression: parity with ``block_dangerous_commands`` for git commit/push.

    Before the fix, ``git commit -a``, ``git commit`` without prior
    ``git add``, and ``git push --force`` all returned ``allow`` from
    the Codex hook because the detector only looped over the static
    ``DANGEROUS_PATTERNS``/``DESTRUCTIVE_GIT_PATTERNS`` lists. AC #9
    requires the Codex hook to mirror the Claude hook's full coverage.
    """

    @pytest.mark.parametrize(
        "command,expected_substring",
        [
            ("git commit -a -m 'wip'", "git commit -a"),
            ("git commit --all -m 'wip'", "git commit -a/--all"),
            ("git commit -m 'wip'", "git commit without prior git add"),
            ("git push --force origin main", "git push --force"),
            ("git push -f origin main", "git push --force"),
        ],
    )
    def test_extended_git_guards_deny(
        self,
        env: dict[str, str],
        command: str,
        expected_substring: str,
    ) -> None:
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), result
        assert expected_substring in deny_reason(result), deny_reason(result)

    def test_force_with_lease_allowed(self, env: dict[str, str]) -> None:
        # ``--force-with-lease`` is the documented safer alternative.
        result = decide(
            make_payload(
                "bash",
                {"command": "git push --force-with-lease origin main"},
            )
        )
        assert is_allow(result)

    def test_atomic_add_commit_allowed(self, env: dict[str, str], repo: Path) -> None:
        # ``git add <file> && git commit -m ...`` is the required pattern.
        result = decide(
            make_payload(
                "bash",
                {"command": "git add f.py && git commit -m 'msg'"},
            )
        )
        assert is_allow(result)


class TestApplyPatchPatchBody:
    """Regression: apply_patch with embedded patch body parses headers.

    Codex apply_patch surfaces the patch content under
    ``tool_input.command`` (or ``input``/``patch`` etc.) rather than as a
    structured ``path`` field. Before the fix, the hook returned an empty
    target list and the *next* gate was a no-op, so the call was allowed
    without any lock check — defeating AC #9 entirely under the dominant
    Codex shape.
    """

    def test_envelope_add_file_extracted(self, env: dict[str, str], repo: Path) -> None:
        body = (
            "*** Begin Patch\n"
            "*** Add File: src/new.py\n"
            "+def f():\n"
            "+    return 1\n"
            "*** End Patch\n"
        )
        result = decide(
            make_payload(
                "apply_patch",
                {"command": body},
                cwd=repo,
            )
        )
        assert is_deny(result), result
        assert "src/new.py" in deny_reason(result)

    def test_envelope_update_file_extracted(
        self, env: dict[str, str], repo: Path
    ) -> None:
        body = "*** Update File: src/foo.py\n@@\n-old\n+new\n"
        result = decide(
            make_payload(
                "apply_patch",
                {"input": body},
                cwd=repo,
            )
        )
        assert is_deny(result)
        assert "src/foo.py" in deny_reason(result)

    def test_unified_diff_header_extracted(
        self, env: dict[str, str], repo: Path
    ) -> None:
        body = (
            "diff --git a/src/foo.py b/src/foo.py\n"
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@\n-x\n+y\n"
        )
        result = decide(
            make_payload(
                "apply_patch",
                {"patch": body},
                cwd=repo,
            )
        )
        assert is_deny(result)
        assert "src/foo.py" in deny_reason(result)

    def test_envelope_locked_path_allowed(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Lock the cwd-resolved path; apply_patch should allow.
        target = repo / "src" / "new.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        body = "*** Begin Patch\n*** Add File: src/new.py\n+x\n*** End Patch\n"
        result = decide(
            make_payload(
                "apply_patch",
                {"command": body},
                cwd=repo,
            )
        )
        assert is_allow(result), result

    def test_apply_patch_no_extractable_paths_denies(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # No ``path``/``paths``-style keys, no parseable patch headers.
        # Must deny fail-closed (Amp parity: mala-safety.ts L1329-L1339).
        result = decide(
            make_payload(
                "apply_patch",
                {"command": "this is not a patch body at all"},
                cwd=repo,
            )
        )
        assert is_deny(result)
        assert "no extractable target paths" in deny_reason(result)


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 3)
# ---------------------------------------------------------------------------


class TestRedirectionFdDuplications:
    """Regression: ``2>&1``/``>&2``/``>&-`` are fd refs, not write paths.

    The previous regex (this iteration) treated ``2>&1`` as a write to
    ``&1``. ``cmd > out.log 2>&1`` then denied even when ``out.log`` was
    locked correctly, because the bogus ``&1`` target failed the lock
    check first. Pin the fd-ref skip behaviour.
    """

    def test_stderr_to_stdout_fd_dup_does_not_block(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "out.log"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        # ``2>&1`` must not be lock-checked as ``&1``.
        result = decide(make_payload("bash", {"command": f"cmd >{target} 2>&1"}))
        assert is_allow(result), result

    def test_close_fd_does_not_block(self, env: dict[str, str], repo: Path) -> None:
        target = repo / "out.log"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        # ``>&-`` closes a fd. Same false-positive risk as ``2>&1``.
        result = decide(make_payload("bash", {"command": f"cmd >{target} >&-"}))
        assert is_allow(result), result

    def test_fd_dup_alone_with_unlocked_redirect_still_denies(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # The lock-check still applies to the *real* redirect target.
        target = repo / "unlocked.log"
        result = decide(make_payload("bash", {"command": f"cmd >{target} 2>&1"}))
        assert is_deny(result)
        assert "unlocked.log" in deny_reason(result)


class TestSedAttachedAndLongScriptFlags:
    """Regression: attached/long script-flag forms keep the file target.

    Before the fix, ``sed -es/a/b/ file`` was tokenized as
    ``['sed', '-es/a/b/', 'file']`` and the only positional ``file`` was
    discarded as the "inline script", so the heuristic emitted no
    targets and the in-place edit slipped past the gate. Same shape for
    ``sed --expression=... file`` and ``awk --source=... file``.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            # Short attached form: -e<script> (no space).
            "sed -i -es/a/b/ {target}",
            # Long form with `=` value.
            "sed --in-place --expression=s/a/b/ {target}",
            # Long form with separate value token.
            "sed --in-place --expression s/a/b/ {target}",
            # Bundled in-place + expression: -ies<script>.
            "sed -ies/a/b/ {target}",
        ],
    )
    def test_sed_attached_or_long_script_flag_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "data.txt"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "has no active lock" in deny_reason(result)

    @pytest.mark.parametrize(
        "command_template",
        [
            "sed -i -es/a/b/ {target}",
            "sed --in-place --expression=s/a/b/ {target}",
        ],
    )
    def test_sed_attached_or_long_allowed_when_locked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "data.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), f"command: {command!r} → {result}"


class TestAwkLongSourceFlag:
    """Regression: ``awk --source=...`` keeps the file target.

    Same shape as the sed/perl long-form bug. Before the fix,
    ``awk -i inplace --source='{print}' file`` had only ``file`` as a
    positional, which was discarded as "the inline program".
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "awk -i inplace --source='{{print}}' {target}",
            "awk -i inplace --source '{{print}}' {target}",
        ],
    )
    def test_awk_long_source_flag_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "data.txt"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "has no active lock" in deny_reason(result)


class TestOnelinerLeadingFlags:
    """Regression: leading interpreter flags don't bypass the heuristic.

    Before the fix, ``flag = tokens[1]; body = tokens[2]`` hardcoded the
    script-flag position, so ``python -u -c "open(...)"`` (with the
    unbuffered ``-u`` flag before ``-c``) had ``flag == "-u"`` and the
    heuristic returned ``[]``, allowing the unowned write.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            # python with -u (unbuffered) before -c.
            "python -u -c \"open('{target}','w').write('x')\"",
            # python3 with -B (no .pyc) before -c.
            "python3 -B -c \"open('{target}','w').write('x')\"",
            # node with --no-warnings before -e.
            "node --no-warnings -e \"fs.writeFileSync('{target}','x')\"",
            # bash with -x (xtrace) before -c.
            "bash -x -c 'echo y > {target}'",
        ],
    )
    def test_leading_flags_do_not_bypass(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.txt"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"


class TestControlOperatorsWithoutSpaces:
    """Regression: ``;`` / ``&&`` / ``||`` / ``|`` without surrounding spaces.

    Before the fix, ``shlex.split`` returned ``true;touch f`` as a single
    token ``true;touch`` plus ``f``, no extractor saw ``touch``, the
    write-target list was empty, and the hook allowed the unowned write.
    Pin the normalized-separator behaviour.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "true;touch {target}",
            "true&&touch {target}",
            "false||touch {target}",
            "echo y|tee {target}",
        ],
    )
    def test_control_op_without_spaces_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.txt"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "has no active lock" in deny_reason(result)

    def test_quoted_separator_not_split(self, env: dict[str, str], repo: Path) -> None:
        # Literal ``;`` inside single quotes must NOT be treated as a
        # control operator. ``echo 'a;b' > out`` should detect ``out``
        # as the only target and deny only on that path.
        target = repo / "out"
        command = f"echo 'a;b' > {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)
        assert "out" in deny_reason(result)


class TestApplyPatchMoveDestination:
    """Regression: ``*** Move to: <new>`` is a write target.

    Before the fix, the regex only matched headers ending in ``File:``
    so ``*** Move to: new.py`` was ignored. A patch that updated
    ``old.py`` and renamed it to ``new.py`` lock-checked only ``old.py``;
    holding the source lock allowed creating an unowned ``new.py``.
    """

    def test_move_to_destination_extracted(
        self, env: dict[str, str], repo: Path
    ) -> None:
        body = (
            "*** Begin Patch\n"
            "*** Update File: old.py\n"
            "*** Move to: new.py\n"
            "@@\n-x\n+y\n"
            "*** End Patch\n"
        )
        # Lock only the SOURCE; the destination is unlocked. The hook
        # must lock-check the destination too and deny.
        assert try_lock(str(repo / "old.py"), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("apply_patch", {"command": body}, cwd=repo))
        assert is_deny(result), result
        assert "new.py" in deny_reason(result)

    def test_move_to_destination_locked_allowed(
        self, env: dict[str, str], repo: Path
    ) -> None:
        body = "*** Update File: old.py\n*** Move to: new.py\n@@\n-x\n+y\n"
        # Lock both source and destination.
        assert try_lock(str(repo / "old.py"), "agent-me", repo_namespace=str(repo))
        assert try_lock(str(repo / "new.py"), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("apply_patch", {"command": body}, cwd=repo))
        assert is_allow(result), result


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 4)
# ---------------------------------------------------------------------------


class TestArbitraryFdRedirections:
    """Regression: POSIX shells accept any numeric fd as a redirect prefix.

    Previously the regex hardcoded ``[012]?>`` which only recognised fd
    0/1/2. ``echo hi 3>file`` truncates ``file`` before the command
    runs, but the heuristic returned no targets and the unowned write
    slipped past the lock check.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "echo hi 3>{target}",
            "cmd 4>>{target}",
            "cmd 9>{target}",
            # Space-separated form for the same higher fds.
            "echo hi 3> {target}",
            "cmd 5>> {target}",
        ],
    )
    def test_high_fd_redirect_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.log"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "out.log" in deny_reason(result)


class TestLegacyCombinedRedirection:
    """Regression: ``>&file`` (legacy) writes both stdout and stderr to file.

    Before the fix, the prefix regex only matched ``>``, leaving the
    target as ``&file``. An agent could lock the literal path ``&file``
    (a junk lock-key) and the hook would allow the command — but bash
    actually writes to ``file``. Pin the legacy form.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "cmd >&{target}",
            "cmd 2>&{target}",
            "cmd >& {target}",
            "cmd 2>& {target}",
        ],
    )
    def test_legacy_redirect_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.log"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "out.log" in deny_reason(result)

    def test_legacy_redirect_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "out.log"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"cmd >&{target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result

    def test_fd_dup_in_combined_form_still_skipped(self, env: dict[str, str]) -> None:
        # Sanity: the legacy fix must not regress the fd-dup skip.
        # ``>&1`` and ``2>&1`` are fd dups, not writes. With no other
        # write target, the command must allow.
        result = decide(make_payload("bash", {"command": "cmd >&1 2>&1"}))
        assert is_allow(result), result


class TestUtilityGNUFlags:
    """Regression: ``cp -t DIR src`` and ``chmod --reference=R target``.

    GNU-specific flags reorder the positional argument layout. Before
    the fix:
    - ``cp -t dir src`` lock-checked ``src`` (the source) and allowed
      writing to the unowned ``dir``.
    - ``chmod --reference=R target`` discarded ``target`` as the "mode
      positional" and skipped the lock-check entirely.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "cp -t {dest_dir} /tmp/src",
            "cp -t{dest_dir} /tmp/src",
            "cp --target-directory={dest_dir} /tmp/src",
            "cp --target-directory {dest_dir} /tmp/src",
            "mv -t {dest_dir} /tmp/src",
            "install -t {dest_dir} -m 0644 /tmp/src",
        ],
    )
    def test_target_directory_flag_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        dest = repo / "outdir"
        command = command_template.format(dest_dir=dest)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "outdir" in deny_reason(result)

    def test_target_directory_overrides_source_lock(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Holding the SOURCE lock must NOT satisfy the destination check.
        src = repo / "src.py"
        dest = repo / "outdir"
        assert try_lock(str(src), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("bash", {"command": f"cp -t {dest} {src}"}))
        assert is_deny(result)
        assert "outdir" in deny_reason(result)

    @pytest.mark.parametrize(
        "command_template",
        [
            "chmod --reference=/tmp/ref {target}",
            "chmod --reference /tmp/ref {target}",
            "chown --reference=/tmp/ref {target}",
            "chown --reference /tmp/ref {target}",
        ],
    )
    def test_reference_flag_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "data.txt"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "data.txt" in deny_reason(result)


class TestPathQualifiedCommandNames:
    """Regression: ``/usr/bin/touch`` dispatches the same as ``touch``.

    Before the fix, every extractor compared ``tokens[0]`` literally to
    the command name, so any absolute or relative path invocation
    (``/usr/bin/touch``, ``./bin/sed``) bypassed the heuristic.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "/usr/bin/touch {target}",
            "/bin/cp /tmp/src {target}",
            "/usr/bin/sed -i s/a/b/ {target}",
            "/usr/bin/awk -i inplace '{{print}}' {target}",
            "./scripts/touch {target}",
        ],
    )
    def test_path_qualified_dispatches_correctly(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.txt"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"

    def test_path_qualified_oneliner_dispatches(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "out.txt"
        command = f"/usr/bin/python -c \"open('{target}','w').write('x')\""
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)

    def test_path_qualified_git_dispatches(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "f.py"
        command = f"/usr/bin/git checkout {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)


class TestEscapedQuotesInNormalizer:
    """Regression: ``\\"`` outside quotes must not flip quote state.

    Bash treats ``\\"`` (backslash-quote) at the top level as a literal
    ``"`` character — it does NOT open a quoted region. Before the fix,
    ``echo \\";touch f`` made the parser think ``"`` opened a quote, so
    the trailing ``;`` was treated as quoted and never split. The
    second command ran at top level in bash but the heuristic never saw
    it.
    """

    def test_escaped_quote_does_not_hide_separator(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "unowned.py"
        # ``echo \";touch <unowned>`` — the \" is a literal " so ; is a
        # top-level separator. The second command must be detected.
        command = f'echo \\";touch {target}'
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_escaped_separator_inside_quotes_still_quoted(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: a ``;`` inside a real quoted region is still quoted.
        # ``echo "a;b" > out`` — ; is inside quotes; only ``out`` is a
        # write target.
        target = repo / "out"
        command = f'echo "a;b" > {target}'
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)
        assert "out" in deny_reason(result)

    def test_single_quotes_treat_backslash_literally(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Inside single quotes, backslash is literal — \" does NOT
        # escape anything. ``echo 'a\";b' > out`` — the ; is inside
        # single quotes, no split.
        target = repo / "out"
        command = f"echo 'a\\\";b' > {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)
        assert "out" in deny_reason(result)


class TestBareBackgroundOperator:
    """Regression: ``touch out&`` is backgrounded, not a write to ``out&``.

    Before the fix, bare ``&`` was never split, so ``touch out&``
    tokenized as ``['touch', 'out&']`` and the heuristic added the
    junk path ``out&`` as the write target. An agent could lock the
    literal ``out&`` and the hook would allow the command — while bash
    runs ``touch out`` and writes the unlocked file in the background.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "touch {target}&",
            "touch {target} &",
            "true & touch {target}",
            "true&touch {target}",
        ],
    )
    def test_bare_background_separator_split(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.txt"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "out.txt" in deny_reason(result)

    def test_fd_dup_with_trailing_background_split(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``cmd >out 2>&1 &`` — the trailing ``&`` is backgrounding,
        # but the inner ``2>&1`` is fd dup. Both must be classified
        # correctly: ``out`` is the only write target.
        target = repo / "out.log"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"cmd >{target} 2>&1 &"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 5)
# ---------------------------------------------------------------------------


class TestShellGroupingOperators:
    """Regression: ``(touch f)`` and ``{ touch f; }`` are subshell/group writes.

    Before the fix, ``(touch unowned.py)`` tokenized as ``['(touch',
    'unowned.py)']`` so no extractor saw ``touch`` and the unlocked
    write was allowed. ``{ touch unowned.py; }`` had ``{`` as the first
    segment token, which the utility extractor's basename matched
    against the literal ``{`` (no match), again silently allowing the
    write.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "(touch {target})",
            "( touch {target} )",
            "{{ touch {target}; }}",
            "{{ touch {target} ; }}",
            "true && (touch {target})",
            "(true; touch {target})",
        ],
    )
    def test_grouped_write_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.txt"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "out.txt" in deny_reason(result)

    def test_subshell_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "out.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"(touch {target})"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result


class TestVariableAllocatedFd:
    """Regression: ``{fd}>file`` (bash 4.1+ var-allocated fd).

    Before the fix, the redirection regex only accepted numeric or
    ``&`` fd prefixes, so ``echo hi {myfd}>unlocked.txt`` produced no
    write target. Bash creates/truncates ``unlocked.txt`` regardless,
    so the unlocked write slipped past the gate.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "echo hi {{myfd}}>{target}",
            "echo hi {{myfd}}>>{target}",
            "echo hi {{fd_a}}>{target}",
            "echo hi {{LOG_FD}}>{target}",
            # Whole-token operator form (with space).
            "echo hi {{myfd}}> {target}",
        ],
    )
    def test_var_allocated_fd_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.log"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "out.log" in deny_reason(result)

    def test_var_allocated_fd_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "out.log"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"echo hi {{myfd}}>{target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result


class TestAttachedAwkInplace:
    """Regression: ``awk -iinplace '{print}' file`` (attached arg).

    Before the fix, the heuristic only matched ``-i inplace`` as two
    separate tokens. gawk also accepts the attached short-flag form
    ``-iinplace``, which collapsed the file detection: ``found_inplace``
    stayed False and the heuristic returned no targets.
    """

    def test_awk_attached_inplace_denies_unlocked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "data.txt"
        command = f"awk -iinplace '{{print}}' {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "data.txt" in deny_reason(result)

    def test_gawk_attached_inplace_denies_unlocked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "data.txt"
        command = f"gawk -iinplace '{{print}}' {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"

    def test_awk_attached_inplace_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "data.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"awk -iinplace '{{print}}' {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result


class TestBundledTargetDirFlag:
    """Regression: ``cp -fvt dir src`` (bundled short flags).

    Before the fix, ``-t`` was only recognised at the start of a flag
    bundle (``tok == "-t"`` or ``tok.startswith("-t")``). When ``-t``
    appeared anywhere else in a bundle (``-fvt``, ``-vt``, etc.), the
    heuristic missed the target-directory flag and lock-checked the
    source file instead — letting an agent who held the source lock
    silently write to an unowned destination.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "cp -fvt {dest} src",
            "cp -vt {dest} src",
            "cp -ft {dest} src",
            "mv -fvt {dest} src",
            "install -m0644 -vt {dest} src",
            # Bundled with attached value: -fvtDIR
            "cp -fvt{dest} src",
        ],
    )
    def test_bundled_target_dir_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        dest = repo / "outdir"
        command = command_template.format(dest=dest)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "outdir" in deny_reason(result)

    def test_bundled_target_dir_overrides_source_lock(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Holding the source lock must NOT satisfy the destination check.
        src = repo / "src.py"
        dest = repo / "outdir"
        assert try_lock(str(src), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("bash", {"command": f"cp -fvt {dest} {src}"}))
        assert is_deny(result)
        assert "outdir" in deny_reason(result)


class TestAnsiCQuoting:
    """Regression: ``$'\\''`` ANSI-C quote-toggle bypass.

    Before the fix, the parser treated ``$'...'`` as a regular single-
    quoted region without recognising backslash escapes inside it. For
    ``$'\\''`` the parser saw three single quotes (``'``, ``'``, ``'``)
    and concluded the *last* one opened a fresh quoted region, so a
    trailing ``;`` looked quoted and was never split. The second
    command then ran in bash but the heuristic never saw it.
    """

    def test_ansi_c_escaped_quote_does_not_invert_state(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "unowned.py"
        # ``echo $'\\'' ; touch <unowned>`` — the inner ``\'`` is an
        # ANSI-C escape for ``'``. The outer ``;`` must remain a
        # top-level separator.
        command = f"echo $'\\'' ; touch {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_ansi_c_string_does_not_break_redirect(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # An ANSI-C string as a normal positional argument should not
        # affect later redirection-target detection.
        target = repo / "out"
        command = f"echo $'a\\nb' > {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)
        assert "out" in deny_reason(result)


class TestNewlineSeparator:
    """Regression: literal ``\\n`` in the command runs as a separator.

    Before the fix, ``true\\ntouch unowned.py`` only got ``;``/``&&``/
    etc. normalisation; ``\\n`` reached ``shlex.split`` as plain
    whitespace and the entire input collapsed into a single segment
    headed by ``true``. The heuristic never saw ``touch`` and the
    unlocked write was allowed.
    """

    def test_newline_separates_commands(self, env: dict[str, str], repo: Path) -> None:
        target = repo / "unowned.py"
        command = f"true\ntouch {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_multiline_script_each_line_extracted(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # A multi-line bash body should still gate every line.
        a = repo / "a.txt"
        b = repo / "b.txt"
        command = f"touch {a}\ntouch {b}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        # First missing lock fires the deny — either path is acceptable.
        assert "a.txt" in deny_reason(result) or "b.txt" in deny_reason(result)


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 6)
# ---------------------------------------------------------------------------


class TestAwkLongIncludeInplace:
    """Regression: ``gawk --include inplace`` long-form activates in-place.

    Before the fix, the heuristic only matched ``-i inplace`` and
    ``-iinplace``. ``gawk --include inplace '{print}' file`` was
    treated as a generic flag, ``found_inplace`` stayed False, and the
    in-place mutation slipped past the gate.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "gawk --include inplace '{{print}}' {target}",
            "gawk --include=inplace '{{print}}' {target}",
            "awk --include inplace '{{print}}' {target}",
            "awk --include=inplace '{{print}}' {target}",
        ],
    )
    def test_include_inplace_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "data.txt"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "data.txt" in deny_reason(result)

    def test_include_inplace_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "data.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"gawk --include inplace '{{print}}' {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result

    def test_include_non_inplace_value_does_not_activate(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``--include readfile`` includes a gawk library named ``readfile``
        # — it does NOT enable in-place mode. Without explicit handling,
        # the value would land in ``positionals`` and confuse the heuristic.
        # The command should NOT deny on lock-mismatch (no in-place active).
        result = decide(
            make_payload(
                "bash",
                {"command": "gawk --include readfile '{print}' some-arg"},
            )
        )
        assert is_allow(result), result


class TestReadWriteRedirection:
    """Regression: ``<>file`` and ``n<>file`` (POSIX read-write open).

    Before the fix, the redirection regex only matched the ``>`` family.
    ``: <>unlocked.txt`` and ``exec 3<>unlocked.txt`` open ``unlocked.txt``
    for read+write and create it if missing — both clear write surfaces —
    yet the heuristic returned no targets and the unlocked write was
    silently allowed.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            ": <>{target}",
            ": <> {target}",
            "exec 3<>{target}",
            "exec 3<> {target}",
            "exec 9<>{target}",
        ],
    )
    def test_read_write_redirect_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.log"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "out.log" in deny_reason(result)

    def test_read_only_redirect_does_not_deny(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``<file`` opens for reading only — NOT a write surface. Must
        # not falsely trigger the lock check.
        target = repo / "input.txt"
        result = decide(make_payload("bash", {"command": f"cat <{target}"}))
        assert is_allow(result), result


class TestBundledValueFlagBeforeT:
    """Regression: ``cp -St dir src dest`` (-S consumes ``t`` as suffix).

    Before the fix, the heuristic blindly scanned the bundle for ``t``
    and treated it as ``-t`` regardless of preceding value-taking flags.
    For ``cp -St dir src dst``, GNU cp parses ``-S t`` (suffix=``t``)
    and ``dest=dst``, but the heuristic extracted ``dir`` as the
    destination. An agent holding ``dir``'s lock could silently write
    to the unowned ``dst``.
    """

    @pytest.mark.parametrize(
        "command_template,suffix_arg,expected_substr",
        [
            # -S<arg> via bundle: -S consumes 't' (no -t flag at all)
            ("cp -St dir src {dest}", None, "out"),
            ("mv -St dir src {dest}", None, "out"),
            # -S<arg> attached: -Sbak still allows -t recognition only
            # when ``t`` is NOT after the S-bundle.
            # Reverse order (`-tS`) is genuinely target-dir taking ``S``.
            # ``-vSt`` — S consumes the rest of the bundle (``t``).
            ("cp -vSt dir src {dest}", None, "out"),
            ("install -mSt dir src {dest}", None, "out"),
        ],
    )
    def test_value_flag_before_t_extracts_real_dest(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
        suffix_arg: str | None,
        expected_substr: str,
    ) -> None:
        del suffix_arg  # placeholder for future params
        dest = repo / "out.txt"
        command = command_template.format(dest=dest)
        result = decide(make_payload("bash", {"command": command}))
        # Real destination is the trailing positional ``out.txt``, not
        # the ``dir`` arg consumed by ``-S``.
        assert is_deny(result), f"command: {command!r} → {result}"
        assert expected_substr in deny_reason(result), deny_reason(result)

    def test_t_before_s_still_extracts_target_dir(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``-tS dir src`` — cp parses ``-t S`` (target-dir=``S``). The
        # heuristic should match cp's interpretation: extract ``S`` as
        # the destination value.
        # Run with no lock anywhere; the heuristic deny on the literal
        # ``S`` proves it dispatched correctly to the target-dir branch.
        result = decide(
            make_payload(
                "bash",
                {"command": "cp -tS dir src extra"},
            )
        )
        # ``-tS`` is treated as ``-t`` with attached value ``S``; a
        # nonexistent ``S`` lock-check denies.
        assert is_deny(result), result
        assert "'S'" in deny_reason(result)


class TestGitGlobalOptions:
    """Regression: git global options before the subcommand.

    Before the fix, the subcommand position was hardcoded to
    ``tokens[1]``. Global options like ``-C /repo``, ``-c k=v``,
    ``--git-dir=...``, ``--work-tree=.`` shifted the real subcommand
    later in the token stream. The flag landed in the
    ``GIT_WRITE_SUBCOMMANDS`` lookup, found no match, and the unlocked
    write slipped past the gate.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "git -C /tmp checkout {target}",
            "git --git-dir=/tmp/.git checkout {target}",
            "git --work-tree=. checkout {target}",
            "git -c core.editor=vi checkout {target}",
            "git --no-pager checkout {target}",
            "git -c k=v -C /tmp checkout {target}",
        ],
    )
    def test_global_options_before_subcommand_denies(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "f.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"

    def test_global_options_before_apply_denies(self, env: dict[str, str]) -> None:
        # ``git apply patch`` returns the unresolved sentinel which
        # always denies. With global options, the subcommand finder
        # must still locate ``apply``.
        result = decide(
            make_payload(
                "bash",
                {"command": "git --work-tree=. apply some.patch"},
            )
        )
        assert is_deny(result)

    def test_non_write_subcommand_allowed(self, env: dict[str, str]) -> None:
        # ``git status`` is not a write subcommand — even with global
        # options, the heuristic must not deny.
        result = decide(make_payload("bash", {"command": "git -C /tmp status"}))
        assert is_allow(result)


class TestForceClobberRedirect:
    """Regression: ``>|`` (force-clobber) preserved through normalization.

    Before the fix, ``_normalize_separators`` split every ``|``
    including the one inside ``>|``. ``echo hi >| unlocked.txt`` got
    rewritten to ``echo hi >  |  unlocked.txt``, the redirection
    extractor saw ``>`` followed by ``|`` (not a target), and the
    write to ``unlocked.txt`` slipped past the lock check.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "echo hi >|{target}",
            "echo hi >| {target}",
            "cmd 2>|{target}",
        ],
    )
    def test_force_clobber_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.log"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "out.log" in deny_reason(result)

    def test_force_clobber_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "out.log"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"echo hi >| {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result

    def test_pipe_alone_still_split(self, env: dict[str, str]) -> None:
        # Sanity: a regular ``|`` (not preceded by ``>``) must still
        # split into separate segments.
        result = decide(
            make_payload(
                "bash",
                {"command": "echo y|tee /tmp/should-not-exist-xyz"},
            )
        )
        # Either deny — the point is that the segment got split and
        # the tee target was extracted (not ``y|tee``).
        assert is_deny(result)


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 7)
# ---------------------------------------------------------------------------


class TestCdInvalidatesRelativeResolution:
    """Regression: ``cd pkg && touch f`` writes to ``cwd/pkg/f``, not ``cwd/f``.

    Before the fix, the hook always resolved relative targets against
    the payload ``cwd``. An agent could ``cd`` into a subdirectory and
    write to a path the hook lock-checked against the parent — letting
    the agent hold the parent-level lock and silently mutate the
    subdirectory file.
    """

    def test_cd_then_relative_write_denies(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # The agent holds the lock on ``cwd/generated.py`` (root file)
        # but the actual write is to ``cwd/pkg/generated.py`` after cd.
        sub = repo / "pkg"
        sub.mkdir()
        assert try_lock(
            str(repo / "generated.py"), "agent-me", repo_namespace=str(repo)
        )
        command = "cd pkg && touch generated.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        # The cd invalidates relative-path resolution — must deny.
        assert is_deny(result), result

    def test_cd_then_absolute_write_uses_absolute(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # An absolute path after ``cd`` is unambiguous; the hook should
        # still lock-check it normally (no over-deny).
        target = repo / "out.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"cd /tmp && touch {target}"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result

    def test_no_cd_relative_write_resolves_normally(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: without a ``cd``, the existing cwd-resolution path
        # still works.
        target = repo / "generated.py"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = "touch generated.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result


class TestAttachedOnelinerScript:
    """Regression: ``python -c"open('f','w')"`` (attached body).

    Before the fix, ``_find_oneliner_body`` only matched
    ``tokens[i] == "-c"`` and returned ``None`` for the attached form.
    shlex tokenises ``python -c"..."`` as ``['python', '-c<body>']``,
    so the body was never inspected and the unowned write slipped past
    the heuristic.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "python -c\"open('{target}','w').write('x')\"",
            "python3 -c\"open('{target}','a').write('x')\"",
            # bash -c with attached redirect
            'bash -c"echo y > {target}"',
            # ruby/node attached -e
            "ruby -e\"File.write('{target}', 'x')\"",
            "node -e\"fs.writeFileSync('{target}','x')\"",
        ],
    )
    def test_attached_oneliner_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.txt"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"

    def test_attached_oneliner_does_not_match_long_flag(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``python --check`` starts with ``-c`` but is a long flag —
        # MUST NOT be treated as the script flag. With no real script
        # body, no write target is detected, command should allow.
        result = decide(make_payload("bash", {"command": "python --check foo.py"}))
        assert is_allow(result), result


class TestUnspacedRedirection:
    """Regression: ``cmd>file`` and ``cat>|file`` (no space before ``>``).

    Before the fix, the normalizer never padded ``>``, so ``shlex.split``
    produced one combined token (``echo hi>file`` →
    ``['echo', 'hi>file']``). The combined-prefix regex requires the
    token to START with the operator/fd-prefix, so ``hi>file`` produced
    no target and the unowned write slipped past the gate.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "echo hi>{target}",
            "echo hi>>{target}",
            "echo hi>|{target}",
            # No space before fd prefix either: ``cmd2>file`` is real bash.
            # (However bash usually treats trailing digits as part of the
            #  preceding word; we still want to handle the safer
            #  ``cmd 2>file`` form.)
            "ls 2>{target}",
            "cat <>{target}",
            "exec 3>{target}",
        ],
    )
    def test_unspaced_redirect_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.log"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "out.log" in deny_reason(result)

    def test_unspaced_redirect_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "out.log"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"echo hi>{target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result


class TestReservedWordSegments:
    """Regression: reserved words leave the real cmd behind a non-cmd token.

    Before the fix, ``if true; then touch unowned.py; fi`` produced a
    segment ``['then', 'touch', 'unowned.py']`` whose first token was
    not the command. The utility extractor saw ``then`` (no match)
    and emitted no targets.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "if true; then touch {target}; fi",
            "for x in 1 2; do touch {target}; done",
            "while false; do touch {target}; done",
            "until true; do touch {target}; done",
            # ``elif`` body
            "if false; then :; elif true; then touch {target}; fi",
            # ``else`` body
            "if false; then :; else touch {target}; fi",
            # leading ``!`` negation — strip and process inner command
            "! touch {target}",
            # ``time touch`` (time is a reserved word)
            "time touch {target}",
        ],
    )
    def test_compound_command_inner_write_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)


class TestBacktickCommandSubstitution:
    """Regression: ``echo `touch unowned.py``` runs the inner ``touch``.

    Before the fix, backticks were not preprocessed and shlex left them
    glued to surrounding tokens (``['echo', '`touch', 'unowned.py`']``).
    No extractor matched the backtick-prefixed token, and the unowned
    write slipped past the gate. This is a standard shell-execution
    form, not one of the documented residual obfuscation gaps.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "echo `touch {target}`",
            "echo `touch {target}` more",
            "x=`touch {target}`",
            # Inside double quotes (still substitutes in bash)
            'echo "result: `touch {target}`"',
            # Multiple backtick blocks
            "echo `true` `touch {target}`",
        ],
    )
    def test_backtick_inner_write_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_backtick_in_single_quotes_is_literal(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Inside single quotes, backticks are literal — bash does NOT
        # execute the inner command. The hook must NOT extract a write
        # target from a literal-quoted backtick.
        result = decide(
            make_payload(
                "bash",
                {"command": "echo 'literal `touch foo` text'"},
            )
        )
        assert is_allow(result), result

    def test_backtick_inner_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "out.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"echo `touch {target}`"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 8)
# ---------------------------------------------------------------------------


class TestExecutionWrappers:
    """Regression: ``env``/``sudo``/``exec``/``command``/etc. wrap inner commands.

    Before the fix, ``env touch unowned.py`` was dispatched with cmd
    basename ``env``; no extractor matched and the unowned write was
    silently allowed. Per AC #19, every shell-invoked write must be
    lock-checked regardless of the wrapper used to invoke it.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "env touch {target}",
            "env -i touch {target}",
            "env FOO=bar touch {target}",
            "env -u SECRET touch {target}",
            "command touch {target}",
            "command -p touch {target}",
            "exec touch {target}",
            "exec -a name touch {target}",
            "sudo touch {target}",
            "sudo -u root touch {target}",
            "sudo FOO=bar touch {target}",
            "nice touch {target}",
            "nice -n 10 touch {target}",
            "nohup touch {target}",
            "timeout 5s touch {target}",
            "timeout -s INT 5s touch {target}",
            # Nested: sudo env touch
            "sudo env touch {target}",
            # Path-qualified wrapper
            "/usr/bin/env touch {target}",
        ],
    )
    def test_wrapper_inner_write_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_wrapper_inner_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "out.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(
            make_payload("bash", {"command": f"sudo -u root touch {target}"})
        )
        assert is_allow(result), result


class TestDollarParenSubstitution:
    """Regression: ``"$(touch f)"`` (command sub inside double quotes).

    Before the fix, ``$()`` was only split into segments via the ``(``/
    ``)`` separator path, which is bypassed inside double quotes. An
    agent could ``echo "$(touch unowned.py)"`` and the inner ``touch``
    was never seen because shlex returned ``$(touch unowned.py)`` as a
    single token. ``$()`` is now extracted alongside backticks via
    ``_extract_command_substitutions`` regardless of quote state.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            'echo "$(touch {target})"',
            "echo $(touch {target})",
            'x="$(touch {target})"',
            # Nested $() inside $()
            'echo "$(echo $(touch {target}))"',
            # Mixed with backtick
            'echo "`touch {target}`"',
            # Inside single quotes — literal, NOT executed (covered below).
        ],
    )
    def test_dollar_paren_inner_write_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_dollar_paren_in_single_quotes_is_literal(
        self, env: dict[str, str]
    ) -> None:
        # Inside single quotes, ``$()`` is literal — bash does NOT
        # execute the inner command. The hook must NOT extract a write
        # target.
        result = decide(
            make_payload(
                "bash",
                {"command": "echo 'literal $(touch foo) text'"},
            )
        )
        assert is_allow(result), result

    def test_arithmetic_expansion_not_extracted(self, env: dict[str, str]) -> None:
        # ``$((arith))`` is arithmetic, NOT command substitution. The
        # body has no command to execute and must not be misclassified.
        result = decide(make_payload("bash", {"command": "echo $((1 + 2))"}))
        assert is_allow(result), result


class TestOuterCdAffectsSubstitutions:
    """Regression: ``cd pkg && echo `touch f``` (or ``$(touch f)``).

    Backtick / ``$()`` subshells inherit the parent's current
    directory. Before the fix, the recursive call to extract write
    targets from the substitution body did not know that an outer
    ``cd`` had already run, so the body's relative ``f`` was lock-
    checked against the original payload ``cwd`` — letting an agent
    holding the parent-level lock silently write to the unowned
    subdirectory file.
    """

    def test_outer_cd_invalidates_backtick_relative(
        self, env: dict[str, str], repo: Path
    ) -> None:
        sub = repo / "pkg"
        sub.mkdir()
        # Lock root-level f.txt (would be matched if hook ignored cd).
        assert try_lock(str(repo / "f.txt"), "agent-me", repo_namespace=str(repo))
        command = "cd pkg && echo `touch f.txt`"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        # The cd invalidates resolution; touch in backtick body should
        # also be flagged unresolved → deny.
        assert is_deny(result), result

    def test_outer_cd_invalidates_dollar_paren_relative(
        self, env: dict[str, str], repo: Path
    ) -> None:
        sub = repo / "pkg"
        sub.mkdir()
        assert try_lock(str(repo / "f.txt"), "agent-me", repo_namespace=str(repo))
        command = "cd pkg && echo $(touch f.txt)"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    def test_outer_cd_then_absolute_in_backtick_allowed(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # An ABSOLUTE path inside the backtick body is unambiguous;
        # the hook should still allow it when the agent holds the lock.
        target = repo / "out.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"cd /tmp && echo `touch {target}`"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result

    def test_no_outer_cd_backtick_relative_resolves_normally(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: without an outer cd, relative paths inside backticks
        # resolve against the payload cwd as before.
        target = repo / "out.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = "echo `touch out.txt`"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 9)
# ---------------------------------------------------------------------------


class TestBooleanLongWrapperFlags:
    """Regression: boolean wrapper long flags must NOT consume the inner cmd.

    Before the fix, ``_skip_wrapper_flag_args`` heuristically treated
    every ``--flag`` (without ``=``) as consuming the next non-``-``
    token. Boolean long flags like ``sudo --preserve-env`` or
    ``env --ignore-environment`` then swallowed the inner ``touch``,
    leaving the heuristic with the trailing args (``unowned.py``)
    headed by what it thought was the cmd — no extractor matched and
    the unlocked write slipped past the gate.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "sudo --preserve-env touch {target}",
            "sudo --background touch {target}",
            "sudo --login touch {target}",
            "sudo --remove-timestamp touch {target}",
            "env --ignore-environment touch {target}",
            "env -i touch {target}",
            "env --null touch {target}",
            "exec --cleanup touch {target}",
            # Mixed: long boolean + short value
            "sudo --preserve-env -u root touch {target}",
        ],
    )
    def test_boolean_long_wrapper_flag_inner_write_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_value_long_wrapper_flag_still_consumes_value(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``sudo --user root touch f`` — ``--user`` IS in the value-long
        # table and should still consume ``root``. The inner ``touch``
        # then dispatches correctly and (with no lock) denies.
        target = repo / "unowned.py"
        result = decide(
            make_payload(
                "bash",
                {"command": f"sudo --user root touch {target}"},
            )
        )
        assert is_deny(result)
        assert "unowned.py" in deny_reason(result)

    def test_value_long_wrapper_flag_with_eq_consumes_inline(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``sudo --user=root touch f`` — value embedded in the same
        # token; no separate-token consumption. Inner ``touch`` runs.
        target = repo / "unowned.py"
        result = decide(
            make_payload(
                "bash",
                {"command": f"sudo --user=root touch {target}"},
            )
        )
        assert is_deny(result)
        assert "unowned.py" in deny_reason(result)


class TestNodeRequireFsWrite:
    """Regression: ``node -e "require('fs').writeFileSync(...)"``.

    Before the fix, the Node oneliner regex only matched calls
    starting with ``fs.`` (after ``const fs = require('fs')``). The
    ``require('fs').writeFileSync(...)`` literal-write form had a
    static target but matched neither the regex nor the hint list, so
    the heuristic returned no targets and the unowned write was
    silently allowed.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "node -e \"require('fs').writeFileSync('{target}','x')\"",
            'node -e "require(\\"fs\\").writeFileSync(\\"{target}\\",\\"x\\")"',
            "node -e \"require('fs').appendFileSync('{target}','x')\"",
            "node -e \"require('fs').createWriteStream('{target}')\"",
            # node:fs prefix
            "node -e \"require('node:fs').writeFileSync('{target}','x')\"",
            # nodejs alias
            "nodejs -e \"require('fs').writeFileSync('{target}','x')\"",
        ],
    )
    def test_require_fs_write_denied_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "out.txt"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "out.txt" in deny_reason(result)

    def test_require_fs_dynamic_target_denied(self, env: dict[str, str]) -> None:
        # Dynamic target (variable) — heuristic detects the require()
        # hint and emits the unresolved sentinel → deny.
        result = decide(
            make_payload(
                "bash",
                {"command": ("node -e \"require('fs').writeFileSync(VAR,'x')\"")},
            )
        )
        assert is_deny(result)


class TestQuotedRedirectArgNotFalseDenied:
    """P2: ``echo ">foo"`` is a positional arg, NOT a redirect.

    Before the fix, ``_extract_redirection_targets`` applied a combined-
    prefix fallback regex that matched the leading ``>`` of any token
    and treated the rest as a write target. That false-positive denied
    legitimate commands that pass ``>``-prefixed arguments to a tool
    (printing them, grepping for them, etc.). After the normalizer pads
    real unquoted redirects, any ``>file``-shaped token reaching the
    extractor must have been quoted/escaped — so the combined fallback
    is removed.
    """

    @pytest.mark.parametrize(
        "command",
        [
            'echo ">foo"',
            'echo ">>foo"',
            'echo "&>foo"',
            'echo "<>foo"',
            "grep '>5' file.txt",
            'printf "%s\\n" ">file"',
        ],
    )
    def test_quoted_redirect_literal_does_not_false_deny(
        self, env: dict[str, str], command: str
    ) -> None:
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), f"command: {command!r} → {result}"

    def test_real_unquoted_redirect_still_denied(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: an actual unquoted redirect must still be detected.
        # The normalizer pads it before shlex sees it.
        target = repo / "unowned.txt"
        result = decide(make_payload("bash", {"command": f"echo hi>{target}"}))
        assert is_deny(result)
        assert "unowned.txt" in deny_reason(result)


class TestUtilityValueFlagSkip:
    """P2: ``mkdir -m 755 dir`` and ``touch -d "time" file`` value-flag values.

    Before the fix, ``-m`` / ``-d`` / ``-r`` / ``-t`` (touch) were
    treated as plain flags and SKIPPED, but the next token (``755``,
    ``"time"``, ``ref``) was added to ``positionals`` and lock-checked
    by the ``all_positional`` strategy — guaranteeing a false-positive
    deny on a legit, fully-locked command.
    """

    def test_mkdir_with_mode_does_not_lock_check_mode(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "newdir"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("bash", {"command": f"mkdir -m 755 {target}"}))
        assert is_allow(result), result

    def test_touch_with_date_flag_does_not_lock_check_date(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "f.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(
            make_payload(
                "bash",
                {"command": f'touch -d "2026-01-01" {target}'},
            )
        )
        assert is_allow(result), result

    def test_touch_with_reference_flag_does_not_lock_check_ref(
        self, env: dict[str, str], repo: Path, tmp_path: Path
    ) -> None:
        # ``touch -r ref-file target`` — ``-r`` consumes ``ref-file``;
        # only ``target`` is the write surface.
        target = repo / "f.txt"
        ref = tmp_path / "ref"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(
            make_payload(
                "bash",
                {"command": f"touch -r {ref} {target}"},
            )
        )
        assert is_allow(result), result

    def test_mkdir_with_value_flag_still_denies_target(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: the value-flag skip must NOT cause the actual target
        # to be missed.
        result = decide(
            make_payload(
                "bash",
                {"command": "mkdir -m 755 unowned-dir"},
            )
        )
        assert is_deny(result)
        assert "unowned-dir" in deny_reason(result)


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 10)
# ---------------------------------------------------------------------------


class TestTimeWrapper:
    """Regression: ``time -p touch f`` and ``/usr/bin/time touch f``.

    ``time`` was previously in ``_SHELL_RESERVED_WORDS`` (stripped
    from segment heads). That handled the bare ``time touch f`` form
    but failed for both ``time -p touch f`` (which left ``-p`` as the
    cmd basename) and ``/usr/bin/time touch f`` (path-qualified, so
    the reserved-word check never matched the basename). Moved
    ``time`` to ``_EXECUTION_PREFIXES`` with proper option handling.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "time touch {target}",
            "time -p touch {target}",
            "time -v touch {target}",
            "time --portability touch {target}",
            "time --verbose touch {target}",
            # Path-qualified — only catches with execution-prefix
            # unwrapping (basename match against _EXECUTION_PREFIXES).
            "/usr/bin/time touch {target}",
            "/usr/bin/time -p touch {target}",
        ],
    )
    def test_time_wrapper_inner_write_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    @pytest.mark.parametrize(
        "command_template",
        [
            "time -o /tmp/timing.log touch {target}",
            "time --output=/tmp/timing.log touch {target}",
            "time --output /tmp/timing.log touch {target}",
        ],
    )
    def test_time_output_flag_target_lock_checked(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        # ``time -o /path/log cmd`` writes timing data to ``/path/log``.
        # The wrapper output AND the inner command's writes must both
        # be lock-checked. With both unlocked, the deny may reference
        # either path; both are correct outcomes.
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        reason = deny_reason(result)
        assert "unowned.py" in reason or "/tmp/timing.log" in reason, reason

    def test_time_wrapper_inner_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "out.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("bash", {"command": f"time -p touch {target}"}))
        assert is_allow(result), result


class TestWrapperChdirOption:
    """Regression: ``env -C dir cmd`` and ``env --chdir=dir cmd``.

    Coreutils 8.32+'s ``env -C dir`` runs the inner command with
    cwd=``dir``. Before the fix, ``_unwrap_execution_prefix`` stripped
    the option silently and the inner command's relative writes
    resolved against the original payload cwd — letting an agent
    holding the parent-level lock write to the unowned subdirectory
    file.
    """

    def test_env_chdir_short_form_invalidates_relative(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # The agent holds the lock on cwd-level f.txt but env -C pkg
        # changes cwd to repo/pkg before running touch.
        sub = repo / "pkg"
        sub.mkdir()
        assert try_lock(
            str(repo / "generated.py"), "agent-me", repo_namespace=str(repo)
        )
        command = "env -C pkg touch generated.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    def test_env_chdir_long_form_invalidates_relative(
        self, env: dict[str, str], repo: Path
    ) -> None:
        sub = repo / "pkg"
        sub.mkdir()
        assert try_lock(
            str(repo / "generated.py"), "agent-me", repo_namespace=str(repo)
        )
        command = "env --chdir pkg touch generated.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    def test_env_chdir_eq_long_form_invalidates_relative(
        self, env: dict[str, str], repo: Path
    ) -> None:
        sub = repo / "pkg"
        sub.mkdir()
        assert try_lock(
            str(repo / "generated.py"), "agent-me", repo_namespace=str(repo)
        )
        command = "env --chdir=pkg touch generated.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    def test_env_no_chdir_relative_resolves_normally(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: ``env touch f`` (no chdir flag) still resolves
        # against the payload cwd.
        target = repo / "out.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = "env touch out.txt"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result


class TestRedirectToDigitFilenameThenBackground:
    """Regression: ``echo hi >123&`` writes ``123``, then backgrounds.

    Before the fix, the normalizer's ``&`` look-back walked through
    digits and matched the ``>`` of a normal redirect, treating the
    ``&`` as part of an fd-redirect operator. The literal filename
    ``123`` then absorbed the trailing ``&`` (no surrounding spaces)
    and the lock was looked up for the bogus path ``123&`` — which an
    agent could lock as a junk path while bash silently wrote to
    ``123`` in the background.
    """

    def test_digit_filename_with_trailing_background(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "123"
        # Lock the REAL file (123); junk lock on "123&" would have
        # bypassed the gate before the fix.
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"echo hi >{target}&"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result

    def test_digit_filename_with_trailing_background_unlocked_denies(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "123"
        command = f"echo hi >{target}&"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)
        # The deny must reference the actual file, not a junk "123&"
        # path.
        assert "/123" in deny_reason(result)
        assert "123&" not in deny_reason(result)

    def test_real_fd_redirect_still_recognised(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: ``cmd 2>&1`` is fd dup, must still allow.
        target = repo / "out.log"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"cmd >{target} 2>&1"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result


class TestEndOfOptionsMarker:
    """Regression: ``touch -- -owned.py`` writes ``-owned.py``.

    Before the fix, every later token starting with ``-`` was treated
    as an option. After ``--``, dash-prefixed tokens are operands —
    bash writes to ``-owned.py`` but the heuristic skipped it as a
    flag and emitted no targets, allowing the unlocked write.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "touch -- {target}",
            "rm -- {target}",
            "mkdir -- {target}",
            "cp /tmp/src -- {target}",
            "mv /tmp/src -- {target}",
        ],
    )
    def test_dash_prefixed_operand_after_dashdash_denies(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "-owned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "-owned.py" in deny_reason(result)

    def test_dashdash_operand_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "-owned.py"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"touch -- {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result


class TestUtilityLongValueFlags:
    """P2: ``touch --date "2026-01-01" file`` and ``mkdir --mode 755 dir``.

    Before the fix, long flags were skipped (``i += 1``) but their
    separate-token values were left in the stream and added to
    ``positionals``. For all_positional utilities this caused a
    false-positive lock-check on the value literal.
    """

    def test_touch_long_date_flag_does_not_lock_check_value(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "f.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(
            make_payload(
                "bash",
                {"command": f'touch --date "2026-01-01" {target}'},
            )
        )
        assert is_allow(result), result

    def test_mkdir_long_mode_flag_does_not_lock_check_value(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "newdir"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(
            make_payload(
                "bash",
                {"command": f"mkdir --mode 755 {target}"},
            )
        )
        assert is_allow(result), result

    def test_touch_long_reference_flag_does_not_lock_check_value(
        self, env: dict[str, str], repo: Path, tmp_path: Path
    ) -> None:
        target = repo / "f.txt"
        ref = tmp_path / "ref"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(
            make_payload(
                "bash",
                {"command": f"touch --reference {ref} {target}"},
            )
        )
        assert is_allow(result), result

    def test_mkdir_long_mode_eq_form_works(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``--mode=755`` (with =) — value embedded, no separate-token
        # consumption. The existing ``= in tok`` branch handles this.
        target = repo / "newdir"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(
            make_payload(
                "bash",
                {"command": f"mkdir --mode=755 {target}"},
            )
        )
        assert is_allow(result), result


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 1, post-restart)
# ---------------------------------------------------------------------------


class TestInstallDirectoryMode:
    """Regression: ``install -d dir1 dir2`` creates ALL listed dirs.

    Before the fix, install used ``last_positional`` so ``install -d
    unowned_dir owned_dir`` only lock-checked ``owned_dir``. An agent
    could place a single owned dir at the end of the list to slip
    arbitrary unowned dirs past the gate.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "install -d unowned_a unowned_b {target}",
            "install --directory unowned_a unowned_b {target}",
            "install --directory=755 unowned_a unowned_b {target}",
            "install -fvd unowned_a unowned_b {target}",
        ],
    )
    def test_install_d_lock_checks_all_dirs(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        # Lock only the trailing dir; install -d would create the
        # leading unowned dirs too. Must deny.
        target = repo / "owned_dir"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"

    def test_install_d_all_locked_allowed(
        self, env: dict[str, str], repo: Path
    ) -> None:
        a = repo / "a"
        b = repo / "b"
        for p in (a, b):
            assert try_lock(str(p), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("bash", {"command": f"install -d {a} {b}"}))
        assert is_allow(result), result

    def test_install_without_d_still_last_positional(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: ``install /tmp/src dest`` (no -d) is the file-copy
        # mode where dest is the only write target.
        dest = repo / "dest"
        assert try_lock(str(dest), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("bash", {"command": f"install /tmp/src {dest}"}))
        assert is_allow(result), result


class TestLnSingleArgForm:
    """Regression: ``ln -s /path/source`` creates ``./source`` link.

    Before the fix, ``ln`` used ``last_positional`` and extracted the
    SOURCE path. An agent owning the source could create a link in
    cwd named after the basename without any lock check.
    """

    def test_ln_single_arg_creates_link_in_cwd(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # The link is created at ./source.txt (resolved against cwd).
        # Lock the cwd-resolved path to prove the heuristic resolves
        # correctly.
        target = repo / "source.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = "ln -s /elsewhere/source.txt"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result

    def test_ln_single_arg_unlocked_link_denies(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # No lock on ./source.txt → must deny. Locking the SOURCE
        # path (/elsewhere/...) would have allowed before the fix.
        result = decide(
            make_payload(
                "bash",
                {"command": "ln -s /elsewhere/source.txt"},
                cwd=repo,
            )
        )
        assert is_deny(result)
        assert "source.txt" in deny_reason(result)

    def test_ln_two_arg_form_still_uses_dest(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: ``ln src dest`` still uses last_positional → dest.
        dest = repo / "dest.txt"
        assert try_lock(str(dest), "agent-me", repo_namespace=str(repo))
        result = decide(
            make_payload(
                "bash",
                {"command": f"ln /tmp/src {dest}"},
            )
        )
        assert is_allow(result), result


class TestTimeOutputFlagWrite:
    """Regression: ``time -o file cmd`` writes timing output to ``file``.

    Before the fix, ``time -o file`` consumed both flag tokens
    silently. The output file is a real write surface that must be
    lock-checked alongside the inner command's writes.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "time -o {target} ls",
            "time --output={target} ls",
            "time --output {target} ls",
            "time -p -o {target} ls",
        ],
    )
    def test_time_output_unlocked_denies(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "timing.log"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "timing.log" in deny_reason(result)

    def test_time_output_locked_inner_readonly_allows(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "timing.log"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("bash", {"command": f"time -o {target} ls"}))
        assert is_allow(result), result


class TestSudoChdirOption:
    """Regression: ``sudo --chdir dir`` and ``sudo -D dir`` change cwd.

    Before the fix, sudo's chdir flag was tracked in the value-flag
    table (so the value got consumed) but NOT in the cwd-change
    tables, so the inner command's relative writes resolved against
    the original payload cwd — letting an agent holding the parent-
    level lock slip past while bash actually wrote the subdirectory
    file.
    """

    def test_sudo_long_chdir_invalidates_relative(
        self, env: dict[str, str], repo: Path
    ) -> None:
        sub = repo / "pkg"
        sub.mkdir()
        assert try_lock(
            str(repo / "generated.py"), "agent-me", repo_namespace=str(repo)
        )
        command = "sudo --chdir pkg touch generated.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    def test_sudo_short_chdir_invalidates_relative(
        self, env: dict[str, str], repo: Path
    ) -> None:
        sub = repo / "pkg"
        sub.mkdir()
        assert try_lock(
            str(repo / "generated.py"), "agent-me", repo_namespace=str(repo)
        )
        command = "sudo -D pkg touch generated.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    def test_sudo_long_chdir_eq_invalidates_relative(
        self, env: dict[str, str], repo: Path
    ) -> None:
        sub = repo / "pkg"
        sub.mkdir()
        assert try_lock(
            str(repo / "generated.py"), "agent-me", repo_namespace=str(repo)
        )
        command = "sudo --chdir=pkg touch generated.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result


class TestInplaceDashDash:
    """Regression: ``perl -i -e '...' -- -owned.py`` writes ``-owned.py``.

    Before the fix, the in-place extractors skipped every dash-prefixed
    token as a flag. After ``--``, dash-prefixed tokens are operands —
    bash writes to ``-owned.py`` but the heuristic emitted no targets.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "sed -i s/a/b/ -- {target}",
            "perl -i -e 's/a/b/' -- {target}",
            "awk -i inplace -- '{{print}}' {target}",
        ],
    )
    def test_inplace_dashdash_operand_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "-owned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "-owned.py" in deny_reason(result)


class TestEnvAssignmentAfterReservedWord:
    """Regression: ``then VAR=val touch f`` strips both prefix layers.

    Before the fix, env-assignments were stripped BEFORE reserved
    words. ``if true; then VAR=val touch f; fi`` produced a segment
    ``['then', 'VAR=val', 'touch', 'f']`` whose head was ``then`` (so
    env-strip was a no-op). After reserved-strip removed ``then``,
    ``VAR=val`` was at the head but never re-stripped. The extractor
    saw ``VAR=val`` as the cmd basename and emitted no targets.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "if true; then VAR=val touch {target}; fi",
            "while false; do FOO=bar touch {target}; done",
            "! VAR=val touch {target}",
            "for x in 1; do VAR=val touch {target}; done",
        ],
    )
    def test_reserved_then_env_assignment_inner_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)


class TestBuiltinCdAndPushd:
    """Regression: ``builtin cd`` / ``pushd`` / ``popd`` change cwd.

    Before the fix, only literal ``cd`` was recognised. Bash also
    supports ``builtin cd dir`` (explicit builtin invocation) and
    ``pushd dir`` / ``popd`` (dir-stack manipulation). All three
    change the shell cwd; their relative writes inherit the change.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "pushd pkg && touch generated.py",
            "pushd /elsewhere && touch generated.py",
            "builtin cd pkg && touch generated.py",
            "builtin pushd pkg && touch generated.py",
            # popd also changes cwd (pops the stack).
            "popd && touch generated.py",
        ],
    )
    def test_builtin_cd_invalidates_relative(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        # Lock the cwd-level path; if the heuristic ignored the cd,
        # it would resolve against payload cwd and allow.
        sub = repo / "pkg"
        sub.mkdir(exist_ok=True)
        assert try_lock(
            str(repo / "generated.py"), "agent-me", repo_namespace=str(repo)
        )
        command = command_template
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), f"command: {command!r} → {result}"


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 2, post-restart)
# ---------------------------------------------------------------------------


class TestBooleanWrapperFlags:
    """Regression: ``sudo -A`` and ``unshare --user`` are BOOLEAN.

    Before the fix, both wrappers had these boolean flags listed in
    their value-flag tables. The flags would consume the inner cmd as
    their "value" and the heuristic would see e.g. ``unowned.py`` as
    the cmd basename — silently allowing the unowned write.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            # sudo boolean flags: -A askpass, -b background, -E preserve-env,
            # -H set-home, -i login, -K kill-timestamp, -k reset-timestamp,
            # -l list, -n non-interactive, -P preserve-groups, -S stdin,
            # -s shell, -V version, -v validate
            "sudo -A touch {target}",
            "sudo -b touch {target}",
            "sudo -E touch {target}",
            "sudo -H touch {target}",
            "sudo -n touch {target}",
            "sudo -S touch {target}",
            # unshare boolean namespace flags: -u uts, -i ipc, -m mount,
            # -n net, -p pid, -U user, -r map-root, -T time, -C cgroup,
            # -l ipc/list (deprecated short for some)
            "unshare --user touch {target}",
            "unshare -U touch {target}",
            "unshare -u touch {target}",
            "unshare --mount touch {target}",
            "unshare -m touch {target}",
            "unshare -n touch {target}",
            "unshare --net touch {target}",
            "unshare -p touch {target}",
            "unshare --pid touch {target}",
            "unshare -r touch {target}",
            "unshare --map-root-user touch {target}",
        ],
    )
    def test_boolean_wrapper_flag_inner_write_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_unshare_value_flag_still_consumes(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``--setuid 0`` IS value-taking; the inner cmd then dispatches
        # correctly.
        target = repo / "unowned.py"
        result = decide(
            make_payload(
                "bash",
                {"command": f"unshare --setuid 0 touch {target}"},
            )
        )
        assert is_deny(result)
        assert "unowned.py" in deny_reason(result)


class TestChrtTasksetWrapper:
    """Regression: ``chrt`` and ``taskset`` need their leading positional dropped.

    ``chrt PRIORITY cmd`` and ``taskset MASK cmd`` put a numeric/hex
    positional before the inner command. Before the fix, only
    ``timeout DURATION cmd`` had its leading positional dropped, so
    ``chrt 10 touch f`` and ``taskset 0x1 touch f`` left ``10`` and
    ``0x1`` as the cmd basename — no extractor matched.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "chrt 10 touch {target}",
            "chrt --rr 5 touch {target}",
            "taskset 0x1 touch {target}",
            "taskset --cpu-list 0,2 touch {target}",
            # Nested with another wrapper.
            "sudo chrt 10 touch {target}",
        ],
    )
    def test_chrt_taskset_inner_write_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)


class TestInstallBackupOptionalArg:
    """Regression: ``install --backup`` takes an OPTIONAL argument.

    Before the fix, ``--backup`` was in install's value-long-flag
    table, so ``install -d --backup dir1 dir2`` would consume ``dir1``
    as the backup-control value — the heuristic then only saw
    ``dir2`` and missed the lock-check on the unowned ``dir1``.
    """

    def test_install_backup_does_not_swallow_dir(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``--backup`` alone has no value; ``dir1 dir2`` are both
        # positional dirs to create. Locking only ``dir2`` should
        # still deny because ``dir1`` is unowned.
        a = repo / "dir1"
        b = repo / "dir2"
        assert try_lock(str(b), "agent-me", repo_namespace=str(repo))
        command = f"install -d --backup {a} {b}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), result
        assert "dir1" in deny_reason(result)

    def test_install_backup_with_eq_value_still_works(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``--backup=numbered`` has the value embedded in the same
        # token; the heuristic must NOT consume the next token.
        a = repo / "dir1"
        b = repo / "dir2"
        for p in (a, b):
            assert try_lock(str(p), "agent-me", repo_namespace=str(repo))
        command = f"install -d --backup=numbered {a} {b}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result


class TestShellGlobMetacharacters:
    """Regression: ``rm *.py`` cannot be statically resolved to files.

    Before the fix, the heuristic treated the literal ``*.py`` as a
    write target. An agent could acquire a lock on the literal pattern
    (``$cwd/*.py``) — the lock-check would succeed and the hook would
    allow ``rm *.py`` while bash expanded it to every matching .py
    file at runtime.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "rm *.py",
            "rm /tmp/*.log",
            "touch a.txt b.txt c[0-9].txt",
            "touch {a,b,c}.txt",
            "cp src *.bak",
            "cp src ?.txt",
            "mkdir test[12]",
        ],
    )
    def test_glob_target_always_denies(
        self, env: dict[str, str], repo: Path, command: str
    ) -> None:
        # Even if the agent acquires a lock on the literal pattern,
        # the hook must still deny — the actual files bash expands
        # cannot be enumerated statically.
        # Pre-acquire a junk lock on the pattern to prove that
        # locking the literal does NOT bypass the gate.
        from src.infra.tools.locking import try_lock as try_lock_

        # Try locking the most likely-relevant glob pattern from the
        # command (best-effort; we don't parse it ourselves here).
        for tok in command.split():
            if any(g in tok for g in "*?[]{}"):
                try_lock_(tok, "agent-me", repo_namespace=str(repo))
                break

        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), f"command: {command!r} → {result}"

    def test_non_glob_path_unchanged(self, env: dict[str, str], repo: Path) -> None:
        # Sanity: a plain literal path with NO globs flows through
        # normally (locked → allow, unlocked → deny).
        target = repo / "plain.py"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("bash", {"command": f"touch {target}"}))
        assert is_allow(result), result


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 3, post-restart)
# ---------------------------------------------------------------------------


class TestTildeAndParameterExpansion:
    """Regression: ``~/foo`` and ``$VAR`` cannot be statically resolved.

    Before the fix, ``touch ~/unowned.py`` had its target lock-checked
    against the literal ``~/unowned.py`` (resolved against cwd). An
    agent could acquire a lock on that exact string while bash
    expanded ``~`` to ``/home/user`` at runtime — writing to a file
    that was never lock-checked.

    Same shape for ``$VAR`` parameter expansion and command-
    substitution placeholders.
    """

    @pytest.mark.parametrize(
        "command",
        [
            # Tilde expansion at start of path.
            "touch ~/unowned.py",
            "touch ~/.bashrc",
            "rm ~/unowned.py",
            "cp /tmp/src ~/dest.py",
            # Mid-path tilde (bash does NOT expand here, but we deny
            # conservatively since we cannot tell if the original was
            # quoted).
            "touch /tmp/~/foo",
            # Parameter expansion.
            "touch $TARGET",
            "touch $HOME/foo",
            "touch ${VAR}",
            "touch ${VAR}/path",
            "rm $1",
            "echo hi > $OUT",
            # Command-substitution placeholder reaching the lock-check.
            # ``cp src $(echo dest)`` — the $(...) was extracted as a
            # body, the surrounding token has the placeholder.
            "cp src $(echo dest)",
            "touch `echo dest`",
        ],
    )
    def test_expansion_target_always_denies(
        self, env: dict[str, str], repo: Path, command: str
    ) -> None:
        # Even if the agent acquires a lock on the literal expansion
        # string, the hook must still deny.
        from src.infra.tools.locking import try_lock as try_lock_

        for tok in command.split():
            if any(c in tok for c in "$~"):
                try_lock_(tok, "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), f"command: {command!r} → {result}"

    def test_dollar_at_in_filename_still_denied(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Even ``touch $$.tmp`` (bash $$ = PID) denies.
        result = decide(make_payload("bash", {"command": "touch $$.tmp"}, cwd=repo))
        assert is_deny(result)


class TestPatchHeaderTimestamp:
    """Regression: ``+++ b/path\\t2026-05-08`` header includes timestamp.

    Before the fix, ``_DIFF_DEST_HEADER_RE`` captured ``path\\t2026-05-08``
    as the path string. An agent could lock that literal junk string
    while ``patch(1)`` ignored everything after the tab and wrote to
    ``path``.
    """

    def test_diff_header_with_timestamp_extracts_path_only(
        self, env: dict[str, str], repo: Path
    ) -> None:
        body = (
            "diff --git a/src/foo.py b/src/foo.py\n"
            "--- a/src/foo.py\t2026-05-08 12:00:00.000\n"
            "+++ b/src/foo.py\t2026-05-08 12:00:01.000\n"
            "@@\n-x\n+y\n"
        )
        # Lock the actual file (without timestamp). If the heuristic
        # extracted ``src/foo.py\t2026-05-08...`` it would deny here.
        target = repo / "src" / "foo.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("apply_patch", {"patch": body}, cwd=repo))
        assert is_allow(result), result

    def test_diff_header_with_timestamp_unlocked_denies_real_path(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # No locks; deny message must reference the real path, not
        # ``path\t2026-...``.
        body = (
            "--- a/src/foo.py\t2026-05-08\n+++ b/src/foo.py\t2026-05-08\n@@\n-x\n+y\n"
        )
        result = decide(make_payload("apply_patch", {"patch": body}, cwd=repo))
        assert is_deny(result)
        assert "src/foo.py" in deny_reason(result)
        assert "2026-05-08" not in deny_reason(result)

    def test_envelope_header_with_tab_extracts_path_only(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Codex apply_patch envelope header tolerant of trailing tab/text.
        body = "*** Update File: src/foo.py\textra-data\n@@\n-x\n+y\n"
        target = repo / "src" / "foo.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("apply_patch", {"command": body}, cwd=repo))
        assert is_allow(result), result


class TestAttachedFdPrefix:
    """Regression: ``unowned_dir123>out`` keeps ``123`` in the filename.

    Before the fix, ``_strip_fd_prefix_from_output`` unconditionally
    popped trailing digits from the output buffer when emitting a
    redirect operator. For ``cp src -t unowned_dir123>out``, the ``123``
    was stripped and treated as an fd prefix, leaving the heuristic
    to lock-check ``unowned_dir`` and ``out`` separately while bash
    silently created the unowned ``unowned_dir123``.
    """

    def test_attached_digit_prefix_stays_in_filename(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Trigger the buggy code path: ``cp src -t {target}>out``
        # invokes ``_strip_fd_prefix_from_output`` (only fires when
        # the normalizer emits a ``>`` operator). The bug stripped
        # the trailing ``123`` from ``unowned_dir123`` and emitted
        # the operator as ``123>``; the heuristic then lock-checked
        # ``unowned_dir`` (the digit-stripped name) and ``out``
        # (the redirect target). Pre-locking BOTH of those wrong
        # paths would have allowed the command — but the actual
        # write goes to ``unowned_dir123``.
        from src.infra.tools.locking import try_lock as try_lock_

        target = repo / "unowned_dir123"
        out_file = repo / "out"
        # Lock the digit-stripped (WRONG) name and the redirect
        # target, simulating an agent that has misidentified the
        # write surfaces. Before the fix, both would match and the
        # command would be allowed.
        try_lock_(str(repo / "unowned_dir"), "agent-me", repo_namespace=str(repo))
        try_lock_(str(out_file), "agent-me", repo_namespace=str(repo))
        command = f"cp src -t {target}>{out_file}"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        # Must deny: ``unowned_dir123`` is the real ``-t`` destination,
        # and the agent did NOT lock that path.
        assert is_deny(result), result
        assert "unowned_dir123" in deny_reason(result)

    def test_attached_digit_with_redirect_still_correct(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``echo hi >file123`` — no space between operator and file.
        # The ``123`` is part of the filename, not an fd prefix.
        target = repo / "file123"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"echo hi >{target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result

    def test_word_then_digits_then_redirect(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``echo word123 >out`` — the word123 is an arg, not an fd.
        target = repo / "out"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"echo word123 >{target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result

    def test_real_fd_prefix_with_whitespace_still_works(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: ``echo hi 2>file`` with `2>` separated from prior
        # word by whitespace is a real fd redirect.
        target = repo / "stderr.log"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"echo hi 2>{target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 4, post-restart)
# ---------------------------------------------------------------------------


class TestNestedBacktickSubstitutions:
    """Regression: ``echo \\`echo \\\\`touch unowned\\\\`\\``` (nested backticks).

    Before the fix, the outer backtick body was extracted with the
    inner ``\\``` escapes preserved. The recursive call then treated
    those backslashes as literal escapes and skipped the inner
    backticks, missing the nested ``touch`` entirely.
    """

    def test_nested_backticks_inner_write_denied(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "unowned.py"
        # Outer backticks contain escaped inner backticks: `... \`...\` ...`
        command = f"echo `echo \\`touch {target}\\``"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)


class TestArithmeticContainsCmdSub:
    """Regression: ``echo $(( $(touch unowned) + 1 ))`` runs touch.

    Before the fix, ``$((arith))`` was passed through verbatim by the
    cmd-substitution preprocessor. Bash, however, evaluates command
    substitutions and backticks INSIDE the arithmetic body before
    computing the result — so the inner ``touch`` runs.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            # Modern $() inside arithmetic
            "echo $(( $(touch {target}) + 1 ))",
            # Backtick inside arithmetic
            "echo $(( `touch {target}; echo 1` + 1 ))",
            # Nested arithmetic with cmd sub
            "echo $(( 2 * $(touch {target}; echo 3) ))",
        ],
    )
    def test_arithmetic_inner_cmd_sub_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)


class TestDiffHeaderSpaceTimestamp:
    """P2: ``+++ b/file 2026-05-08`` (space-separated timestamp).

    GNU patch truncates the path at the first whitespace. Before the
    fix, the regex captured the whole ``file 2026-05-08`` string; an
    agent could lock that literal junk while patch wrote to ``file``.
    """

    def test_diff_header_with_space_timestamp_extracts_path_only(
        self, env: dict[str, str], repo: Path
    ) -> None:
        body = (
            "diff --git a/src/foo.py b/src/foo.py\n"
            "--- a/src/foo.py 2026-05-08 12:00:00\n"
            "+++ b/src/foo.py 2026-05-08 12:00:01\n"
            "@@\n-x\n+y\n"
        )
        target = repo / "src" / "foo.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("apply_patch", {"patch": body}, cwd=repo))
        assert is_allow(result), result

    def test_diff_header_space_timestamp_unlocked_denies_real_path(
        self, env: dict[str, str], repo: Path
    ) -> None:
        body = "--- a/src/foo.py 2026-05-08\n+++ b/src/foo.py 2026-05-08\n@@\n-x\n+y\n"
        result = decide(make_payload("apply_patch", {"patch": body}, cwd=repo))
        assert is_deny(result)
        assert "src/foo.py" in deny_reason(result)
        # The deny reason must reference the real path, not the
        # date-suffixed string.
        assert "2026-05-08" not in deny_reason(result)


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 5, post-restart)
# ---------------------------------------------------------------------------


class TestEnvSplitString:
    """Regression: ``env -S 'touch unowned'`` runs the inner command.

    Before the fix, the ``-S`` flag was treated as a regular value-
    consuming flag — the value was discarded and the heuristic
    continued with whatever followed. ``env -S 'touch unowned.py'``
    has nothing after the value, so the wrapper-strip yielded an
    empty token list and the inner ``touch`` was never seen. The
    same shape via ``--split-string=`` was also bypassed.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            # Short form, separate-token value.
            "env -S 'touch {target}'",
            # Long form, attached value.
            "env --split-string='touch {target}'",
            # Long form, separate value.
            "env --split-string 'touch {target}'",
            # Mixed with other flags.
            "env -i -S 'touch {target}'",
            "env --ignore-environment -S 'touch {target}'",
            # Followed by a normal cmd (still must extract -S body).
            "env -S 'touch {target}' /bin/true",
            # Nested wrapper (sudo wrapping env -S).
            "sudo env -S 'touch {target}'",
        ],
    )
    def test_env_split_string_inner_write_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_env_split_string_locked_allows(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "out.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"env -S 'touch {target}'"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result

    def test_env_split_string_with_redirect(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # The body itself contains a redirect; it must be detected.
        target = repo / "out.txt"
        command = f"env -S 'echo hi > {target}'"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)
        assert "out.txt" in deny_reason(result)

    def test_env_without_s_unaffected(self, env: dict[str, str], repo: Path) -> None:
        # Sanity: ``env touch f`` (no -S) still works via the regular
        # wrapper unwrap path.
        target = repo / "out.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"env touch {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 6, post-restart)
# ---------------------------------------------------------------------------


class TestDoasAuthStyleFlag:
    """Regression: ``doas -a auth_style touch unowned`` consumes ``auth_style``.

    Before the fix, ``a`` was missing from doas's value-letter table.
    The wrapper-strip then treated ``-a`` as boolean and consumed the
    inner ``touch`` as the auth-style value, leaving ``unowned`` as
    the cmd basename — no extractor matched, write was allowed.
    """

    def test_doas_a_flag_value_consumed_correctly(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "unowned.py"
        command = f"doas -a persist touch {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)


class TestBraceExpansion:
    """Regression: ``{touch,unowned.py}`` is brace expansion.

    Before the fix, ``{touch,unowned.py}`` was one shlex token; the
    extractor saw ``{touch,unowned.py}`` as the cmd basename and no
    write was detected. Bash brace-expands to ``touch unowned.py``
    and runs the write.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "{touch,unowned.py}",
            "{touch,/tmp/unowned}",
            "{cp,/tmp/src,unowned}",
            # Range expansion
            "{touch,a{1..3}.txt}",
            # Nested braces
            "{cp,src,{a,b}}",
            # arg-position brace expansion (already covered by
            # ``_check_lock`` glob check, but should be uniformly denied)
            "touch a{b,c}.txt",
        ],
    )
    def test_brace_expansion_command_denied(
        self, env: dict[str, str], repo: Path, command: str
    ) -> None:
        # Even pre-locking the literal brace pattern doesn't bypass.
        from src.infra.tools.locking import try_lock as try_lock_

        for tok in command.split():
            if "{" in tok and "}" in tok:
                try_lock_(tok, "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), f"command: {command!r} → {result}"

    def test_param_expansion_not_brace(self, env: dict[str, str], repo: Path) -> None:
        # ``${VAR}`` is parameter expansion (denied by the existing
        # ``$`` check in _SHELL_EXPANSION_RE), NOT brace expansion. The
        # brace-expansion detector should NOT fire on bare ``${...}``.
        # Either way the command denies (via the $ check).
        result = decide(make_payload("bash", {"command": "touch ${VAR}"}, cwd=repo))
        assert is_deny(result)


class TestInterpreterValueFlagsDontMatchScriptFlag:
    """Regression: ``python -W -c -c "..."`` — first ``-c`` is ``-W``'s value.

    Before the fix, ``_find_oneliner_body`` naively returned the FIRST
    occurrence of ``-c`` as the script flag. With ``-W -c`` (where
    ``-c`` is the value of ``-W``), the heuristic returned ``-c`` as
    the body — completely missing the actual script.
    """

    def test_python_W_then_c_does_not_match_W_value(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "unowned.py"
        # ``-W default`` sets a warning filter; ``-c "script"`` runs
        # the script. Before the fix, the heuristic matched the first
        # ``-c`` (the value of ``-W``) — but since python ignores
        # unknown ``-W`` arg, this is exploitable. Use a clearer form:
        # ``python -W <flag-value> -c "<script>"``.
        command = f"python -W default -c \"open('{target}','w').write('x')\""
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_python_X_value_does_not_match(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``-X`` also takes a value.
        target = repo / "unowned.py"
        command = f"python -X dev -c \"open('{target}','w').write('x')\""
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)
        assert "unowned.py" in deny_reason(result)

    def test_perl_I_and_M_value_flags(self, env: dict[str, str], repo: Path) -> None:
        # ``perl -I path -M Module -e '<script>'`` — both ``-I`` and
        # ``-M`` take values; the script is the ``-e`` body.
        target = repo / "unowned.py"
        command = f"perl -I /tmp -M strict -e \"open(F, '>{target}'); print F 'x'\""
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)


class TestGitCResolvesPath:
    """Regression: ``git -C pkg checkout file.py`` writes ``pkg/file.py``.

    Before the fix, ``-C pkg`` was stripped as a global option but the
    extracted target was ``file.py``, resolved against payload cwd as
    ``$cwd/file.py``. Git actually writes ``$cwd/pkg/file.py``. An
    agent holding the parent-level lock could bypass the gate.
    """

    def test_git_C_short_form_resolves_target(
        self, env: dict[str, str], repo: Path
    ) -> None:
        sub = repo / "pkg"
        sub.mkdir()
        # Lock only the parent-level path; the real write surface is
        # in pkg/. Must deny.
        from src.infra.tools.locking import try_lock as try_lock_

        try_lock_(str(repo / "file.py"), "agent-me", repo_namespace=str(repo))
        command = "git -C pkg checkout file.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    def test_git_work_tree_long_form_resolves_target(
        self, env: dict[str, str], repo: Path
    ) -> None:
        sub = repo / "pkg"
        sub.mkdir()
        from src.infra.tools.locking import try_lock as try_lock_

        try_lock_(str(repo / "file.py"), "agent-me", repo_namespace=str(repo))
        command = "git --work-tree=pkg checkout file.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    def test_git_C_with_correct_lock_allows(
        self, env: dict[str, str], repo: Path
    ) -> None:
        sub = repo / "pkg"
        sub.mkdir()
        # Lock the cwd-resolved -C path; the heuristic should match.
        target = sub / "file.py"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = "git -C pkg checkout file.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result


class TestGitGlobalOptionsDestructiveGate:
    """Regression: ``git -C dir stash`` is destructive but lacks ``git stash``.

    Before the fix, the dangerous-command gate only used substring
    matches. ``git -C ../repo stash`` does not contain ``git stash``,
    so the substring check missed it; the heuristic emitted no write
    targets for plain ``stash`` and the bash branch fell through to
    allow even though stash is explicitly destructive.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "git -C /tmp stash",
            "git -C ../repo stash push -u",
            "git --work-tree=. restore file.py",
            "git --git-dir=.git/ rebase main",
            "git -C /tmp reset --hard HEAD",
            "git -c k=v stash",
            "git -C /tmp commit --amend",
            "git -C /tmp clean -fd",
            "git -C /tmp branch -D feature",
            "git -C /tmp merge --abort",
            "git -C /tmp cherry-pick --abort",
            "git -C /tmp worktree remove wt",
        ],
    )
    def test_destructive_git_with_global_opts_denied(
        self, env: dict[str, str], command: str
    ) -> None:
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        # Deny reason must reference a destructive git pattern, not
        # a missing-lock or other heuristic miss.
        reason = deny_reason(result)
        assert "git " in reason or "rebase" in reason or "stash" in reason


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 7, post-restart)
# ---------------------------------------------------------------------------


class TestCoprocReservedWord:
    """Regression: ``coproc touch unowned`` runs touch in a co-process.

    Before the fix, ``coproc`` was missing from both
    ``_SHELL_RESERVED_WORDS`` and ``_EXECUTION_PREFIXES``. The segment
    ``[coproc, touch, unowned.py]`` was dispatched with cmd basename
    ``coproc`` — no extractor matched and the unowned write was
    silently allowed.
    """

    def test_coproc_inner_write_denied(self, env: dict[str, str], repo: Path) -> None:
        target = repo / "unowned.py"
        command = f"coproc touch {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)


class TestCommentBypass:
    """Regression: ``cmd # "`` (comment with unclosed quote).

    Before the fix, ``shlex.split`` was called with ``comments=False``,
    so a trailing ``# "`` (which bash treats as a comment) raised
    ValueError on the unclosed quote. The hook caught the ValueError
    and silently returned an empty target list — letting ``touch
    unowned.py # "`` slip past the gate.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            'touch {target} # "',
            'touch {target} #unclosed"',
            "touch {target} # 'still bad",
            # Multiple commands separated by ; with a trailing comment
            'true ; touch {target} # "',
        ],
    )
    def test_comment_with_unclosed_quote_inner_write_denied(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"

    def test_quoted_hash_is_literal(self, env: dict[str, str], repo: Path) -> None:
        # Sanity: a quoted ``#`` is literal, not a comment marker.
        # The quoted region preserves the ``#`` so the heuristic
        # operates on the full command. Should still detect the
        # touch as a write.
        target = repo / "unowned.py"
        command = f"touch '{target}' '# not a comment'"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)
        assert "unowned.py" in deny_reason(result)


class TestGitGlobalOptionsCommitPushGuards:
    """Regression: ``git -C dir commit -a`` and ``git -c k=v push --force``.

    Before the fix, the dangerous-cmd gate's ``git commit -a`` and
    ``git push --force`` checks used substring matching against the
    full command. Global options spliced into the command broke the
    substring match: ``git -C dir commit -a -m x`` does not contain
    ``git commit -a``, so the substring detector missed it.
    """

    @pytest.mark.parametrize(
        "command",
        [
            # commit -a / --all with global options
            "git -C /tmp commit -a -m x",
            "git -C /tmp commit --all -m x",
            "git -C /tmp commit -am x",
            "git -c key=val commit -a -m x",
            "git --git-dir=.git/ commit --all -m x",
            "git --work-tree=. commit -a",
            # commit without prior git add (atomic-add-commit)
            "git -C /tmp commit -m x",
            "git -c k=v commit -m x",
            # push --force with global options
            "git -C /tmp push --force origin main",
            "git -c k=v push --force",
            "git -C /tmp push -f origin",
            "git --git-dir=.git/ push --force",
        ],
    )
    def test_global_options_commit_push_destructive_denied(
        self, env: dict[str, str], command: str
    ) -> None:
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        reason = deny_reason(result)
        # Reason must reference a git destructive pattern — not a
        # missing-lock or heuristic miss.
        assert any(
            keyword in reason for keyword in ("git commit", "git push", "git add")
        ), reason

    def test_global_options_force_with_lease_allowed(self, env: dict[str, str]) -> None:
        # ``--force-with-lease`` is the documented safer alternative
        # and must be allowed even with global options.
        result = decide(
            make_payload(
                "bash",
                {"command": "git -C /tmp push --force-with-lease origin main"},
            )
        )
        assert is_allow(result)

    def test_atomic_add_commit_with_global_options_allows(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``git add f && git -C /tmp commit -m x`` — git add precedes
        # the commit, so the atomic-add check must pass.
        result = decide(
            make_payload(
                "bash",
                {"command": "git add f.py && git -C /tmp commit -m x"},
            )
        )
        assert is_allow(result)


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 8, post-restart)
# ---------------------------------------------------------------------------


class TestAnsiCBypassesGitDangerCheck:
    """Regression: ``git ... $'\\''`` must not bypass token-based git checks.

    Before the fix, ``_detect_dangerous_pattern`` called
    ``shlex.split(command, ...)`` directly on the raw command string.
    An ANSI-C quoted region with an escaped single quote (``$'\\''``)
    raised ValueError, the token-walk was skipped, and the global-
    option destructive git rules silently allowed the command.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "git -C /repo push --force origin main {ansic}",
            "git -c k=v commit -a -m x {ansic}",
            "git -C /tmp stash {ansic}",
            "git --git-dir=.git/ rebase main {ansic}",
        ],
    )
    def test_ansi_c_quoted_string_does_not_bypass(
        self,
        env: dict[str, str],
        command_template: str,
    ) -> None:
        # Inject an ANSI-C string with an escaped single quote.
        ansic = "$'\\''"
        command = command_template.format(ansic=ansic)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        # Reason must reference the destructive git pattern.
        reason = deny_reason(result)
        assert any(
            keyword in reason
            for keyword in (
                "git push",
                "git commit",
                "git stash",
                "git rebase",
                "git add",
            )
        ), reason


class TestBranchDashDFToken:
    """Regression: ``git -C dir branch -d -f feat`` deletes a branch by force.

    Before the fix, the token-aware detector only matched ``-D`` and
    the substring detector only matched the literal ``git branch -d -f``.
    With global options spliced in, the substring missed and the
    detector returned None — the shell branch then allowed the force
    branch deletion.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "git -C /tmp branch -d -f feature",
            "git -c k=v branch -d -f feat",
            "git --git-dir=.git/ branch -d -f feat",
            # Bundled short flags
            "git -C /tmp branch -df feature",
            "git -C /tmp branch -fd feature",
            # Long form
            "git -C /tmp branch --delete --force feat",
            "git -C /tmp branch --force --delete feat",
        ],
    )
    def test_branch_dash_d_f_with_global_opts_denied(
        self, env: dict[str, str], command: str
    ) -> None:
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "git branch" in deny_reason(result)


class TestUnspacedControlOperatorDangerCheck:
    """Regression: ``git add f;git commit -a -m x`` (no spaces around ``;``).

    Before the fix, ``_detect_dangerous_pattern`` tokenized the raw
    command without first running ``_normalize_separators``. shlex
    treats ``f;git`` as a single token, so the second ``git``
    invocation was never seen by the token-walk. The destructive
    commit-with-all-flag check missed the command.
    """

    @pytest.mark.parametrize(
        "command",
        [
            # No spaces around ;
            "git add f;git commit -a -m x",
            "true;git push --force origin main",
            "true;git stash",
            # No spaces around &&
            "git add f&&git commit -a -m x",
            "true&&git -C /tmp commit -m x",
            # No spaces around control operator with global option
            "true;git -C /tmp push --force origin",
            # Newline as separator
            "true\ngit commit -a",
        ],
    )
    def test_unspaced_separator_destructive_git_denied(
        self, env: dict[str, str], command: str
    ) -> None:
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        reason = deny_reason(result)
        assert any(
            keyword in reason
            for keyword in (
                "git push",
                "git commit",
                "git stash",
                "git add",
            )
        ), reason

    def test_unspaced_atomic_add_commit_with_add_first_allowed(
        self, env: dict[str, str]
    ) -> None:
        # Sanity: ``git add f;git commit -m x`` (with leading add)
        # must still be allowed.
        result = decide(
            make_payload(
                "bash",
                {"command": "git add f;git commit -m x"},
            )
        )
        assert is_allow(result)


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 9, post-restart)
# ---------------------------------------------------------------------------


class TestTimeWrapperEnvAssignments:
    """Regression: ``time VAR=1 touch f`` runs touch with VAR=1.

    Before the fix, ``time`` was an execution prefix but missing from
    the env-assignment-stripping wrapper set. ``time VAR=1 touch f``
    left ``VAR=1`` as the cmd basename — no extractor matched and the
    write was silently allowed.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "time VAR=1 touch {target}",
            "time -p VAR=1 FOO=bar touch {target}",
            "time --portability VAR=1 touch {target}",
            "/usr/bin/time VAR=1 touch {target}",
            # nested wrappers: env outer, time inner, then assign
            "env time VAR=1 touch {target}",
        ],
    )
    def test_time_with_env_assignment_inner_write_denied(
        self, env: dict[str, str], repo: Path, command_template: str
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)


class TestNewlineCommentBypass:
    """Regression: ``# foo\\ntouch unowned`` (comment on first line).

    Before the fix, ``_normalize_separators`` replaced every ``\\n``
    with ``;`` BEFORE shlex's comment handling. With ``comments=True``
    shlex only terminates a comment at the next ``\\n``; since all
    newlines were rewritten, the ``#`` opened a comment that consumed
    the entire remainder of the input — a zero-token segment list
    and a silent allow.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "# leading comment\ntouch {target}",
            "# multi-line comment\n# second line\ntouch {target}",
            "echo hi # trailing comment\ntouch {target}",
            # First line ends in `;`, then comment, then second cmd.
            ":; # comment\ntouch {target}",
        ],
    )
    def test_comment_then_newline_does_not_swallow_next_command(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_quoted_hash_with_newline_not_a_comment(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``echo "# not a comment"`` — the ``#`` is inside quotes, so
        # it's a literal char, NOT a comment marker. Even though there's
        # no newline, the heuristic must not eat the rest of the input.
        target = repo / "unowned.py"
        command = f'echo "# not a comment" ; touch {target}'
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result)
        assert "unowned.py" in deny_reason(result)


class TestEscapedRedirectBeforePipe:
    """Regression: ``\\>|`` and ``\\>&`` (escaped redirect).

    Before the fix, the ``|`` and ``&`` handlers checked if the
    immediately preceding char in ``out`` (or input) was ``>``, but
    didn't verify the ``>`` was unescaped. ``cmd \\>| tee unowned``
    has ``\\>`` (literal ``>``) followed by a real pipeline ``| tee``
    — but the heuristic glued ``|`` to the literal ``>`` as if it
    were a redirect operator and left ``unowned`` as a passive arg
    that no extractor saw.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "cmd \\>| tee {target}",
            "cmd \\>& touch {target}",
            "echo \\> | tee {target}",
        ],
    )
    def test_escaped_redirect_does_not_hide_pipe(
        self,
        env: dict[str, str],
        repo: Path,
        command_template: str,
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_unescaped_force_clobber_still_works(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: real ``>|`` (force-clobber redirect) is still
        # recognised as a redirect operator.
        target = repo / "out.log"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"echo hi >| {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result


class TestDangerousCmdsInsideSubstitution:
    """Regression: destructive git inside backticks/$() bypasses gate.

    Before the fix, ``_detect_dangerous_pattern`` ran only on the
    outer command (post-substitution-extraction). ``echo `git -C
    /tmp stash``` had its dangerous body extracted as a substitution
    body but only ``_extract_shell_write_paths`` saw it; that
    extractor returns no write targets for ``git stash`` and the
    shell branch fell through to allow.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            # backticks
            "echo `git -C /tmp stash`",
            "x=`git -C /tmp commit -a -m x`",
            "echo `git push --force origin main`",
            # $() form
            "echo $(git -C /tmp stash)",
            'echo "$(git --git-dir=.git/ rebase main)"',
            # Nested
            "echo $(echo $(git -C /tmp stash))",
            # Unspaced control op inside body
            "echo $(true;git -C /tmp stash)",
        ],
    )
    def test_dangerous_inside_substitution_denied(
        self, env: dict[str, str], command_template: str
    ) -> None:
        result = decide(make_payload("bash", {"command": command_template}))
        assert is_deny(result), f"command: {command_template!r} → {result}"
        reason = deny_reason(result)
        assert any(
            keyword in reason
            for keyword in (
                "git push",
                "git commit",
                "git stash",
                "git rebase",
            )
        ), reason


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 10, post-restart)
# ---------------------------------------------------------------------------


class TestTeeDashDashOperand:
    """Regression: ``tee -- -owned.py`` writes to a dash-prefixed file.

    Before the fix, ``_extract_tee_targets`` skipped every
    dash-prefixed token including operands after ``--``. The write
    target list was empty and the bash branch allowed the command.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "echo x | tee -- {target}",
            "echo x | tee -a -- {target}",
            "echo x | tee --append -- {target}",
        ],
    )
    def test_tee_dash_dash_operand_denied(
        self, env: dict[str, str], repo: Path, command_template: str
    ) -> None:
        target = repo / "-owned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "-owned.py" in deny_reason(result)


class TestChmodSymbolicMode:
    """Regression: ``chmod -w unowned.py`` is a symbolic-mode chmod.

    Before the fix, ``_extract_utility_targets`` treated every
    dash-prefixed token as a flag. For ``chmod -w f``, ``-w`` was
    skipped, leaving ``[f]`` as the only positional. The
    ``skip_first_positional`` strategy then dropped ``f`` as the
    "mode" — silently allowing the chmod against the unlocked file.
    """

    @pytest.mark.parametrize(
        "command_template",
        [
            "chmod -w {target}",
            "chmod -r {target}",
            "chmod -x {target}",
            "chmod +w {target}",
            "chmod +x {target}",
            "chmod =rw {target}",
            "chmod u-w {target}",
            "chmod ugo+r {target}",
            "chmod u+rwx,g-x {target}",
            # With a real flag mixed in
            "chmod -R -w {target}",
            "chmod -v -w {target}",
        ],
    )
    def test_chmod_symbolic_mode_denied(
        self, env: dict[str, str], repo: Path, command_template: str
    ) -> None:
        target = repo / "unowned.py"
        command = command_template.format(target=target)
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), f"command: {command!r} → {result}"
        assert "unowned.py" in deny_reason(result)

    def test_chmod_symbolic_mode_allowed_when_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "f.py"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("bash", {"command": f"chmod -w {target}"}))
        assert is_allow(result), result

    def test_chmod_real_flags_still_skip(self, env: dict[str, str], repo: Path) -> None:
        # Sanity: real chmod flags (-R, -v, -c, -f) without a
        # symbolic mode are still skipped (and the numeric-mode form
        # ``chmod 644 f`` still uses skip_first_positional).
        target = repo / "f.py"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        result = decide(make_payload("bash", {"command": f"chmod -R 644 {target}"}))
        assert is_allow(result), result


class TestCommentedSubstitutionsNotExtracted:
    """P2: ``# $(touch unowned)`` is commented out — must NOT extract.

    Before the fix, ``_extract_command_substitutions`` ran on the raw
    command BEFORE comments were stripped. A commented-out
    substitution body was extracted and processed as if bash would
    execute it — false-positive deny on a no-op command.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "# $(touch unowned.py)\necho ok",
            "echo ok # `touch unowned.py`",
            "# `touch unowned.py`\necho ok",
            'echo ok # outer "$(touch unowned.py)"',
        ],
    )
    def test_commented_substitution_does_not_deny(
        self, env: dict[str, str], command: str
    ) -> None:
        # Bash does not execute the commented substitution — the hook
        # must not deny on the inner write that never runs.
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), f"command: {command!r} → {result}"

    def test_uncommented_substitution_still_denied(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: a real substitution in a non-comment context is
        # still detected and denied.
        target = repo / "unowned.py"
        result = decide(
            make_payload("bash", {"command": f"echo $(touch {target})"}, cwd=repo)
        )
        assert is_deny(result)
        assert "unowned.py" in deny_reason(result)


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 3 — P1 findings)
# ---------------------------------------------------------------------------


class TestAnsiCBacktickBypass:
    """Regression: ``$'\\''`` must not desync the cmd-sub extractor.

    Before the fix, ``_extract_command_substitutions`` lacked ANSI-C
    handling. The escaped single quote inside ``$'\\''`` toggled its
    plain-quote tracker into a stuck "inside-single-quotes" state, so
    any backtick substitution that followed was treated as quoted and
    NOT extracted as a separate segment. The bash command
    ``echo $'\\'' `touch unowned.py``` then produced no extracted
    targets; ``decide()`` returned allow while bash actually executed
    the inner ``touch``.
    """

    def test_backtick_after_ansi_c_is_extracted(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "unowned.py"
        # Inner ``touch unowned.py`` is a real write target that bash
        # will execute. The hook must extract the backtick body and
        # deny on the missing lock.
        command = f"echo $'\\'' `touch {target}`"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result
        assert "unowned.py" in deny_reason(result)

    def test_dollar_paren_after_ansi_c_is_extracted(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # ``$()`` was already rescued by separator splitting on
        # ``(``/``)``; pin it so a future regression in that path is
        # caught.
        target = repo / "unowned.py"
        command = f"echo $'\\'' $(touch {target})"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result
        assert "unowned.py" in deny_reason(result)

    def test_ansi_c_with_no_inner_write_is_allowed(self, env: dict[str, str]) -> None:
        # Sanity: an ANSI-C string by itself has no write surface and
        # must not produce a spurious deny.
        result = decide(make_payload("bash", {"command": "echo $'\\''"}))
        assert is_allow(result), result


class TestAttachedAmpersandRedirectBypass:
    """Regression: ``cmd dst&>out`` must split ``&`` from ``dst``.

    Before the fix, ``_normalize_separators`` appended ``&`` (when
    followed by ``>``) without padding, leaving ``dst&`` glued in the
    output. ``_strip_fd_prefix_from_output`` then refused to strip the
    ``&`` because its whitespace-boundary check failed (the previous
    out-char was a non-space letter), so the cp destination was
    lock-checked as the literal junk path ``dst&`` — an agent locking
    that junk path was allowed to write while bash actually wrote to
    ``dst``.
    """

    def test_attached_amp_redirect_locks_real_destination(
        self, env: dict[str, str], repo: Path
    ) -> None:
        owned = repo / "owned.txt"
        unowned = repo / "unowned.txt"
        out = repo / "out.txt"
        # Lock the junk path the buggy parser used as the destination.
        # Pre-fix the parser would lock-check ``unowned.txt&`` (junk),
        # find the agent owns it, and ALLOW the write while bash wrote
        # to the real ``unowned.txt``. Post-fix the parser splits the
        # ``&`` from the cp destination word, lock-checks the real
        # ``unowned.txt`` (no lock), and DENIES.
        junk = repo / "unowned.txt&"
        assert try_lock(str(junk), "agent-me", repo_namespace=str(repo))
        assert try_lock(str(owned), "agent-me", repo_namespace=str(repo))
        # Lock the redirect target so the only failing check is the cp
        # destination — pinning that the post-fix denial is on the
        # split-off real path, not on the redirect target.
        assert try_lock(str(out), "agent-me", repo_namespace=str(repo))
        command = f"cp {owned} {unowned}&>{out}"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result
        # The real path must appear in the reason (not the junk one).
        reason = deny_reason(result)
        assert "unowned.txt" in reason
        assert "unowned.txt&" not in reason

    def test_attached_amp_append_redirect_locks_real_destination(
        self, env: dict[str, str], repo: Path
    ) -> None:
        unowned = repo / "unowned.txt"
        out = repo / "out.txt"
        junk = repo / "unowned.txt&"
        # Lock the junk-path and the redirect target so the only
        # un-owned write is the real ``unowned.txt``. Post-fix the
        # parser must split the ``&`` from the redirection target.
        assert try_lock(str(junk), "agent-me", repo_namespace=str(repo))
        assert try_lock(str(out), "agent-me", repo_namespace=str(repo))
        # ``&>>`` (append form) must split the ``&`` the same way.
        command = f"echo data >{unowned}&>>{out}"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    def test_attached_amp_redirect_allows_when_locks_match_bash(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: when the agent holds the lock on the real bash-side
        # destination (``unowned.txt`` and ``out.txt``), the write is
        # allowed — proving the parser now uses bash's split.
        unowned = repo / "unowned.txt"
        out = repo / "out.txt"
        owned = repo / "owned.txt"
        assert try_lock(str(owned), "agent-me", repo_namespace=str(repo))
        assert try_lock(str(unowned), "agent-me", repo_namespace=str(repo))
        assert try_lock(str(out), "agent-me", repo_namespace=str(repo))
        command = f"cp {owned} {unowned}&>{out}"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result


class TestNodeFsPromisesWrites:
    """Regression: ``require('fs/promises')`` writes must be detected.

    Before the fix, the Node hint table only mentioned ``require('fs')``
    / ``require('node:fs')``. A normal command like
    ``node -e "const fsp=require('fs/promises'); fsp.writeFile(...)"``
    matched no regex and no hint, so ``decide()`` returned allow even
    though the agent did not hold a lock on the target.
    """

    @pytest.mark.parametrize(
        ("require_form", "body_quote"),
        [
            # require with single-quoted module name → wrap body in
            # double quotes for the bash -e arg.
            ("require('fs/promises')", '"'),
            ("require('node:fs/promises')", '"'),
            # require with double-quoted module name → wrap body in
            # single quotes for the bash -e arg.
            ('require("fs/promises")', "'"),
            ('require("node:fs/promises")', "'"),
        ],
    )
    def test_fs_promises_dynamic_binding_denies(
        self,
        env: dict[str, str],
        repo: Path,
        require_form: str,
        body_quote: str,
    ) -> None:
        # The promise-API binding through a fresh identifier is dynamic
        # — no literal target the regex can extract — so the hook must
        # fall back to the unresolved-sentinel deny via the hint check.
        target = repo / "unlocked.txt"
        # Use the opposite quote style for the writeFile literal so the
        # body's quoting is consistent with body_quote.
        target_quote = "'" if body_quote == '"' else '"'
        body = (
            f"const fsp={require_form}; "
            f"fsp.writeFile({target_quote}{target}{target_quote},"
            f"{target_quote}x{target_quote})"
        )
        command = f"node -e {body_quote}{body}{body_quote}"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    def test_fs_promises_direct_call_extracts_literal(
        self,
        env: dict[str, str],
        repo: Path,
    ) -> None:
        # Direct ``require('fs/promises').writeFile(<lit>)`` form: the
        # extended ``_NODE_REQUIRE_FS_WRITE_RE`` extracts the literal
        # target so an unlocked path denies with the explicit message.
        target = repo / "unlocked.txt"
        body = f"require('fs/promises').writeFile('{target}','x')"
        command = f'node -e "{body}"'
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result
        assert "unlocked.txt" in deny_reason(result)


class TestBsdSedEmptySuffix:
    """Regression: ``sed -i '' SCRIPT file`` (BSD/macOS) must not deny
    the script as a write target.

    Before the fix, the BSD-style empty in-place suffix collected as
    a positional and the script-positional logic dropped it as the
    "first positional", returning ``SCRIPT`` plus the file as write
    targets. An agent that correctly held ``file`` was denied for
    missing a lock on ``s/a/b/`` — a routine macOS sed invocation.
    """

    def test_bsd_sed_empty_suffix_allows_when_only_file_locked(
        self, env: dict[str, str], repo: Path
    ) -> None:
        target = repo / "data.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        # The agent does NOT lock ``s/a/b/`` (which isn't a real path
        # and never could be locked sensibly). Pre-fix this denied.
        command = f"sed -i '' 's/a/b/' {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result

    def test_bsd_sed_empty_suffix_denies_unlocked_file(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: the heuristic still locks-checks the real file
        # target. Without a lock on ``data.txt`` the write is denied.
        target = repo / "data.txt"
        command = f"sed -i '' 's/a/b/' {target}"
        result = decide(make_payload("bash", {"command": command}))
        assert is_deny(result), result
        assert "data.txt" in deny_reason(result)


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 4 — P1 findings)
# ---------------------------------------------------------------------------


class TestNodeEsmImportBypass:
    """Regression: ESM ``import { X } from 'node:fs'`` bindings must
    surface to the heuristic's write-detection.

    Before the fix, ``_NODE_FS_WRITE_RE`` required the ``fs.`` method-
    chain prefix and the hint table only knew about ``require()``
    forms. ``node --input-type=module -e "import { writeFileSync }
    from 'node:fs'; writeFileSync('unowned', 'x')"`` matched no
    regex and triggered no hint, so ``write_targets`` was empty and
    ``decide()`` returned allow.
    """

    @pytest.mark.parametrize(
        ("import_form", "module_quote"),
        [
            # Static named imports.
            ("import { writeFileSync } from", "'"),
            ("import { writeFileSync } from", '"'),
            # Default import.
            ("import fs from", "'"),
            # Namespace import.
            ("import * as fs from", "'"),
        ],
    )
    @pytest.mark.parametrize(
        "module_name",
        ["fs", "node:fs", "fs/promises", "node:fs/promises"],
    )
    def test_esm_import_with_literal_target_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        import_form: str,
        module_quote: str,
        module_name: str,
    ) -> None:
        target = repo / "unlocked.txt"
        # Build the body. For default and namespace imports use
        # ``fs.writeFileSync``; for the named import use bare
        # ``writeFileSync``.
        if "{" in import_form:
            call = f"writeFileSync('{target}','x')"
        else:
            call = f"fs.writeFileSync('{target}','x')"
        body = f"{import_form} {module_quote}{module_name}{module_quote}; {call}"
        command = f'node --input-type=module -e "{body}"'
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result
        assert "unlocked.txt" in deny_reason(result)

    def test_esm_import_with_dynamic_target_denies_via_hint(
        self,
        env: dict[str, str],
        repo: Path,
    ) -> None:
        # Dynamic target — the literal-target regex cannot extract it,
        # but the ``from 'node:fs'`` hint must trigger the
        # unresolved-sentinel deny.
        body = "import { writeFileSync } from 'node:fs'; writeFileSync(p, 'x')"
        command = f'node --input-type=module -e "{body}"'
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result


class TestNodeOpenSyncWrite:
    """Regression: ``fs.openSync(path, 'w')`` must be detected.

    Before the fix, the Node detector only looked for
    ``writeFile``/``appendFile``/``createWriteStream`` and only
    fell back to unresolved-deny for those hints. A standard
    write-open such as ``fs.openSync('/repo/unlocked.txt', 'w')``
    matched no regex and no hint, so the gate allowed it even
    though ``openSync`` with a write-mode flag creates/truncates
    the file.
    """

    @pytest.mark.parametrize(
        "flag",
        ["w", "wx", "w+", "a", "ax", "a+", "r+"],
    )
    def test_opensync_write_flag_with_literal_target_denies_unlocked(
        self,
        env: dict[str, str],
        repo: Path,
        flag: str,
    ) -> None:
        target = repo / "unlocked.txt"
        body = f"const fs=require('fs'); fs.openSync('{target}','{flag}')"
        command = f'node -e "{body}"'
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result
        assert "unlocked.txt" in deny_reason(result)

    def test_esm_bare_opensync_write_denies(
        self,
        env: dict[str, str],
        repo: Path,
    ) -> None:
        # ESM ``import { openSync } from 'node:fs'`` binds openSync
        # directly. The bare-call regex with the write-mode check
        # must still extract the literal target.
        target = repo / "unlocked.txt"
        body = f"import {{ openSync }} from 'node:fs'; openSync('{target}','w')"
        command = f'node --input-type=module -e "{body}"'
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    def test_opensync_write_flag_with_dynamic_target_denies_via_hint(
        self,
        env: dict[str, str],
        repo: Path,
    ) -> None:
        # Dynamic target ``p`` — the literal regex cannot extract,
        # but the openSync write-flag hint regex catches the call
        # shape and triggers unresolved-sentinel deny.
        body = "import { openSync } from 'node:fs'; openSync(p,'w')"
        command = f'node --input-type=module -e "{body}"'
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result


class TestGitChdirComposition:
    """Regression: repeated ``git -C dir`` options must compose.

    Per ``git`` semantics, multiple ``-C`` options accumulate: each
    subsequent non-absolute ``-C`` is interpreted relative to the
    preceding one; an absolute ``-C`` resets; an empty ``-C ""`` is
    a no-op. Before the fix, ``_find_git_subcommand_info`` overwrote
    ``chdir_value`` on each ``-C``, so ``git -C pkg -C sub checkout
    file.py`` lock-checked ``$cwd/sub/file.py`` while git actually
    wrote ``$cwd/pkg/sub/file.py``. An agent holding only
    ``$cwd/sub/file.py`` could mutate the unowned nested file under
    ``$cwd/pkg/sub/``.
    """

    def test_repeated_relative_dash_c_composes(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Lock only the wrong path the buggy parser would have
        # picked: ``$repo/sub/file.py``. Pre-fix ALLOW; post-fix
        # DENY because git actually writes ``$repo/pkg/sub/file.py``.
        wrong = repo / "sub" / "file.py"
        wrong.parent.mkdir(parents=True)
        assert try_lock(str(wrong), "agent-me", repo_namespace=str(repo))
        command = "git -C pkg -C sub checkout file.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result
        # The deny reason must mention the *composed* path, not the
        # wrong one.
        reason = deny_reason(result)
        assert "pkg/sub/file.py" in reason

    def test_composed_path_lock_allows(self, env: dict[str, str], repo: Path) -> None:
        # Sanity: locking the *composed* path lets the write through.
        composed = repo / "pkg" / "sub" / "file.py"
        composed.parent.mkdir(parents=True)
        assert try_lock(str(composed), "agent-me", repo_namespace=str(repo))
        command = "git -C pkg -C sub checkout file.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result

    def test_absolute_dash_c_resets(self, env: dict[str, str], repo: Path) -> None:
        # ``git -C pkg -C /abs checkout f`` discards ``pkg`` and
        # resolves ``f`` relative to the absolute path. With the
        # absolute path under tmp ``repo``, the resolved write is
        # ``$abs/file.py``.
        abs_dir = repo / "abs"
        abs_dir.mkdir()
        target = abs_dir / "file.py"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"git -C pkg -C {abs_dir} checkout file.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result

    def test_empty_dash_c_is_noop(self, env: dict[str, str], repo: Path) -> None:
        # ``git -C pkg -C '' checkout f`` keeps ``pkg`` and resolves
        # to ``$cwd/pkg/file.py``.
        target = repo / "pkg" / "file.py"
        target.parent.mkdir(parents=True)
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = "git -C pkg -C '' checkout file.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result


# ---------------------------------------------------------------------------
# Review-fix regressions (attempt 5 — P1 + P2 findings)
# ---------------------------------------------------------------------------


class TestPerlBareOpenWrite:
    """Regression: Perl ``open`` without parens must be detected.

    The previous regex/hint required ``open(`` (with parenthesis), but
    Perl supports bareword call style: ``open my $fh, '>', '/path';``.
    Pre-fix this matched no regex and no hint, so ``decide()`` returned
    allow on a write-open that creates/truncates the unowned file.
    """

    @pytest.mark.parametrize(
        "body_template",
        [
            # 3-arg bareword form (idiomatic modern Perl).
            "open my $fh, '>', '{target}'; print $fh 'x'",
            # 3-arg bareword with double-quoted args.
            'open my $fh, ">", "{target}"; print $fh "x"',
            # 3-arg parens (was already covered).
            "open(my $fh, '>', '{target}'); print $fh 'x'",
            # 3-arg bareword, append mode.
            "open my $fh, '>>', '{target}'; print $fh 'x'",
            # 3-arg parens with whitespace inside the mode literal —
            # Perl strips it so this opens for write. The literal
            # extractor must match so the path is denied via the
            # standard unowned-path check rather than the hint.
            "open(my $fh, ' > ', '{target}'); print $fh 'x'",
            # 3-arg bareword, append mode with surrounding whitespace.
            "open my $fh, ' >> ', '{target}'; print $fh 'x'",
        ],
    )
    def test_perl_three_arg_open_extracts_literal(
        self,
        env: dict[str, str],
        repo: Path,
        body_template: str,
    ) -> None:
        target = repo / "unowned.txt"
        body = body_template.format(target=target)
        # The body uses single quotes for Perl args; wrap in double
        # quotes for the bash -e arg.
        if "'" in body:
            command = f'perl -e "{body}"'
        else:
            command = f"perl -e '{body}'"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result
        assert "unowned.txt" in deny_reason(result)

    @pytest.mark.parametrize(
        "body",
        [
            # 2-arg form with mode-in-path: not literal-extractable but
            # the hint must trigger.
            "open FH, '>/tmp/unowned.txt'; print FH 'x'",
            # ``q(>)`` quote-like operator: hint must trigger.
            "open my $fh, q(>), q(/tmp/unowned.txt); print $fh q(x)",
            # ``qq(>)`` with double-quote semantics: hint must trigger.
            "open my $fh, qq(>), qq(/tmp/unowned.txt); print $fh qq(x)",
            # 2-arg form with whitespace before the ``>`` marker.
            "open FH, ' >/tmp/unowned.txt'; print FH 'x'",
        ],
    )
    def test_perl_open_write_hint_triggers_unresolved(
        self,
        env: dict[str, str],
        repo: Path,
        body: str,
    ) -> None:
        # Choose outer quote to avoid clashing with body quotes.
        outer = '"' if "'" in body and '"' not in body else "'"
        if outer == "'" and "'" in body:
            outer = '"'
        command = f"perl -e {outer}{body}{outer}"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    @pytest.mark.parametrize(
        "body",
        [
            # Sanity: read-mode ``open my $fh, '<', '/path'`` has no
            # ``>`` marker and must NOT be flagged as a write.
            "open my $fh, '<', '/tmp/file'; my $line = <$fh>",
            # Read-mode dynamic open whose statement happens to
            # contain a ``>``-prefixed token *outside* the mode slot
            # (here: a ``die`` message). The hint must be scoped to
            # the mode argument and not match the unrelated ``'>'``.
            "open my $fh, '<', $path or die '>'; my $line = <$fh>",
            # Read-mode 3-arg parens form with ``>`` elsewhere in the
            # statement.
            "open(my $fh, '<', $path) or warn '>>>'; my $line = <$fh>",
            # 1-arg ``open FH;`` with NO trailing comma in its own
            # statement, followed by a *later* statement that has
            # ``,`` + ``'>'``. The filehandle group must stop at
            # ``;`` and not span into the next statement's comma.
            "open FH; print STDERR, '>'",
            # ``open()`` with 0 args followed by a list literal that
            # contains ``,`` + ``'>'``.
            "open(); my @x = ('foo', '>')",
            # Method call ``$x->open()`` is matched by ``\\bopen\\b``;
            # the trailing ``;`` must terminate the filehandle group
            # so the next statement's ``,`` + ``'>'`` is not consumed.
            "$app->open(); foo('bar', '>')",
        ],
    )
    def test_perl_read_open_does_not_trigger(
        self,
        env: dict[str, str],
        body: str,
    ) -> None:
        command = f'perl -e "{body}"'
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result

    @pytest.mark.parametrize(
        "mode",
        [" > ", " >> ", " >  ", "  >> "],
    )
    def test_perl_space_padded_mode_extracts_allowed_path(
        self,
        env: dict[str, str],
        repo: Path,
        mode: str,
    ) -> None:
        # Perl strips whitespace from the mode string, so `' > '` is a
        # write mode. The literal extractor must recognise the path
        # so a write to a locked/allowed file is permitted; otherwise
        # the call falls through to the hint and is incorrectly denied
        # as an unresolved write.
        target = repo / "owned.txt"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        body = f"open(my $fh, '{mode}', '{target}'); print $fh 'x'"
        command = f'perl -e "{body}"'
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result


class TestNodeReadOnlyDoesNotDeny:
    """Regression: read-only Node fs commands must not deny.

    The previous ``_NODE_WRITE_HINTS`` table included module-import
    substrings (``require('fs')``, ``from 'fs'``, ``import('fs')``).
    Any node command that imported fs — even a read-only one — hit
    the substring fallback and was denied. Tighten hints to write-call
    substrings only so ``node -e "const fs=require('fs');
    fs.readFileSync('/repo/file')"`` allows.
    """

    @pytest.mark.parametrize(
        "body",
        [
            "const fs=require('fs'); fs.readFileSync('/tmp/file')",
            "const fs=require('fs/promises'); await fs.readFile('/tmp/file')",
            "const {readFileSync}=require('fs'); readFileSync('/tmp/file')",
        ],
    )
    def test_node_read_only_require_allows(
        self,
        env: dict[str, str],
        body: str,
    ) -> None:
        command = f'node -e "{body}"'
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result

    @pytest.mark.parametrize(
        "body",
        [
            "import { readFileSync } from 'node:fs'; readFileSync('/tmp/file')",
            "import fs from 'fs'; fs.readFileSync('/tmp/file')",
        ],
    )
    def test_node_read_only_esm_import_allows(
        self,
        env: dict[str, str],
        body: str,
    ) -> None:
        command = f'node --input-type=module -e "{body}"'
        result = decide(make_payload("bash", {"command": command}))
        assert is_allow(result), result

    def test_node_dynamic_target_write_still_denies(
        self,
        env: dict[str, str],
        repo: Path,
    ) -> None:
        # The tightened hints must still catch dynamic-target writes
        # via the bare-call substring (``writeFile``).
        body = "const fs=require('fs'); fs.writeFile(p, 'x')"
        command = f'node -e "{body}"'
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result


class TestGitWorkTreeAndGitDir:
    """Regression: ``--git-dir`` must be ignored; ``--work-tree`` must
    compose with ``-C``.

    Pre-fix, ``_find_git_subcommand_info`` treated both as setting
    ``chdir_value`` (with last-value-wins overwrite). That allowed
    two distinct bypasses:
      * ``git --git-dir=/tmp/junk checkout f`` — agent locks
        ``/tmp/junk/f`` (a junk path) while git writes ``$cwd/f``.
      * ``git -C pkg --work-tree=wt checkout f`` — agent locks
        ``wt/f`` while git writes ``pkg/wt/f``.
    """

    def test_git_dir_does_not_shift_resolution_base(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Lock a junk path under /tmp/junk; the agent does NOT own
        # the real write target ``$cwd/f.py``. Pre-fix this allowed
        # because the heuristic resolved the write under ``/tmp/junk/``.
        junk_dir = repo / "junk"
        junk_dir.mkdir()
        junk = junk_dir / "f.py"
        assert try_lock(str(junk), "agent-me", repo_namespace=str(repo))
        command = f"git --git-dir={junk_dir} checkout f.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result

    def test_dash_c_with_work_tree_composes(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Lock the path the buggy parser used: ``$repo/wt/f.py``.
        # Pre-fix ALLOW; post-fix DENY because git writes
        # ``$repo/pkg/wt/f.py``.
        wrong = repo / "wt" / "f.py"
        wrong.parent.mkdir(parents=True)
        assert try_lock(str(wrong), "agent-me", repo_namespace=str(repo))
        command = "git -C pkg --work-tree=wt checkout f.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_deny(result), result
        reason = deny_reason(result)
        assert "pkg/wt/f.py" in reason

    def test_dash_c_with_work_tree_composed_lock_allows(
        self, env: dict[str, str], repo: Path
    ) -> None:
        # Sanity: locking the *composed* path lets the write through.
        composed = repo / "pkg" / "wt" / "f.py"
        composed.parent.mkdir(parents=True)
        assert try_lock(str(composed), "agent-me", repo_namespace=str(repo))
        command = "git -C pkg --work-tree=wt checkout f.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result

    def test_absolute_work_tree_resets(self, env: dict[str, str], repo: Path) -> None:
        # ``--work-tree=/abs`` (absolute) discards any preceding ``-C``.
        abs_dir = repo / "abs"
        abs_dir.mkdir()
        target = abs_dir / "f.py"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        command = f"git -C pkg --work-tree={abs_dir} checkout f.py"
        result = decide(make_payload("bash", {"command": command}, cwd=repo))
        assert is_allow(result), result


@pytest.mark.unit
def test_selftest_sentinel_allow_shape_is_stable() -> None:
    """Lock-step pin: ``codex_provider._default_selftest_probe`` hardcodes
    the expected decision shape for a sentinel ``echo`` invocation
    (the architectural rule forbids ``src.infra.clients`` from
    importing ``src.infra.hooks``, so the shape is duplicated). This
    test fails if either side drifts so the cross-module contract
    stays in sync.
    """
    sentinel_input = {
        "tool_name": "bash",
        "tool_input": {"command": "echo mala-codex-selftest-sentinel"},
        "session_id": "mala-selftest-session",
        "cwd": "/tmp/mala-selftest",
    }
    expected = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
        },
    }
    assert decide(sentinel_input) == expected


# ---------------------------------------------------------------------------
# Selftest marker (Phase E5 / AC-5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_emits_selftest_marker_when_env_var_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main()`` writes the live-hook marker to ``SELFTEST_MARKER_ENV``
    *and* still produces normal hook output, so the provider's selftest
    can both observe the marker and the JSON allow/deny decision.
    """
    import json as _json

    from src.infra.hooks.codex.selftest_policy import (
        SELFTEST_MARKER_ENV,
        _compute_selftest_identity_hash,
    )
    from src.infra.hooks.codex_pre_tool_use import main

    marker_path = tmp_path / "marker.json"
    monkeypatch.setenv(SELFTEST_MARKER_ENV, str(marker_path))
    monkeypatch.setattr(
        "sys.stdin",
        type(
            "FakeStdin",
            (),
            {"read": staticmethod(lambda: _json.dumps({"tool_name": "noop"}))},
        )(),
    )

    rc = main()
    assert rc == 0

    payload = _json.loads(marker_path.read_text(encoding="utf-8"))
    assert payload["mala_codex_hook"] == "loaded"
    assert payload["version"] == _compute_selftest_identity_hash()

    # main() still wrote a JSON decision to stdout — the marker emit
    # must not short-circuit normal hook processing.
    captured = capsys.readouterr().out.strip()
    decision = _json.loads(captured)
    assert "hookSpecificOutput" in decision


@pytest.mark.unit
def test_main_does_not_emit_marker_when_env_var_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``SELFTEST_MARKER_ENV`` is not set, the hook does not write
    any marker. Real Codex tool calls must not leave stray files."""
    import json as _json

    from src.infra.hooks.codex.selftest_policy import SELFTEST_MARKER_ENV
    from src.infra.hooks.codex_pre_tool_use import main

    monkeypatch.delenv(SELFTEST_MARKER_ENV, raising=False)
    monkeypatch.setattr(
        "sys.stdin",
        type(
            "FakeStdin",
            (),
            {"read": staticmethod(lambda: _json.dumps({"tool_name": "noop"}))},
        )(),
    )

    main()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_marker_write_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for review-4 P2 TOCTOU: a polling reader observing
    the marker path must never see a partial / empty file.

    Surrogate proof: assert the implementation routes the write
    through a sibling temp file + ``os.replace`` (atomic at the
    directory-entry level on POSIX). A direct ``Path.write_text``
    would expose ``open + write`` non-atomicity to a parallel
    reader.
    """
    import json as _json

    from src.infra.hooks.codex.selftest_policy import SELFTEST_MARKER_ENV
    from src.infra.hooks.codex_pre_tool_use import main

    marker_path = tmp_path / "marker.json"
    monkeypatch.setenv(SELFTEST_MARKER_ENV, str(marker_path))
    monkeypatch.setattr(
        "sys.stdin",
        type(
            "FakeStdin",
            (),
            {"read": staticmethod(lambda: _json.dumps({"tool_name": "noop"}))},
        )(),
    )

    # Track every os.replace call so we can assert atomicity.
    import os as _os

    replace_calls: list[tuple[str, str]] = []
    real_replace = _os.replace

    def _tracking_replace(src: object, dst: object) -> None:
        replace_calls.append((str(src), str(dst)))
        real_replace(str(src), str(dst))

    monkeypatch.setattr(_os, "replace", _tracking_replace)

    main()

    assert marker_path.is_file()
    # Exactly one os.replace went to the marker path; the source
    # path is a sibling temp (under the same parent dir) so the
    # rename is atomic.
    matching = [src for src, dst in replace_calls if dst == str(marker_path)]
    assert matching, f"marker write did not go through os.replace: {replace_calls!r}"
    from pathlib import Path as _Path

    src_path = _Path(matching[0])
    assert src_path.parent == marker_path.parent

    # Marker contents are valid JSON — ``read_text`` from a parallel
    # poller would either see "no file yet" or the fully-written
    # JSON, never an empty / partial body.
    marker_data = _json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker_data["mala_codex_hook"] == "loaded"


@pytest.mark.unit
def test_main_swallows_marker_write_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwritable marker path must not break the hook's primary
    decision flow — the provider's selftest reads the absence of the
    marker as the fail-closed signal."""
    import json as _json

    from src.infra.hooks.codex.selftest_policy import SELFTEST_MARKER_ENV
    from src.infra.hooks.codex_pre_tool_use import main

    # Point at a path under a non-existent directory so write fails.
    bad_path = tmp_path / "missing-dir" / "marker.json"
    monkeypatch.setenv(SELFTEST_MARKER_ENV, str(bad_path))
    monkeypatch.setattr(
        "sys.stdin",
        type(
            "FakeStdin",
            (),
            {"read": staticmethod(lambda: _json.dumps({"tool_name": "noop"}))},
        )(),
    )

    rc = main()
    assert rc == 0
    assert not bad_path.exists()


# ---------------------------------------------------------------------------
# SessionStart handler (Phase E5 / AC-5 — live-Codex dispatch probe)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_aborts_session_start_in_selftest_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A SessionStart payload + ``SELFTEST_MARKER_ENV`` set produces
    ``{"continue": false, "stopReason": ...}`` so Codex aborts the
    turn before any LLM call.

    This is the contract the live-Codex hook-dispatch probe relies on:
    SessionStart fires → marker emitted → continue=false → turn aborts
    with no model token spent. Without this branch, Codex would parse
    the PreToolUse-shaped output as a SessionStart response, hit
    ``deny_unknown_fields`` on the ``decision`` key, and treat the
    hook as failed — defeating the whole proof.
    """
    import json as _json

    from src.infra.hooks.codex.selftest_policy import SELFTEST_MARKER_ENV
    from src.infra.hooks.codex_pre_tool_use import main

    marker_path = tmp_path / "marker.json"
    monkeypatch.setenv(SELFTEST_MARKER_ENV, str(marker_path))
    monkeypatch.setattr(
        "sys.stdin",
        type(
            "FakeStdin",
            (),
            {
                "read": staticmethod(
                    lambda: _json.dumps(
                        {
                            "hook_event_name": "SessionStart",
                            "session_id": "sess-1",
                            "cwd": str(tmp_path),
                            "model": "gpt-test",
                            "permission_mode": "auto",
                            "transcript_path": None,
                            "source": "user",
                        }
                    )
                )
            },
        )(),
    )

    main()

    output = _json.loads(capsys.readouterr().out.strip())
    assert output["continue"] is False
    assert "stopReason" in output
    # Marker is still written so the provider can verify hook fired.
    assert marker_path.exists()


@pytest.mark.unit
def test_main_continues_session_start_in_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Outside selftest mode, the SessionStart handler returns
    ``{"continue": true}`` so real Codex sessions proceed normally.

    Adding SessionStart to the bundled plugin's hooks.json means the
    hook fires on EVERY Codex session start in production. The handler
    must be a quiet no-op there (no marker written, no abort signal),
    just a small subprocess-spawn latency.
    """
    import json as _json

    from src.infra.hooks.codex.selftest_policy import SELFTEST_MARKER_ENV
    from src.infra.hooks.codex_pre_tool_use import main

    monkeypatch.delenv(SELFTEST_MARKER_ENV, raising=False)
    monkeypatch.setattr(
        "sys.stdin",
        type(
            "FakeStdin",
            (),
            {
                "read": staticmethod(
                    lambda: _json.dumps(
                        {"hook_event_name": "SessionStart", "session_id": "s"}
                    )
                )
            },
        )(),
    )

    main()

    output = _json.loads(capsys.readouterr().out.strip())
    assert output == {"continue": True}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_main_pre_tool_use_selftest_omits_universal_continue_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """PreToolUse selftest must use only the per-event deny fields.

    Codex's PreToolUse output parser rejects the universal
    ``continue``/``stopReason`` fields as unsupported, and the
    PreToolUse runner fails open on invalid hook output. Emitting
    them would let the sentinel tool call proceed even though the
    selftest marker was written and the probe appeared to pass —
    silently defeating the deny/abort the live dispatch probe relies
    on.
    """
    import json as _json

    from src.infra.hooks.codex.selftest_policy import (
        SELFTEST_MARKER_ENV,
        SELFTEST_TARGET_EVENT_ENV,
    )
    from src.infra.hooks.codex_pre_tool_use import main

    marker_path = tmp_path / "PreToolUse.json"
    monkeypatch.setenv(SELFTEST_MARKER_ENV, str(marker_path))
    monkeypatch.setenv(SELFTEST_TARGET_EVENT_ENV, "PreToolUse")
    monkeypatch.setattr(
        "sys.stdin",
        type(
            "FakeStdin",
            (),
            {
                "read": staticmethod(
                    lambda: _json.dumps(
                        {
                            "hook_event_name": "PreToolUse",
                            "session_id": "sess-1",
                            "cwd": str(tmp_path),
                            "tool_name": "shell",
                            "tool_input": {},
                        }
                    )
                )
            },
        )(),
    )

    main()

    output = _json.loads(capsys.readouterr().out.strip())
    assert "continue" not in output
    assert "stopReason" not in output
    assert output["decision"] == "block"
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
