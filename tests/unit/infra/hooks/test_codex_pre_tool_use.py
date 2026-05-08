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
    return spec.get("permissionDecision") == "allow"


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
# State-file fallback transport (plan L817-L825)
# ---------------------------------------------------------------------------


class TestStateFileFallback:
    def test_reads_state_file_when_env_missing(
        self,
        repo: Path,
        lock_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force HOME to a temp dir so the hook reads from a controlled
        # state-file location instead of the real user's config.
        fake_home = tmp_path / "home"
        state_dir = fake_home / ".config" / "mala" / "agent-state"
        state_dir.mkdir(parents=True)
        session_id = "thr_state_only"
        (state_dir / f"{session_id}.env").write_text(
            f"MALA_AGENT_ID=agent-me\n"
            f"MALA_LOCK_DIR={lock_dir}\n"
            f"MALA_REPO_NAMESPACE={repo}\n"
        )
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.delenv("MALA_AGENT_ID", raising=False)
        monkeypatch.delenv("MALA_REPO_NAMESPACE", raising=False)
        monkeypatch.delenv("MALA_LOCK_DIR", raising=False)
        # Lock the file *after* unsetting env so try_lock uses the
        # state-file lock_dir via a manual override.
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_dir))
        target = repo / "src.py"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))
        # Now drop MALA_LOCK_DIR again so the hook is forced to source it
        # from the state file. The hook re-sets MALA_LOCK_DIR internally
        # before consulting the lock store.
        monkeypatch.delenv("MALA_LOCK_DIR", raising=False)

        result = decide(
            make_payload("apply_patch", {"path": str(target)}, session_id=session_id)
        )

        assert is_allow(result), result

    def test_env_overrides_state_file(
        self,
        repo: Path,
        lock_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        state_dir = fake_home / ".config" / "mala" / "agent-state"
        state_dir.mkdir(parents=True)
        session_id = "thr_env_wins"
        (state_dir / f"{session_id}.env").write_text(
            "MALA_AGENT_ID=agent-state\n"
            f"MALA_LOCK_DIR={lock_dir}\n"
            f"MALA_REPO_NAMESPACE={repo}\n"
        )
        monkeypatch.setenv("HOME", str(fake_home))
        # Env says agent-me; state file says agent-state. Lock is held by
        # agent-me, so allow path proves env won.
        monkeypatch.setenv("MALA_AGENT_ID", "agent-me")
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_dir))
        monkeypatch.setenv("MALA_REPO_NAMESPACE", str(repo))
        target = repo / "src.py"
        assert try_lock(str(target), "agent-me", repo_namespace=str(repo))

        result = decide(
            make_payload("apply_patch", {"path": str(target)}, session_id=session_id)
        )

        assert is_allow(result), result

    def test_state_file_missing_env_missing_denies(
        self,
        repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
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


# ---------------------------------------------------------------------------
# Output-shape contract
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_allow_shape(self, env: dict[str, str]) -> None:
        result = decide(make_payload("bash", {"command": "echo hi"}))
        assert result["decision"] == "approve"
        spec = result["hookSpecificOutput"]
        assert spec["permissionDecision"] == "allow"

    def test_deny_shape(self, env: dict[str, str]) -> None:
        result = decide(make_payload("bash", {"command": "rm -rf /"}))
        assert result["decision"] == "block"
        spec = result["hookSpecificOutput"]
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


class TestEmptyEnvOverridesStateFile:
    """Regression: ``MALA_DISALLOWED_TOOLS=""`` clears the state-file value.

    Before the fix, ``if env_val:`` treated an explicitly-empty env as
    absent and silently fell back to the state file, so an operator
    could not unset a stale state-file disallowed-tools list without
    deleting the file.
    """

    def test_empty_env_clears_state_disallowed_tools(
        self,
        repo: Path,
        lock_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        state_dir = fake_home / ".config" / "mala" / "agent-state"
        state_dir.mkdir(parents=True)
        session_id = "thr_clear"
        (state_dir / f"{session_id}.env").write_text(
            "MALA_AGENT_ID=agent-me\n"
            f"MALA_LOCK_DIR={lock_dir}\n"
            f"MALA_REPO_NAMESPACE={repo}\n"
            "MALA_DISALLOWED_TOOLS=TodoWrite\n"
        )
        monkeypatch.setenv("HOME", str(fake_home))
        # Explicitly-empty env should clear the state-file value.
        monkeypatch.setenv("MALA_DISALLOWED_TOOLS", "")

        # TodoWrite was disallowed by the state file but env clears it.
        result = decide(make_payload("TodoWrite", {}, session_id=session_id))

        assert is_allow(result), result


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
