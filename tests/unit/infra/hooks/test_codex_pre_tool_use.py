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
