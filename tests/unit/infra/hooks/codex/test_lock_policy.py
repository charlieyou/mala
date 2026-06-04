"""Focused unit tests for ``src.infra.hooks.codex.lock_policy``.

These cover every lock state and write-permission decision called out
by issue T_B10:

* **unresolved sentinel** target — the parseable-but-unresolved write
  expression always denies (plan L846).
* **substitution placeholder** target — tokens carrying the
  ``__MALA_CMDSUB_LITERAL__`` / ``__MALA_ANSI_C_LITERAL__`` markers
  always deny.
* **shell expansion metachar** target — glob / brace / parameter /
  tilde characters always deny.
* **no lock holder** — ``get_lock_holder`` returns ``None`` → deny
  with the ``no active lock`` message.
* **wrong owner** — ``get_lock_holder`` returns a different agent
  → deny with the ``locked by agent`` message.
* **correct owner** — ``get_lock_holder`` returns ``agent_id`` →
  allow (return ``None``).
* **gate iteration** — :func:`_gate_write_targets` returns the FIRST
  deny reason and short-circuits, returns ``None`` when every target
  is owned.
* **cwd resolution** — :func:`_resolve_against_cwd` is a pure helper
  that joins relative targets against the payload ``cwd`` (absolute
  / empty / unresolved-sentinel / missing-cwd are pass-throughs).

The end-to-end byte-identical contract for ``codex_pre_tool_use`` is
preserved by the golden corpus harness; these tests pin the extracted
policy's internal shapes so a regression here is caught without
rerunning the whole hook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.infra.hooks.codex.lock_policy import (
    _MSG_ENV_MISSING,
    _SHELL_EXPANSION_RE,
    _check_lock,
    _gate_write_targets,
    _is_validation_log_path,
    _msg_no_lock,
    _msg_wrong_owner,
    _resolve_against_cwd,
)
from src.infra.hooks.codex.shell_parser import (
    _ANSI_C_PLACEHOLDER,
    _CMDSUB_PLACEHOLDER,
)
from src.infra.hooks.codex.write_targets import _UNRESOLVED_SENTINEL
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


# ----- message templates ------------------------------------------------


@pytest.mark.unit
class TestMessageTemplates:
    def test_msg_no_lock_uses_target_and_agent(self) -> None:
        # Pinned wording — golden corpus and reviewer grep depend on it.
        assert (
            _msg_no_lock("/repo/file.py", "agent-me")
            == "path '/repo/file.py' has no active lock for agent 'agent-me'"
        )

    def test_msg_wrong_owner_names_both_agents(self) -> None:
        assert (
            _msg_wrong_owner("/repo/file.py", "agent-other", "agent-me")
            == "path '/repo/file.py' is locked by agent 'agent-other'; "
            "current agent is 'agent-me'"
        )

    def test_msg_env_missing_is_the_documented_string(self) -> None:
        # The env-missing message is a constant rather than a function
        # because the lock-env triad is not target-specific.
        assert _MSG_ENV_MISSING == (
            "MALA_AGENT_ID/MALA_LOCK_DIR/MALA_REPO_NAMESPACE missing; "
            "refusing to evaluate lock"
        )


# ----- _resolve_against_cwd helper --------------------------------------


@pytest.mark.unit
class TestResolveAgainstCwd:
    def test_absolute_target_is_passed_through(self) -> None:
        assert _resolve_against_cwd("/abs/path.py", "/cwd") == "/abs/path.py"

    def test_relative_target_is_joined_against_cwd(self) -> None:
        # Without resolution a cwd-relative write would be lock-checked
        # against the namespace root and a wrong-directory lock would
        # silently allow the write.
        assert _resolve_against_cwd("file.py", "/cwd/pkg") == "/cwd/pkg/file.py"

    def test_empty_target_is_passed_through(self) -> None:
        assert _resolve_against_cwd("", "/cwd") == ""

    def test_unresolved_sentinel_is_passed_through(self) -> None:
        # The sentinel is not a real path; joining it with cwd would
        # produce a meaningless lock key.
        assert (
            _resolve_against_cwd(_UNRESOLVED_SENTINEL, "/cwd") == _UNRESOLVED_SENTINEL
        )

    def test_missing_cwd_returns_target_unchanged(self) -> None:
        assert _resolve_against_cwd("file.py", None) == "file.py"
        assert _resolve_against_cwd("file.py", "") == "file.py"


# ----- _SHELL_EXPANSION_RE ----------------------------------------------


@pytest.mark.unit
class TestShellExpansionRegex:
    @pytest.mark.parametrize(
        "target",
        [
            "*.py",  # glob star
            "f?.py",  # glob question
            "[ab].py",  # glob char class
            "f{1,2}.py",  # brace expansion
            "$VAR",  # parameter expansion
            "~/foo",  # tilde expansion
        ],
    )
    def test_matches_metachars(self, target: str) -> None:
        assert _SHELL_EXPANSION_RE.search(target) is not None

    @pytest.mark.parametrize(
        "target",
        [
            "plain/file.py",
            "/abs/path",
            "with-dashes_and_dots.py",
        ],
    )
    def test_does_not_match_plain_paths(self, target: str) -> None:
        assert _SHELL_EXPANSION_RE.search(target) is None


# ----- validation-log exemption ------------------------------------------


@pytest.mark.unit
class TestValidationLogPath:
    def test_matches_directory_and_file_under_configured_dir(
        self, tmp_path: Path
    ) -> None:
        log_dir = tmp_path / "validation-logs"
        assert _is_validation_log_path(str(log_dir), str(log_dir), cwd=None)
        assert _is_validation_log_path(
            str(log_dir / "issue.test.log"), str(log_dir), cwd=None
        )

    def test_does_not_match_sibling_prefix_or_traversal(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "validation-logs"
        assert not _is_validation_log_path(
            str(tmp_path / "validation-logs-evil" / "issue.test.log"),
            str(log_dir),
            cwd=None,
        )
        assert not _is_validation_log_path(
            str(log_dir / ".." / "repo" / "file.py"),
            str(log_dir),
            cwd=None,
        )

    def test_does_not_match_root_or_dynamic_targets(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "validation-logs"
        assert not _is_validation_log_path("/tmp/anything.log", "/", cwd=None)
        assert not _is_validation_log_path(
            str(log_dir / "$name.log"), str(log_dir), cwd=None
        )
        assert not _is_validation_log_path(_CMDSUB_PLACEHOLDER, str(log_dir), cwd=None)

    def test_gate_skips_validation_logs_without_locks(
        self, lock_dir: Path, repo: Path, tmp_path: Path
    ) -> None:
        log_dir = tmp_path / "validation-logs"
        reason = _gate_write_targets(
            [str(log_dir), str(log_dir / "issue.test.log")],
            agent_id="agent-me",
            lock_dir=str(lock_dir),
            repo_namespace=str(repo),
            cwd=None,
            canonical_validation_log_dir=str(log_dir),
            canonical_validation_log_targets=frozenset(
                {str(log_dir), str(log_dir / "issue.test.log")}
            ),
        )
        assert reason is None

    def test_gate_still_denies_non_exempt_validation_descendant(
        self, lock_dir: Path, repo: Path, tmp_path: Path
    ) -> None:
        log_dir = tmp_path / "validation-logs"
        target = log_dir / "extra.log"
        reason = _gate_write_targets(
            [str(target)],
            agent_id="agent-me",
            lock_dir=str(lock_dir),
            repo_namespace=str(repo),
            cwd=None,
            canonical_validation_log_dir=str(log_dir),
            canonical_validation_log_targets=frozenset(
                {str(log_dir), str(log_dir / "issue.test.log")}
            ),
        )
        assert reason == _msg_no_lock(str(target), "agent-me")


# ----- _check_lock: pre-lookup deny branches ----------------------------


@pytest.mark.unit
class TestCheckLockPreLookup:
    """States that deny BEFORE any filesystem lock-holder lookup."""

    def test_unresolved_sentinel_denies_with_no_lock_message(
        self, lock_dir: Path, repo: Path
    ) -> None:
        # plan L846: unresolved-target sentinel always denies.
        reason = _check_lock(
            _UNRESOLVED_SENTINEL,
            "agent-me",
            str(lock_dir),
            str(repo),
            cwd=None,
        )
        assert reason == _msg_no_lock(_UNRESOLVED_SENTINEL, "agent-me")

    def test_cmdsub_placeholder_in_target_denies(
        self, lock_dir: Path, repo: Path
    ) -> None:
        # Tokens carrying the cmdsub marker resolve to whatever bash
        # produces at runtime — locking the literal would be a bypass.
        target = f"out_{_CMDSUB_PLACEHOLDER}_suffix"
        reason = _check_lock(target, "agent-me", str(lock_dir), str(repo), cwd=None)
        assert reason == _msg_no_lock(target, "agent-me")

    def test_ansi_c_placeholder_in_target_denies(
        self, lock_dir: Path, repo: Path
    ) -> None:
        target = f"out_{_ANSI_C_PLACEHOLDER}"
        reason = _check_lock(target, "agent-me", str(lock_dir), str(repo), cwd=None)
        assert reason == _msg_no_lock(target, "agent-me")

    @pytest.mark.parametrize(
        "target",
        ["*.py", "f?.py", "[ab].py", "f{1,2}.py", "$VAR", "~/foo"],
    )
    def test_shell_expansion_metachar_denies(
        self, target: str, lock_dir: Path, repo: Path
    ) -> None:
        reason = _check_lock(target, "agent-me", str(lock_dir), str(repo), cwd=None)
        assert reason == _msg_no_lock(target, "agent-me")


# ----- _check_lock: filesystem holder-lookup branches -------------------


@pytest.mark.unit
class TestCheckLockHolderLookup:
    """States that depend on the on-disk lock state."""

    def test_no_holder_denies_with_no_lock_message(
        self, lock_dir: Path, repo: Path
    ) -> None:
        target = str(repo / "src.py")
        reason = _check_lock(target, "agent-me", str(lock_dir), str(repo), cwd=None)
        assert reason == _msg_no_lock(target, "agent-me")

    def test_wrong_owner_denies_with_owner_message(
        self, lock_dir: Path, repo: Path
    ) -> None:
        target = str(repo / "src.py")
        assert try_lock(target, "agent-other", repo_namespace=str(repo))

        reason = _check_lock(target, "agent-me", str(lock_dir), str(repo), cwd=None)
        assert reason == _msg_wrong_owner(target, "agent-other", "agent-me")

    def test_correct_owner_allows(self, lock_dir: Path, repo: Path) -> None:
        target = str(repo / "src.py")
        assert try_lock(target, "agent-me", repo_namespace=str(repo))

        reason = _check_lock(target, "agent-me", str(lock_dir), str(repo), cwd=None)
        assert reason is None

    def test_relative_target_resolves_against_cwd_before_lookup(
        self, lock_dir: Path, repo: Path
    ) -> None:
        # Lock the absolute path; pass the relative form + cwd to
        # _check_lock and confirm it finds the lock (i.e. the cwd
        # resolution happened before lock-key derivation).
        pkg = repo / "pkg"
        pkg.mkdir()
        abs_path = pkg / "f.py"
        assert try_lock(str(abs_path), "agent-me", repo_namespace=str(repo))

        reason = _check_lock(
            "f.py",
            "agent-me",
            str(lock_dir),
            str(repo),
            cwd=str(pkg),
        )
        assert reason is None

    def test_deny_message_uses_original_target_not_resolved_path(
        self, lock_dir: Path, repo: Path
    ) -> None:
        # The lock-key is computed from the cwd-resolved absolute path,
        # but the deny message preserves the caller-visible target for
        # readability.
        pkg = repo / "pkg"
        pkg.mkdir()
        reason = _check_lock(
            "f.py",
            "agent-me",
            str(lock_dir),
            str(repo),
            cwd=str(pkg),
        )
        assert reason == _msg_no_lock("f.py", "agent-me")


# ----- _gate_write_targets ---------------------------------------------


@pytest.mark.unit
class TestGateWriteTargets:
    def test_empty_iterable_allows(self, lock_dir: Path, repo: Path) -> None:
        reason = _gate_write_targets(
            [],
            agent_id="agent-me",
            lock_dir=str(lock_dir),
            repo_namespace=str(repo),
            cwd=None,
        )
        assert reason is None

    def test_all_owned_allows(self, lock_dir: Path, repo: Path) -> None:
        a = str(repo / "a.py")
        b = str(repo / "b.py")
        assert try_lock(a, "agent-me", repo_namespace=str(repo))
        assert try_lock(b, "agent-me", repo_namespace=str(repo))

        reason = _gate_write_targets(
            [a, b],
            agent_id="agent-me",
            lock_dir=str(lock_dir),
            repo_namespace=str(repo),
            cwd=None,
        )
        assert reason is None

    def test_returns_first_deny_short_circuiting(
        self, lock_dir: Path, repo: Path
    ) -> None:
        # First target has no lock → deny on it (short-circuit). Second
        # target is unowned too, but the gate must not advance past the
        # first deny.
        first = str(repo / "first.py")
        second = str(repo / "second.py")
        reason = _gate_write_targets(
            [first, second],
            agent_id="agent-me",
            lock_dir=str(lock_dir),
            repo_namespace=str(repo),
            cwd=None,
        )
        assert reason == _msg_no_lock(first, "agent-me")

    def test_middle_target_deny_is_returned(self, lock_dir: Path, repo: Path) -> None:
        a = str(repo / "a.py")
        b = str(repo / "b.py")
        c = str(repo / "c.py")
        assert try_lock(a, "agent-me", repo_namespace=str(repo))
        # b has no lock — must be the returned deny reason.
        assert try_lock(c, "agent-me", repo_namespace=str(repo))

        reason = _gate_write_targets(
            [a, b, c],
            agent_id="agent-me",
            lock_dir=str(lock_dir),
            repo_namespace=str(repo),
            cwd=None,
        )
        assert reason == _msg_no_lock(b, "agent-me")

    def test_wrong_owner_propagates_through_gate(
        self, lock_dir: Path, repo: Path
    ) -> None:
        target = str(repo / "src.py")
        assert try_lock(target, "agent-other", repo_namespace=str(repo))

        reason = _gate_write_targets(
            [target],
            agent_id="agent-me",
            lock_dir=str(lock_dir),
            repo_namespace=str(repo),
            cwd=None,
        )
        assert reason == _msg_wrong_owner(target, "agent-other", "agent-me")
