"""Focused unit tests for ``src.infra.hooks.codex.dangerous_commands``.

These cover every destructive-command classification rule called out by
issue T_B11 (B.3 sub-commit), one rule per test:

* **deny-message template** — ``_msg_dangerous`` produces the exact
  pinned wording the hook surfaces and the golden corpus depends on.
* **dangerous substring patterns** — every literal in
  ``DANGEROUS_PATTERNS`` (``rm -rf /``, fork bomb, ``mkfs.``, ``dd if=``,
  etc.) is matched when present in the command.
* **destructive git substring patterns** — every literal in
  ``DESTRUCTIVE_GIT_PATTERNS`` is matched (``git reset --hard``,
  ``git clean -f``, ``git stash``, ``git rebase``, ``git restore``,
  ``git commit --amend``, ``git worktree remove`` etc.).
* **token-walk destructive git** — ``_detect_destructive_git_in_subtokens``
  recognises destructive subcommands when global options
  (``git -C dir <sub>``) shift the subcommand position past the
  substring matcher's window. One test per detected variant.
* **token-walk via _detect_dangerous_pattern** — separator normalisation
  lets the token-walk recover destructive git invocations chained
  through ``;``/``&&`` glue (``git add f;git commit -a``).
* **atomic add+commit gate** — a ``git commit`` with no preceding
  ``git add`` in the same command returns the
  ``"git commit without prior git add"`` label.
* **safe alternatives** — ``--force-with-lease`` is allowed, ``git
  branch -d`` without ``-f`` is allowed, plain reads are allowed.
* **substitution recursion** — ``_detect_dangerous_pattern_recursive``
  walks into backtick / ``$()`` bodies AND short-circuits on commented
  out destructive commands AND tolerates malformed substitutions.
* **shlex ValueError tolerance** — an unbalanced quote in the outer
  command does not raise; classification proceeds with an empty token
  stream and falls back to the substring-only path.

The end-to-end byte-identical contract for ``codex_pre_tool_use`` is
preserved by the golden corpus harness; these tests pin the extracted
classification's internal shapes so a regression here is caught without
rerunning the whole hook.
"""

from __future__ import annotations

import pytest

from src.infra.hooks.codex.dangerous_commands import (
    _detect_dangerous_pattern,
    _detect_dangerous_pattern_recursive,
    _detect_destructive_git_in_subtokens,
    _msg_dangerous,
)
from src.infra.hooks.dangerous_commands import (
    DANGEROUS_PATTERNS,
    DESTRUCTIVE_GIT_PATTERNS,
)


# ----- message template -------------------------------------------------


@pytest.mark.unit
class TestMsgDangerous:
    def test_pinned_wording(self) -> None:
        # Golden corpus and reviewer grep depend on this exact string.
        assert (
            _msg_dangerous("git stash")
            == "command matches dangerous pattern: git stash"
        )

    def test_arbitrary_pattern_is_interpolated(self) -> None:
        assert (
            _msg_dangerous("custom label")
            == "command matches dangerous pattern: custom label"
        )


# ----- substring patterns ------------------------------------------------


@pytest.mark.unit
class TestDangerousSubstringPatterns:
    """Every literal in ``DANGEROUS_PATTERNS`` is matched as a substring."""

    @pytest.mark.parametrize("pattern", DANGEROUS_PATTERNS)
    def test_each_dangerous_pattern_is_detected(self, pattern: str) -> None:
        assert _detect_dangerous_pattern(pattern) == pattern

    def test_dangerous_pattern_inside_larger_command_detected(self) -> None:
        # Substring-anywhere semantics: pattern wrapped in other tokens.
        cmd = "echo hello && rm -rf / && echo done"
        assert _detect_dangerous_pattern(cmd) == "rm -rf /"

    def test_first_dangerous_pattern_in_list_wins(self) -> None:
        # Substring scan order is the list order; ``rm -rf /`` precedes
        # ``mkfs.`` so a command containing both returns ``rm -rf /``.
        cmd = "rm -rf / && mkfs.ext4 /dev/sda1"
        assert _detect_dangerous_pattern(cmd) == "rm -rf /"


@pytest.mark.unit
class TestDestructiveGitSubstringPatterns:
    """Every literal in ``DESTRUCTIVE_GIT_PATTERNS`` is classified as destructive.

    A few patterns are prefixes of others (``git checkout --`` is a
    prefix of ``git checkout --force``; ``git rebase`` is a prefix of
    ``git rebase --abort``). The substring scan is first-match-wins on
    list order, so the returned label is the first prefix in the list,
    not the input string. The contract these tests pin is "every entry
    is detected as destructive (returns a non-empty label that itself
    appears in the deny list)" — not strict label-equality.
    """

    @pytest.mark.parametrize("pattern", DESTRUCTIVE_GIT_PATTERNS)
    def test_each_destructive_git_pattern_is_detected(self, pattern: str) -> None:
        result = _detect_dangerous_pattern(pattern)
        assert result is not None
        assert result in DESTRUCTIVE_GIT_PATTERNS


# ----- _detect_destructive_git_in_subtokens (per-rule) ------------------


@pytest.mark.unit
class TestDestructiveGitInSubtokens:
    """One test per rule encoded in the token-walk helper."""

    def test_non_git_first_token_returns_none(self) -> None:
        assert _detect_destructive_git_in_subtokens(["echo", "hi"]) is None

    def test_empty_toks_returns_none(self) -> None:
        assert _detect_destructive_git_in_subtokens([]) is None

    def test_git_with_no_subcommand_returns_none(self) -> None:
        # Only global options, no subcommand → cannot classify.
        assert _detect_destructive_git_in_subtokens(["git", "-C", "dir"]) is None

    # --- stash / rebase / restore (bare destructive subcommands) ---

    def test_git_stash_after_global_option(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "-C", "/tmp", "stash"])
            == "git stash"
        )

    def test_git_rebase_after_global_option(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "-C", "/tmp", "rebase"])
            == "git rebase"
        )

    def test_git_restore_after_global_option(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "-C", "/tmp", "restore"])
            == "git restore"
        )

    # --- reset variants ---

    def test_git_reset_hard(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "reset", "--hard"])
            == "git reset --hard"
        )

    def test_git_reset_mixed(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "reset", "--mixed"])
            == "git reset --mixed"
        )

    def test_git_reset_soft(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "reset", "--soft"])
            == "git reset --soft"
        )

    def test_git_reset_head(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "reset", "HEAD"])
            == "git reset HEAD"
        )

    def test_git_reset_without_destructive_arg_returns_none(self) -> None:
        # Bare ``git reset`` (no --hard/--mixed/--soft/HEAD) is not flagged
        # by the token-walk; substring-only paths may still flag based on
        # DESTRUCTIVE_GIT_PATTERNS, but the token-walk is conservative.
        assert _detect_destructive_git_in_subtokens(["git", "reset"]) is None

    # --- checkout variants ---

    def test_git_checkout_double_dash(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "checkout", "--", "file.py"])
            == "git checkout --"
        )

    def test_git_checkout_force_short(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "checkout", "-f", "branch"])
            == "git checkout -f"
        )

    def test_git_checkout_force_long(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "checkout", "--force"])
            == "git checkout -f"
        )

    def test_git_checkout_safe_returns_none(self) -> None:
        # A plain ``git checkout branch`` is not destructive on its own.
        assert _detect_destructive_git_in_subtokens(["git", "checkout", "main"]) is None

    # --- clean ---

    def test_git_clean_bundled_short_with_f(self) -> None:
        # ``-fd`` bundles ``-f`` + ``-d``; both forms must be detected.
        assert (
            _detect_destructive_git_in_subtokens(["git", "clean", "-fd"])
            == "git clean -f"
        )

    def test_git_clean_separate_short_f(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "clean", "-f"])
            == "git clean -f"
        )

    def test_git_clean_without_force_returns_none(self) -> None:
        # Token-walk only flags clean when a ``-...f...`` short flag
        # appears; ``-d`` alone is safe (dry-run-ish).
        assert _detect_destructive_git_in_subtokens(["git", "clean", "-d"]) is None

    # --- branch ---

    def test_git_branch_capital_D(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "branch", "-D", "feature"])
            == "git branch -D"
        )

    def test_git_branch_delete_force_long(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(
                ["git", "branch", "--delete", "--force", "feature"]
            )
            == "git branch -d -f"
        )

    def test_git_branch_delete_force_separate_short(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(
                ["git", "branch", "-d", "-f", "feature"]
            )
            == "git branch -d -f"
        )

    def test_git_branch_bundled_df(self) -> None:
        # ``-df`` bundles ``-d`` + ``-f`` and is equally destructive.
        assert (
            _detect_destructive_git_in_subtokens(["git", "branch", "-df", "feature"])
            == "git branch -d -f"
        )

    def test_git_branch_bundled_fd(self) -> None:
        # Reverse short-flag order is also detected.
        assert (
            _detect_destructive_git_in_subtokens(["git", "branch", "-fd", "feature"])
            == "git branch -d -f"
        )

    def test_git_branch_safe_delete_returns_none(self) -> None:
        # Plain ``git branch -d`` (without -f) is not destructive — git
        # refuses to delete unmerged branches without force.
        assert (
            _detect_destructive_git_in_subtokens(["git", "branch", "-d", "feature"])
            is None
        )

    # --- commit ---

    def test_git_commit_amend(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "commit", "--amend"])
            == "git commit --amend"
        )

    def test_git_commit_all_long(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "commit", "--all", "-m", "x"])
            == "git commit -a/--all"
        )

    def test_git_commit_a_short(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "commit", "-a", "-m", "x"])
            == "git commit -a/--all"
        )

    def test_git_commit_bundled_short_with_a(self) -> None:
        # ``-am`` bundles ``-a`` + ``-m``; the bundled ``a`` must be
        # recognised the same as ``-a``.
        assert (
            _detect_destructive_git_in_subtokens(["git", "commit", "-am", "x"])
            == "git commit -a/--all"
        )

    def test_git_commit_with_only_m_is_not_destructive(self) -> None:
        # ``git commit -m`` (no ``a``/``--all``) is not flagged by the
        # commit-rule branch.
        assert (
            _detect_destructive_git_in_subtokens(["git", "commit", "-m", "msg"]) is None
        )

    # --- push / force ---

    def test_git_push_force_long(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "push", "--force"])
            == "git push --force"
        )

    def test_git_push_force_short(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "push", "-f"])
            == "git push --force"
        )

    def test_git_push_force_bundled_short(self) -> None:
        # Bundled short flag containing ``f`` (e.g. ``-uf``) is a force.
        assert (
            _detect_destructive_git_in_subtokens(["git", "push", "-uf", "origin"])
            == "git push --force"
        )

    def test_git_push_force_with_lease_is_allowed(self) -> None:
        # The documented safer alternative is explicitly allowed.
        assert (
            _detect_destructive_git_in_subtokens(
                ["git", "push", "--force-with-lease", "origin"]
            )
            is None
        )

    def test_git_push_force_with_lease_overrides_force(self) -> None:
        # When BOTH appear, the with-lease guard wins (no --force/-f
        # branch is taken).
        assert (
            _detect_destructive_git_in_subtokens(
                ["git", "push", "--force-with-lease", "--force", "origin"]
            )
            is None
        )

    def test_git_push_safe_returns_none(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "push", "origin", "main"])
            is None
        )

    # --- merge / cherry-pick abort ---

    def test_git_merge_abort(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "merge", "--abort"])
            == "git merge --abort"
        )

    def test_git_cherry_pick_abort(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(["git", "cherry-pick", "--abort"])
            == "git cherry-pick --abort"
        )

    def test_git_merge_safe_returns_none(self) -> None:
        # Plain merge is not aborted; not destructive.
        assert _detect_destructive_git_in_subtokens(["git", "merge", "feature"]) is None

    # --- worktree / submodule ---

    def test_git_worktree_remove(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(
                ["git", "worktree", "remove", "/tmp/wt"]
            )
            == "git worktree remove"
        )

    def test_git_worktree_safe_returns_none(self) -> None:
        # ``git worktree list`` is read-only.
        assert _detect_destructive_git_in_subtokens(["git", "worktree", "list"]) is None

    def test_git_submodule_deinit_force(self) -> None:
        assert (
            _detect_destructive_git_in_subtokens(
                ["git", "submodule", "deinit", "-f", "lib"]
            )
            == "git submodule deinit -f"
        )

    def test_git_submodule_deinit_without_force_returns_none(self) -> None:
        # The plain deinit form (no -f) is allowed.
        assert (
            _detect_destructive_git_in_subtokens(["git", "submodule", "deinit", "lib"])
            is None
        )


# ----- _detect_dangerous_pattern: token-walk recovery -------------------


@pytest.mark.unit
class TestTokenWalkRecovery:
    """Behaviour that depends on _normalize_separators + shlex tokenization."""

    def test_separator_normalisation_recovers_glued_git_commit(self) -> None:
        # ``git add f;git commit -a`` would tokenize the second ``git``
        # as ``f;git`` without normalisation — the destructive commit
        # would slip past the token-walk. The fix is in
        # _normalize_separators.
        cmd = "git add f;git commit -a -m x"
        assert _detect_dangerous_pattern(cmd) == "git commit -a/--all"

    def test_separator_normalisation_recovers_after_double_amp(self) -> None:
        cmd = "echo hi&&git stash"
        assert _detect_dangerous_pattern(cmd) == "git stash"

    def test_destructive_git_after_global_option_via_token_walk(self) -> None:
        # ``git -C /tmp stash`` does not contain the literal substring
        # ``git stash`` — the token-walk is the only path that catches it.
        cmd = "git -C /tmp stash"
        assert _detect_dangerous_pattern(cmd) == "git stash"

    def test_shlex_value_error_falls_back_to_substring_only(self) -> None:
        # An unbalanced quote raises ValueError in shlex.split. The
        # try/except keeps the function alive; the substring-only path
        # still catches the dangerous pattern.
        cmd = "rm -rf / 'unbalanced"
        assert _detect_dangerous_pattern(cmd) == "rm -rf /"

    def test_shlex_value_error_with_no_substring_match_returns_none(self) -> None:
        # Same path, but no dangerous substring — returns None rather
        # than raising.
        cmd = "echo 'unbalanced"
        assert _detect_dangerous_pattern(cmd) is None


# ----- atomic add+commit gate -------------------------------------------


@pytest.mark.unit
class TestAtomicAddCommitGate:
    def test_commit_without_prior_add_is_flagged(self) -> None:
        # ``git commit -m 'msg'`` with no preceding ``git add`` is the
        # canonical case the gate rejects.
        cmd = "git commit -m 'msg'"
        assert _detect_dangerous_pattern(cmd) == "git commit without prior git add"

    def test_commit_after_prior_add_is_allowed(self) -> None:
        cmd = "git add file.py && git commit -m 'msg'"
        assert _detect_dangerous_pattern(cmd) is None

    def test_commit_before_add_in_same_command_is_flagged(self) -> None:
        # Order matters: ``git commit && git add`` still counts as
        # "commit without prior add" because the token-walk sees the
        # commit first.
        cmd = "git commit -m x && git add file.py"
        assert _detect_dangerous_pattern(cmd) == "git commit without prior git add"


# ----- _detect_dangerous_pattern_recursive ------------------------------


@pytest.mark.unit
class TestDetectDangerousPatternRecursive:
    def test_top_level_dangerous_pattern_is_detected(self) -> None:
        assert _detect_dangerous_pattern_recursive("git stash") == "git stash"

    def test_safe_command_returns_none(self) -> None:
        assert _detect_dangerous_pattern_recursive("echo hello") is None

    def test_destructive_git_in_backtick_body_is_detected(self) -> None:
        # ``echo `git -C /tmp stash``` runs the inner stash for real
        # when bash evaluates the substitution; the recursion must look
        # into backtick bodies. (Plan: closes the substitution-bypass
        # gap left open by the outer token-walk's placeholder
        # substitution.)
        cmd = "echo `git -C /tmp stash`"
        assert _detect_dangerous_pattern_recursive(cmd) == "git stash"

    def test_destructive_git_in_dollar_paren_body_is_detected(self) -> None:
        cmd = "echo $(git -C /tmp stash)"
        assert _detect_dangerous_pattern_recursive(cmd) == "git stash"

    def test_nested_substitution_body_is_detected(self) -> None:
        # Substitution within substitution: the recursion must drain
        # every level of the pending stack.
        cmd = "echo $(echo $(git stash))"
        assert _detect_dangerous_pattern_recursive(cmd) == "git stash"

    def test_commented_destructive_git_is_not_flagged(self) -> None:
        # ``_strip_shell_comments`` is applied at every level so a
        # commented-out destructive command is ignored.
        cmd = "# git -C /tmp stash"
        assert _detect_dangerous_pattern_recursive(cmd) is None

    def test_commented_destructive_git_inside_substitution_is_not_flagged(
        self,
    ) -> None:
        # Comment-stripping must apply to every recursion level, not
        # just the outer command.
        cmd = "echo $(\n# git stash\necho hi)"
        assert _detect_dangerous_pattern_recursive(cmd) is None
