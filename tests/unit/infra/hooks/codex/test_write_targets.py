"""Focused unit tests for ``src.infra.hooks.codex.write_targets``.

These cover the five branches called out by issue T_B9:

* **redirection** — shell redirection operators (``> file``, ``2>> log``,
  ``&> all``, ``<> rw``) flowing through ``_extract_shell_write_paths``,
* **utilities** — file-mutation utilities (cp/mv/rm/touch/mkdir/chmod/
  ``-t target-dir``, ``tee``, sed/perl in-place, awk in-place, language
  one-liner ``-c``/``-e`` write calls),
* **git** — ``git checkout``/``restore``/``apply``/``stash`` with
  ``-C`` / ``--work-tree`` path-resolution base composition,
* **apply_patch** — Codex envelope (``*** Update File: ...``) and
  unified-diff (``+++ b/...``) header extraction,
* **unresolved** — every fallback to ``_UNRESOLVED_SENTINEL`` (cd-shifted
  cwd, brace expansion, ``git apply``, ``patch``, dynamic Python/Node
  open targets, ``apply_patch`` Move-to dest).

The full byte-identical end-to-end contract for ``codex_pre_tool_use``
is preserved by the golden corpus harness; these tests pin the
extracted helpers' internal shapes so a regression here is caught
without rerunning the whole hook.
"""

from __future__ import annotations

import pytest

from src.infra.hooks.codex.write_targets import (
    GIT_WRITE_SUBCOMMANDS,
    UTILITY_STRATEGIES,
    _UNRESOLVED_SENTINEL,
    _apply_patch_paths,
    _extract_awk_inplace_targets,
    _extract_git_targets,
    _extract_inplace_targets,
    _extract_oneliner_targets,
    _extract_patch_targets,
    _extract_shell_write_paths,
    _extract_tee_targets,
    _extract_utility_targets,
)


# ----- redirection branch -----------------------------------------------


@pytest.mark.unit
class TestRedirectionBranch:
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("echo hi > out", ["out"]),
            ("echo hi >> log", ["log"]),
            ("cmd 2> stderr", ["stderr"]),
            ("cmd 2>> stderr.log", ["stderr.log"]),
            ("cmd &> all", ["all"]),
            ("cmd &>> all", ["all"]),
            ("cmd <> rwfile", ["rwfile"]),
            # ``>|`` force-clobber redirect.
            ("cmd >| forced", ["forced"]),
            # ``/dev/null`` is a documented non-write redirect target;
            # the redirection extractor drops it.
            ("cmd > /dev/null", []),
        ],
    )
    def test_basic_redirection_targets(self, command: str, expected: list[str]) -> None:
        assert _extract_shell_write_paths(command) == expected

    def test_redirection_glued_to_command_without_whitespace(self) -> None:
        # ``echo hi>file`` (no space) must still tokenise as a redirect;
        # the normalisation pass pads the operator before shlex.split.
        assert _extract_shell_write_paths("echo hi>file") == ["file"]

    def test_multiple_redirects_in_one_segment_are_all_extracted(self) -> None:
        # Both stdout and stderr surfaces get lock-checked.
        assert _extract_shell_write_paths("cmd > out 2>> err") == ["out", "err"]


# ----- utilities branch -------------------------------------------------


@pytest.mark.unit
class TestUtilitiesBranch:
    def test_cp_mv_install_ln_use_last_positional(self) -> None:
        assert UTILITY_STRATEGIES["cp"] == "last_positional"
        assert _extract_utility_targets(["cp", "src.py", "dest.py"]) == ["dest.py"]
        assert _extract_utility_targets(["mv", "a", "b"]) == ["b"]

    def test_touch_rm_mkdir_use_all_positional(self) -> None:
        assert _extract_utility_targets(["touch", "a", "b", "c"]) == ["a", "b", "c"]
        assert _extract_utility_targets(["rm", "x", "y"]) == ["x", "y"]
        assert _extract_utility_targets(["mkdir", "d1", "d2"]) == ["d1", "d2"]

    def test_chmod_skips_first_positional_for_mode(self) -> None:
        assert _extract_utility_targets(["chmod", "644", "f"]) == ["f"]

    def test_chmod_symbolic_mode_is_the_mode_positional(self) -> None:
        # ``chmod -w unowned.py`` — ``-w`` matches the symbolic-mode
        # pattern and is appended as the first positional (the mode);
        # ``skip_first_positional`` then drops it, leaving ``unowned.py``
        # as the lock-checked write target. Without the symbolic-mode
        # branch the ``-w`` would be skipped as a flag and the single
        # remaining positional dropped — silently allowing the write.
        assert _extract_utility_targets(["chmod", "-w", "unowned.py"]) == ["unowned.py"]

    def test_dd_extracts_of_argument(self) -> None:
        assert _extract_utility_targets(["dd", "if=/dev/zero", "of=disk.img"]) == [
            "disk.img"
        ]

    def test_cp_target_directory_flag_overrides_trailing_positional(self) -> None:
        # ``cp -t DIR src`` lock-checks DIR (not src). Per-utility ``-t``
        # bundle handling per ``_scan_short_bundle_for_target_dir``.
        assert _extract_utility_targets(["cp", "-t", "dir", "src"]) == ["dir"]
        assert _extract_utility_targets(["cp", "--target-directory=dir", "a", "b"]) == [
            "dir"
        ]

    def test_tee_targets_extracted(self) -> None:
        assert _extract_tee_targets(["tee", "-a", "out.log"]) == ["out.log"]

    def test_sed_inplace_emits_trailing_positionals(self) -> None:
        # ``sed -i 's/a/b/' f`` — first positional is the script, the
        # rest are file targets.
        assert _extract_inplace_targets(["sed", "-i", "s/a/b/", "f"]) == ["f"]

    def test_awk_inplace_emits_trailing_positionals(self) -> None:
        # gawk's documented ``-i inplace`` form.
        assert _extract_awk_inplace_targets(
            ["awk", "-i", "inplace", "{print}", "f"]
        ) == ["f"]

    def test_python_oneliner_open_write_extracts_literal_target(self) -> None:
        assert _extract_oneliner_targets(
            ["python", "-c", "open('out.txt', 'w').write('x')"]
        ) == ["out.txt"]

    def test_node_oneliner_writefilesync_extracts_literal_target(self) -> None:
        assert _extract_oneliner_targets(
            ["node", "-e", "fs.writeFileSync('out.js', 'x')"]
        ) == ["out.js"]


# ----- git branch -------------------------------------------------------


@pytest.mark.unit
class TestGitBranch:
    def test_git_write_subcommands_constant(self) -> None:
        assert GIT_WRITE_SUBCOMMANDS == frozenset(
            {"checkout", "restore", "apply", "stash"}
        )

    def test_git_checkout_emits_positional_pathspecs(self) -> None:
        assert _extract_git_targets(["git", "checkout", "HEAD", "file.py"]) == [
            "HEAD",
            "file.py",
        ]

    def test_git_restore_emits_pathspecs_after_dash_dash(self) -> None:
        assert _extract_git_targets(["git", "restore", "--", "f.py"]) == ["f.py"]

    def test_git_chdir_prefixes_relative_pathspecs(self) -> None:
        # ``-C pkg`` shifts the effective write base; relative pathspecs
        # are resolved against ``pkg/`` (per _git_effective_base).
        result = _extract_git_targets(["git", "-C", "pkg", "checkout", "f.py"])
        assert result == ["pkg/f.py"]

    def test_git_chdir_composes_multiple_C_flags(self) -> None:
        # ``-C pkg -C sub`` composes (per _compose_git_chdir).
        result = _extract_git_targets(
            ["git", "-C", "pkg", "-C", "sub", "checkout", "f.py"]
        )
        assert result == ["pkg/sub/f.py"]

    def test_git_work_tree_overrides_for_relative_paths(self) -> None:
        # ``--work-tree=wt`` resolves against post-``-C`` cwd.
        result = _extract_git_targets(
            ["git", "-C", "pkg", "--work-tree=wt", "checkout", "f"]
        )
        assert result == ["pkg/wt/f"]

    def test_git_non_write_subcommand_returns_empty(self) -> None:
        # ``git status`` / ``git log`` are read-only — no write targets.
        assert _extract_git_targets(["git", "status"]) == []


# ----- apply_patch branch -----------------------------------------------


@pytest.mark.unit
class TestApplyPatchBranch:
    def test_direct_path_key_extracted(self) -> None:
        assert _apply_patch_paths({"path": "file.py"}) == ["file.py"]

    def test_paths_list_extracted(self) -> None:
        assert _apply_patch_paths({"paths": ["a.py", "b.py"]}) == ["a.py", "b.py"]

    def test_codex_envelope_update_file_header(self) -> None:
        body = (
            "*** Begin Patch\n"
            "*** Update File: src/foo.py\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )
        assert _apply_patch_paths({"command": body}) == ["src/foo.py"]

    def test_codex_envelope_add_and_delete_headers(self) -> None:
        body = "*** Add File: created.py\n+new content\n*** Delete File: removed.py\n"
        assert _apply_patch_paths({"input": body}) == ["created.py", "removed.py"]

    def test_codex_envelope_move_to_destination_included(self) -> None:
        # ``*** Update File: old`` followed by ``*** Move to: new`` writes
        # both paths; lock-check on the destination prevents an agent
        # owning only the source from creating an unowned new file.
        body = "*** Update File: old.py\n*** Move to: new.py\n@@\n-x\n+y\n"
        assert _apply_patch_paths({"patch": body}) == ["old.py", "new.py"]

    def test_unified_diff_dest_header_extracted(self) -> None:
        body = "--- a/old.py\n+++ b/new.py\n@@ -1 +1 @@\n-x\n+y\n"
        assert _apply_patch_paths({"diff": body}) == ["new.py", "old.py"]

    def test_dev_null_excluded_from_diff_headers(self) -> None:
        body = "--- /dev/null\n+++ b/new.py\n@@\n+x\n"
        assert _apply_patch_paths({"diff": body}) == ["new.py"]

    def test_path_capture_strips_tab_separated_timestamp(self) -> None:
        # Envelope regex stops at ``\t`` so a trailing timestamp is not
        # folded into the path (avoids the literal ``path\t2026-...``
        # bypass).
        body = "*** Update File: foo.py\t2026-05-08 12:00\n@@\n"
        assert _apply_patch_paths({"command": body}) == ["foo.py"]

    def test_empty_input_yields_empty_paths(self) -> None:
        assert _apply_patch_paths({}) == []

    def test_dedup_preserves_first_match_order(self) -> None:
        body = "*** Update File: a.py\n*** Update File: a.py\n*** Update File: b.py\n"
        assert _apply_patch_paths({"command": body}) == ["a.py", "b.py"]


# ----- unresolved branch ------------------------------------------------


@pytest.mark.unit
class TestUnresolvedBranch:
    def test_cd_makes_following_relative_writes_unresolved(self) -> None:
        # Once ``cd`` runs, the hook can't trust relative-path resolution
        # against the payload cwd; every following relative write becomes
        # the sentinel so the lock-check fails closed.
        result = _extract_shell_write_paths("cd /tmp && touch f")
        assert _UNRESOLVED_SENTINEL in result

    def test_brace_expansion_in_segment_yields_sentinel(self) -> None:
        # ``{touch,unowned}`` expands to ``touch unowned`` at runtime;
        # shlex sees one token so the sentinel is emitted to fail closed.
        result = _extract_shell_write_paths("{touch,unowned}")
        assert _UNRESOLVED_SENTINEL in result

    def test_git_apply_yields_sentinel(self) -> None:
        assert _extract_git_targets(["git", "apply", "patchfile"]) == [
            _UNRESOLVED_SENTINEL
        ]

    def test_git_stash_apply_yields_sentinel(self) -> None:
        assert _extract_git_targets(["git", "stash", "apply"]) == [_UNRESOLVED_SENTINEL]

    def test_patch_command_yields_sentinel(self) -> None:
        # The patch utility writes paths from its diff body; the body
        # is not available statically so the call denies unless the
        # parent directory lock is held.
        assert _extract_patch_targets(["patch", "-p1"]) == [_UNRESOLVED_SENTINEL]

    def test_dynamic_python_open_target_yields_sentinel(self) -> None:
        # ``open(path_var, 'w')`` — dynamic target, not statically
        # extractable. The hint check triggers the sentinel.
        result = _extract_oneliner_targets(
            ["python", "-c", "open(path_var, 'w').write('x')"]
        )
        assert result == [_UNRESOLVED_SENTINEL]

    def test_dynamic_node_writefile_yields_sentinel(self) -> None:
        result = _extract_oneliner_targets(["node", "-e", "fs.writeFile(dyn, 'x', cb)"])
        assert result == [_UNRESOLVED_SENTINEL]

    def test_malformed_shell_input_yields_sentinel(self) -> None:
        # An unclosed quote that shlex can't parse must fail closed,
        # not silently allow.
        result = _extract_shell_write_paths('touch "unclosed')
        assert _UNRESOLVED_SENTINEL in result
