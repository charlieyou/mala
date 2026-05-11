"""Focused unit tests for ``src.infra.hooks.codex.shell_parser``.

These cover the four behavior buckets called out by issue T_B8:

* tokenization (segment splitting, env / reserved-word stripping,
  shell-comment removal, fd-prefix handling),
* quoted strings (single, double, ANSI-C, and backslash-escaped quotes),
* nested command substitution (legacy backticks, ``$(...)``, arithmetic,
  recursive bodies),
* redirection (whole-token operators, fd dups, ``/dev/null``-like targets,
  ``<>`` read-write open, normalization padding).

The full byte-identical end-to-end contract for ``codex_pre_tool_use``
is preserved by the golden corpus harness; these tests pin the
extracted helpers' internal shapes so a regression here is caught
without rerunning the whole hook.
"""

from __future__ import annotations

import shlex

import pytest

from src.infra.hooks.codex.shell_parser import (
    SHELL_SEPARATORS,
    _ANSI_C_PLACEHOLDER,
    _CMDSUB_PLACEHOLDER,
    _SHELL_RESERVED_WORDS,
    _drop_redirections,
    _extract_command_substitutions,
    _extract_redirection_targets,
    _normalize_separators,
    _split_segments,
    _strip_env_assignments,
    _strip_fd_prefix_from_output,
    _strip_leading_reserved,
    _strip_segment_prefix,
    _strip_shell_comments,
)


# ----- tokenization: _split_segments ------------------------------------


@pytest.mark.unit
class TestSplitSegments:
    @pytest.mark.parametrize(
        ("tokens", "expected"),
        [
            ([], []),
            (["touch", "f"], [["touch", "f"]]),
            (["touch", "a", ";", "touch", "b"], [["touch", "a"], ["touch", "b"]]),
            (["echo", "x", "&&", "touch", "f"], [["echo", "x"], ["touch", "f"]]),
            (["a", "||", "b"], [["a"], ["b"]]),
            (["a", "|", "b"], [["a"], ["b"]]),
            (["a", "|&", "b"], [["a"], ["b"]]),
            (["a", "\n", "b"], [["a"], ["b"]]),
            (["(", "touch", "f", ")"], [["touch", "f"]]),
            (["{", "touch", "f", ";", "}"], [["touch", "f"]]),
        ],
    )
    def test_basic_separators(
        self, tokens: list[str], expected: list[list[str]]
    ) -> None:
        assert _split_segments(tokens) == expected

    def test_separator_constants_are_all_recognised(self) -> None:
        # Every documented separator must split a segment.
        for sep in SHELL_SEPARATORS:
            segs = _split_segments(["a", sep, "b"])
            assert segs == [["a"], ["b"]], sep

    def test_leading_and_trailing_separators_dropped(self) -> None:
        assert _split_segments([";", "touch", "f", ";"]) == [["touch", "f"]]


# ----- tokenization: env / reserved-word stripping ----------------------


@pytest.mark.unit
class TestStripEnvAssignments:
    @pytest.mark.parametrize(
        ("tokens", "expected"),
        [
            ([], []),
            (["touch", "f"], ["touch", "f"]),
            (["A=1", "touch", "f"], ["touch", "f"]),
            (["A=1", "B=2", "touch", "f"], ["touch", "f"]),
            (["_A=1", "touch", "f"], ["touch", "f"]),
            (["A=", "touch", "f"], ["touch", "f"]),
            (["touch", "A=1"], ["touch", "A=1"]),
            (["1A=1", "touch", "f"], ["1A=1", "touch", "f"]),
        ],
    )
    def test_strips_only_leading_env_pairs(
        self, tokens: list[str], expected: list[str]
    ) -> None:
        assert _strip_env_assignments(tokens) == expected


@pytest.mark.unit
class TestStripLeadingReserved:
    @pytest.mark.parametrize(
        ("tokens", "expected"),
        [
            ([], []),
            (["if", "true", "then"], ["true", "then"]),
            (["then", "touch", "f"], ["touch", "f"]),
            (["!", "touch", "f"], ["touch", "f"]),
            (["coproc", "touch", "f"], ["touch", "f"]),
            (["touch", "if"], ["touch", "if"]),
        ],
    )
    def test_drops_each_reserved_word(
        self, tokens: list[str], expected: list[str]
    ) -> None:
        assert _strip_leading_reserved(tokens) == expected

    def test_time_is_not_a_reserved_word(self) -> None:
        # ``time`` is handled as an execution-prefix wrapper, NOT as a
        # reserved word here. The shell_parser must leave it in place.
        assert "time" not in _SHELL_RESERVED_WORDS
        assert _strip_leading_reserved(["time", "touch", "f"]) == [
            "time",
            "touch",
            "f",
        ]


@pytest.mark.unit
class TestStripSegmentPrefix:
    def test_strips_interleaved_reserved_and_env_until_stable(self) -> None:
        # ``then VAR=val touch f`` requires looping: reserved-strip
        # exposes VAR=val which env-strip must then also remove.
        assert _strip_segment_prefix(["then", "VAR=val", "touch", "f"]) == [
            "touch",
            "f",
        ]

    def test_keeps_command_word_intact(self) -> None:
        assert _strip_segment_prefix(["touch", "f"]) == ["touch", "f"]


# ----- quoting: _strip_shell_comments ----------------------------------


@pytest.mark.unit
class TestStripShellComments:
    def test_removes_word_starting_comment(self) -> None:
        assert _strip_shell_comments("touch f # tail") == "touch f "

    def test_preserves_newline_after_comment(self) -> None:
        out = _strip_shell_comments("# top\ntouch f")
        assert out == "\ntouch f"

    def test_does_not_strip_hash_inside_single_quotes(self) -> None:
        assert _strip_shell_comments("echo '# not a comment'") == (
            "echo '# not a comment'"
        )

    def test_does_not_strip_hash_inside_double_quotes(self) -> None:
        assert _strip_shell_comments('echo "x # y"') == 'echo "x # y"'

    def test_does_not_strip_attached_hash(self) -> None:
        # ``foo#bar`` is one bash word, not a comment.
        assert _strip_shell_comments("foo#bar") == "foo#bar"

    def test_escaped_hash_is_literal(self) -> None:
        # ``\#`` is a literal ``#`` and is kept; nothing should be
        # stripped here.
        assert _strip_shell_comments(r"echo \# tail") == r"echo \# tail"

    def test_ansi_c_quoted_hash_is_replaced_with_placeholder(self) -> None:
        # ``$'...'`` body is opaque; the parser emits a placeholder so
        # the quote tracker stays in sync. The hash inside the body is
        # never treated as a comment.
        out = _strip_shell_comments(r"echo $'\'# inside' tail")
        assert _ANSI_C_PLACEHOLDER in out
        # The trailing literal text outside the ANSI-C block survives.
        assert " tail" in out


# ----- command substitution --------------------------------------------


@pytest.mark.unit
class TestExtractCommandSubstitutions:
    def test_no_substitution_returns_input_unchanged(self) -> None:
        cleaned, bodies = _extract_command_substitutions("touch f")
        assert cleaned == "touch f"
        assert bodies == []

    def test_extracts_backtick_body(self) -> None:
        cleaned, bodies = _extract_command_substitutions("echo `touch f`")
        assert bodies == ["touch f"]
        assert _CMDSUB_PLACEHOLDER in cleaned
        assert "`" not in cleaned

    def test_extracts_dollar_paren_body(self) -> None:
        cleaned, bodies = _extract_command_substitutions("echo $(touch f)")
        assert bodies == ["touch f"]
        assert _CMDSUB_PLACEHOLDER in cleaned

    def test_inside_single_quotes_is_preserved(self) -> None:
        # ``'$(...)'`` is literal in bash; nothing should be extracted.
        cleaned, bodies = _extract_command_substitutions("echo '$(touch f)'")
        assert bodies == []
        assert "$(touch f)" in cleaned

    def test_inside_double_quotes_is_extracted(self) -> None:
        cleaned, bodies = _extract_command_substitutions('echo "$(touch f)"')
        assert bodies == ["touch f"]
        assert _CMDSUB_PLACEHOLDER in cleaned

    def test_nested_dollar_paren_pulls_inner_body(self) -> None:
        # The OUTER body keeps its inner ``$(touch g)`` text. The
        # extractor pulls one level of nesting per call; ``decide``
        # then recurses on each body. We assert only the outer here.
        cleaned, bodies = _extract_command_substitutions("$(echo $(touch g))")
        assert _CMDSUB_PLACEHOLDER in cleaned
        assert bodies == ["echo $(touch g)"]
        # And a second pass on the body extracts the inner write.
        _, inner_bodies = _extract_command_substitutions(bodies[0])
        assert inner_bodies == ["touch g"]

    def test_arithmetic_expansion_extracts_inner_substitution(self) -> None:
        # ``$(( $(touch f) + 1 ))`` runs ``touch f`` for real; the
        # arithmetic wrapper is preserved but the inner cmd-sub must be
        # surfaced in ``bodies``.
        _, bodies = _extract_command_substitutions("echo $(( $(touch f) + 1 ))")
        assert "touch f" in bodies

    def test_unmatched_backtick_falls_through(self) -> None:
        # Should not raise; the parser leaves the unmatched ``\``` in
        # place. No body is extracted.
        cleaned, bodies = _extract_command_substitutions("echo `touch f")
        assert bodies == []
        assert "`" in cleaned

    def test_ansi_c_outside_quotes_replaced_with_placeholder(self) -> None:
        _, _ = _extract_command_substitutions("echo $'\\''")  # smoke
        # ANSI-C with an escaped single quote should not desync the
        # parser — a following backtick body must still be extracted.
        cleaned, bodies = _extract_command_substitutions("echo $'\\'' `touch f`")
        assert bodies == ["touch f"]
        assert _ANSI_C_PLACEHOLDER in cleaned


# ----- redirection -----------------------------------------------------


@pytest.mark.unit
class TestExtractRedirectionTargets:
    def test_simple_stdout_redirect(self) -> None:
        assert _extract_redirection_targets(["echo", "hi", ">", "out"]) == ["out"]

    def test_append_operator(self) -> None:
        assert _extract_redirection_targets(["echo", "hi", ">>", "out"]) == ["out"]

    def test_fd_prefixed_operator(self) -> None:
        assert _extract_redirection_targets(["cmd", "2>", "err"]) == ["err"]

    def test_combined_stdout_stderr(self) -> None:
        assert _extract_redirection_targets(["cmd", "&>", "both"]) == ["both"]

    def test_force_clobber(self) -> None:
        assert _extract_redirection_targets(["cmd", ">|", "out"]) == ["out"]

    def test_read_write_open_is_a_write(self) -> None:
        assert _extract_redirection_targets(["cmd", "<>", "out"]) == ["out"]

    def test_fd_duplication_is_skipped(self) -> None:
        # ``2>&1`` after normalization tokenizes as ``2>&`` ``1``;
        # the trailing numeric target is fd-dup, not a path.
        assert _extract_redirection_targets(["cmd", "2>&", "1"]) == []

    def test_dev_null_like_targets_skipped(self) -> None:
        assert _extract_redirection_targets(["cmd", ">", "/dev/null"]) == []
        assert _extract_redirection_targets(["cmd", ">", "/dev/stderr"]) == []

    def test_quoted_combined_token_is_not_a_redirect(self) -> None:
        # ``>out`` arriving as a single token (quoted/escaped at the
        # source) must NOT be treated as a redirect.
        assert _extract_redirection_targets(["echo", ">out"]) == []


@pytest.mark.unit
class TestDropRedirections:
    def test_removes_operator_and_target(self) -> None:
        assert _drop_redirections(["cp", "a", "b", ">", "out"]) == ["cp", "a", "b"]

    def test_handles_mid_arglist_redirect(self) -> None:
        assert _drop_redirections(["cmd", ">", "out", "arg"]) == ["cmd", "arg"]

    def test_leaves_non_redirect_tokens(self) -> None:
        assert _drop_redirections(["cmd", "arg"]) == ["cmd", "arg"]


# ----- normalization ---------------------------------------------------


@pytest.mark.unit
class TestNormalizeSeparators:
    @pytest.mark.parametrize(
        ("command", "expected_shlex"),
        [
            ("true;touch f", ["true", ";", "touch", "f"]),
            ("true&&touch f", ["true", "&&", "touch", "f"]),
            ("a||b", ["a", "||", "b"]),
            ("a|b", ["a", "|", "b"]),
            ("(touch f)", ["(", "touch", "f", ")"]),
            ("touch f&", ["touch", "f", "&"]),
        ],
    )
    def test_operators_become_separate_tokens(
        self, command: str, expected_shlex: list[str]
    ) -> None:
        # Real downstream consumer is shlex.split; assert the result is
        # tokenized cleanly after normalization.
        normalized = _normalize_separators(command)
        assert shlex.split(normalized, posix=True, comments=True) == expected_shlex

    def test_newline_rewritten_to_semicolon(self) -> None:
        normalized = _normalize_separators("true\ntouch f")
        assert shlex.split(normalized, posix=True, comments=True) == [
            "true",
            ";",
            "touch",
            "f",
        ]

    def test_force_clobber_redirect_is_kept_intact(self) -> None:
        # ``>|`` must survive normalization as a single operator token.
        normalized = _normalize_separators("cmd >|out")
        assert ">|" in normalized.split()

    def test_quoted_separators_not_split(self) -> None:
        # Literal ``;`` inside a single-quoted argument must not split.
        normalized = _normalize_separators("echo 'a;b' > out")
        tokens = shlex.split(normalized, posix=True, comments=True)
        assert "a;b" in tokens

    def test_backslash_escaped_quote_does_not_open_quote_region(self) -> None:
        # ``\"`` outside a quoted region is a literal ``"``; a
        # following ``;`` must still split.
        normalized = _normalize_separators(r"echo \";touch f")
        tokens = shlex.split(normalized, posix=True, comments=True)
        assert "touch" in tokens
        assert "f" in tokens

    def test_ansi_c_quoting_does_not_desync(self) -> None:
        # ``$'\\''`` looks like a mismatched quote to plain shlex; the
        # ANSI-C handling must hide it as a single placeholder token so
        # later operators (here ``;``) still split.
        normalized = _normalize_separators("echo $'\\'';touch f")
        tokens = shlex.split(normalized, posix=True, comments=True)
        assert "touch" in tokens

    def test_fd_redirect_pad_keeps_2_to_1_dup_intact(self) -> None:
        normalized = _normalize_separators("cmd 2>&1")
        assert "2>&" in normalized.split()


@pytest.mark.unit
class TestStripFdPrefixFromOutput:
    def test_pops_numeric_prefix_when_preceded_by_space(self) -> None:
        out = list("foo 2")
        prefix = _strip_fd_prefix_from_output(out)
        assert prefix == "2"
        assert "".join(out) == "foo "

    def test_pops_ampersand_prefix(self) -> None:
        out = list("foo &")
        prefix = _strip_fd_prefix_from_output(out)
        assert prefix == "&"
        assert "".join(out) == "foo "

    def test_pops_brace_varname_prefix(self) -> None:
        out = list("foo {fd}")
        prefix = _strip_fd_prefix_from_output(out)
        assert prefix == "{fd}"
        assert "".join(out) == "foo "

    def test_refuses_to_pop_attached_prefix(self) -> None:
        # ``foo123>`` must NOT strip ``123`` — it's part of the word.
        out = list("foo123")
        prefix = _strip_fd_prefix_from_output(out)
        assert prefix == ""
        assert "".join(out) == "foo123"

    def test_empty_input(self) -> None:
        out: list[str] = []
        assert _strip_fd_prefix_from_output(out) == ""
        assert out == []
