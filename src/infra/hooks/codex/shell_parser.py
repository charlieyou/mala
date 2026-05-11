"""Pure shell tokenization + command-substitution extraction helpers.

Extracted from ``src.infra.hooks.codex_pre_tool_use`` (plan B.3 sub-commit
T_B8). These are the side-effect-free shell-parsing primitives used by
the Codex pre-tool-use write-path heuristic: separator normalisation,
segment splitting, env-assignment / reserved-word stripping, shell
comment stripping, command-substitution body extraction, and redirection
operator handling.

The hook imports the helpers it actually consumes from this module;
unit tests under ``tests/unit/infra/hooks/codex/`` import the broader
set declared in ``__all__`` (constants, regex tables, fd helpers). Each
public name here may be referenced from either consumer, so don't rely
on the hook's import list as an exhaustive index.
"""

from __future__ import annotations

import re


__all__ = [
    "REDIR_OPERATORS",
    "SHELL_SEPARATORS",
    "_ANSI_C_PLACEHOLDER",
    "_CMDSUB_PLACEHOLDER",
    "_COMBINED_REDIR_PREFIX_RE",
    "_DEV_NULL_LIKE",
    "_FD_REF_RE",
    "_FD_REF_TARGET_RE",
    "_REDIR_FD_PREFIX",
    "_REDIR_OP_TOKEN_RE",
    "_REDIR_TAIL",
    "_SHELL_RESERVED_WORDS",
    "_drop_redirections",
    "_extract_command_substitutions",
    "_extract_redirection_targets",
    "_normalize_separators",
    "_split_segments",
    "_strip_env_assignments",
    "_strip_fd_prefix_from_output",
    "_strip_leading_reserved",
    "_strip_segment_prefix",
    "_strip_shell_comments",
]


# Shell separators that end a "simple command" segment for heuristic
# parsing. Includes pipeline operators (``|``/``&&``/``||``), control
# operators (``;``/``&``/``\n``), and shell grouping delimiters
# (``(``/``)``/``{``/``}``) so a write inside a subshell or brace group
# is still seen by the per-segment extractor (``(touch unowned)`` and
# ``{ touch unowned; }`` both run their inner commands).
SHELL_SEPARATORS = frozenset(
    {";", "&&", "||", "|", "|&", "&", "\n", "(", ")", "{", "}"}
)

# Redirection operators that consume the next token as a write target.
REDIR_OPERATORS = frozenset(
    {">", ">>", "&>", "&>>", "2>", "2>>", "1>", "1>>", ">|", ">&"}
)


# Placeholder emitted in place of a bash ANSI-C-quoted string (``$'...'``)
# so shlex.split treats the entire literal as a single token. The content
# of an ANSI-C string is never a command word or write target, so the
# heuristic does not need its actual value.
_ANSI_C_PLACEHOLDER = "__MALA_ANSI_C_LITERAL__"

# Placeholder emitted in place of a command-substitution body (either
# legacy ```...``` or modern ``$(...)``). Bodies are extracted and
# processed as their own segments via ``_extract_command_substitutions``
# before the main heuristic runs.
_CMDSUB_PLACEHOLDER = "__MALA_CMDSUB_LITERAL__"

# Shell reserved words that introduce a compound-command body but are
# never themselves a command word. ``_split_segments`` produces segments
# headed by these tokens (``[..., 'then', 'touch', 'unowned']``); we
# strip them so the inner extractors see the actual command word.
_SHELL_RESERVED_WORDS = frozenset(
    {
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "for",
        "while",
        "do",
        "done",
        "until",
        "case",
        "esac",
        "select",
        "in",
        "function",
        "!",
        # ``coproc cmd`` runs cmd in a co-process; the inner cmd is
        # what actually executes (and writes). Strip ``coproc`` so the
        # extractor sees the real cmd word.
        "coproc",
        # ``time`` is intentionally NOT a reserved word here — it's
        # handled as an execution-prefix wrapper instead so its options
        # (``time -p cmd``) and path-qualified form (``/usr/bin/time
        # cmd``) are correctly unwrapped to the inner command.
    }
)


_DEV_NULL_LIKE = frozenset({"/dev/null", "/dev/stderr", "/dev/stdout", "/dev/tty"})
# File-descriptor duplications: leading-``&`` form (``&1`` / ``&-``).
# These appear as the *post-prefix* tail when the operator regex consumed
# only a leading-``&``-less prefix (e.g. ``2>`` from ``2>&1``); the residue
# ``&1`` is an fd reference, not a filesystem path.
_FD_REF_RE = re.compile(r"^&[0-9\-]+$")
# Pure fd-numeric / close-fd target. Used after a ``>&`` / ``n>&`` operator
# to distinguish ``>&1`` (fd dup, skip) from ``>&file`` (write, lock-check).
_FD_REF_TARGET_RE = re.compile(r"^[0-9]+$|^-$")
# Whole-token redirection operators. Accepted fd prefixes:
#   - numeric (``1``/``2``/``3``/...) — POSIX
#   - ``&`` — combined stdout+stderr (``&>``, ``&>>``)
#   - ``{varname}`` — bash 4.1+ variable-allocated fd (``{fd}>file``);
#     bash auto-assigns a free fd to ``$varname`` and applies the redirect.
#
# Recognised operator forms:
#   - ``>``/``>>``/``n>``/``n>>`` — write
#   - ``&>``/``&>>``/``>&``/``n>&`` — combined / legacy combined
#   - ``>|`` / ``n>|`` — force-clobber write
#   - ``<>`` / ``n<>`` — POSIX read-write open (creates file if missing,
#     so it IS a write surface)
# ``<`` alone (read-only), ``<<`` (here-doc), and ``<<<`` (here-string)
# are deliberately excluded — they don't open a file for writing.
_REDIR_FD_PREFIX = r"(?:[0-9]+|&|\{[A-Za-z_][A-Za-z0-9_]*\})"
_REDIR_TAIL = r"(?:<>|>{1,2}[&|]?)"
_REDIR_OP_TOKEN_RE = re.compile(rf"^{_REDIR_FD_PREFIX}?{_REDIR_TAIL}$")
# Prefix regex for combined-token forms like ``>file``, ``2>file``,
# ``&>file``, ``>&file``, ``3>>file``, ``{fd}>file``, ``<>file``,
# ``3<>file``. Greedy on the optional trailing ``&`` so ``2>&1`` matches
# the prefix ``2>&`` (target ``1`` → fd dup).
_COMBINED_REDIR_PREFIX_RE = re.compile(rf"^{_REDIR_FD_PREFIX}?{_REDIR_TAIL}")


def _split_segments(tokens: list[str]) -> list[list[str]]:
    """Split a token stream at shell separators into simple-command segments."""
    segments: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok in SHELL_SEPARATORS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(tok)
    if current:
        segments.append(current)
    return segments


def _strip_env_assignments(tokens: list[str]) -> list[str]:
    """Drop leading ``KEY=VALUE`` env-assignment tokens before the command word."""
    i = 0
    while i < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[i]):
        i += 1
    return tokens[i:]


def _strip_segment_prefix(tokens: list[str]) -> list[str]:
    """Strip leading reserved words and env-assignments until stable.

    Calling ``_strip_env_assignments`` once before ``_strip_leading_reserved``
    misses the case where a reserved word HIDES an env assignment, e.g.
    ``then VAR=val touch f``: ``then`` is at the head so env-strip is a
    no-op; ``then`` is then removed by reserved-strip; but ``VAR=val``
    is now at the head and never re-stripped. The result is a segment
    whose "command word" is ``VAR=val``, which no extractor matches —
    the inner ``touch`` is silently allowed. Looping until both passes
    are no-ops handles this and any other interleaving (``! VAR=val
    cmd``, ``time VAR=val cmd``).
    """
    while True:
        before = len(tokens)
        tokens = _strip_env_assignments(tokens)
        tokens = _strip_leading_reserved(tokens)
        if len(tokens) == before:
            break
    return tokens


def _strip_leading_reserved(tokens: list[str]) -> list[str]:
    """Drop leading shell reserved words (``if``/``then``/``do``/...).

    Compound commands (``if ...; then touch f; fi``) leave the actual
    command word behind a non-mutating reserved word in the resulting
    segment. Without stripping, the utility extractor sees ``then`` as
    the basename and emits no targets, silently allowing the unowned
    write that follows.
    """
    while tokens and tokens[0] in _SHELL_RESERVED_WORDS:
        tokens = tokens[1:]
    return tokens


def _strip_shell_comments(command: str) -> str:
    """Remove unquoted ``#``-comments from ``command``.

    Comments run from a word-starting ``#`` (preceded by whitespace,
    a separator, or start-of-input) up to the next newline. The
    newline itself is preserved so it can later be rewritten to ``;``
    by ``_normalize_separators`` and split into a fresh segment.

    Used as a pre-pass before ``_extract_command_substitutions`` so a
    substitution inside a comment (``# $(touch unowned)\\necho ok``)
    is NOT extracted as if bash would execute it. Also used by the
    dangerous-cmd recursive detector for the same reason.
    """
    out: list[str] = []
    i = 0
    n = len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        # Backslash escape outside single quotes consumes the next
        # char verbatim — including a ``#`` so ``\\#`` is literal.
        if c == "\\" and i + 1 < n and quote != "'":
            out.append(c)
            out.append(command[i + 1])
            i += 2
            continue
        if quote is not None:
            out.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        # ``$'...'`` ANSI-C: skip the entire block and emit a
        # placeholder so the quote tracker stays in sync. The previous
        # implementation tracked ANSI-C as a regular single-quoted
        # region, which desynced on inputs like ``$'\\''`` (it treated
        # the escaped ``\\'`` as the closer and the third ``'`` as a
        # fresh opener). The body is never a comment and never a write
        # target, so a placeholder is sufficient.
        if c == "$" and quote is None and i + 1 < n and command[i + 1] == "'":
            j = i + 2
            while j < n:
                if command[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if command[j] == "'":
                    j += 1
                    break
                j += 1
            out.append(_ANSI_C_PLACEHOLDER)
            i = j
            continue
        if c in ('"', "'"):
            out.append(c)
            quote = c
            i += 1
            continue
        # Word-starting ``#`` outside quotes: skip to next newline.
        if c == "#" and (
            i == 0
            or command[i - 1].isspace()
            or command[i - 1] in (";", "&", "|", "(", ")")
        ):
            while i < n and command[i] != "\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _extract_command_substitutions(command: str) -> tuple[str, list[str]]:
    """Replace ```...``` and ``$(...)`` with a placeholder, returning bodies.

    Returns ``(cleaned, bodies)`` — ``cleaned`` is the input string with
    every command-substitution block replaced by ``_CMDSUB_PLACEHOLDER``,
    and ``bodies`` is the list of extracted body strings. Both legacy
    backticks and modern ``$()`` substitutions are pulled out so the
    inner command is processed by the recursive ``_extract_shell_write_paths``
    call.

    Inside single quotes (``'...'``) substitutions are preserved verbatim
    — bash treats them literally there. Inside double quotes both forms
    still trigger command substitution and ARE extracted.

    ``$((arith))`` (arithmetic expansion) is detected by a leading
    ``$((`` and skipped intact so its inner ``(`` / ``)`` don't confuse
    the cmd-substitution body parser.

    Without this preprocessing, ``echo `touch unowned``` and
    ``echo "$(touch unowned)"`` would leave the inner ``touch`` invisible
    to every extractor — both are standard shell-execution forms, not
    residual obfuscation gaps.
    """
    out: list[str] = []
    bodies: list[str] = []
    i = 0
    n = len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        if c == "\\" and i + 1 < n and quote != "'":
            out.append(c)
            out.append(command[i + 1])
            i += 2
            continue
        # ANSI-C quoting (``$'...'``) outside other quoted regions: skip
        # the entire block and emit ``_ANSI_C_PLACEHOLDER`` so the
        # quote/sub tracker stays in sync. Without this, an escaped
        # single quote (``$'\\''``) desyncs the parser — it treats the
        # escaped ``\\'`` as the closer and the third ``'`` as a fresh
        # opener, so any backtick / ``$()`` substitution that follows is
        # seen as "inside single quotes" and never extracted into
        # ``bodies``. Concrete bypass before this fix:
        # ``echo $'\\'' `touch unowned.py``` left the inner ``touch``
        # invisible to every extractor and the hook returned allow.
        # (Inside ``"..."`` bash treats ``$'...'`` literally, so the
        # ANSI-C skip is gated on ``quote is None``.)
        if c == "$" and quote is None and i + 1 < n and command[i + 1] == "'":
            j = i + 2
            while j < n:
                if command[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if command[j] == "'":
                    j += 1
                    break
                j += 1
            out.append(_ANSI_C_PLACEHOLDER)
            i = j
            continue
        # Legacy backtick command substitution.
        if c == "`" and quote != "'":
            j = i + 1
            body_chars: list[str] = []
            while j < n:
                if command[j] == "\\" and j + 1 < n and command[j + 1] == "`":
                    # Escaped inner backtick (`\``). bash treats this
                    # as a literal backtick in the OUTER body but as a
                    # nested-substitution delimiter when the body is
                    # re-parsed. Unescape here so the recursive call
                    # to ``_extract_command_substitutions`` on the
                    # extracted body sees real backticks and pulls
                    # the nested ``touch f`` out as its own segment.
                    body_chars.append("`")
                    j += 2
                    continue
                if command[j] == "\\" and j + 1 < n:
                    body_chars.append(command[j])
                    body_chars.append(command[j + 1])
                    j += 2
                    continue
                if command[j] == "`":
                    break
                body_chars.append(command[j])
                j += 1
            if j < n:
                bodies.append("".join(body_chars))
                out.append(_CMDSUB_PLACEHOLDER)
                i = j + 1
                continue
            # Unmatched: fall through.
        # ``$(...)`` command substitution / ``$((...))`` arithmetic.
        if c == "$" and quote != "'" and i + 1 < n and command[i + 1] == "(":
            if i + 2 < n and command[i + 2] == "(":
                # ``$(( arith ))`` — arithmetic expansion. The OUTER
                # value is numeric so the surrounding token is not a
                # write surface, but bash STILL evaluates command
                # substitutions and backticks inside the arithmetic
                # body before computing the result. ``echo $(( $(touch
                # unowned) + 1 ))`` runs ``touch unowned`` for real;
                # passing the entire ``$((...))`` block through
                # verbatim would silently allow it.
                depth = 2
                j = i + 3
                inner_quote: str | None = None
                while j < n and depth > 0:
                    cj = command[j]
                    if cj == "\\" and j + 1 < n and inner_quote != "'":
                        j += 2
                        continue
                    if inner_quote is not None:
                        if cj == inner_quote:
                            inner_quote = None
                        j += 1
                        continue
                    if cj in ('"', "'"):
                        inner_quote = cj
                        j += 1
                        continue
                    if cj == "(":
                        depth += 1
                    elif cj == ")":
                        depth -= 1
                    j += 1
                if depth == 0:
                    out.append(command[i:j])
                    # Recursively pull any cmd substitutions out of the
                    # arithmetic body so their writes are gated.
                    arith_body = command[i + 3 : j - 2]
                    _, inner_bodies = _extract_command_substitutions(arith_body)
                    bodies.extend(inner_bodies)
                    i = j
                    continue
                # Unmatched: fall through.
            else:
                # ``$( cmd )`` — find matching close paren, respecting
                # nesting and quote state inside the body.
                depth = 1
                j = i + 2
                inner_quote = None
                while j < n and depth > 0:
                    cj = command[j]
                    if cj == "\\" and j + 1 < n and inner_quote != "'":
                        j += 2
                        continue
                    if inner_quote is not None:
                        if cj == inner_quote:
                            inner_quote = None
                        j += 1
                        continue
                    if cj in ('"', "'"):
                        inner_quote = cj
                        j += 1
                        continue
                    if cj == "(":
                        depth += 1
                    elif cj == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if depth == 0:
                    bodies.append(command[i + 2 : j])
                    out.append(_CMDSUB_PLACEHOLDER)
                    i = j + 1
                    continue
                # Unmatched: fall through.
        if quote is not None:
            out.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ('"', "'"):
            out.append(c)
            quote = c
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out), bodies


def _drop_redirections(tokens: list[str]) -> list[str]:
    """Return ``tokens`` with redirect operators and their targets removed.

    Used by non-redirect segment extractors so a redirect operator+target
    that appears in the middle of an argument list (between the command
    word and the trailing positional, or after it) does not become a
    positional that the utility/oneliner extractors mistakenly treat as
    a write surface.

    Without this, ``cp owned dst >out`` left ``>`` and ``out`` in cp's
    positional list and ``last_positional`` returned ``out`` instead
    of ``dst``, so the real cp destination was never lock-checked.
    Combined with the ``&>`` split fix in ``_normalize_separators``,
    ``cp owned unowned&>out`` now lock-checks the real ``unowned``
    instead of leaking to a redirect target the agent legitimately
    holds.
    """
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if _REDIR_OP_TOKEN_RE.match(tok):
            # Operator + its target word are both removed; ``cmd > out``
            # becomes ``cmd``. fd-dup target words (``2>&1``) are also
            # removed — fine, they aren't write surfaces.
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def _extract_redirection_targets(tokens: list[str]) -> list[str]:
    """Find ``>``/``>>``/etc. targets in a token stream.

    Operates on tokens AFTER ``_normalize_separators`` has padded every
    unquoted redirect operator with surrounding whitespace; the
    extractor therefore only needs to recognise WHOLE-token operators.
    A ``>file`` token that survives shlex unchanged was either inside
    quotes (``echo ">file"``) or escaped (``grep \\>5``) — both are
    legitimate positional arguments, not redirects, so we no longer
    apply the combined-prefix fallback (which used to false-positive
    deny on those quoted args).

    Recognised forms — POSIX shells accept ANY numeric fd prefix, not just
    ``1``/``2``, and the legacy ``>&file`` operator is equivalent to
    ``> file 2> file`` (write both stdout and stderr to ``file``):

    - ``>file`` / ``>>file`` / ``n>file`` / ``n>>file`` — write to file
    - ``&>file`` / ``&>>file`` — write stdout+stderr (modern syntax)
    - ``>&file`` / ``n>&file`` — write stdout+stderr (legacy syntax)
    - ``>&n`` / ``n>&m`` / ``>&-`` — fd duplication / close (NOT writes)
    - ``<>file`` / ``n<>file`` — POSIX read-write open (creates file)

    Fd duplications and close-fd are skipped so common idioms like
    ``cmd >out.log 2>&1`` don't false-deny when ``out.log`` is locked.
    """
    out: list[str] = []
    for i, tok in enumerate(tokens):
        # Whole-token operator: next token is the redirect target.
        if _REDIR_OP_TOKEN_RE.match(tok) and i + 1 < len(tokens):
            target = tokens[i + 1]
            # ``>&`` / ``n>&`` followed by a numeric/dash word is fd dup.
            if tok.endswith("&") and _FD_REF_TARGET_RE.match(target):
                continue
            if target in _DEV_NULL_LIKE:
                continue
            out.append(target)
            continue
        # The combined-prefix fallback (``>file`` as a single token) is
        # intentionally NOT applied here — see the docstring above. Any
        # ``>file``-shaped token reaching this point is quoted/escaped
        # and is not a real redirect surface.
    return out


def _strip_fd_prefix_from_output(out: list[str]) -> str:
    """Strip a contiguous fd prefix (digits, ``&``, or ``{varname}``) from
    the end of ``out`` IFF it is preceded by whitespace (or starts ``out``).

    Used by ``_normalize_separators`` when emitting a redirect operator:
    the fd prefix that should belong to the operator was already
    appended char-by-char as part of the surrounding word. We need to
    pop it off so the emitted operator (with surrounding spaces) carries
    its prefix in a single token (``2>``/``&>``/``{fd}>``) for shlex.

    The whitespace boundary is critical: bash only treats a numeric
    prefix as an fd selector when it stands as a separate word (or at
    the start of the redirect). Attached to a word it is part of the
    filename — ``cp src -t unowned_dir123>out`` writes to
    ``unowned_dir123``, NOT to fd 123. Without the boundary check the
    heuristic stripped ``123`` and lock-checked ``unowned_dir`` and
    ``out`` separately while bash silently created the unowned
    ``unowned_dir123``.

    Returns ``""`` if the trailing chars don't form a valid fd prefix
    or are attached to a non-whitespace previous char.
    """
    if not out:
        return ""
    last = out[-1]
    # Handle multi-char appends (placeholders, escape pairs): they are
    # never an fd prefix.
    if len(last) != 1:
        return ""

    # Locate the start index of the candidate prefix span. We don't
    # mutate ``out`` until we've validated the whitespace boundary.
    if last == "&":
        start = len(out) - 1
    elif last == "}":
        # ``{varname}`` form. Walk back through varname chars to ``{``.
        j = len(out) - 2
        while j >= 0:
            ch = out[j]
            if len(ch) != 1:
                return ""
            if ch == "{":
                break
            if not (ch.isalnum() or ch == "_"):
                return ""
            j -= 1
        else:
            return ""
        if j < 0 or out[j] != "{":
            return ""
        # Body must start with letter or ``_`` (varname rule).
        if j + 1 >= len(out) - 1:
            return ""
        first = out[j + 1]
        if not (len(first) == 1 and (first.isalpha() or first == "_")):
            return ""
        start = j
    elif last.isdigit():
        j = len(out) - 1
        while j >= 0 and len(out[j]) == 1 and out[j].isdigit():
            j -= 1
        start = j + 1
    else:
        return ""

    # Whitespace boundary: the prefix must stand alone (start of out
    # or preceded by a whitespace char). Otherwise it's part of a word.
    if start > 0:
        prev = out[start - 1]
        if not (len(prev) == 1 and prev.isspace()):
            return ""

    prefix = "".join(out[start:])
    del out[start:]
    return prefix


def _normalize_separators(command: str) -> str:
    """Insert spaces around shell control operators outside quoted regions.

    ``shlex.split`` does not split on ``;``/``&&``/``||``/``|``/``&``/
    newlines/``(``/``)`` unless the operators are already surrounded by
    whitespace, so an agent-written ``true;touch unowned.py`` or
    ``touch out&`` or ``(touch unowned.py)`` previously tokenized as a
    single attached token — no extractor saw the inner command and the
    write slipped past AC #19. This helper normalises the command string
    so the downstream shlex split + segment-extractor pipeline sees the
    pipeline boundaries.

    Newlines (``\\n``) are converted to ``;`` so they survive
    ``shlex.split`` (which would otherwise eat them as whitespace and
    let ``true\\ntouch unowned`` collapse into a single segment).

    Quote tracking is required so a literal ``;``/``|``/``&`` inside a
    single- or double-quoted argument (e.g., ``echo 'a;b' > out``) is
    not treated as a control operator.

    Backslash handling is required so an agent cannot hide control
    operators by escaping a quote outside an actual quoted region.
    Without it, ``echo \\";touch f`` would look quoted to the parser,
    leave ``;`` un-split, and silently allow the unowned write — even
    though bash treats ``\\"`` as a literal ``"`` and runs the second
    command at top level.

    ANSI-C quoting (``$'...'``) supports backslash escapes inside what
    looks like a single-quoted region. Without recognising ``$'``, the
    parser miscounts quote toggles for inputs like ``$'\\''`` (an escaped
    single-quote) and incorrectly treats subsequent control operators
    as quoted, hiding them from the heuristic.

    Bare ``&`` is treated as a backgrounding operator (and split)
    EXCEPT when it follows ``>`` or ``digit+>`` (forming the fd
    redirect operators ``>&``/``n>&``). This catches ``touch out&``
    while still leaving ``2>&1`` intact for the redirection extractor.

    ``(`` / ``)`` (subshell delimiters) are split outside quotes so
    ``(touch unowned)`` becomes a segment whose first token is
    ``touch``. ``{`` / ``}`` are left to the existing whitespace-based
    tokenisation (bash already requires whitespace around brace-group
    delimiters), and ``_split_segments`` treats them as separators.
    """
    out: list[str] = []
    i = 0
    n = len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        # Backslash escape outside quotes / inside double quotes:
        # ``\<x>`` is a literal pair. Inside regular single quotes the
        # backslash is literal.
        if c == "\\" and i + 1 < n:
            if quote is None or quote == '"':
                out.append(c)
                out.append(command[i + 1])
                i += 2
                continue
        if quote is not None:
            out.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        # ANSI-C quoting: ``$'...'`` is a bash single-quoted region in
        # which ``\<x>`` is an escape. shlex.split has no ANSI-C support,
        # so a body containing ``\'`` (escaped single quote) tricks
        # shlex's plain single-quote parser into thinking the escaped
        # quote *closes* the region and the next ``'`` *opens* a new
        # one — which then never closes (ValueError) or, worse, swallows
        # subsequent control operators.
        #
        # We don't need the literal content of the ANSI-C string for
        # write-path detection (it's a positional arg, not a write
        # target). Skip the whole block and emit a placeholder identifier
        # so shlex treats it as a single non-quote token.
        if c == "$" and i + 1 < n and command[i + 1] == "'":
            j = i + 2  # past ``$'``
            while j < n:
                if command[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if command[j] == "'":
                    j += 1
                    break
                j += 1
            out.append(_ANSI_C_PLACEHOLDER)
            i = j
            continue
        if c in ('"', "'"):
            out.append(c)
            quote = c
            i += 1
            continue
        # ``#`` comment: from ``#`` (when starting a word — i.e.,
        # preceded by whitespace, separator, or start-of-input) up to
        # the next newline. Must be handled BEFORE the newline-to-``;``
        # rewrite below: otherwise ``# foo\ntouch f`` becomes
        # ``# foo ; touch f``, and shlex's comment handling (which
        # only terminates at a newline) treats the entire remainder as
        # comment — silently dropping the second command.
        if c == "#" and (
            i == 0
            or command[i - 1].isspace()
            or command[i - 1] in (";", "&", "|", "(", ")")
        ):
            while i < n and command[i] != "\n":
                i += 1
            continue
        # Newlines are top-level command separators in shells; rewrite to
        # ``;`` so ``shlex.split`` does not collapse them into whitespace.
        if c == "\n":
            out.append(" ; ")
            i += 1
            continue
        if i + 1 < n:
            two = command[i : i + 2]
            if two in ("&&", "||", "|&", ";;"):
                out.append(" ")
                out.append(two)
                out.append(" ")
                i += 2
                continue
        if c == ";":
            out.append(" ")
            out.append(c)
            out.append(" ")
            i += 1
            continue
        if c == "|":
            # Preserve ``>|`` (force-clobber redirect): if the previous
            # output char is ``>`` we must NOT split the ``|`` away from
            # it. Splitting breaks the operator into ``>`` and ``|``,
            # the redirection regex no longer recognises the operator,
            # and the destination silently becomes a separate token
            # that no extractor can lock-check. Multi-char ``||``/``|&``
            # are already handled by the lookahead above.
            #
            # The ``>`` must be unescaped — the unescaped ``>|``
            # operator is normally consumed by the ``>`` handler
            # above, so any ``>`` left in ``out`` for the ``|``
            # handler to see was preserved by the escape branch
            # (``\>``). In that case the ``>`` is literal text and
            # the ``|`` is a real pipeline operator that MUST split.
            if out and out[-1] == ">" and not (len(out) >= 2 and out[-2] == "\\"):
                out.append(c)
                i += 1
                continue
            out.append(" ")
            out.append(c)
            out.append(" ")
            i += 1
            continue
        if c == "&":
            # ``&`` followed by ``>`` is the start of the ``&>`` redirect
            # operator. Pad the ``&`` with a leading space so it is
            # detached from any preceding word (``&`` is a bash
            # metacharacter and always terminates a word, even without
            # whitespace), then append it so the upcoming ``>`` handler
            # picks it up via ``_strip_fd_prefix_from_output`` and emits
            # the full operator together. Without the leading pad,
            # ``cp owned unowned&>out`` left ``unowned&`` glued in
            # ``out``; ``_strip_fd_prefix_from_output`` then refused to
            # strip the ``&`` (no whitespace boundary), so the cp
            # destination was lock-checked as the literal ``unowned&``
            # — an agent locking that junk path would be allowed to
            # write while bash actually wrote to ``unowned``.
            if i + 1 < n and command[i + 1] == ">":
                out.append(" ")
                out.append(c)
                i += 1
                continue
            # ``&`` is the trailing half of an fd redirect ONLY when it
            # IMMEDIATELY follows an UNESCAPED ``>``. ``\>&`` (escaped
            # ``>``) is literal-``>`` followed by backgrounding ``&``;
            # treating it as a redirect would split off ``&`` as part
            # of the operator and the real backgrounding op is lost.
            if (
                i > 0
                and command[i - 1] == ">"
                and not (i >= 2 and command[i - 2] == "\\")
            ):
                out.append(c)
                i += 1
                continue
            out.append(" ")
            out.append(c)
            out.append(" ")
            i += 1
            continue
        if c in ("(", ")"):
            # Subshell delimiters are control operators in bash and
            # always start/end a command segment.
            out.append(" ")
            out.append(c)
            out.append(" ")
            i += 1
            continue
        if c == ">":
            # Pad redirect operators so ``shlex.split`` (which does not
            # tokenise on ``>``) splits them off from adjacent words.
            # ``echo hi>file`` tokenises as the single token
            # ``hi>file``; without padding, the redirection extractor's
            # combined-prefix regex never matches and the unowned write
            # to ``file`` slips past the gate.
            #
            # The fd prefix (digits, ``&``, or ``{varname}``) is part of
            # the operator and was already appended to ``out``; strip it
            # back, build the full operator (including any trailing
            # ``>`` / ``&`` / ``|``), and emit it space-padded.
            prefix = _strip_fd_prefix_from_output(out)
            op = ">"
            next_i = i + 1
            if next_i < n and command[next_i] == ">":
                op += ">"
                next_i += 1
            if next_i < n and command[next_i] in "&|":
                op += command[next_i]
                next_i += 1
            out.append(" ")
            if prefix:
                out.append(prefix)
            out.append(op)
            out.append(" ")
            i = next_i
            continue
        if c == "<" and i + 1 < n and command[i + 1] == ">":
            # ``<>`` (POSIX read-write open) is a write surface — bash
            # creates the file if missing. Pad so the extractor sees it
            # as a separate token.
            prefix = _strip_fd_prefix_from_output(out)
            out.append(" ")
            if prefix:
                out.append(prefix)
            out.append("<>")
            out.append(" ")
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)
