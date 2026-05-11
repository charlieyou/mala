## External Review Failed (Attempt {attempt}/{max_attempts})

Continue following the implementer prompt. This prompt overrides it only where explicitly stated.

The external reviewers found these issues:

{review_issues}

## Required Actions

1. Fix all P0/P1 findings. Triage P2/P3 findings; fix important ones and defer the rest with rationale.
2. For each P0/P1, identify internally the violated invariant or contract, the enforcing mechanism of the fix, and the regression test or surrogate proof.
3. Add a regression test or assertion that would fail before the fix and pass after when practical. If not practical, record the surrogate evidence.
4. Do not replace the overall approach without proof that the previous approach fails, such as a counterexample, failing test, or contradicted API/spec behavior.
5. Use subagents only when the implementer prompt's subagent threshold is met.

## Validation

Re-run configured validations after edits. Each snippet below is the exact canonical wrapper for one validation command; run each fenced snippet as its own single Bash tool call and do not manually echo `MALA_EVIDENCE`:

Format:
```bash
{format_command}
```

Lint:
```bash
{lint_command}
```

Typecheck:
```bash
{typecheck_command}
```

Custom commands, if configured:
{custom_commands_section}

Test:
```bash
{test_command}
```

If formatting changes files, re-run validations from the start.

## Commit

Commit fixes with:

```bash
git add <explicit files> && git commit -m "bd-{issue_id}: <description>"
```

Use explicit file paths only. Do not use `-A`, `-u`, `--all`, directories, globs, or `git commit -a`.

## Reviewer Context

Your final response is passed to the reviewer on retry. In `Reviewer context`, include only useful retry context:

- `Resolved`: exact prior finding title plus file:line or test evidence.
- `False Positives`: exact prior finding title plus file:line or API evidence proving it is false.
- `Questions`: answers to reviewer questions or narrow uncertainties.

Use the standard final output template from the implementer prompt.

The orchestrator will re-run both the quality gate and external review after your fixes.
