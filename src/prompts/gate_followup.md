## Quality Gate Failed (Attempt {attempt}/{max_attempts})

Continue following the implementer prompt. This prompt overrides it only where explicitly stated.

The quality gate failed with these issues:

{failure_reasons}

## Stale Commit Detection

No git archaeology except this stale-commit check.

If the failure reason mentions "stale commits from previous runs are rejected", the orchestrator is telling you that any existing `bd-{issue_id}` commit was created before this run's baseline timestamp.

If the work is already complete from a prior run and the working tree is clean:

1. Verify a `bd-{issue_id}` commit exists with `git log --oneline --grep="bd-{issue_id}"`.
2. Return `ISSUE_ALREADY_COMPLETE: <rationale>` as your final output.
3. Include the commit hash and `bd-{issue_id}` message in the rationale.

If work is not complete, treat this as a normal validation failure and create a new commit.

## Required Actions

This follow-up overrides the baseline "do not fix untouched files" validation rule. The full quality gate must pass, so fix all errors required by the gate, even if they appear in files you did not touch before this retry.

1. Fix all validation failures listed above.
2. Re-run the canonical wrapper for each missing or invalid command. Each snippet below is the exact wrapper the gate recognizes — run it as-is in a single Bash tool call. Do not manually echo `MALA_EVIDENCE`. Rerun the provided snippet exactly as shown — only the snippet is recognized as evidence.

```bash
mkdir -p {validation_log_dir}
{missing_command_wrappers}
```

If the failure mentions missing custom evidence keys such as `python_test` or `python_lint`, the snippets above already include the canonical wrapper for that exact key — run each wrapper unmodified so its `MALA_EVIDENCE name=<key>` line attributes the rerun. A generic `custom.log` or hand-written `MALA_EVIDENCE` line will not be attributed.

3. If formatting changes files, re-run the wrappers from the start.
4. Commit fixes with:

```bash
git add <explicit files> && git commit -m "bd-{issue_id}: <description>"
```

Use explicit file paths only. Do not use `-A`, `-u`, `--all`, directories, globs, or `git commit -a`.

The orchestrator requires new validation evidence, so re-run the wrappers even if they passed before.
