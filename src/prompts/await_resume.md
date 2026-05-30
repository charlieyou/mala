## Background Task Finished — Finalize `bd-{issue_id}`

Continue following the implementer prompt. This prompt overrides it only where explicitly stated.

Earlier in this session you launched long-running work with `Bash(run_in_background=true)` and yielded. That backgrounded task has now finished and mala has resumed you on the same session to finalize the issue. Do not launch the same long-running work again.

## Completion Report

- Status: {status}
- Exit / summary: {summary}
- Captured output: `{output_file}`

Read `{output_file}` to inspect what the backgrounded task actually produced before deciding what to do. Treat the captured output as the source of truth, not your memory of how the task was expected to behave.

## Branch on Outcome

First determine whether the task succeeded. Treat the run as a **failure** when `Status` is `failed` or `stopped`, when `Exit / summary` reports a non-zero exit code, or when `{output_file}` shows errors, a traceback, or incomplete output. Otherwise treat it as **completed**.

### If the task failed (non-zero exit, `failed`, or `stopped`)

1. Diagnose the failure using `{output_file}`: identify the first real error and its root cause.
2. Do **not** commit a broken or partial result, and do not fabricate validation evidence.
3. If you can fix it within scope, make the smallest correct fix and re-run the required validation wrappers before committing anything.
4. If it cannot be fixed safely in this session, leave the worktree without a misleading commit, record the failure and remaining work under `Follow-ups`, release locks, and return the standard final template (or `ISSUE_NO_CHANGE` with the blocker rationale if no usable change was produced).

### If the task completed successfully

1. Verify the deliverable in `{output_file}` matches the issue requirements.
2. Run any required validation wrappers that depend on the backgrounded result so the gate has fresh evidence.
3. Finalize `bd-{issue_id}`: commit the deliverable with explicit `git add <files> && git commit -m "bd-{issue_id}: <summary>"` (never `git add .`, `-A`, `-u`, directories, globs, or `git commit -a`), and continue any dependent work that the backgrounded step unblocked.
4. Release locks only after the commit succeeds.

## After Finalizing

Release locks and return exactly one final output mode per the implementer prompt — the standard final template for code/config/test changes, or the appropriate marker-only line. Do not push and do not close the issue.
