# Lock Wait Finished — Finalize `bd-{issue_id}`

Continue following the implementer prompt. This prompt overrides it only where explicitly stated.

Earlier in this session you called `lock_wait` on files held by a peer and yielded. mala parked the session, waited on those locks, and has now resumed you on the same session to finalize the issue. Do not call `lock_wait` again on files you have not first tried to re-acquire.

## Parked Files

- Status: {status}
- Files you parked on: {wait_paths}

## Branch on Status

### If status is `free`

The files you parked on are now unlocked.

1. Re-acquire them with `lock_acquire(filepaths=[...], timeout_seconds=0)`, using the parked file list above.
2. Finalize `bd-{issue_id}`: edit the locked files, run the required validation wrappers, commit the deliverable with explicit `git add <files> && git commit -m "bd-{issue_id}: <summary>"` (never `git add .`, `-A`, `-u`, directories, globs, or `git commit -a`), then release locks.
3. If any file was taken again before you could re-acquire it, you may call `lock_wait` once more on the still-blocked files, or wrap up with what you have.

### If status is `unavailable`

The locks were not free within the budget.

1. Re-acquire what you can with `lock_acquire(filepaths=[...], timeout_seconds=0)` and finalize as much of `bd-{issue_id}` as is safe with those files.
2. If you are still blocked on files you need to complete the issue, release locks and return `ISSUE_NO_CHANGE` with the contention rationale (which files stayed locked and why no usable change was produced).

## After Finalizing

Release locks and return exactly one final output mode per the implementer prompt — the standard final template for code/config/test changes, or the appropriate marker-only line. Do not push and do not close the issue.
