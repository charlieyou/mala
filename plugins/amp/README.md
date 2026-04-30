# mala-safety (Amp plugin)

Bundled TypeScript plugin that mirrors mala's Claude-path safety hooks for the
Amp coder. It is the safety surface under `--dangerously-allow-all`: if this
plugin does not load, an Amp run has no replacement gating and is unsafe.

The mala orchestrator copies this directory to `~/.config/amp/plugins/mala-safety/`
on every run (handled by `AmpPluginInstaller`, T010) and verifies at startup
that the plugin actually loaded via a runtime self-test (T013). The repository
copy under `plugins/amp/` is the source of truth.

## Scope (MVP)

This plugin enforces two invariants and nothing else:

1. **Dangerous-command block** — mirrors `src/infra/hooks/dangerous_commands.py`.
   Rejects shell commands matching `DANGEROUS_PATTERNS`,
   `DESTRUCTIVE_GIT_PATTERNS`, `git commit -a/--all`, non-atomic
   add+commit, and force-pushes (without `--force-with-lease`). Both `Bash`
   and `shell_command` tool surfaces are gated; either name routes through
   the same dangerous-command check so a blocked pattern cannot slip through
   whichever surface the model uses.
2. **Lock-ownership** — mirrors
   `src/infra/hooks/locking.py::make_lock_enforcement_hook`. Rejects file-write
   tool calls (`edit_file`, `create_file`, `undo_edit`, `apply_patch`) unless
   this Amp agent holds the lock for **every** file the call would write.
   For `apply_patch` the path list is extracted from direct keys (`path`,
   `filepath`, `file_path`), array keys (`paths`, `files`, `filepaths`), and
   embedded patch bodies (`input`, `patch`, `diff`, `patch_text`) — both
   Codex-style `*** Update/Add/Delete/Move File: <path>` and unified-diff
   `+++ [b/]<path>` headers are parsed. If extraction yields zero paths, the
   call is rejected fail-closed rather than allowed.

Out of scope for MVP (deferred to follow-ups):

- `MALA_DISALLOWED_TOOLS` parity (token-saver tool denylist).
- File-write surfaces beyond the three tools above.
- Read-cache redundancy blocking (the Claude-side `FileReadCache` hook).

## Plugin API caveat (verbatim)

The Amp plugin API is officially experimental ("expect many breaking changes")
and only loads under the binary install with `PLUGINS=all` set and a working
Bun runtime. Per Amp's plugin requirements, every plugin file must start with
the exact verbatim acknowledgment header:

```
// @i-know-the-amp-plugin-api-is-wip-and-very-experimental-right-now
```

This line is the first line of `mala-safety.ts` and **must be preserved
character-for-character** through any installer / packaging step. Tests
asserting this header live in `tests/unit/infra/clients/test_amp_plugin_installer.py`
(T010) and in the CI smoke job.

## Sentinel marker (loaded-proof)

On `session.start` the plugin writes one line to `stderr`:

```json
{"mala_plugin":"loaded","version":"<sha256-prefix-of-this-file>"}
```

The `version` field is the first 16 hex chars of the SHA-256 of this very
plugin file's bytes (computed at plugin start via `import.meta.url`). The
orchestrator's runtime self-test (T013) computes the same hash from the
installed file and compares; if the marker is missing or the version differs,
the run aborts with `AmpPluginNotActiveError` rather than silently running
under `--dangerously-allow-all` with no gating.

There is also a fallback path: any `Bash` `tool.call` whose command begins
with `__mala_safety_self_test__` re-emits the marker and synthesizes an OK
result without executing the command. This lets the self-test drive Amp end
to end (model issues a sentinel bash call) and still observe the marker even
if `session.start` stderr is buffered or interleaved.

## Fail-closed mode

If any of `MALA_AGENT_ID`, `MALA_LOCK_DIR`, or `MALA_REPO_NAMESPACE` is unset
when `session.start` fires, the plugin enters fail-closed mode: every
file-write tool call is rejected with a `"lock-ownership env missing"` message.
This guarantees that a builder bug or env-stripping change in Amp upstream
cannot produce a quietly unguarded run. Dangerous-command checks remain
active in fail-closed mode (they don't depend on the env).

## Cross-language lock-file contract

The plugin **reads** the same `<hash>.lock` file format that
`src/infra/tools/locking.py::try_lock` writes. It never writes lock files —
lock acquisition continues to flow through the `lock_acquire` MCP tool (the
same path the Claude coder uses).

```
lock-key       = f"{MALA_REPO_NAMESPACE}:{canonical_path}"
canonical_path = realpath(filepath)                              if exists
                 else first-existing-ancestor + remaining parts  otherwise
filename       = sha256(lock-key).hexdigest()[:16] + ".lock"
location       = $MALA_LOCK_DIR/<filename>
body           = "<agent_id>\n"  (single line; trailing newline)
companion      = $MALA_LOCK_DIR/<sha256-prefix>.meta  (diagnostics; ignored here)
```

If `<lock_dir>/<hash>.lock` does not exist → reject (no lock held).
If it exists and its first line equals `MALA_AGENT_ID` → allow.
If it exists and the first line differs → reject (held by another agent).

**Format-drift policy:** any change to the lock-key derivation, filename
hashing, body format, or lock-dir layout must be made in lockstep across
`src/infra/tools/locking.py` AND this plugin in the same commit. Format-drift
regression tests live in `tests/integration/test_amp_lock_enforcement.py`
(T014) and generate fixtures from the real Python `try_lock` so the TS reader
is checked against the Python writer.

## Tool-name → input-key mapping

The plugin matches Python's `FILE_WRITE_TOOLS` / `FILE_PATH_KEYS` shape but
with Amp's tool names and field names (confirmed against Amp appendix docs):

| Amp tool      | Path input key       | Notes                                    |
| ------------- | -------------------- | ---------------------------------------- |
| `edit_file`   | `path`               | edit-in-place                            |
| `create_file` | `path`               | create or overwrite                      |
| `undo_edit`   | `path`               | conservative include                     |
| `apply_patch` | multi-key extraction | direct/array keys or patch-body parsing  |

`apply_patch` does not have a single canonical path key; the plugin's
`extractApplyPatchPaths()` enumerates every file the call would write so the
lock-ownership check sees the same surface as Amp's own `FilesModifiedByToolCall`
helper (which "supports edit/create/apply_patch tools"). Each extracted path
is lock-checked individually; the call is allowed only if every path is
owned by this agent.

Shell tools: `Bash` and `shell_command` are both checked. Their command is
read from `cmd` (Amp) with a defensive fallback to `command` (Anthropic-shaped
or alternative toolbox payloads).

Adding a new file-write or shell tool to Amp upstream requires updating
`FILE_WRITE_TOOLS` / `FILE_PATH_KEYS` / `BASH_TOOL_NAMES` in this file in the
same commit as Python's mirrors.

## Reject-message parity

Reject reasons are kept verbatim with the Python hook so user-visible UX is
consistent across coders. Drift here is a UX bug; assert in tests.

## Verification

This plugin is a single TypeScript file run by Bun directly (no build step).
To type-check the source:

```bash
bun build plugins/amp/mala-safety.ts --target=bun --outfile=/tmp/mala-safety-check.js
```

A non-zero exit (or a TypeScript diagnostic) is a failure. The orchestrator
does **not** rely on a build artifact — Amp itself executes `mala-safety.ts`
from `~/.config/amp/plugins/mala-safety/`. The build command above exists
purely to surface syntax/type errors during local development and CI; the
installer (T010) ships the `.ts` file unchanged.

## Code-review checklist

Before merging changes to this file:

- [ ] First line is exactly `// @i-know-the-amp-plugin-api-is-wip-and-very-experimental-right-now`.
- [ ] `session.start` reads the three env vars and emits the sentinel marker
      on stderr in both "config OK" and "fail-closed" paths.
- [ ] Fail-closed mode rejects every `FILE_WRITE_TOOLS` `tool.call` with a
      "lock-ownership env missing" message.
- [ ] Lock-key derivation matches `src/infra/tools/locking._lock_key` +
      `_canonicalize_path` + `lock_path` byte-for-byte (including the SHA-256
      truncation at 16 hex chars and the namespace-prefixed key).
- [ ] Reject messages for "no lock" and "lock held by another agent" match
      `src/infra/hooks/locking.make_lock_enforcement_hook` verbatim.
- [ ] Dangerous-pattern lists, destructive-git list, and safe-alternative map
      match `src/infra/hooks/dangerous_commands.py` exactly.
- [ ] `BASH_TOOL_NAMES` covers every shell-execution surface Amp exposes
      (currently `Bash` and `shell_command`).
- [ ] `FILE_WRITE_TOOLS` covers every file-modification surface Amp exposes,
      including `apply_patch`. New file-write tools require a corresponding
      entry in `FILE_PATH_KEYS` (single-path) or extraction logic in
      `extractFileWritePaths` (multi-path / patch-text).
- [ ] `apply_patch` is rejected fail-closed when no paths can be extracted
      from input.
- [ ] Plugin never writes lock files (no `try_lock` equivalent here).
- [ ] No `MALA_DISALLOWED_TOOLS` parity yet (out of MVP).

## Versioning

The marker `version` field is a content hash of this file, not a manually-bumped
semver. Any edit to `mala-safety.ts` therefore changes the version reported on
`session.start`; T013's self-test will compute the same hash from the installed
file and compare. Manual semver bumps live in `package.json` for human-facing
purposes only.
