// @i-know-the-amp-plugin-api-is-wip-and-very-experimental-right-now
//
// mala-safety: bundled Amp plugin enforcing the same dangerous-command and
// lock-ownership invariants that mala's Claude path enforces via
// `src/infra/hooks/dangerous_commands.py` and
// `src/infra/hooks/locking.py::make_lock_enforcement_hook`.
//
// This plugin is the safety surface under `--dangerously-allow-all`. It must
// load (PLUGINS=all + binary install + Bun runtime) for `coder=amp` runs to
// be safe. The orchestrator runs a runtime self-test before spawning issue
// agents and fails closed if the sentinel marker emitted here is not seen.
//
// Cross-language contract: this plugin READS `<hash>.lock` files written by
// `src/infra/tools/locking.py::try_lock`. Any change to lock-key derivation,
// filename hashing, or lock-file body format must be made in lockstep across
// both languages in the same commit. This plugin never writes lock files.

import { createHash } from "node:crypto";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { basename, dirname, isAbsolute, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// --- Sentinel marker -------------------------------------------------------

// The marker version is a content hash of this plugin file. The orchestrator's
// self-test (T013) computes the same hash from the installed file and
// compares; matching hashes prove that the expected plugin code is loaded.
function computeOwnVersionHash(): string {
  try {
    const filePath = fileURLToPath(import.meta.url);
    const content = readFileSync(filePath);
    return createHash("sha256").update(content).digest("hex").slice(0, 16);
  } catch {
    return "unknown";
  }
}

const PLUGIN_VERSION = computeOwnVersionHash();

const SENTINEL_TOOL_PREFIX = "__mala_safety_self_test__";

function emitSentinelMarker(): void {
  const payload = JSON.stringify({
    mala_plugin: "loaded",
    version: PLUGIN_VERSION,
  });
  process.stderr.write(payload + "\n");
}

// --- Tool name / path key map ---------------------------------------------
//
// Mirrors `FILE_WRITE_TOOLS` / `FILE_PATH_KEYS` in
// `src/infra/hooks/file_cache.py`. Amp's edit_file / create_file / undo_edit
// use the single `path` input key (confirmed against Amp appendix docs).
// `apply_patch` is also gated: it is the third file-write surface Amp's own
// `FilesModifiedByToolCall` helper recognizes ("supports edit/create/apply_patch
// tools"), and any unified-diff/Codex-style patch can edit arbitrary files,
// so leaving it ungated would defeat the lock-ownership invariant. Path
// extraction for apply_patch handles direct path keys, array path keys, and
// embedded patch text (Codex `*** Update File:` and unified `+++ b/...`
// headers); if no paths can be extracted, the call is rejected fail-closed.

const FILE_WRITE_TOOLS: ReadonlySet<string> = new Set([
  "edit_file",
  "create_file",
  "undo_edit",
  "apply_patch",
]);

// Single-path tools: the literal input key holding the file path. apply_patch
// is intentionally absent — it routes through `extractApplyPatchPaths` instead
// because its input shape is multi-path / patch-text rather than `{path: ...}`.
const FILE_PATH_KEYS: Readonly<Record<string, string>> = {
  edit_file: "path",
  create_file: "path",
  undo_edit: "path",
};

// Shell tool surfaces. Amp's primary shell tool is `Bash` (input field `cmd`),
// but custom toolboxes / MCP servers / future Amp builds may expose the same
// capability under `shell_command`. Both names share the dangerous-command
// gate so a blocked pattern cannot slip through whichever surface the model
// happens to use. Defensive `command` fallback handles Anthropic-shaped or
// alternative toolbox payloads.
const BASH_TOOL_NAMES: ReadonlySet<string> = new Set(["Bash", "shell_command"]);
const BASH_INPUT_KEYS: readonly string[] = ["cmd", "command"];

// --- Dangerous-command policy --------------------------------------------
//
// Mirrors src/infra/hooks/dangerous_commands.py exactly. Reject reasons are
// kept verbatim so user-visible UX is consistent across coders.

const DANGEROUS_PATTERNS: readonly string[] = [
  "rm -rf /",
  "rm -rf ~",
  "rm -rf $HOME",
  ":(){:|:&};:",
  "mkfs.",
  "dd if=",
  "> /dev/sd",
  "chmod -R 777 /",
  "curl | bash",
  "wget | bash",
  "curl | sh",
  "wget | sh",
];

const DESTRUCTIVE_GIT_PATTERNS: readonly string[] = [
  "git reset --hard",
  "git reset --mixed",
  "git reset --soft",
  "git reset HEAD",
  "git clean -fd",
  "git clean -f",
  "git clean -df",
  "git clean -d -f",
  "git checkout --",
  "git checkout -f",
  "git checkout --force",
  "git restore",
  "git rebase",
  "git branch -D",
  "git branch -d -f",
  "git stash",
  "git commit --amend",
  "git merge --abort",
  "git rebase --abort",
  "git cherry-pick --abort",
  "git worktree remove",
  "git submodule deinit -f",
];

const SAFE_GIT_ALTERNATIVES: Readonly<Record<string, string>> = {
  "git stash": "commit changes instead: git add . && git commit -m 'WIP: ...'",
  "git reset --hard":
    "commit first, or use git diff to review changes before discarding",
  "git reset --mixed": "commit staged changes first",
  "git reset --soft": "create a new commit instead of rewriting history",
  "git reset HEAD": "commit staged changes first",
  "git rebase": "use git merge instead, or coordinate with other agents",
  "git checkout --":
    "commit changes first, or use git diff to review before discarding",
  "git checkout -f": "commit changes first",
  "git checkout --force": "commit changes first",
  "git restore":
    "commit changes first, or use git diff to review before discarding",
  "git clean -f": "manually remove specific untracked files with rm",
  "git merge --abort": "resolve merge conflicts instead of aborting",
  "git rebase --abort": "resolve rebase conflicts instead of aborting",
  "git cherry-pick --abort": "resolve cherry-pick conflicts instead of aborting",
  "git worktree remove": "commit changes in worktree first",
  "git submodule deinit -f": "use git submodule deinit without -f",
  "git commit --amend": "create a new commit instead of amending history",
};

// --- Path canonicalization -------------------------------------------------
//
// Mirrors src/infra/tools/locking._canonicalize_path and _resolve_with_parents.
// For non-existent paths, walk up to the first existing ancestor, resolve its
// symlinks, then append the missing components. Without this parity, lock
// keys would diverge between Python writers and the TS reader for paths
// through symlinked directories.

const MAX_RESOLVE_ITERATIONS = 100;

function tryRealpath(p: string): string {
  try {
    return realpathSync(p);
  } catch {
    return p;
  }
}

function resolveWithParents(p: string): string {
  if (existsSync(p)) {
    return tryRealpath(p);
  }
  const missing: string[] = [];
  let current = p;
  let iterations = 0;
  while (!existsSync(current) && iterations < MAX_RESOLVE_ITERATIONS) {
    iterations++;
    missing.push(basename(current));
    const parent = dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }
  let base = existsSync(current) ? tryRealpath(current) : current;
  for (let i = missing.length - 1; i >= 0; i--) {
    base = join(base, missing[i]);
  }
  return base;
}

function canonicalizePath(filepath: string, repoNamespace: string): string {
  if (isAbsolute(filepath)) {
    if (existsSync(filepath)) {
      return tryRealpath(filepath);
    }
    return resolveWithParents(normalize(filepath));
  }
  // Relative path: resolve relative to a realpath'd namespace
  const namespacePath = tryRealpath(repoNamespace) || resolve(repoNamespace);
  const candidate = join(namespacePath, filepath);
  if (existsSync(candidate)) {
    return tryRealpath(candidate);
  }
  return resolveWithParents(normalize(candidate));
}

function lockKeyHash(filepath: string, repoNamespace: string): string {
  const canonical = canonicalizePath(filepath, repoNamespace);
  const key = `${repoNamespace}:${canonical}`;
  return createHash("sha256").update(key).digest("hex").slice(0, 16);
}

function lockFilePath(
  filepath: string,
  lockDir: string,
  repoNamespace: string,
): string {
  return join(lockDir, `${lockKeyHash(filepath, repoNamespace)}.lock`);
}

function getLockHolder(lockFile: string): string | null {
  if (!existsSync(lockFile)) {
    return null;
  }
  try {
    const content = readFileSync(lockFile, "utf8");
    const firstLine = content.split("\n", 1)[0]?.trim() ?? "";
    return firstLine || null;
  } catch {
    return null;
  }
}

// Return a rejection ToolDecision if the current agent does not hold the
// lock for `filePath`, else null. Shared between the FILE_WRITE_TOOLS
// branch and the Bash shell-write branch so both paths emit identical
// "is not locked" / "is locked by <holder>" messages and use the same
// lock-key derivation. Reads `cfg` from the module-level config populated
// by `session.start`; callers must have already ensured `!cfg.failClosed`.
function checkLockOwnership(filePath: string): ToolDecision | null {
  let lockFile: string;
  let holder: string | null;
  try {
    lockFile = lockFilePath(filePath, cfg.lockDir, cfg.repoNamespace);
    holder = getLockHolder(lockFile);
  } catch (err) {
    return {
      action: "reject-and-continue",
      message: `mala-safety: lock check failed for ${filePath}: ${
        err instanceof Error ? err.message : String(err)
      }`,
    };
  }

  if (holder === null) {
    return {
      action: "reject-and-continue",
      message: `File ${filePath} is not locked. Use lock_acquire tool with filepaths: ["${filePath}"]`,
    };
  }
  if (holder !== cfg.agentId) {
    return {
      action: "reject-and-continue",
      message: `File ${filePath} is locked by ${holder}. Wait or coordinate to acquire the lock.`,
    };
  }
  return null;
}

// --- File-write path extraction ------------------------------------------
//
// `apply_patch`'s input shape is unstable across Amp versions / toolboxes: it
// may carry a single `path`, an array `paths`, or an embedded patch body that
// names files inline. The lock-ownership invariant requires lock-checking
// EVERY file the call would write, so we must enumerate them up front. We
// try, in order: direct keys, array keys, and patch-text body parsing for
// both Codex (`*** Update File:`) and unified-diff (`+++ b/...`) headers.
// If extraction yields zero paths, the caller fails closed.

const APPLY_PATCH_DIRECT_PATH_KEYS: readonly string[] = [
  "path",
  "filepath",
  "file_path",
];
const APPLY_PATCH_ARRAY_PATH_KEYS: readonly string[] = [
  "paths",
  "files",
  "filepaths",
];
const APPLY_PATCH_BODY_KEYS: readonly string[] = [
  "input",
  "patch",
  "diff",
  "patch_text",
];

function extractPathsFromPatchText(text: string): string[] {
  const out = new Set<string>();
  let m: RegExpExecArray | null;

  // Codex apply_patch primary directives: every line of the form
  //   *** Update File: <path>
  //   *** Add File: <path>
  //   *** Delete File: <path>
  // names a file the patch intends to read or write.
  const codexFileRe = /^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s+(.+?)\s*$/gm;
  while ((m = codexFileRe.exec(text)) !== null) {
    const p = m[1].trim();
    if (p) out.add(p);
  }

  // Codex move sub-directive: "*** Move to: <new>" appears INSIDE an
  // "*** Update File: <old>" block and renames the file in addition to
  // editing it. Both <old> and <new> are write targets — leaving <new>
  // unchecked would let an agent that owns only <old> write to an
  // unowned <new>, breaking AC#9 for rename/move edits. (Older drafts of
  // this regex used the non-existent "*** Move File:" directive.)
  const codexMoveToRe = /^\*\*\*\s+Move\s+to:\s+(.+?)\s*$/gm;
  while ((m = codexMoveToRe.exec(text)) !== null) {
    const p = m[1].trim();
    if (p) out.add(p);
  }

  // Unified diff "new file" header: "+++ [b/]<path>[\t<timestamp>]".
  // GNU diff -u separates filename from timestamp with a literal TAB
  // (POSIX-conformant; widely implemented). Anchoring the boundary on
  // \t — instead of "any whitespace followed by a digit" — preserves
  // filenames containing spaces+digits (e.g., "file 1.ts"). A
  // greedier pattern would clip "file 1.ts" to "file" and lock-check
  // the wrong path, which is fail-open: if "file" is owned but
  // "file 1.ts" isn't, the wrong filename returns allow.
  const unifiedDestRe = /^\+\+\+\s+(?:b\/)?([^\t\r\n]+?)(?:\t.*)?$/gm;
  while ((m = unifiedDestRe.exec(text)) !== null) {
    const p = m[1].trim();
    if (p && p !== "/dev/null") out.add(p);
  }

  // Unified diff "old file" header: "--- [a/]<path>[\t<timestamp>]". Captured
  // in addition to "+++" because each side independently names a file the
  // patch touches: a rename writes to BOTH old and new (`--- a/old` →
  // `+++ b/new`), and a deletion writes only to the `---` side while `+++`
  // is `/dev/null`. Without this, an agent that owns only the destination
  // could rename or delete an unowned source — fail-open. /dev/null is
  // excluded because it represents "no file" (adds use `--- /dev/null`).
  const unifiedSourceRe = /^---\s+(?:a\/)?([^\t\r\n]+?)(?:\t.*)?$/gm;
  while ((m = unifiedSourceRe.exec(text)) !== null) {
    const p = m[1].trim();
    if (p && p !== "/dev/null") out.add(p);
  }

  return [...out];
}

function extractApplyPatchPaths(input: Record<string, unknown>): string[] {
  const paths = new Set<string>();
  for (const k of APPLY_PATCH_DIRECT_PATH_KEYS) {
    const v = input[k];
    if (typeof v === "string" && v) paths.add(v);
  }
  for (const k of APPLY_PATCH_ARRAY_PATH_KEYS) {
    const v = input[k];
    if (Array.isArray(v)) {
      for (const p of v) {
        if (typeof p === "string" && p) paths.add(p);
      }
    }
  }
  for (const k of APPLY_PATCH_BODY_KEYS) {
    const v = input[k];
    if (typeof v === "string") {
      for (const p of extractPathsFromPatchText(v)) paths.add(p);
    }
  }
  return [...paths];
}

function extractFileWritePaths(
  tool: string,
  input: Record<string, unknown>,
): string[] {
  if (tool === "apply_patch") {
    return extractApplyPatchPaths(input);
  }
  const pathKey = FILE_PATH_KEYS[tool];
  if (!pathKey) return [];
  const v = input[pathKey];
  return typeof v === "string" && v ? [v] : [];
}

// --- Session state --------------------------------------------------------

interface SafetyConfig {
  agentId: string;
  lockDir: string;
  repoNamespace: string;
  failClosed: boolean;
  failClosedReason: string;
}

let cfg: SafetyConfig = {
  agentId: "",
  lockDir: "",
  repoNamespace: "",
  failClosed: true,
  failClosedReason:
    "mala-safety: session.start has not run yet (lock-ownership env unknown)",
};

function loadConfig(): SafetyConfig {
  const agentId = process.env.MALA_AGENT_ID ?? "";
  const lockDir = process.env.MALA_LOCK_DIR ?? "";
  const repoNamespace = process.env.MALA_REPO_NAMESPACE ?? "";
  const missing: string[] = [];
  if (!agentId) missing.push("MALA_AGENT_ID");
  if (!lockDir) missing.push("MALA_LOCK_DIR");
  if (!repoNamespace) missing.push("MALA_REPO_NAMESPACE");
  if (missing.length > 0) {
    return {
      agentId,
      lockDir,
      repoNamespace,
      failClosed: true,
      failClosedReason: `mala-safety: lock-ownership env missing (${missing.join(", ")}); refusing all file-write tool calls`,
    };
  }
  return {
    agentId,
    lockDir,
    repoNamespace,
    failClosed: false,
    failClosedReason: "",
  };
}

// --- Bash policy ----------------------------------------------------------

interface ToolDecision {
  action: "allow" | "reject-and-continue" | "synthesize";
  message?: string;
  result?: { output: string; exitCode?: number };
  input?: Record<string, unknown>;
}

function checkBashCommand(command: string): ToolDecision | null {
  for (const pattern of DANGEROUS_PATTERNS) {
    if (command.includes(pattern)) {
      return {
        action: "reject-and-continue",
        message: `Blocked dangerous command pattern: ${pattern}`,
      };
    }
  }

  for (const pattern of DESTRUCTIVE_GIT_PATTERNS) {
    if (command.includes(pattern)) {
      const alternative = SAFE_GIT_ALTERNATIVES[pattern] ?? "";
      let reason = `Blocked destructive git command: ${pattern}`;
      if (alternative) {
        reason = `${reason}. Safe alternative: ${alternative}`;
      }
      return { action: "reject-and-continue", message: reason };
    }
  }

  const lower = command.toLowerCase();
  if (lower.includes("git commit")) {
    if (lower.includes(" --all") || lower.includes("git commit -a")) {
      return {
        action: "reject-and-continue",
        message:
          'Blocked git commit -a/--all. Stage explicit files: git add <files> && git commit -m "...".',
      };
    }
    const commitIdx = lower.indexOf("git commit");
    const addIdx = lower.indexOf("git add");
    if (addIdx === -1 || addIdx > commitIdx) {
      return {
        action: "reject-and-continue",
        message:
          'Atomic add+commit required. Use `git add <files> && git commit -m "..."`.',
      };
    }
  }

  if (command.includes("git push")) {
    if (!command.includes("--force-with-lease")) {
      if (command.includes("--force") || command.includes("-f ")) {
        return {
          action: "reject-and-continue",
          message: "Blocked force push (use --force-with-lease if needed)",
        };
      }
    }
  }

  return null;
}

// --- File-modifying shell policy -----------------------------------------
//
// Closes a fail-open in the Bash branch: in-place shell editors modify files
// without routing through Amp's file-write tools (`edit_file`/`create_file`/
// `apply_patch`), so the FILE_WRITE_TOOLS lock-ownership gate never sees the
// write. Without this list, a command like `sed -i '' 's/foo/bar/' src/file.py`
// would be allowed under --dangerously-allow-all even when this agent does
// not hold the file's lock.
//
// Regex matching (rather than substring) is required because in-place flags
// can be reordered, combined into a flag bundle, or carry a backup-extension
// suffix — `sed -E -i ''`, `sed -ni`, `sed -i.bak`, `perl -pi -e`, etc. —
// and substring `sed -i` only catches one syntactic form. Each pattern
// captures `i` anywhere inside a `-`-prefixed flag bundle (`[A-Za-z]*i[A-Za-z]*`)
// preceded by zero or more reorder-tolerant non-pipeline-separator tokens
// (`(?:[^|;&\n]+\s)*`). Pipe / `;` / `&` / newline are excluded so the
// match doesn't span across separate pipeline stages.
//
// In-place editors (sed -i, perl -i, awk -i inplace) are rejected outright —
// they always rewrite their input file as their primary side-effect, so
// there is no useful "lock-check" semantic for them; the agent is told to
// route through the file-write tools instead.
//
// The OTHER shell primitives that can write repo files — output redirects
// (`>`, `>>`, `1>`, `2>`, `&>`), `tee`, `dd of=`, `mv`, `cp` — DO have a
// useful lock-check: they take an explicit target path. Those are handled
// by `extractShellWritePaths` below; each extracted path is run through
// the same lock-ownership gate as the FILE_WRITE_TOOLS branch, so a
// `lock_acquire` of the redirect target unblocks the write the same way
// it would for `edit_file`.

// Flag-bundle alphabet for sed/perl: letters, digits, underscore, dot. Digits
// matter because backup-extension forms (`sed -i1`, `perl -i0`) and digit-
// bearing flag bundles (`perl -0pi`) are real and must not be matched by a
// letters-only class. Dot covers `-i.bak`. Underscore is harmless and avoids
// drift between `\w` and `[A-Za-z0-9_]`.
//
// Trailing boundary uses a lookahead `(?=[\s'"|;&]|$)` instead of `\b` so
// that digits or quotes following `i` (e.g. `-i1`, `-i''`) do not cancel the
// boundary the way `\b` would (digits and `i` are both word chars → no
// `\b`). The lookahead admits whitespace, end-of-string, shell separators,
// and either quote — every valid token boundary.
//
// Leading `['"]?-` admits a single optional opening quote immediately before
// the flag-bundle's `-`, so whole-token-quoted flags like `sed "-i"` or
// `sed '-Ei'` match. The trailing optional `['"]?` consumes a closing quote
// before the lookahead checks the post-token boundary.
//
// Reorder-tolerance treats each chunk as "non-separator-non-quote chars OR a
// fully-quoted region" so that quoted shell separators (the literal `;`
// inside `'s/foo/bar/g;'` or the literal `;` inside `'{print;}'`) are
// absorbed by a quoted-region alternative instead of breaking the chunk.
// Without this, common in-place commands such as
// `sed -e 's/foo/bar/g;' -i file` and `awk '{print;}' -i inplace file`
// passed through. Real (unquoted) pipeline boundaries `|`/`;`/`&`/newline
// continue to terminate the gap so the gate cannot fire across pipelines.

const FILE_MODIFYING_SHELL_PATTERNS: readonly { pattern: RegExp; label: string }[] = [
  // sed in-place: any flag bundle containing `i` — `-i`, `-i.bak`, `-i1`,
  // `-Ei`, `-ni`, `-iE`, `-i''`, `"-i"`, `'-i'` — with quote-aware
  // reorder-tolerance for preceding/intervening flags.
  {
    pattern:
      /\bsed\s+(?:(?:[^|;&\n'"]|'[^']*'|"[^"]*")+\s+)*['"]?-[\w.]*i[\w.]*['"]?(?=[\s'"|;&]|$)/,
    label: "sed -i (in-place edit)",
  },
  // sed --in-place long form (with optional whole-token quoting)
  {
    pattern: /\bsed\s+(?:(?:[^|;&\n'"]|'[^']*'|"[^"]*")+\s+)*['"]?--in-place\b/,
    label: "sed --in-place",
  },
  // perl in-place: `-i`, `-pi`, `-i.bak`, `-ipe`, `-0pi`, `-i0`, etc.
  {
    pattern:
      /\bperl\s+(?:(?:[^|;&\n'"]|'[^']*'|"[^"]*")+\s+)*['"]?-[\w.]*i[\w.]*['"]?(?=[\s'"|;&]|$)/,
    label: "perl -i (in-place edit)",
  },
  // gawk/awk: literal `-i inplace` extension form (with optional quoting on
  // `inplace` so that shell-quoted forms like `gawk -i 'inplace' ...` and
  // `awk -i "inplace" ...` are caught after shell quote-removal yields the
  // same arg).
  {
    pattern:
      /\bg?awk\s+(?:(?:[^|;&\n'"]|'[^']*'|"[^"]*")+\s+)*-i\s+['"]?inplace['"]?(?=[\s'"|;&]|$)/,
    label: "awk -i inplace",
  },
];

function checkFileModifyingShell(command: string): ToolDecision | null {
  for (const { pattern, label } of FILE_MODIFYING_SHELL_PATTERNS) {
    if (pattern.test(command)) {
      return {
        action: "reject-and-continue",
        message:
          `Blocked file-modifying shell command pattern: ${label}. ` +
          "In-place shell editing bypasses the file-write tool gate and " +
          "would skip the lock-ownership check. Use edit_file / " +
          "create_file / apply_patch instead — these route through the " +
          "plugin's lock-check.",
      };
    }
  }
  return null;
}

// --- Shell file-write target extraction ----------------------------------
//
// Extracts the file paths a shell command would write to via the primitives
// (output redirects, `tee`, `dd of=`, `mv`, `cp`) that bypass Amp's
// file-write tools. Each returned path is then lock-checked by the Bash
// branch with the same allow/reject semantics as the FILE_WRITE_TOOLS
// branch.
//
// Special device targets that are not real files (`/dev/null`,
// `/dev/stderr`, `/dev/stdout`, `/dev/tty`, `/dev/fd/N`) are excluded so
// `cmd > /dev/null` and similar discard-output idioms don't trigger a
// lock-check. Every other captured target — including ones containing
// unexpanded shell variables (`$VAR`, `$(...)`, `~`) — is forwarded to the
// lock-check, where it will fail-closed (no matching <hash>.lock) unless
// the agent took the lock under the literal string the plugin sees.
//
// Why a hand-written quote-aware scanner (and not just a global regex):
// commands like `echo "echo X > foo" > script.sh` contain a `>` inside a
// quoted argument that is NOT a redirect. A pure regex would match it and
// produce a false-positive lock-check on `foo`, fail-closed-rejecting a
// legitimate command. Tracking quote state on the way through the string
// suppresses those interior matches without depending on full shell
// tokenization.

const SHELL_WRITE_DENYLIST_PATHS: ReadonlySet<string> = new Set([
  "/dev/null",
  "/dev/stderr",
  "/dev/stdout",
  "/dev/tty",
]);

function isExcludedShellWritePath(p: string): boolean {
  if (SHELL_WRITE_DENYLIST_PATHS.has(p)) return true;
  if (p.startsWith("/dev/fd/")) return true;
  return false;
}

function unquoteShellPath(s: string): string {
  if (s.length >= 2) {
    const first = s[0];
    const last = s[s.length - 1];
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return s.slice(1, -1);
    }
  }
  return s;
}

// Tokenize a single pipeline stage's argument string into whitespace-
// separated tokens, preserving quoted regions as single tokens. This is a
// poor-man's shell tokenizer: it does not implement word splitting,
// backslash escapes, brace/glob/parameter expansion, or here-docs. It is
// sufficient to identify positional vs flag arguments for `tee`, `mv`, and
// `cp` invocations whose target paths are literal strings.
function simpleShellTokenize(s: string): string[] {
  const tokens: string[] = [];
  let cur = "";
  let inQuote: '"' | "'" | null = null;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (inQuote) {
      cur += c;
      if (c === inQuote) inQuote = null;
      continue;
    }
    if (c === '"' || c === "'") {
      inQuote = c as '"' | "'";
      cur += c;
      continue;
    }
    if (/\s/.test(c)) {
      if (cur) {
        tokens.push(cur);
        cur = "";
      }
      continue;
    }
    cur += c;
  }
  if (cur) tokens.push(cur);
  return tokens;
}

// Scan `command` for output-redirect operators (`>`, `>>`, `1>`, `2>`,
// `&>`, with optional trailing `>`) at top-level (not inside quotes) and
// return each captured target path. Skips `>(` (process substitution) and
// `>&N` (file-descriptor duplication) which do not write to a path.
function extractRedirectTargets(command: string): string[] {
  const out: string[] = [];
  let i = 0;
  let inQuote: '"' | "'" | null = null;

  while (i < command.length) {
    const c = command[i];

    if (inQuote) {
      if (c === inQuote) inQuote = null;
      i++;
      continue;
    }
    if (c === '"' || c === "'") {
      inQuote = c as '"' | "'";
      i++;
      continue;
    }

    // Match a redirect operator at this position. Possible operators:
    //   >, >>, 1>, 1>>, 2>, 2>>, &>, &>>
    // Guard against double-counting >> by looking back at prev char: if
    // it's part of a wider operator, skip.
    let opStart = i;
    let opEnd = -1;
    if (c === "&" && command[i + 1] === ">") {
      opEnd = i + 2;
      if (command[opEnd] === ">") opEnd++;
    } else if (
      (c === "1" || c === "2") &&
      command[i + 1] === ">"
    ) {
      opEnd = i + 2;
      if (command[opEnd] === ">") opEnd++;
    } else if (c === ">") {
      // Bare > or >>: must not be preceded by `>` (already consumed) or
      // a digit/`&` (handled by the earlier branches). Also skip if
      // preceded by `<` (here-doc / fd dup like `<>`) — `<>` opens a
      // file for read+write and is ambiguous; treat as not a redirect
      // for this purpose.
      const prev = i > 0 ? command[i - 1] : "";
      if (prev === ">" || prev === "<") {
        i++;
        continue;
      }
      // `2>>`, `1>>` etc are caught by the digit branch above; if we
      // reach here with prev === '1'|'2'|'&', it means the digit/`&`
      // wasn't followed by `>` immediately so this `>` is independent.
      opEnd = i + 1;
      if (command[opEnd] === ">") opEnd++;
    }

    if (opEnd === -1) {
      i++;
      continue;
    }

    // Consumed operator at [opStart, opEnd). Skip whitespace.
    let j = opEnd;
    while (j < command.length && /\s/.test(command[j])) j++;

    // Reject fd-dup or process substitution.
    if (j < command.length && (command[j] === "&" || command[j] === "(")) {
      i = j;
      continue;
    }

    // Capture target token: quoted or unquoted.
    let target = "";
    if (j < command.length && (command[j] === '"' || command[j] === "'")) {
      const q = command[j++];
      while (j < command.length && command[j] !== q) {
        target += command[j++];
      }
      if (j < command.length && command[j] === q) j++;
    } else {
      while (
        j < command.length &&
        !/[\s|;&<>'"`(]/.test(command[j])
      ) {
        target += command[j++];
      }
    }

    if (target && !isExcludedShellWritePath(target)) {
      out.push(target);
    }
    i = j;
  }

  return out;
}

// Scan `command` for `tee` invocations and return file targets. Tokenizes
// each pipeline stage that mentions `tee` with quote awareness. Flags
// (`-a`, `--append`, ...) are skipped; remaining positional tokens are
// targets. Multiple files per `tee` (a documented form) are all returned.
function extractTeeTargets(command: string): string[] {
  const out: string[] = [];
  // Split into pipeline stages; pipes/`;`/`&&`/`||`/newlines separate.
  // Use a quote-aware split so separators inside quotes are kept.
  const stages = splitPipelineStages(command);
  for (const stage of stages) {
    // Find `tee` at a token boundary.
    const teeRe = /\btee\b(.*)$/m;
    const m = teeRe.exec(stage);
    if (!m) continue;
    const tokens = simpleShellTokenize(m[1]);
    for (const tok of tokens) {
      // Skip flags (unquoted leading `-`).
      if (tok.startsWith("-")) continue;
      // Skip a stray `--` separator.
      if (tok === "--") continue;
      const unq = unquoteShellPath(tok);
      if (unq && !isExcludedShellWritePath(unq)) {
        out.push(unq);
      }
    }
  }
  return out;
}

// Scan `command` for `dd of=PATH` and return the captured PATH. Paths may
// be quoted; whitespace before `of=` is handled by the pipeline-stage
// split. `dd if=...` (input) is not a write and is intentionally ignored.
function extractDdOfTargets(command: string): string[] {
  const out: string[] = [];
  const stages = splitPipelineStages(command);
  for (const stage of stages) {
    if (!/\bdd\b/.test(stage)) continue;
    const tokens = simpleShellTokenize(stage);
    for (const tok of tokens) {
      // `of=PATH` or `of="..."` or `of='...'`
      if (tok.startsWith("of=")) {
        const raw = tok.slice("of=".length);
        const unq = unquoteShellPath(raw);
        if (unq && !isExcludedShellWritePath(unq)) {
          out.push(unq);
        }
      }
    }
  }
  return out;
}

// Scan `command` for `mv` / `cp` invocations and return the destination
// argument (the last positional token). With multiple sources + a directory
// destination (`cp a b c destdir/`), the last token is still the
// destination, and lock-checking that path is the right invariant — the
// agent must hold a lock on the destination (file or directory) to write
// into it.
function extractMvCpTargets(command: string): string[] {
  const out: string[] = [];
  const stages = splitPipelineStages(command);
  const cmdRe = /\b(mv|cp)\b(.*)$/m;
  for (const stage of stages) {
    const m = cmdRe.exec(stage);
    if (!m) continue;
    const tokens = simpleShellTokenize(m[2]);
    // Strip flags. Treat `--` as a separator (everything after is positional).
    let positional: string[] = [];
    let afterDoubleDash = false;
    for (const tok of tokens) {
      if (afterDoubleDash) {
        positional.push(tok);
        continue;
      }
      if (tok === "--") {
        afterDoubleDash = true;
        continue;
      }
      if (tok.startsWith("-")) continue;
      positional.push(tok);
    }
    if (positional.length >= 2) {
      const dst = unquoteShellPath(positional[positional.length - 1]);
      if (dst && !isExcludedShellWritePath(dst)) {
        out.push(dst);
      }
    }
  }
  return out;
}

// Split a command into pipeline stages on top-level `|`, `;`, `&&`, `||`,
// `&`, newline, and parens (subshells / process substitution). Quote-aware
// so separators inside quotes are not split-on. Used so per-stage commands
// (`tee`, `dd`, `mv`, `cp`) are matched independently. Splitting on `(`
// and `)` is required so that `cmd > >(tee log)` yields a clean stage
// `tee log` for the tee extractor — without the split, the trailing `)`
// gets glued onto the path token and the lock-check sees `log)`.
function splitPipelineStages(command: string): string[] {
  const stages: string[] = [];
  let cur = "";
  let inQuote: '"' | "'" | null = null;
  for (let i = 0; i < command.length; i++) {
    const c = command[i];
    if (inQuote) {
      cur += c;
      if (c === inQuote) inQuote = null;
      continue;
    }
    if (c === '"' || c === "'") {
      inQuote = c as '"' | "'";
      cur += c;
      continue;
    }
    if (
      c === "|" ||
      c === ";" ||
      c === "&" ||
      c === "\n" ||
      c === "(" ||
      c === ")"
    ) {
      // Lookahead skip duplicates (`||`, `&&`) — they're the same separator.
      if ((c === "|" && command[i + 1] === "|") || (c === "&" && command[i + 1] === "&")) {
        i++;
      }
      stages.push(cur);
      cur = "";
      continue;
    }
    cur += c;
  }
  if (cur) stages.push(cur);
  return stages;
}

function extractShellWritePaths(command: string): string[] {
  const seen = new Set<string>();
  for (const p of extractRedirectTargets(command)) seen.add(p);
  for (const p of extractTeeTargets(command)) seen.add(p);
  for (const p of extractDdOfTargets(command)) seen.add(p);
  for (const p of extractMvCpTargets(command)) seen.add(p);
  return [...seen];
}

// --- Plugin entrypoint ----------------------------------------------------

// `amp` is typed as `unknown` here because `@ampcode/plugin` types are not
// vendored into this repo (the plugin is shipped as a source file and run
// under Bun against whatever Amp version the user has installed). The tiny
// surface area we touch (`amp.on(event, handler)`) is checked at runtime by
// the orchestrator's plugin-load self-test.
type AmpEventHandler = (event: Record<string, unknown>) => unknown;
interface AmpPluginAPI {
  on(event: string, handler: AmpEventHandler): void;
}

export default function plugin(amp: AmpPluginAPI): void {
  amp.on("session.start", () => {
    cfg = loadConfig();
    emitSentinelMarker();
  });

  amp.on("tool.call", (event: Record<string, unknown>): ToolDecision => {
    const tool = typeof event.tool === "string" ? event.tool : "";
    const rawInput = event.input;
    const input: Record<string, unknown> =
      rawInput && typeof rawInput === "object"
        ? (rawInput as Record<string, unknown>)
        : {};

    // Sentinel tool.call fallback for T013's runtime self-test: any Bash
    // invocation whose command begins with the sentinel prefix re-emits the
    // load marker on stderr regardless of fail-closed state so the
    // orchestrator can confirm the plugin loaded. The synthesized OK result
    // is replaced with reject-and-continue when fail-closed (the plan
    // requires every tool.call to reject when env is missing); the marker
    // on stderr still proves the plugin code ran.
    if (BASH_TOOL_NAMES.has(tool)) {
      const cmd = readBashCommand(input);
      if (cmd.startsWith(SENTINEL_TOOL_PREFIX)) {
        emitSentinelMarker();
        if (cfg.failClosed) {
          return {
            action: "reject-and-continue",
            message: `${cfg.failClosedReason} (sentinel reached: mala-safety v${PLUGIN_VERSION} active, but rejecting fail-closed)`,
          };
        }
        return {
          action: "synthesize",
          result: {
            output: `mala-safety v${PLUGIN_VERSION} active`,
            exitCode: 0,
          },
        };
      }
    }

    // Fail-closed: every non-sentinel tool.call rejects when lock-ownership
    // env is missing. Plan L418-L421: "every tool.call rejected with
    // explanatory message" so the agent (and the orchestrator) cannot
    // accidentally run under --dangerously-allow-all without the
    // lock-ownership gate. The dangerous-command and in-place-editor checks
    // below would also reject these calls eventually, but anchoring the
    // fail-closed gate at the top keeps the invariant explicit and covers
    // tools (e.g. plain `Bash echo x > file`) that neither the
    // dangerous-pattern nor the in-place-editor regex would otherwise
    // catch.
    if (cfg.failClosed) {
      return {
        action: "reject-and-continue",
        message: cfg.failClosedReason,
      };
    }

    if (BASH_TOOL_NAMES.has(tool)) {
      const cmd = readBashCommand(input);
      const decision = checkBashCommand(cmd);
      if (decision) {
        return decision;
      }

      const fileModDecision = checkFileModifyingShell(cmd);
      if (fileModDecision) {
        return fileModDecision;
      }

      // Lock-check shell write primitives that can modify repo files
      // (output redirects, tee, dd of=, mv, cp). Mirrors the FILE_WRITE_TOOLS
      // branch below: same lock-key derivation, same allow/reject messages,
      // so `lock_acquire` of the redirect target unblocks the write the
      // same way it would for `edit_file`.
      const shellWritePaths = extractShellWritePaths(cmd);
      for (const filePath of shellWritePaths) {
        const rejection = checkLockOwnership(filePath);
        if (rejection) return rejection;
      }

      return { action: "allow" };
    }

    if (FILE_WRITE_TOOLS.has(tool)) {
      // failClosed already handled above. The lock-ownership env is
      // guaranteed populated at this point.
      const filePaths = extractFileWritePaths(tool, input);

      // For apply_patch we MUST be able to enumerate every path the call
      // would write before allowing it; fail-closed if extraction returns
      // empty. For single-path tools (edit_file/create_file/undo_edit) we
      // mirror the Python hook and allow when the path is absent (the tool
      // will surface its own error and no actual write can happen without
      // a path).
      if (filePaths.length === 0) {
        if (tool === "apply_patch") {
          return {
            action: "reject-and-continue",
            message:
              "mala-safety: apply_patch input did not expose any file paths " +
              "to lock-check. Refusing the call to preserve the lock-ownership " +
              "invariant. If you need apply_patch, ensure the input includes a " +
              "`path`/`paths` field or a Codex/unified-diff patch body whose " +
              "headers name the target files.",
          };
        }
        return { action: "allow" };
      }

      for (const filePath of filePaths) {
        const rejection = checkLockOwnership(filePath);
        if (rejection) return rejection;
      }

      return { action: "allow" };
    }

    return { action: "allow" };
  });

  amp.on("tool.result", (_event: Record<string, unknown>) => {
    // Diagnostics-only hook: nothing to enforce post-hoc. Returning undefined
    // (no decision) leaves the result unmodified.
    return undefined;
  });
}

function readBashCommand(input: Record<string, unknown>): string {
  for (const key of BASH_INPUT_KEYS) {
    const v = input[key];
    if (typeof v === "string") {
      return v;
    }
  }
  return "";
}
