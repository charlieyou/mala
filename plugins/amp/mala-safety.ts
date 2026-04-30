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
  const unifiedRe = /^\+\+\+\s+(?:b\/)?([^\t\r\n]+?)(?:\t.*)?$/gm;
  while ((m = unifiedRe.exec(text)) !== null) {
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
    // invocation whose command begins with the sentinel prefix is intercepted,
    // re-emits the load marker on stderr, and synthesizes an OK result so the
    // self-test does not actually execute the command.
    if (BASH_TOOL_NAMES.has(tool)) {
      const cmd = readBashCommand(input);
      if (cmd.startsWith(SENTINEL_TOOL_PREFIX)) {
        emitSentinelMarker();
        return {
          action: "synthesize",
          result: {
            output: `mala-safety v${PLUGIN_VERSION} active`,
            exitCode: 0,
          },
        };
      }

      const decision = checkBashCommand(cmd);
      if (decision) {
        return decision;
      }
      return { action: "allow" };
    }

    if (FILE_WRITE_TOOLS.has(tool)) {
      if (cfg.failClosed) {
        return {
          action: "reject-and-continue",
          message: cfg.failClosedReason,
        };
      }

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
