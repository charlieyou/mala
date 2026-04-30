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
// all use the `path` input key (confirmed against Amp appendix docs); MVP
// scope per the plan covers exactly these tools.

const FILE_WRITE_TOOLS: ReadonlySet<string> = new Set([
  "edit_file",
  "create_file",
  "undo_edit",
]);

const FILE_PATH_KEYS: Readonly<Record<string, string>> = {
  edit_file: "path",
  create_file: "path",
  undo_edit: "path",
};

// Amp's shell tool is named `Bash` and accepts the command in `cmd`
// (unlike Anthropic's `command`). Confirmed against Amp appendix docs.
const BASH_TOOL_NAMES: ReadonlySet<string> = new Set(["Bash"]);
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
      const pathKey = FILE_PATH_KEYS[tool];
      const filePathRaw = pathKey ? input[pathKey] : undefined;
      if (typeof filePathRaw !== "string" || !filePathRaw) {
        // Path absent or non-string. Mirrors Python hook: allow and let the
        // tool surface its own error (we can't compute a lock key with no path).
        return { action: "allow" };
      }
      const filePath = filePathRaw;

      let lockFile: string;
      let holder: string | null;
      try {
        lockFile = lockFilePath(filePath, cfg.lockDir, cfg.repoNamespace);
        holder = getLockHolder(lockFile);
      } catch (err) {
        // Fail closed on any unexpected error in lock-key derivation.
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
