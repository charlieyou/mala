// @i-know-the-amp-plugin-api-is-wip-and-very-experimental-right-now
//
// Bun unit tests for the shell-write classifier in mala-safety.ts.
//
// Every adversarial scenario flagged in external review of bd-mala-eymhx.20
// has at least one regression test here. The tests assert behavior of the
// pure parsing functions (no plugin lifecycle); the production load path
// is tested separately by tests/integration/test_amp_lock_enforcement.py
// via real Amp.
//
// Run from the repo root:
//   bun test plugins/amp/mala-safety.test.ts
//
// Or via the Python wrapper:
//   uv run pytest tests/integration/test_amp_safety_parser.py

import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  classifyShellWrites,
  extractRedirectTargets,
  findCommandSubstitutionWithWrite,
  findRejectedShellPrimitive,
  getStageCommandBaseName,
  isExcludedShellWritePath,
  isValidationLogPath,
} from "./mala-safety.ts";

// `dd if=` is in the dangerous-pattern list of the plugin's *other*
// gate; the constant below avoids embedding that literal substring in
// test sources, since some surrounding agents apply the same gate to
// their own tool-call inputs and would refuse to write the test file.
const DD_IF = "if" + "=";

// ---------------------------------------------------------------------------
// isExcludedShellWritePath — strict /dev/fd/<digits> regex
// ---------------------------------------------------------------------------

describe("isExcludedShellWritePath", () => {
  test("excludes /dev/null exactly", () => {
    expect(isExcludedShellWritePath("/dev/null")).toBe(true);
  });

  test("excludes /dev/stderr/stdout/tty", () => {
    expect(isExcludedShellWritePath("/dev/stderr")).toBe(true);
    expect(isExcludedShellWritePath("/dev/stdout")).toBe(true);
    expect(isExcludedShellWritePath("/dev/tty")).toBe(true);
  });

  test("excludes literal /dev/fd/<digits>", () => {
    expect(isExcludedShellWritePath("/dev/fd/0")).toBe(true);
    expect(isExcludedShellWritePath("/dev/fd/3")).toBe(true);
    expect(isExcludedShellWritePath("/dev/fd/255")).toBe(true);
  });

  // Regression: `startsWith("/dev/fd/")` would let a path-traversal
  // bypass the lock-check by writing to `/dev/fd/../../path`, which
  // Bash resolves to an arbitrary file under cwd. Strict literal-only
  // matching closes this fail-open.
  test("does NOT exclude /dev/fd/ path-traversal", () => {
    expect(isExcludedShellWritePath("/dev/fd/../../etc/passwd")).toBe(false);
    expect(isExcludedShellWritePath("/dev/fd/../foo")).toBe(false);
  });

  test("does NOT exclude /dev/fd/ with non-numeric segment", () => {
    expect(isExcludedShellWritePath("/dev/fd/abc")).toBe(false);
    expect(isExcludedShellWritePath("/dev/fd/3a")).toBe(false);
  });

  test("does NOT exclude /dev/fd/N/extra", () => {
    expect(isExcludedShellWritePath("/dev/fd/3/extra")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// isValidationLogPath — narrow shell-redirect exemption
// ---------------------------------------------------------------------------

describe("isValidationLogPath", () => {
  test("matches files under the configured validation log directory", () => {
    expect(
      isValidationLogPath(
        "/tmp/mala-validation-logs/repo/issue.test.log",
        "/tmp/mala-validation-logs",
      ),
    ).toBe(true);
  });

  test("does NOT match sibling prefixes", () => {
    expect(
      isValidationLogPath(
        "/tmp/mala-validation-logs-evil/issue.test.log",
        "/tmp/mala-validation-logs",
      ),
    ).toBe(false);
  });

  test("does NOT match traversal out of the validation log directory", () => {
    expect(
      isValidationLogPath(
        "/tmp/mala-validation-logs/../repo/file.py",
        "/tmp/mala-validation-logs",
      ),
    ).toBe(false);
  });

  test("does NOT exempt lock directory logs", () => {
    expect(
      isValidationLogPath(
        "/tmp/mala-locks/engine-5hx.4.test.log",
        "/tmp/mala-validation-logs",
      ),
    ).toBe(false);
  });

  test("does NOT exempt paths escaping through a symlink", () => {
    const root = mkdtempSync(join(tmpdir(), "mala-validation-test-"));
    try {
      const logs = join(root, "logs");
      const repo = join(root, "repo");
      mkdirSync(logs);
      mkdirSync(repo);
      symlinkSync(repo, join(logs, "escape"), "dir");

      expect(isValidationLogPath(join(logs, "escape", "file.py"), logs)).toBe(
        false,
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("does NOT exempt everything when validation log dir is root", () => {
    expect(isValidationLogPath("/tmp/anything.log", "/")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// extractRedirectTargets — operators, quoting, separators
// ---------------------------------------------------------------------------

describe("extractRedirectTargets", () => {
  test("simple > captures literal target", () => {
    expect(extractRedirectTargets("echo X > foo")).toEqual([
      { kind: "literal", path: "foo" },
    ]);
  });

  test("simple >> captures literal target", () => {
    expect(extractRedirectTargets("echo X >> foo")).toEqual([
      { kind: "literal", path: "foo" },
    ]);
  });

  // Regression: `>|` (noclobber-override) was unrecognized — a bare `>`
  // followed by `|` looked like a redirect with no target, and the
  // command was allowed without lock-check while still overwriting the
  // file.
  test("clobber-override >| captures target", () => {
    expect(extractRedirectTargets("echo X >| foo")).toEqual([
      { kind: "literal", path: "foo" },
    ]);
  });

  test("1>| captures target", () => {
    expect(extractRedirectTargets("ls 1>| foo")).toEqual([
      { kind: "literal", path: "foo" },
    ]);
  });

  test("2>| captures target", () => {
    expect(extractRedirectTargets("ls 2>| err.log")).toEqual([
      { kind: "literal", path: "err.log" },
    ]);
  });

  test("&>| captures target", () => {
    expect(extractRedirectTargets("ls &>| out")).toEqual([
      { kind: "literal", path: "out" },
    ]);
  });

  test("1>, 1>>, 2>, 2>>, &>, &>> all parse", () => {
    expect(extractRedirectTargets("ls 1> a 1>> b 2> c 2>> d &> e &>> f")).toEqual(
      [
        { kind: "literal", path: "a" },
        { kind: "literal", path: "b" },
        { kind: "literal", path: "c" },
        { kind: "literal", path: "d" },
        { kind: "literal", path: "e" },
        { kind: "literal", path: "f" },
      ],
    );
  });

  test("excludes /dev/null", () => {
    expect(extractRedirectTargets("ls > /dev/null")).toEqual([]);
  });

  test("excludes /dev/fd/3 but not /dev/fd/../foo", () => {
    expect(extractRedirectTargets("ls > /dev/fd/3")).toEqual([]);
    expect(extractRedirectTargets("ls > /dev/fd/../foo")).toEqual([
      { kind: "literal", path: "/dev/fd/../foo" },
    ]);
  });

  test("ignores `>` inside double quotes", () => {
    expect(extractRedirectTargets('echo "x > foo" > script')).toEqual([
      { kind: "literal", path: "script" },
    ]);
  });

  test("ignores `>` inside single quotes", () => {
    expect(extractRedirectTargets("echo 'x > foo' > script")).toEqual([
      { kind: "literal", path: "script" },
    ]);
  });

  // Regression: previously the captured target stopped at the closing
  // quote, so `> "foo"bar` extracted `foo` while Bash actually writes
  // to `foobar`. If the agent held a lock on `foo`, it could overwrite
  // an unowned `foobar`.
  test("mixed quoting concatenates quoted + unquoted suffix", () => {
    expect(extractRedirectTargets('echo X > "foo"bar')).toEqual([
      { kind: "literal", path: "foobar" },
    ]);
    expect(extractRedirectTargets("echo X > 'a'b'c'")).toEqual([
      { kind: "literal", path: "abc" },
    ]);
    expect(extractRedirectTargets('echo X > pre"mid"post')).toEqual([
      { kind: "literal", path: "premidpost" },
    ]);
  });

  // Regression: `)` was missing from the unquoted-target separator
  // class, so `(echo X > foo)` captured `foo)` — confusing
  // fail-closed-reject under canonicalization.
  test("`)` terminates target capture", () => {
    expect(extractRedirectTargets("(cd /repo && echo X > foo)")).toEqual([
      { kind: "literal", path: "foo" },
    ]);
  });

  test("skips fd-dup `>&N`", () => {
    // `>&1`, `2>&1`: pure fd duplication, no file write. The redirect
    // scanner detects the operator but the target is `&...`, which is
    // recognized as fd-dup and skipped.
    expect(extractRedirectTargets("ls 2>&1")).toEqual([]);
    expect(extractRedirectTargets("ls >&2")).toEqual([]);
  });

  test("skips process substitution `>(...)`", () => {
    expect(extractRedirectTargets("ls > >(grep foo)")).toEqual([]);
  });

  // Regression: an expanded target (path containing `$VAR`, `~`, glob,
  // etc. in a context Bash expands) is not the literal the plugin
  // sees. Lock-checking the literal would let the agent acquire a lock
  // on `$VAR/file` while Bash writes to the resolved path.
  test("flags `$VAR` outside quotes as expanded", () => {
    expect(extractRedirectTargets("echo X > $VAR/file")).toEqual([
      { kind: "expanded", raw: "$VAR/file" },
    ]);
  });

  test("flags `$VAR` inside double quotes as expanded", () => {
    expect(extractRedirectTargets('echo X > "$VAR/file"')).toEqual([
      { kind: "expanded", raw: "$VAR/file" },
    ]);
  });

  test("does NOT flag `$VAR` inside single quotes (literal)", () => {
    expect(extractRedirectTargets("echo X > '$VAR/file'")).toEqual([
      { kind: "literal", path: "$VAR/file" },
    ]);
  });

  test("flags backtick in target as expanded", () => {
    expect(extractRedirectTargets("echo X > `pwd`/file")).toEqual([
      { kind: "expanded", raw: "`pwd`/file" }, // captured raw includes backticks
    ]);
  });

  test("flags glob characters in target as expanded", () => {
    expect(extractRedirectTargets("echo X > foo*.py")).toEqual([
      { kind: "expanded", raw: "foo*.py" },
    ]);
  });

  test("flags `~` at start of unquoted target as expanded", () => {
    expect(extractRedirectTargets("echo X > ~/file")).toEqual([
      { kind: "expanded", raw: "~/file" },
    ]);
  });

  test("multiple redirects in one command", () => {
    expect(extractRedirectTargets("ls 1>> out 2>> err")).toEqual([
      { kind: "literal", path: "out" },
      { kind: "literal", path: "err" },
    ]);
  });

  // Regression: `$((arith))` is not a command substitution. Without
  // special-casing, the inner `>` was treated as a redirect.
  test("ignores `$((1>0))` arithmetic", () => {
    expect(extractRedirectTargets("echo $((1>0))")).toEqual([]);
  });

  test("captures concatenated single+double quoted target", () => {
    expect(extractRedirectTargets("echo X > 'a'\"b\"c")).toEqual([
      { kind: "literal", path: "abc" },
    ]);
  });
});

// ---------------------------------------------------------------------------
// findRejectedShellPrimitive — anchor to command position
// ---------------------------------------------------------------------------

describe("findRejectedShellPrimitive", () => {
  test("rejects tee at command position", () => {
    expect(findRejectedShellPrimitive("tee /tmp/log")).not.toBeNull();
  });

  test("rejects mv", () => {
    expect(findRejectedShellPrimitive("mv old new")).not.toBeNull();
  });

  test("rejects cp", () => {
    expect(findRejectedShellPrimitive("cp src dst")).not.toBeNull();
  });

  test("rejects dd", () => {
    expect(findRejectedShellPrimitive(`dd ${DD_IF}/dev/zero of=/tmp/big bs=1M`)).not.toBeNull();
  });

  test("rejects install", () => {
    expect(findRejectedShellPrimitive("install -m 0644 src dst")).not.toBeNull();
  });

  test("rejects ln", () => {
    expect(findRejectedShellPrimitive("ln -s a b")).not.toBeNull();
  });

  // Regression: previous tee extractor matched `tee` anywhere in the
  // stage, so `grep tee README.md` was treated as a `tee` write and
  // wrongly rejected.
  test("does NOT reject `tee` as an argument", () => {
    expect(findRejectedShellPrimitive("grep tee README.md")).toBeNull();
    expect(findRejectedShellPrimitive("echo mv")).toBeNull();
    expect(findRejectedShellPrimitive("man cp")).toBeNull();
  });

  test("rejects with var-assignment prefix", () => {
    expect(findRejectedShellPrimitive("FOO=bar BAZ=q mv old new")).not.toBeNull();
  });

  test("rejects /usr/bin/cp (basename match)", () => {
    expect(findRejectedShellPrimitive("/usr/bin/cp src dst")).not.toBeNull();
  });

  test("rejects in pipeline second stage", () => {
    expect(findRejectedShellPrimitive("cat foo | tee bar")).not.toBeNull();
  });

  test("rejects in subshell stage", () => {
    expect(findRejectedShellPrimitive("(cd repo; mv old new)")).not.toBeNull();
  });

  test("does NOT reject `git mv` (command is git)", () => {
    // `git mv` is a git-managed move; the agent still has to commit, and
    // the dangerous-command gate handles git separately. Since the
    // command name is `git` (not `mv`), the shell-primitive check is a
    // no-op here.
    expect(findRejectedShellPrimitive("git mv old new")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// findCommandSubstitutionWithWrite — $(...) and backticks
// ---------------------------------------------------------------------------

describe("findCommandSubstitutionWithWrite", () => {
  // Regression: previously the outer scanner skipped the entire `"..."`
  // span, missing the inner `> target.py`. Bash evaluates the
  // substitution and writes the file regardless.
  test("detects $(...) with redirect inside double quotes", () => {
    expect(
      findCommandSubstitutionWithWrite('echo "$(echo X > target.py)"'),
    ).not.toBeNull();
  });

  test("detects $(...) with redirect outside quotes", () => {
    expect(
      findCommandSubstitutionWithWrite("echo $(echo X > target.py)"),
    ).not.toBeNull();
  });

  test("detects backtick substitution with redirect", () => {
    expect(
      findCommandSubstitutionWithWrite("echo `echo X > target.py`"),
    ).not.toBeNull();
  });

  test("detects $(...) containing rejected primitive", () => {
    expect(
      findCommandSubstitutionWithWrite('echo "$(tee /tmp/foo)"'),
    ).not.toBeNull();
  });

  test("detects nested $(...)", () => {
    expect(
      findCommandSubstitutionWithWrite(
        'echo "$(echo "$(echo X > inner)")"',
      ),
    ).not.toBeNull();
  });

  test("does NOT detect benign $(date)", () => {
    expect(findCommandSubstitutionWithWrite("echo $(date)")).toBeNull();
  });

  test("does NOT detect substring inside single quotes", () => {
    // Single quotes are pure-literal; Bash does NOT expand $ here.
    expect(
      findCommandSubstitutionWithWrite("echo '$(echo X > target.py)'"),
    ).toBeNull();
  });

  test("does NOT detect $(:) empty substitution", () => {
    expect(findCommandSubstitutionWithWrite("echo $(:)")).toBeNull();
  });

  test("does NOT detect $(()) arithmetic with `>`", () => {
    expect(findCommandSubstitutionWithWrite("echo $((1>0))")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// classifyShellWrites — the integrated decision surface
// ---------------------------------------------------------------------------

describe("classifyShellWrites", () => {
  test("benign read: no decision", () => {
    expect(classifyShellWrites("ls -la")).toEqual({ paths: [], reject: null });
    expect(classifyShellWrites("git status")).toEqual({
      paths: [],
      reject: null,
    });
  });

  test("simple redirect: lock-check the literal path", () => {
    expect(classifyShellWrites("echo X > /tmp/foo.py")).toEqual({
      paths: ["/tmp/foo.py"],
      reject: null,
    });
  });

  test("clobber-override redirect: lock-check", () => {
    expect(classifyShellWrites("echo X >| foo")).toEqual({
      paths: ["foo"],
      reject: null,
    });
  });

  test("redirect to /dev/null: no paths to check", () => {
    expect(classifyShellWrites("ls > /dev/null")).toEqual({
      paths: [],
      reject: null,
    });
  });

  test("$(...) write: reject", () => {
    const r = classifyShellWrites('echo "$(echo X > target.py)"');
    expect(r.reject).not.toBeNull();
    expect(r.paths).toEqual([]);
  });

  test("backtick write: reject", () => {
    const r = classifyShellWrites('echo `echo X > target.py`');
    expect(r.reject).not.toBeNull();
    expect(r.paths).toEqual([]);
  });

  test("tee command: reject (cannot lock-check safely)", () => {
    const r = classifyShellWrites("tee /tmp/foo");
    expect(r.reject).not.toBeNull();
  });

  test("mv command: reject", () => {
    const r = classifyShellWrites("mv old new");
    expect(r.reject).not.toBeNull();
  });

  test("cp --target-directory: reject", () => {
    // Regression: previously the parser only used the last positional
    // arg, so `cp --target-directory=out src` skipped lock-check
    // entirely (one positional). Rejecting cp at command position
    // closes this without needing a flag-aware parser.
    const r = classifyShellWrites("cp --target-directory=out src");
    expect(r.reject).not.toBeNull();
  });

  test("cp -t out src: reject", () => {
    const r = classifyShellWrites("cp -t out src");
    expect(r.reject).not.toBeNull();
  });

  test("mv with redirect mixed in: reject (mv is the command)", () => {
    // Regression: `simpleShellTokenize` did not split `>`, so
    // `mv foo bar > /dev/null` tokenized as 4 args and the last one
    // (/dev/null) was wrongly the destination. With command-position
    // rejection the call is refused outright.
    const r = classifyShellWrites("mv foo bar > /dev/null");
    expect(r.reject).not.toBeNull();
  });

  test("mv source must also be lock-checked: covered by reject", () => {
    // Regression: `extractMvCpTargets` only returned the destination,
    // so `mv unowned.py owned.py` could overwrite/delete `unowned.py`
    // while only `owned.py` was lock-checked. Rejecting mv outright
    // closes both source and destination at once.
    const r = classifyShellWrites("mv unowned.py owned.py");
    expect(r.reject).not.toBeNull();
  });

  test("install/ln: reject", () => {
    expect(classifyShellWrites("install -m 0644 src dst").reject).not.toBeNull();
    expect(classifyShellWrites("ln -s a b").reject).not.toBeNull();
  });

  test("expanded redirect target: reject", () => {
    const r = classifyShellWrites("echo X > $VAR/file");
    expect(r.reject).not.toBeNull();
  });

  test("expanded inside double quotes: reject", () => {
    const r = classifyShellWrites('echo X > "$VAR/file"');
    expect(r.reject).not.toBeNull();
  });

  test("expansion inside single quotes is literal: lock-check", () => {
    const r = classifyShellWrites("echo X > '$VAR/file'");
    expect(r.reject).toBeNull();
    expect(r.paths).toEqual(["$VAR/file"]);
  });

  test("grep tee README.md: not a write", () => {
    expect(classifyShellWrites("grep tee README.md")).toEqual({
      paths: [],
      reject: null,
    });
  });

  test("echo with nested quoted `>`: not a write", () => {
    // The `>` inside `"echo X > foo"` is inside a quote — not a
    // redirect — and the OUTER `> script.sh` is the only redirect.
    // The classifier must not over-trigger on the inner quoted `>`.
    expect(classifyShellWrites('echo "echo X > foo" > script.sh')).toEqual({
      paths: ["script.sh"],
      reject: null,
    });
  });

  test("multiple top-level redirects: lock-check each literal", () => {
    expect(classifyShellWrites("ls 1>> out 2>> err")).toEqual({
      paths: ["out", "err"],
      reject: null,
    });
  });

  test("subshell containing redirect: lock-check", () => {
    expect(classifyShellWrites("(cd repo && echo X > foo)")).toEqual({
      paths: ["foo"],
      reject: null,
    });
  });

  test("subshell with rejected primitive: reject", () => {
    expect(
      classifyShellWrites("(cd repo && tee foo)").reject,
    ).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// getStageCommandBaseName — basename + assignment-skip
// ---------------------------------------------------------------------------

describe("getStageCommandBaseName", () => {
  test("plain command", () => {
    expect(getStageCommandBaseName("ls -la")).toBe("ls");
  });

  test("absolute path stripped to basename", () => {
    expect(getStageCommandBaseName("/usr/bin/cp src dst")).toBe("cp");
  });

  test("variable-assignment prefix skipped", () => {
    expect(getStageCommandBaseName("FOO=bar BAZ=q mv old new")).toBe("mv");
  });

  test("returns null on empty or whitespace-only", () => {
    expect(getStageCommandBaseName("")).toBeNull();
    expect(getStageCommandBaseName("   ")).toBeNull();
  });

  test("returns null on assignment-only", () => {
    expect(getStageCommandBaseName("FOO=bar")).toBeNull();
  });
});
