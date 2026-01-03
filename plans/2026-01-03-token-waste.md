# Token Waste Analysis Report

**Date**: 2026-01-03  
**Sessions Analyzed**: 76  
**Total Log Size**: ~15MB (~3.75M tokens)  
**Estimated Waste**: ~800K tokens (21%) *after warmup fix*

## Executive Summary

Analysis of 76 mala orchestrator sessions revealed systematic token waste patterns affecting 27% of all tokens. The highest-impact issues are tool I/O related (full-file reads, TodoWrite spam) rather than verbose narration, which is more visible but less costly.

**Key Insight**: Fix tool-level issues first (full-file reads, state rewrites, warmup hallucinations), then address behavioral patterns (lock thrashing, narration).

---

## Token Waste Taxonomy

### Tier 1: Critical (Fix Immediately)

| Pattern | Sessions | Est. Waste | Root Cause |
|---------|----------|------------|------------|
| Full file reads (no line ranges) | 48/76 (63%) | ~150K tokens | Agent reads entire files when only small sections needed |
| Re-reading same files | 32/76 (42%) | ~55K tokens | No content caching; agent forgets what it already read |
| TodoWrite spam | 35/76 (46%) | ~90K tokens | Full list rewritten on every status change |
| ~~Warmup hallucinations~~ | 12/76 (16%) | ~~200K tokens~~ | ~~Fixed: tools now hooked up properly~~ |

### Tier 2: High Priority

| Pattern | Sessions | Est. Waste | Root Cause |
|---------|----------|------------|------------|
| Lock thrashing | 28/76 (37%) | ~75K tokens | try→fail→retry loops without backoff |
| Edit before lock | 18/76 (24%) | ~50K tokens | Hook denials from missing locks |
| Sequential reads | 40/76 (53%) | ~60K tokens | Reads files one-by-one instead of parallel batching |
| Verbose narration | 52/76 (68%) | ~180K tokens | "Let me...", "Now I understand..." preambles |

### Tier 3: Medium Priority

| Pattern | Sessions | Est. Waste | Root Cause |
|---------|----------|------------|------------|
| Git archaeology | 14/76 (18%) | ~40K tokens | Investigating commit history for simple fixes |
| Grep→Read→Grep redundancy | 25/76 (33%) | ~35K tokens | Searching files already in context |
| API trial-and-error | 8/76 (11%) | ~60K tokens | Wrong endpoints/auth without loading skills first |

### Tier 4: Low Priority

| Pattern | Sessions | Est. Waste | Root Cause |
|---------|----------|------------|------------|
| `python` vs `uv run python` | 15/76 (20%) | ~12K tokens | Wrong Python invocation causing retries |
| Verbose final summaries | 38/76 (50%) | ~30K tokens | Restating what's already visible |
| Post-commit verification | 12/76 (16%) | ~8K tokens | Running `git log -1` after successful commit |

---

## Waste by Session Type

| Session Type | Count | Avg. Waste | Top Issues |
|--------------|-------|------------|------------|
| Warmup/Prefill | 12 | 85-95% | Hallucinated tool calls, full file dumps |
| Code Review | 22 | 35-50% | Verbose narration, excessive exploration |
| Implementation | 30 | 20-35% | TodoWrite spam, lock thrashing |
| Bug Fix | 12 | 30-45% | Git archaeology, re-reading files |

---

## Priority-Ordered Fixes

### Priority #1: Full-File Reads + Re-reads

**Problem**: Tool responses returning thousands of lines dwarf all other token costs.

**Fixes**:
1. Enforce `read_range` parameter in system prompt
2. Reject or auto-slice reads over threshold (e.g., 200 lines)
3. Add content cache keyed by `(path, [start,end])`
4. Log/expose "read budget per session"

**Expected Impact**: 20-30% token reduction

### Priority #2: TodoWrite Spam

**Problem**: Full list rewritten on every tick instead of delta updates.

**Fixes**:
1. Change tool contract to accept incremental updates only
2. Have orchestrator delta-compress: store canonical state, only echo diffs
3. Limit to 2-3 TodoWrite calls per session (start + end)

**Expected Impact**: 10-15% token reduction

### ~~Priority #3: Warmup Hallucinations~~ (FIXED)

*Tools are now hooked up properly - this issue no longer applies.*

### Priority #3: Lock Thrashing

**Problem**: Repeated try→fail→retry cycles without backoff.

**Fixes**:
1. Move lock acquisition earlier in agent flow
2. Add backoff and cap: after 3 failures, pick another file or pause
3. Detect and short-circuit "same file, same error, >3 times"
4. Document correct locking pattern in AGENTS.md

**Expected Impact**: 5-10% token reduction

### Priority #4: Sequential vs Parallel Reads

**Problem**: Reading files one-by-one when batching is possible.

**Fixes**:
1. Encourage batch Read/Grep calls in system prompt
2. Orchestrator automatically batches independent read requests
3. Combine with content caching to avoid duplicates

**Expected Impact**: 5% token reduction (mostly latency improvement)

### Priority #5: Verbose Narration

**Problem**: "Let me check...", "Now I understand..." preambles.

**Fixes**:
1. System prompt: "Limit responses to 3 sentences. Don't narrate mechanics."
2. Lower temperature slightly
3. Bias toward concise examples in few-shot prompt

**Expected Impact**: 3-5% token reduction

---

## Patterns to Monitor (Potentially Missing)

Based on Oracle review, also watch for:

1. **Oversized system prompts**: Long preambles copied every turn
2. **Echoing large context**: Model re-printing tool outputs verbatim
3. **Over-summarization**: Recaps before each step
4. **Cross-agent duplication**: Multiple agents reading same core files
5. **Tool-returned noise**: Large envelopes when subsets suffice

---

## AGENTS.md Token Hygiene Rules

Add this block to AGENTS.md:

```markdown
## Token Efficiency (MUST Follow)

### Tool Usage
- NEVER read entire files. ALWAYS use `read_range` parameter.
- NEVER re-read a file you already have in context.
- Batch independent Read/Grep calls in single message.
- Use `grep -n <pattern>` first, then Read with line range.

### Locking
- ALWAYS acquire locks BEFORE attempting edit_file.
- Pattern: lock-try.sh → edit → unlock
- After 3 lock failures, stop touching that file.

### TodoWrite
- MAX 2-3 TodoWrite calls per session.
- Write at start (plan) and end (complete). Skip intermediate updates.
- Send only changed items, not full list.

### Response Style
- DO NOT narrate actions: "Let me...", "Now I will...", "I understand..."
- Execute tools directly without preamble.
- Keep explanations to 3 sentences max unless asked.
- Front-load conclusions before explanations.

### Commands
- ALWAYS use `uv run python`, never bare `python`.
- Use `pytest -o cache_dir=/tmp/... --reruns 2 -q`.
- Combine validation: `uvx ruff check . && uvx ruff format --check . && uvx ty check`

### Warmup
- Warmup response MUST be ≤ 500 tokens.
- DO NOT simulate tool calls. Output only bullet points.
```

---

## Monitoring Recommendations

### Per-Session Metrics

- Total tokens, tool-input tokens, tool-output tokens
- Count/size of Read calls (full vs ranged)
- Duplicate read count (same path + same range)
- TodoWrite calls and cumulative payload size
- Lock failures and retries per file
- Warmup-only sessions (no commits but tokens > threshold)

### Threshold Alerts

| Metric | Threshold | Action |
|--------|-----------|--------|
| Full-file reads | > 3 per session | Log warning, clamp |
| Lock failures on file | > 3 per file | Force stop touching file |
| TodoWrite calls | > 5 per session | Switch to minimal mode |
| Warmup tokens | > 1000 | Flag for review |

### Regression Checks

- Re-run against fixed task set on each deploy
- Compare token breakdowns by task type
- Track "token efficiency score" over time

---

## Expected Outcomes

| Fix Category | Projected Savings |
|--------------|-------------------|
| Full-file reads + re-reads | 20-30% |
| TodoWrite batching | 10-15% |
| ~~Warmup fixes~~ | ~~5-8%~~ (fixed) |
| Lock thrashing | 5-10% |
| Parallelization | 5% |
| Narration reduction | 3-5% |
| **Total (with overlap)** | **40-50%** |

---

## Appendix: Session Analysis Details

### Session Type Breakdown

- **Warmup/Prefill**: 12 sessions, 85-95% waste
- **Code Review**: 22 sessions, 35-50% waste  
- **Implementation**: 30 sessions, 20-35% waste
- **Bug Fix**: 12 sessions, 30-45% waste

### Largest Individual Waste Sessions

1. `1e00d0d5` (1.7MB): Braintrust API confusion - 37 failed API attempts
2. `3593f230` (841KB): Lock polling loop - 15+ lock commands waiting
3. `512f7d9c` (882KB): TodoWrite spam - 9 calls rewriting 8-9 items each
4. `17dabd27` (930KB): Multiple concurrent issues with all patterns

### Tool Frequency Across All Sessions

| Tool | Total Uses | Avg/Session | Waste Contribution |
|------|------------|-------------|-------------------|
| Bash | ~2,400 | 32 | Lock commands, git archaeology |
| Read | ~1,200 | 16 | Full-file reads, re-reads |
| Edit | ~450 | 6 | Pre-lock failures |
| TodoWrite | ~280 | 4 | Full list rewrites |
| Grep | ~350 | 5 | Redundant searches |
| Glob | ~150 | 2 | .venv pollution |
