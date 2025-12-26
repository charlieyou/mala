---
description: Convert a review or feature plan into small, parallelizable Beads issues with rich descriptions and dependencies.
argument-hint: <issue-scope>
---

# Beads Issue Breakdown (Reviews or Plans)

> **Upstream**: This skill accepts output from `/healthcheck` or any structured review/plan.

You are creating **bd (beads)** issues from a code review or a feature plan. Optimize for **parallel execution** by a swarm of agents and ensure each issue is **small enough for one agent to finish in a single thread**.

**Issue creation instructions (from command arguments):** $ARGUMENTS

Use the arguments above as the **authoritative scope** for what to create issues for. If they are missing, vague, or conflict with the input, ask up to 3 clarifying questions before producing issues.

If input is missing critical context (scope, expected behavior, or affected areas), ask **up to 3 clarifying questions** before producing issues.

Use the **beads skill** for this task. Produce bd-ready issues and dependency links.
Execute the `bd` commands to create issues and dependencies (don't just print them).

## Prompting Principles (Best Practices)

- **Be grounded**: Only create issues supported by the provided review/plan. If uncertain, say "Needs verification."
- **Be explicit**: State assumptions and missing context up front.
- **Be atomic**: One issue = one clear outcome; avoid bundling unrelated fixes.
- **Be parallel-first**: Avoid file overlap unless strictly required.
- **Be consistent**: Use stable priority mapping and uniform wording.
- **Be actionable**: Every issue must have acceptance criteria and a test plan.

## Goals

- Create small, high-signal issues that are easy to pick up without prior context.
- Minimize file overlap across issues so agents can work in parallel.
- Use dependencies to serialize work **only when two issues must touch the same file**.
- Include TDD guidance when appropriate.

## Sizing Rules

- Each issue must be completable by a single agent within **100k tokens**.
- If a task would exceed **100k tokens**, split it into a **parent epic** plus child issues.
- Use **parent-child dependencies** when splitting into an epic with children.
- Each issue should touch **1-2 primary files** whenever possible.
- If a change spans 3+ files, consider a coordinating epic and split by file/domain.
- **Epic default**: when creating 3+ related issues from a single review/plan scope, first create an umbrella epic for that scope and attach all issues (or sub-epics) as parent-child dependencies. If a sub-area needs 3+ issues, create a sub-epic and attach it to the umbrella epic.

## Parallelization Rules

- Prefer splits by module/file/feature boundary.
- **Do not** create two issues that edit the same file unless one blocks the other.
- If overlap is unavoidable, add a dependency and explain the ordering.
- Avoid cross-cutting refactors that force multiple agents to touch the same files.

## Priority Mapping

- Critical -> P0
- High -> P1
- Medium -> P2
- Low -> P3 or P4 (use P4 for backlog cleanups)

## Type Derivation

When input comes from `/healthcheck`, derive Type from the issue's Category and Type fields:

| Category | Default Type | Notes |
|----------|--------------|-------|
| Correctness | bug | Broken behavior |
| Dead Code | chore | Cleanup, no behavior change |
| AI Smell | bug or task | bug if behavior broken, task if just smelly |
| Structure | task | Refactor work |
| Hygiene | chore | Cleanup |
| Architecture | task or epic | epic if spans 3+ files |
| Config Drift | task | Alignment work |

If the healthcheck issue already specifies a `Type` field, use that directly.

## TDD Guidance

Add explicit TDD instructions **when changing logic or fixing a bug**:
- Add/adjust failing test first.
- Implement minimal fix.
- Refactor and run relevant tests.

## Output Format

Start with a short **Method** block (3-6 bullets): input type (review/plan), assumptions, and any missing context.

Then produce **Issue Specs** in the format below:

```
### [Handle] Title

Type: bug | feature | task | epic | chore
Priority: P0 | P1 | P2 | P3 | P4
Labels: optional (comma-separated)
Primary files: path1, path2
Dependencies: [HandleA, HandleB] (only if required)

Context:
- 2-5 bullets explaining the background and why this matters
- Include exact file/line pointers when available

Scope:
- In: what will be changed
- Out: explicit non-goals to prevent scope creep

Acceptance Criteria:
- Bullet list, testable outcomes

Test Plan:
- Bullet list (include TDD steps if applicable)

Notes for Agent:
- Constraints, edge cases, or gotchas
```

After the issue specs, **run bd commands** in this order:

1) `bd create` commands in the same order as the issue list
2) `bd label add` commands (if labels specified)
3) `bd dep add` commands for dependencies

Use **issue handles as placeholders** for IDs:

```
bd create "Title..." -p 1 --type bug --description "..."
# returns: ISSUE-A
bd dep add ISSUE-B ISSUE-A
```

## Quality Checks Before Finalizing

- No duplicate issues
- No two parallel issues edit the same file
- Dependencies only where file overlap exists
- Every issue has clear acceptance criteria and test plan
- TDD included when appropriate
- All issues are grounded in the provided input; if input has `Confidence: Low` or `Medium`, add "Needs verification" to Context
- Each issue is within the 100k-token limit (otherwise split into epic + children)
