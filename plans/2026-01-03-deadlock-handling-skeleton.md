# Implementation Plan: Deadlock Handling for Multi-Agent Coordination

## Context & Goals
- **Spec**: N/A — derived from user description
- Detect deadlocks between concurrent agents (e.g., two agents waiting on each other's locks)
- Surface deadlocks to users/orchestrator with actionable information
- Provide resolution paths: alerting, timeouts, or automatic recovery

## Scope & Non-Goals

### In Scope
- Deadlock detection mechanism between agents
- [TBD: Which detection approach to implement]
- Alerting/surfacing deadlock conditions
- [TBD: Recovery mechanism]

### Out of Scope (Non-Goals)
- Changes to the core locking primitives (hardlink-based locks work correctly)
- [TBD: Confirm other exclusions with user]

## Assumptions & Constraints

### Implementation Constraints
- Must integrate with existing `LockManager` and lock scripts without breaking changes
- Must work with current agent prompt protocol (3 retries + lock-wait.sh pattern)
- Should not add significant overhead to lock acquisition path
- [TBD: Performance budget]

### Testing Constraints
- 85% coverage threshold (enforced by quality gate)
- Must include unit tests for detection logic
- Integration tests for multi-agent deadlock scenarios
- [TBD: E2E test requirements]

## Prerequisites
- [ ] Understand current lock state visibility (can we query all locks and their holders?)
- [ ] Verify lock files contain enough metadata for deadlock detection
- [ ] [TBD: Other prerequisites]

## High-Level Approach

[TBD: Select from approaches below after user input]

### Approach A: Wait-For Graph Detection (Centralized)

Build a directed graph of agent → lock dependencies, detect cycles.

**How it works:**
1. Orchestrator maintains a wait-for graph: nodes = agents + locks, edges = "waits for" / "held by"
2. When agent A calls `lock-wait.sh <file>`, orchestrator records A → lock_X edge
3. Lock holder already known from lock files, so we have lock_X → agent_B edge
4. Cycle detection runs on graph: if A → lock_X → B → lock_Y → A found, deadlock detected

**Pros:**
- Precise detection, no false positives
- Can detect multi-party deadlocks (A→B→C→A)
- Classical algorithm, well-understood

**Cons:**
- Requires orchestrator to intercept/track wait operations
- Additional state management
- Slight latency to check graph on each wait

### Approach B: Timeout with Holder Introspection (Decentralized)

Enhance timeout handling with deadlock awareness.

**How it works:**
1. When `lock-wait.sh` times out, agent checks who holds the lock
2. Agent then checks if holder is also waiting (via a new "wait-status" query)
3. If holder is waiting on something this agent holds → deadlock
4. Report deadlock with specific details

**Pros:**
- No centralized state needed
- Detection only happens when there's already a timeout
- Simpler implementation

**Cons:**
- Only detects after timeout (not proactive)
- May miss complex multi-party deadlocks
- Requires new "what is agent X waiting for?" capability

### Approach C: Heartbeat + Probe (Hybrid)

Combine timeout detection with active probing.

**How it works:**
1. Waiting agents periodically "probe" the lock holder
2. Probe message asks: "Are you also waiting? If so, for what?"
3. Chain the probes: if holder is waiting on lock Z, probe Z's holder
4. If probe chain returns to original agent → deadlock

**Pros:**
- Proactive detection without central state
- Works with existing architecture
- Self-healing potential (abort lowest-priority agent)

**Cons:**
- Probe overhead during waits
- Complex message passing between agents
- Race conditions possible

### Approach D: Lock Ordering + Prevention (Preventive)

Prevent deadlocks rather than detect them.

**How it works:**
1. Assign a total order to all lockable resources (e.g., by canonical path hash)
2. Require agents to acquire locks in order: lock_A < lock_B < lock_C
3. If agent needs lock_C but already holds lock_D where D > C, must release D first

**Pros:**
- Prevents deadlocks entirely (no detection needed)
- No runtime overhead after initial sort
- Proven technique (OS kernels use this)

**Cons:**
- Requires changing agent lock acquisition behavior
- May require releasing and reacquiring locks
- Work lost if agent must release mid-task

## Technical Design

### Architecture
[TBD: Design details based on chosen approach]

### Data Model
[TBD: New data structures needed]

Current lock file format: `<agent_id>\n`
[TBD: Additional metadata needed?]

### API/Interface Design
[TBD: New functions/hooks/commands]

### File Impact Summary
- `src/infra/tools/locking.py` — Exists (likely modify for detection state)
- `src/infra/hooks/locking.py` — Exists (may need new hooks)
- `src/orchestration/orchestrator.py` — Exists (may need deadlock handling in main loop)
- `src/prompts/implementer_prompt.md` — Exists (update agent instructions)
- `src/scripts/lock-wait.sh` — Exists (may need deadlock-aware version)
- [TBD: New files needed]

## Risks, Edge Cases & Breaking Changes
- [TBD: Risk assessment based on chosen approach]
- Backwards compatibility concerns: Must not break single-agent workflows
- Race condition between detection and resolution
- False positives could incorrectly abort agents

## Testing & Validation Strategy
- [TBD: Test types and coverage]
- Unit tests for deadlock detection algorithm
- Integration tests with multiple mock agents
- [TBD: Manual validation steps]

### Acceptance Criteria Coverage
| Criterion | Approach |
|-----------|----------|
| Detect classic A↔B deadlock | [TBD] |
| Surface deadlock to user | [TBD] |
| Provide resolution path | [TBD] |
| No false positives | [TBD] |

## Rollback Strategy
- [TBD: How to revert if deployment fails]
- Feature flag strategy: [TBD]

## Open Questions
1. What is the acceptable detection latency? (proactive vs on-timeout)
2. Should deadlock resolution be automatic (kill one agent) or manual (alert user)?
3. What priority scheme for automatic resolution? (oldest agent? agent with less work done?)
4. Should we prevent deadlocks (lock ordering) or detect them?
5. Performance budget: how much overhead is acceptable?

## Next Steps
After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
