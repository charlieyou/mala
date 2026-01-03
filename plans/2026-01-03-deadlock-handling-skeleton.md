# Implementation Plan: Deadlock Handling for Multi-Agent Coordination

## Context & Goals
- **Spec**: N/A — derived from user description
- Detect deadlocks between concurrent agents (e.g., two agents waiting on each other's locks)
- Surface deadlocks to users/orchestrator with actionable information
- Provide resolution paths: alerting, timeouts, or automatic recovery

## Scope & Non-Goals

### In Scope
- Wait-For Graph based deadlock detection (centralized in orchestrator)
- Real-time detection on lock contention (not timeout-based)
- Auto-resolution: kill youngest agent, release locks, add dependency
- Event emission from PostToolUse hooks for lock state changes
- Alerting/surfacing deadlock conditions via event sink

### Out of Scope (Non-Goals)
- Changes to the core locking primitives (hardlink-based locks work correctly)
- Lock ordering / deadlock prevention (too disruptive to agent behavior)
- Agent-side deadlock detection (centralized approach chosen)
- Progress-based victim selection (youngest is simpler, can add later)

## Assumptions & Constraints

### Implementation Constraints
- Must integrate with existing `LockManager` and lock scripts without breaking changes
- Must work with current agent prompt protocol (3 retries + lock-wait.sh pattern)
- Should not add significant overhead to lock acquisition path
- Performance budget: O(n) cycle detection per WAITING event where n = active agents (typically ≤10)

### Testing Constraints
- 85% coverage threshold (enforced by quality gate)
- Must include unit tests for cycle detection logic
- Integration tests for multi-agent deadlock scenarios with mocked SDK
- No E2E tests required (deadlock scenarios are too complex to orchestrate reliably in E2E)

## Prerequisites
- [x] Understand current lock state visibility (lock files contain agent_id, can query via `get_lock_holder`)
- [x] Verify lock files contain enough metadata (agent_id is sufficient; orchestrator tracks issue_id mapping)
- [x] Verify PostToolUse hooks can intercept Bash tool results (hook infrastructure exists in `src/infra/hooks/`)

## High-Level Approach

**User Requirements:**
- Real-time detection (on lock contention)
- Auto-resolution: kill one agent, release their locks
- Add dependency to killed agent's issue so it doesn't immediately restart

Below is a detailed evaluation of each approach with concrete implementation designs.

---

### Approach A: Wait-For Graph Detection (Centralized)

**Overview:** Orchestrator maintains a directed graph of agent → lock dependencies, detects cycles in real-time.

**Concrete Implementation:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  WaitForGraph                                            │    │
│  │  ─────────────                                           │    │
│  │  nodes: Set[str]  # agent IDs + lock paths               │    │
│  │  edges: Dict[str, Set[str]]  # waits_for relationships   │    │
│  │                                                          │    │
│  │  Agent-A ──waits──► /repo/file1.py ◄──holds── Agent-B   │    │
│  │  Agent-B ──waits──► /repo/file2.py ◄──holds── Agent-A   │    │
│  │           ▲                                              │    │
│  │           └── CYCLE DETECTED = DEADLOCK                  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

Data Flow:
1. Agent calls lock-try.sh → fails (blocked)
2. PreToolUse hook intercepts → emits "wait_started" event to orchestrator
3. Orchestrator adds edge: agent → lock_path
4. Orchestrator runs DFS cycle detection
5. If cycle found → select victim, kill agent, add dependency
```

**New Components:**
```python
# src/domain/deadlock.py
@dataclass
class WaitForGraph:
    """Directed graph for deadlock detection."""
    holds: dict[str, set[str]]      # agent_id -> set of lock paths held
    waits_for: dict[str, str | None] # agent_id -> lock path waiting for (or None)

    def add_hold(self, agent_id: str, lock_path: str) -> None: ...
    def add_wait(self, agent_id: str, lock_path: str) -> None: ...
    def remove_agent(self, agent_id: str) -> None: ...
    def detect_cycle(self) -> list[str] | None:
        """Return list of agent_ids in cycle, or None if no deadlock."""
        ...

# src/infra/hooks/deadlock.py
def make_deadlock_detection_hook(
    graph: WaitForGraph,
    on_deadlock: Callable[[list[str]], None]
) -> PreToolUseHook:
    """Hook that updates wait-for graph and triggers deadlock check."""
    ...
```

**Victim Selection & Resolution:**
```python
def select_victim(cycle: list[str], agent_metadata: dict) -> str:
    """Select which agent to kill. Options:
    - Youngest (most recently started)
    - Least progress (fewest tool calls / commits)
    - Random (fair but unpredictable)
    """
    # Recommend: youngest agent (least likely to have significant work)
    return min(cycle, key=lambda a: agent_metadata[a].start_time)

async def resolve_deadlock(victim_id: str, blocked_on: str):
    """Kill victim, release locks, add dependency."""
    # 1. Cancel the agent's task
    task = orchestrator.active_tasks[victim_issue_id]
    task.cancel()

    # 2. Release all locks held by victim
    cleanup_agent_locks(victim_id)

    # 3. Add dependency: victim's issue depends on blocker's issue
    blocker_issue_id = extract_issue_id(get_lock_holder(blocked_on))
    await beads.add_dependency(victim_issue_id, blocker_issue_id)

    # 4. Mark issue for retry (but dependency prevents immediate restart)
    await beads.mark_needs_followup(victim_issue_id,
        f"Deadlock: blocked on {blocked_on}, waiting for {blocker_issue_id}")
```

**Pros:**
- Precise detection, zero false positives
- Real-time: detects immediately when cycle forms
- Can detect multi-party deadlocks (A→B→C→A)
- Classical algorithm with formal guarantees
- Clean separation: detection in orchestrator, not agents

**Cons:**
- Requires hooks to report "wait started" events to orchestrator
- Central state must be kept in sync (agent crash = stale edge)
- Slight complexity in wiring events from hooks to graph
- ~50 lines of cycle detection code

**Estimated Complexity:** Medium (~200 LOC new code)

---

### Approach B: Timeout with Holder Introspection (Decentralized)

**Overview:** No real-time detection. Only check for deadlocks when `lock-wait.sh` times out.

**Concrete Implementation:**

```
Timeline:
────────────────────────────────────────────────────────────────►
Agent-A: lock-try(file1) ✓    lock-try(file2) ✗    lock-wait(file2, 300s)
Agent-B: lock-try(file2) ✓    lock-try(file1) ✗    lock-wait(file1, 300s)
                                                    │
                                                    ▼ (5 min later)
                                              BOTH TIMEOUT
                                                    │
                                              Check: "Who holds file2?"
                                              Answer: Agent-B
                                              Check: "What is Agent-B waiting for?"
                                              Answer: file1
                                              Check: "Who holds file1?"
                                              Answer: Agent-A (ME!)
                                                    │
                                              DEADLOCK DETECTED
```

**New Components:**
```python
# src/infra/tools/locking.py
def get_wait_status(agent_id: str) -> str | None:
    """Return the lock path this agent is waiting for, or None."""
    # Requires new wait-status file: LOCK_DIR/.wait-{agent_id}
    wait_file = get_lock_dir() / f".wait-{agent_id}"
    if wait_file.exists():
        return wait_file.read_text().strip()
    return None

def record_wait_status(agent_id: str, lock_path: str) -> None:
    """Record that this agent is waiting for a lock."""
    wait_file = get_lock_dir() / f".wait-{agent_id}"
    wait_file.write_text(lock_path)

def clear_wait_status(agent_id: str) -> None:
    """Clear wait status (lock acquired or gave up)."""
    wait_file = get_lock_dir() / f".wait-{agent_id}"
    wait_file.unlink(missing_ok=True)

# Modified lock-wait.sh:
# 1. Before waiting: record_wait_status(agent_id, filepath)
# 2. On success: clear_wait_status(agent_id)
# 3. On timeout: check for deadlock using chain inspection
```

**Deadlock Chain Detection (in agent):**
```python
def check_deadlock_chain(my_agent_id: str, target_lock: str, my_locks: set[str]) -> bool:
    """Trace the wait-for chain to see if it loops back to me."""
    visited = set()
    current_lock = target_lock

    while current_lock:
        if current_lock in visited:
            return False  # Cycle but doesn't include me
        visited.add(current_lock)

        holder = get_lock_holder(current_lock)
        if holder is None:
            return False  # Lock released, no deadlock
        if holder == my_agent_id:
            return True   # DEADLOCK: chain leads back to me!

        # What is the holder waiting for?
        current_lock = get_wait_status(holder)
        if current_lock and current_lock in my_locks:
            return True   # DEADLOCK: holder waiting on something I hold

    return False
```

**Pros:**
- No central state needed
- Simple file-based coordination (wait-status files)
- Detection logic runs only on timeout (low overhead)
- Agent-side detection, no orchestrator changes

**Cons:**
- NOT real-time: 5+ minutes before detection
- May miss complex multi-party deadlocks
- Stale wait-status files if agent crashes
- Both agents timeout = both try to resolve = race condition

**Estimated Complexity:** Low (~100 LOC new code)

**❌ REJECTED: Does not meet real-time requirement**

---

### Approach C: Heartbeat + Probe (Hybrid)

**Overview:** Waiting agents periodically probe lock holders to detect cycles.

**Concrete Implementation:**

```
┌─────────────┐         probe(file1)          ┌─────────────┐
│   Agent-A   │ ──────────────────────────►   │   Agent-B   │
│             │                               │             │
│ holds:file2 │   ◄──────────────────────────│ holds:file1 │
│ waits:file1 │     "I'm waiting for file2"   │ waits:file2 │
│             │                               │             │
│             │  file2 is mine!               │             │
│             │  DEADLOCK DETECTED            │             │
└─────────────┘                               └─────────────┘

Probe Protocol (via MCP or shared filesystem):
1. Agent-A waiting for file1
2. Every 5s, A sends probe: "Agent-B, what are you waiting for?"
3. B responds: "file2"
4. A checks: "Do I hold file2?" → YES → DEADLOCK
```

**New Components:**
```python
# Option 1: File-based probing
# LOCK_DIR/.probe-{agent_id}-{nonce} = request
# LOCK_DIR/.probe-response-{agent_id}-{nonce} = response

# Option 2: Orchestrator-mediated probing
# Agent emits "probe_request" event, orchestrator routes to target agent
# Target agent responds via "probe_response" event

# Option 3: Direct MCP communication (if agents have MCP servers)
```

**Probe Loop (in lock-wait.sh or Python equivalent):**
```python
async def wait_for_lock_with_probing(
    filepath: str,
    agent_id: str,
    my_locks: set[str],
    timeout: float = 300,
    probe_interval: float = 5
) -> bool:
    """Wait for lock with periodic deadlock probing."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if try_lock(filepath, agent_id):
            return True  # Got the lock

        # Probe the holder
        holder = get_lock_holder(filepath)
        if holder:
            holder_waiting_for = await probe_agent(holder, "what_waiting_for")
            if holder_waiting_for and holder_waiting_for in my_locks:
                raise DeadlockDetected(filepath, holder, holder_waiting_for)

        await asyncio.sleep(probe_interval)

    return False  # Timeout
```

**Pros:**
- Proactive detection (every 5s vs 5min timeout)
- No central state in orchestrator
- Agent-to-agent communication = horizontally scalable

**Cons:**
- Complex message passing infrastructure needed
- Race conditions: probe response may be stale
- Each waiting agent polls → O(n²) probes with n waiting agents
- Must handle agent crashes during probe

**Estimated Complexity:** High (~300 LOC new code + message infrastructure)

---

### Approach D: Lock Ordering + Prevention (Preventive)

**Overview:** Prevent deadlocks entirely by requiring locks to be acquired in a deterministic order.

**Concrete Implementation:**

```
Rule: Always acquire locks in sorted order by canonical path hash.

Example:
  hash(/repo/src/a.py) = 0x1234
  hash(/repo/src/b.py) = 0x5678
  hash(/repo/tests/c.py) = 0x2345

  Sorted order: a.py < c.py < b.py

  Agent needs: [b.py, a.py, c.py]
  Must acquire: a.py → c.py → b.py (sorted order)

  If agent already holds b.py and now needs a.py:
  - VIOLATION: a.py < b.py but we hold b.py
  - Must release b.py, acquire a.py, then reacquire b.py
```

**New Components:**
```python
# src/infra/tools/locking.py
def lock_order_key(filepath: str) -> int:
    """Return sort key for lock ordering."""
    canonical = _canonicalize_path(filepath)
    return int(hashlib.sha256(canonical.encode()).hexdigest()[:8], 16)

class OrderedLockManager:
    """Lock manager that enforces acquisition order."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.held_locks: list[tuple[int, str]] = []  # sorted by key

    def acquire(self, filepath: str) -> bool:
        key = lock_order_key(filepath)

        # Check ordering violation
        if self.held_locks and key < self.held_locks[-1][0]:
            raise LockOrderingViolation(
                f"Cannot acquire {filepath} (key={key}) while holding "
                f"{self.held_locks[-1][1]} (key={self.held_locks[-1][0]})"
            )

        if try_lock(filepath, self.agent_id):
            self.held_locks.append((key, filepath))
            return True
        return False

    def acquire_all(self, filepaths: list[str]) -> bool:
        """Acquire all locks in correct order."""
        sorted_paths = sorted(filepaths, key=lock_order_key)
        for fp in sorted_paths:
            if not self.acquire(fp):
                # Rollback: release what we got
                self.release_all()
                return False
        return True
```

**Agent Prompt Changes:**
```markdown
### Lock Ordering Protocol (MANDATORY)

Deadlock prevention via ordered acquisition:

1. Before editing, list ALL files you will modify
2. Sort files by lock order: `lock-order.sh file1 file2 file3`
3. Acquire locks in the printed order (NOT your preferred order)
4. If you discover you need another file mid-task:
   - Check its order vs your current locks
   - If order violation: release ALL locks, re-plan, start over
```

**Pros:**
- PREVENTS deadlocks entirely (mathematical guarantee)
- No detection logic needed
- No runtime overhead for cycle detection
- Proven technique (used in databases, OS kernels)

**Cons:**
- Must release and reacquire if order violated → work loss risk
- Requires agents to know ALL files upfront (may not be possible)
- Changes agent behavior significantly
- Doesn't work well with "discover files as you go" workflow

**Estimated Complexity:** Medium (~150 LOC) but high agent behavior change

---

## Recommendation: Approach A (Wait-For Graph)

**Why Wait-For Graph is the best fit:**

| Requirement | A: Graph | B: Timeout | C: Probe | D: Ordering |
|-------------|----------|------------|----------|-------------|
| Real-time detection | ✅ Immediate | ❌ 5+ min | ⚠️ 5s delay | ✅ Prevention |
| Auto-resolution | ✅ Central control | ⚠️ Race conditions | ⚠️ Race conditions | N/A |
| Add dependency | ✅ Orchestrator has context | ⚠️ Agent-side | ⚠️ Agent-side | N/A |
| No agent behavior change | ✅ Hooks only | ✅ | ⚠️ Probe responses | ❌ Major change |
| Handles complex cycles | ✅ A→B→C→A | ⚠️ Only A↔B | ⚠️ Chain depth limit | ✅ |
| Implementation complexity | Medium | Low | High | Medium |

**The Wait-For Graph approach:**
1. Meets real-time requirement (cycle detected on contention, not timeout)
2. Keeps resolution logic in orchestrator (clean victim selection, dependency addition)
3. Minimal agent changes (just emit events from existing hooks)
4. Handles multi-party deadlocks correctly
5. Well-understood algorithm with formal correctness guarantees

**Trade-off accepted:** Requires orchestrator state management, but this is reasonable since orchestrator already tracks `active_tasks`, `agent_ids`, etc.

## Technical Design

### Architecture

**Selected Approach:** Wait-For Graph (Centralized) with hook-based event emission.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR                                    │
│  ┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐   │
│  │ DeadlockMonitor │◄────│ WaitForGraph    │     │ IssueCoordinator │   │
│  │                 │     │                 │     │                  │   │
│  │ - on_lock_event │     │ - holds: dict   │     │ - active_tasks   │   │
│  │ - check_cycle() │     │ - waits: dict   │     │ - agent_ids      │   │
│  │ - resolve()     │     │ - detect_cycle()│     │ - start_times    │   │
│  └────────┬────────┘     └─────────────────┘     └────────┬─────────┘   │
│           │                                                │             │
│           │  on deadlock detected                          │             │
│           ├────────────────────────────────────────────────┤             │
│           │  1. select_victim(youngest)                    │             │
│           │  2. cancel task                                │             │
│           │  3. cleanup_agent_locks()                      │             │
│           │  4. add_dependency(victim → blocker)           │             │
│           │  5. mark_needs_followup()                      │             │
│           ▼                                                ▼             │
└──────────────────────────────────────────────────────────────────────────┘
                           ▲
                           │ LockEvent (acquired/waiting/released)
                           │
┌──────────────────────────┴───────────────────────────────────────────────┐
│                           AGENT PROCESS                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ PostToolUse Hook (lock event emission based on exit code)           │ │
│  │                                                                      │ │
│  │ on Bash(lock-try.sh file) completed:                                │ │
│  │   exit_code = get_exit_code(tool_result)                            │ │
│  │   if exit_code == 0:                                                │ │
│  │     emit LockEvent(type=ACQUIRED, agent_id, file)                   │ │
│  │   else:                                                             │ │
│  │     emit LockEvent(type=WAITING, agent_id, file)                    │ │
│  │                                                                      │ │
│  │ on Bash(lock-release.sh file):                                      │ │
│  │   emit LockEvent(type=RELEASED, agent_id, file)                     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

**Event Flow:**
1. Agent calls `lock-try.sh file.py` via Bash tool
2. PostToolUse hook intercepts result:
   - Success (exit 0) → emit `LockEvent(ACQUIRED, agent_id, file.py)`
   - Failure (exit 1) → emit `LockEvent(WAITING, agent_id, file.py)`
3. Orchestrator's `DeadlockMonitor` receives event, updates `WaitForGraph`
4. After each WAITING event, run `detect_cycle()`
5. If cycle found → `resolve_deadlock()` with youngest agent as victim

### Data Model

```python
# src/domain/deadlock.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable

class LockEventType(Enum):
    ACQUIRED = "acquired"
    WAITING = "waiting"
    RELEASED = "released"

@dataclass
class LockEvent:
    """Event emitted by hooks when lock state changes."""
    event_type: LockEventType
    agent_id: str
    lock_path: str  # Canonical absolute path
    timestamp: float = field(default_factory=time.monotonic)

@dataclass
class WaitForGraph:
    """Directed graph for deadlock detection.

    Edges:
    - agent --holds--> lock (tracked in `holds`)
    - agent --waits_for--> lock (tracked in `waits_for`)
    - lock --held_by--> agent (tracked in `held_by` for O(1) lookup)
    """
    holds: dict[str, set[str]] = field(default_factory=dict)
    # agent_id -> set of lock paths held

    waits_for: dict[str, str | None] = field(default_factory=dict)
    # agent_id -> lock path waiting for (None if not waiting)

    held_by: dict[str, str] = field(default_factory=dict)
    # lock_path -> agent_id (inverse of holds for O(1) lookup)

    def add_hold(self, agent_id: str, lock_path: str) -> None:
        """Record that agent acquired a lock."""
        if agent_id not in self.holds:
            self.holds[agent_id] = set()
        self.holds[agent_id].add(lock_path)
        self.held_by[lock_path] = agent_id
        # Clear wait status since they got the lock
        self.waits_for[agent_id] = None

    def add_wait(self, agent_id: str, lock_path: str) -> None:
        """Record that agent is waiting for a lock."""
        self.waits_for[agent_id] = lock_path

    def remove_hold(self, agent_id: str, lock_path: str) -> None:
        """Record that agent released a lock."""
        if agent_id in self.holds:
            self.holds[agent_id].discard(lock_path)
        if self.held_by.get(lock_path) == agent_id:
            del self.held_by[lock_path]

    def remove_agent(self, agent_id: str) -> None:
        """Remove all state for an agent (on termination)."""
        # Remove from held_by inverse mapping
        for lock_path in self.holds.get(agent_id, set()):
            self.held_by.pop(lock_path, None)
        self.holds.pop(agent_id, None)
        self.waits_for.pop(agent_id, None)

    def get_holder(self, lock_path: str) -> str | None:
        """Return agent_id holding the lock, or None. O(1) via inverse mapping."""
        return self.held_by.get(lock_path)

    def detect_cycle(self) -> list[str] | None:
        """Detect deadlock cycle using DFS.

        Returns list of agent_ids in cycle, or None if no deadlock.

        Algorithm:
        For each waiting agent, follow the wait chain:
        agent_A waits_for lock_X -> lock_X held_by agent_B -> agent_B waits_for lock_Y -> ...
        If we return to a visited agent, we have a cycle.
        """
        for start_agent, waiting_lock in self.waits_for.items():
            if waiting_lock is None:
                continue

            path = [start_agent]
            visited = {start_agent}
            current_lock = waiting_lock

            while current_lock:
                holder = self.get_holder(current_lock)
                if holder is None:
                    break  # Lock not held, no deadlock via this path

                if holder in visited:
                    # Found cycle - extract the cycle portion
                    cycle_start = path.index(holder)
                    return path[cycle_start:]

                path.append(holder)
                visited.add(holder)
                current_lock = self.waits_for.get(holder)

        return None

@dataclass
class DeadlockInfo:
    """Information about a detected deadlock."""
    cycle: list[str]  # Agent IDs in the cycle
    victim_id: str    # Selected victim
    victim_issue_id: str
    blocked_on: str   # Lock path victim was waiting for
    blocker_id: str   # Agent holding that lock
    blocker_issue_id: str
```

### API/Interface Design

```python
# src/domain/deadlock.py

DeadlockCallback = Callable[[DeadlockInfo], Awaitable[None]]

@dataclass
class DeadlockMonitor:
    """Monitors lock events and detects deadlocks."""

    graph: WaitForGraph = field(default_factory=WaitForGraph)
    agent_start_times: dict[str, float] = field(default_factory=dict)
    agent_to_issue: dict[str, str] = field(default_factory=dict)
    on_deadlock: DeadlockCallback | None = None

    def register_agent(self, agent_id: str, issue_id: str) -> None:
        """Register an agent when it starts."""
        self.agent_start_times[agent_id] = time.monotonic()
        self.agent_to_issue[agent_id] = issue_id

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent when it stops."""
        self.graph.remove_agent(agent_id)
        self.agent_start_times.pop(agent_id, None)
        self.agent_to_issue.pop(agent_id, None)

    async def handle_event(self, event: LockEvent) -> None:
        """Process a lock event and check for deadlocks."""
        if event.event_type == LockEventType.ACQUIRED:
            self.graph.add_hold(event.agent_id, event.lock_path)
        elif event.event_type == LockEventType.WAITING:
            self.graph.add_wait(event.agent_id, event.lock_path)
            await self._check_and_resolve_deadlock(event.agent_id)
        elif event.event_type == LockEventType.RELEASED:
            self.graph.remove_hold(event.agent_id, event.lock_path)

    async def _check_and_resolve_deadlock(self, triggering_agent: str) -> None:
        """Check for deadlock and resolve if found."""
        cycle = self.graph.detect_cycle()
        if cycle is None:
            return

        # Select victim: youngest agent in cycle (highest start time = most recent)
        victim_id = max(cycle, key=lambda a: self.agent_start_times.get(a, 0))
        victim_issue_id = self.agent_to_issue.get(victim_id, "unknown")

        blocked_on = self.graph.waits_for.get(victim_id, "unknown")
        blocker_id = self.graph.get_holder(blocked_on) if blocked_on else None
        blocker_issue_id = self.agent_to_issue.get(blocker_id, "unknown") if blocker_id else "unknown"

        info = DeadlockInfo(
            cycle=cycle,
            victim_id=victim_id,
            victim_issue_id=victim_issue_id,
            blocked_on=blocked_on or "unknown",
            blocker_id=blocker_id or "unknown",
            blocker_issue_id=blocker_issue_id,
        )

        if self.on_deadlock:
            await self.on_deadlock(info)


# src/infra/hooks/deadlock.py

def make_lock_event_hook(
    agent_id: str,
    emit_event: Callable[[LockEvent], None],
    get_held_locks: Callable[[str], set[str]],  # agent_id -> lock paths
) -> PostToolUseHook:
    """Create a PostToolUse hook that emits lock events."""

    async def emit_lock_events(
        hook_input: PostToolUseHookInput,
        stderr: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        tool_name = hook_input["tool_name"]
        tool_input = hook_input["tool_input"]
        tool_result = hook_input.get("tool_result", {})

        if tool_name.lower() != "bash":
            return {}

        command = tool_input.get("command", "")

        # Detect lock-try.sh
        if "lock-try.sh" in command:
            lock_path = _extract_lock_path(command)
            if lock_path:
                exit_code = _get_exit_code(tool_result)
                if exit_code == 0:
                    emit_event(LockEvent(LockEventType.ACQUIRED, agent_id, lock_path))
                else:
                    emit_event(LockEvent(LockEventType.WAITING, agent_id, lock_path))

        # Detect lock-release.sh
        elif "lock-release.sh" in command:
            lock_path = _extract_lock_path(command)
            if lock_path:
                emit_event(LockEvent(LockEventType.RELEASED, agent_id, lock_path))

        # Detect lock-release-all.sh
        elif "lock-release-all.sh" in command:
            # Release all locks held by this agent to avoid stale graph state
            # (agent may release all locks without terminating)
            for lock_path in list(get_held_locks(agent_id)):
                emit_event(LockEvent(LockEventType.RELEASED, agent_id, lock_path))

        return {}

    return emit_lock_events
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/domain/deadlock.py` | **New** | WaitForGraph, DeadlockMonitor, DeadlockInfo, LockEvent |
| `src/infra/hooks/deadlock.py` | **New** | make_lock_event_hook PostToolUse hook |
| `src/orchestration/orchestrator.py` | Modify | Add DeadlockMonitor, wire up event handling and resolution |
| `src/orchestration/orchestration_wiring.py` | Modify | Wire DeadlockMonitor into agent hooks |
| `src/pipeline/agent_session_runner.py` | Modify | Add PostToolUse hook for lock events |
| `src/infra/io/event_protocol.py` | Modify | Add on_deadlock_detected, on_deadlock_resolved events |
| `tests/domain/test_deadlock.py` | **New** | Unit tests for WaitForGraph cycle detection |
| `tests/integration/test_deadlock_integration.py` | **New** | Integration tests for deadlock resolution |

## Risks, Edge Cases & Breaking Changes

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Event lost (agent crash) | Medium | Low | Orchestrator cleanup removes agent from graph on task cancel |
| Race condition: detect after resolve | Low | Medium | Graph updates are synchronous, resolution locks graph |
| Victim selection unfair | Low | Low | Youngest heuristic is consistent; can add progress tracking later |
| Hook overhead | Low | Low | Event emission is O(1), cycle detection O(n) where n = agents |
| False positive | Very Low | Medium | Graph is precise; only real cycles detected |

### Edge Cases

- **Agent crashes mid-wait**: `_cleanup_agent_locks` already called; add `deadlock_monitor.unregister_agent()`
- **Lock released while cycle detected**: Cycle detection re-validates holder exists before declaring deadlock
- **Multi-party cycle (A→B→C→A)**: DFS handles arbitrary cycle lengths
- **Self-deadlock (agent waits on own lock)**: Detected as cycle of length 1; should not happen due to lock ownership check
- **Victim task already completed**: Check `task.done()` before cancel; if done, pick next youngest

### Breaking Changes

- **None expected**: All additions are new code paths
- Existing single-agent workflows unaffected (no cycle possible with 1 agent)
- Hook additions are opt-in via wiring

## Testing & Validation Strategy

### Unit Tests (tests/domain/test_deadlock.py)
- `WaitForGraph.detect_cycle()` with no agents → None
- `WaitForGraph.detect_cycle()` with single agent waiting → None (no cycle)
- `WaitForGraph.detect_cycle()` with A↔B cycle → returns [A, B]
- `WaitForGraph.detect_cycle()` with A→B→C→A cycle → returns [A, B, C]
- `WaitForGraph.add_hold`/`remove_hold` state management
- Victim selection: youngest of 3 agents

### Integration Tests (tests/integration/test_deadlock_integration.py)
- Mock two agents, simulate A acquires lock1, B acquires lock2
- A waits on lock2, B waits on lock1 → deadlock detected
- Verify victim (younger) is cancelled, dependency added
- Verify older agent continues successfully

### Manual Validation
- Run `mala` with 2 agents on issues that edit overlapping files
- Force deadlock by having both wait for each other's locks
- Observe log output: "Deadlock detected", "Victim: ...", "Added dependency: ..."

### Acceptance Criteria Coverage

| Criterion | Implementation |
|-----------|----------------|
| Detect classic A↔B deadlock | `WaitForGraph.detect_cycle()` DFS algorithm |
| Surface deadlock to user | `event_sink.on_deadlock_detected(info)` + log output |
| Provide resolution path | Kill youngest, release locks, add dependency |
| No false positives | Precise graph-based detection, only real cycles |

## Rollback Strategy

1. **Quick Disable**: Set `deadlock_detection_enabled: false` in orchestrator config
2. **Feature Flag**: Add `--disable-validations=deadlock-detection` CLI flag
3. **Full Revert**: Revert PR; no database/external state involved

### Rollback Safety
- No persistent state: graph lives in memory only
- No beads schema changes: uses existing `add_dependency` API
- Safe to disable mid-run: existing locks/agents continue normally

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Detection latency? | Real-time (on lock contention) |
| Auto or manual resolution? | Auto: kill victim, add dependency |
| Victim selection? | Youngest agent (most recently started) |
| Prevent or detect? | Detect via Wait-For Graph |
| Performance budget? | O(n) cycle detection per WAITING event; acceptable |

## Next Steps
After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
