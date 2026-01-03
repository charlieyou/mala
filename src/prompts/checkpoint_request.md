# Checkpoint Request

Context usage is high. Before we run out of space, output a checkpoint so work can continue in a fresh session.

Output a `<checkpoint>` block with these sections:

```
<checkpoint>
## Goal
<One sentence: what is this task trying to accomplish?>

## Completed Work
<Bulleted list of what's done. Include file:line references.>

## Remaining Tasks
<Numbered list of what's left, in priority order.>

## Do Not Redo
<Files already modified, tests already passing, locks already held.>

## Key Decisions
<Important choices made during implementation that should be preserved.>
</checkpoint>
```

After outputting this checkpoint, STOP.
