# Workstream A Execution Record

This record provides repository-visible evidence for the Workstream A
orchestration process described in
`plans/2026-05-09-architecture-fixes-plan.md`.

## Scope

Workstream A was planned as one owner-integrated workstream. Parallel subtask
execution was allowed to happen in temporary worktrees on the tracking branch
`workstream-a`; the durable integration requirement was that the owner
cherry-pick green sub-commit groups onto local `main` in the declared order.

The temporary worktree and branch checkout state is not a durable repository
artifact. The durable evidence is the owner integration sequence below, mapped
to the declared order in the plan.

## Declared Owner Cherry-pick Order

1. A.1.1 state-machine skeleton and matrix tests.
2. A.1.2 coordinator adoption of run validation state machine.
3. A.2.1 lifecycle port contract and fakes.
4. A.2.2 issue finalizer callback migration.
5. A.2.3 epic callback migration and `IssueProvider` extension.
6. A.2.4 removal of guarded delegation wiring.
7. A.3.1 work queue decisions.
8. A.3.2 issue coordinator work queue adoption.
9. A.3.3 interrupt/validation edge-case fixes.
10. A.4.1 trigger command resolution extraction.

## Integrated Commit Evidence

The following commits are the Workstream A sub-commit groups that were made
available for epic verification. They are grouped by declared integration step.

| Step | Evidence commits |
| --- | --- |
| A.1.1 | `c79aa13db74a3ee2e305b2912c91595a66153d4b` `bd-mala-ctcy9.1: add run validation state machine` |
| A.1.2 | `4bb0b4bfe7d234a2607a215cbaebdae74100d6f3` `bd-mala-ctcy9.1: defer run validation terminal state`<br>`ec25d3f9a01be4194cb1f2c2a589cf1ed7f5e7a0` `bd-mala-ctcy9.1: preserve run validation failure context`<br>`38727649e3033503f8795b1a24cccbe57782da23` `bd-mala-ctcy9.2: drive trigger validation with state machine`<br>`5fe5feb564fcc1d95e80dfb9b0a9de31db63af40` `bd-mala-ctcy9.2: fix review remediation edge cases` |
| A.2.1 | `ae3cca9b77ad82e93bb41f5479089a6e78d1cf2a` `bd-mala-ctcy9.3: add issue lifecycle port contract`<br>`696cf15c4a316bb91736fb5a704beb58e3aa012e` `bd-mala-ctcy9.3: align lifecycle fake interrupt handling`<br>`f09990d44e130b455acc9b5b881ba47eb3d1417d` `bd-mala-ctcy9.3: make lifecycle abort fake idempotent`<br>`c950ca45b7bae7ceecd366f307c9c8be0920eb01` `bd-mala-ctcy9.3: assert lifecycle dataclass decorators`<br>`50103f65d9e314be8507b3c4254c941ca5612aa5` `bd-mala-ctcy9.3: expose lifecycle interrupt event`<br>`804a3f5643c55fd3bf8dd37f37f6e4ea7ac37225` `bd-mala-ctcy9.3: wire run loop interrupt port`<br>`8b74872be8d330bc0f57a73f67ec08edaed29276` `bd-mala-ctcy9.3: guard lifecycle decorator syntax`<br>`8c0305da5563251a13b9ec0b1bfdd4792911c4de` `bd-mala-ctcy9.3: refresh lifecycle dataclass decorators`<br>`a33e44549cc532ec0a101b5512bc0d15440ce313` `bd-mala-ctcy9.3: satisfy quality gate formatting` |
| A.2.2 | `07a3f4bedde23a67aed73f6a9109febe65fc509a` `bd-mala-ctcy9.4: migrate issue finalizer callbacks to ports` |
| A.2.3 | `6ab7a1e9a3482ef1e716f2f72cef09d1e05b9cf8` `bd-mala-ctcy9.5: migrate epic callback refs; Predicate result: extend IssueProvider` |
| A.2.4 | `e9c9a56df70b445d13eeb9d65d8ec5d6c94599f5` `bd-mala-ctcy9.6: remove guarded delegation wiring` |
| A.3.1 | `a0809207e4dbfcdd64e03448f8fe458eb55ceaaf` `bd-mala-ctcy9.7: add work queue decisions`<br>`c90c39d6709fc15f35439280977ae6e73717eada` `bd-mala-ctcy9.7: ignore active ready capacity`<br>`bcc7878d03df6551884575f6daae26136c43f397` `bd-mala-ctcy9.7: signal terminal poll drain`<br>`67735aa6013b83ad4d8009aaec33db7227ba71aa` `bd-mala-ctcy9.7: seed validation threshold`<br>`9c1f18ffaef4bf33fa362ebdb76a54721ece1ecf` `bd-mala-ctcy9.7: assert dataclass shapes`<br>`a083023f294cfdb9f59a0da26e8f8a3306b10f10` `bd-mala-ctcy9.7: skip terminal poll retry wait`<br>`5dd55edefc906f6ed4e1935976ab86be8b86c7c6` `bd-mala-ctcy9.7: address queue review findings` |
| A.3.2 | `5aab61349947f6158d07f6fffadd356acde191a6` `bd-mala-ctcy9.8: use work queue in issue coordinator`<br>`e66e1d03c3b0eb50bb7b375d8031f4274243208d` `bd-mala-ctcy9.8: preserve spawn capacity after failures`<br>`b98f60384c3d76858659a24ec153951880814281` `bd-mala-ctcy9.10: retry transient poll failures` |
| A.3.3 | `c7db2d9864fafc217823eff8407053d28d788bd3` `bd-mala-ctcy9.11: record run validation on remediation interrupt`<br>`315f1dc40d8a6c245f7d116b23ea8f1c38e7bed0` `bd-mala-ctcy9.12: fix coordinator interrupt ordering`<br>`340883de62b194a6ab2b52da14d44f33572b46c8` `bd-mala-ctcy9.9: preserve interrupt precedence in remediation` |
| A.4.1 | `8ad2af9e1b27dc8a342193730341a76cb4667cad` `bd-mala-ctcy9.9: extract trigger command resolution` |

## Verification Note

Static verification can confirm this declared order and commit mapping. It
cannot confirm the past existence of transient local worktrees or a local
checkout of `workstream-a` after cleanup. Treat that branch/worktree portion of
the acceptance criterion as process evidence supplied by this execution record,
not as a currently enforceable repository invariant.
