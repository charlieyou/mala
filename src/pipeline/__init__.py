"""Pipeline stage modules for MalaOrchestrator decomposition.

This package contains extracted pipeline stages that were previously inline
in the orchestrator. Each module represents a stage with explicit inputs/outputs
and can be tested in isolation.

Modules:
    agent_session_runner: Agent session execution with SDK streaming
    epic_verification_coordinator: Epic closure verification with retry loop
    gate_runner: Quality gate checking with retry/fixer logic
    issue_execution_coordinator: Main loop coordination for issue processing
    run_coordinator: Global coordination and validation
"""

from src.pipeline.agent_session_runner import (
    AgentSessionConfig,
    AgentSessionInput,
    AgentSessionOutput,
    AgentSessionRunner,
)
from src.pipeline.epic_verification_coordinator import (
    EpicVerificationConfig,
    EpicVerificationCoordinator,
)
from src.pipeline.gate_runner import (
    GateRunner,
    PerSessionGateInput,
    PerSessionGateOutput,
)
from src.pipeline.issue_execution_coordinator import (
    CoordinatorConfig,
    IssueExecutionCoordinator,
)
from src.pipeline.issue_finalizer import (
    IssueFinalizeConfig,
    IssueFinalizeInput,
    IssueFinalizeOutput,
    IssueFinalizer,
)
from src.pipeline.run_coordinator import (
    RunCoordinator,
    SpecResultBuilder,
)
from src.pipeline.session_callback_factory import SessionCallbackFactory

__all__ = [
    "AgentSessionConfig",
    "AgentSessionInput",
    "AgentSessionOutput",
    "AgentSessionRunner",
    "CoordinatorConfig",
    "EpicVerificationConfig",
    "EpicVerificationCoordinator",
    "GateRunner",
    "IssueExecutionCoordinator",
    "IssueFinalizeConfig",
    "IssueFinalizeInput",
    "IssueFinalizeOutput",
    "IssueFinalizer",
    "PerSessionGateInput",
    "PerSessionGateOutput",
    "RunCoordinator",
    "SessionCallbackFactory",
    "SpecResultBuilder",
]
