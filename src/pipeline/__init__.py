"""Pipeline stage modules for MalaOrchestrator decomposition.

This package contains extracted pipeline stages that were previously inline
in the orchestrator. Each module represents a stage with explicit inputs/outputs
and can be tested in isolation.

Modules:
    agent_session_runner: Agent session execution with SDK streaming
    gate_runner: Quality gate checking with retry/fixer logic
    run_coordinator: Run-level coordination and validation
"""

from src.pipeline.agent_session_runner import (
    AgentSessionConfig,
    AgentSessionInput,
    AgentSessionOutput,
    AgentSessionRunner,
    SDKClientFactory,
    SDKClientProtocol,
    SessionCallbacks,
)
from src.pipeline.gate_runner import (
    GateRunner,
    GateRunnerConfig,
    PerIssueGateInput,
    PerIssueGateOutput,
)
from src.pipeline.run_coordinator import (
    RunCoordinator,
    RunCoordinatorConfig,
    RunLevelValidationInput,
    RunLevelValidationOutput,
    SpecResultBuilder,
)

__all__ = [
    "AgentSessionConfig",
    "AgentSessionInput",
    "AgentSessionOutput",
    "AgentSessionRunner",
    "GateRunner",
    "GateRunnerConfig",
    "PerIssueGateInput",
    "PerIssueGateOutput",
    "RunCoordinator",
    "RunCoordinatorConfig",
    "RunLevelValidationInput",
    "RunLevelValidationOutput",
    "SDKClientFactory",
    "SDKClientProtocol",
    "SessionCallbacks",
    "SpecResultBuilder",
]
