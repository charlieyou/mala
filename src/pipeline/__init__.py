"""Pipeline stage modules for MalaOrchestrator decomposition.

This package contains extracted pipeline stages that were previously inline
in the orchestrator. Each module represents a stage with explicit inputs/outputs
and can be tested in isolation.

Modules:
    gate_runner: Quality gate checking with retry/fixer logic
"""

from src.pipeline.gate_runner import (
    GateRunner,
    GateRunnerConfig,
    PerIssueGateInput,
    PerIssueGateOutput,
)

__all__ = [
    "GateRunner",
    "GateRunnerConfig",
    "PerIssueGateInput",
    "PerIssueGateOutput",
]
