"""RuntimeDeps: Named dependency assembly for orchestrator construction.

Replaces the positional dependency tuple previously returned by
`_build_dependencies` in `factory.py` with an explicit frozen dataclass so
callers consume named attributes rather than tuple indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.protocols.agent_provider import AgentProvider
    from src.core.protocols.evidence import EvidenceProvider
    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.infra import (
        CommandRunnerPort,
        EnvConfigPort,
        LockManagerPort,
    )
    from src.core.protocols.issue import IssueProvider
    from src.core.protocols.review import CodeReviewer
    from src.core.protocols.validation import EpicVerifierProtocol, GateChecker
    from src.infra.io.config import MalaConfig
    from src.infra.telemetry import TelemetryProvider


@dataclass(frozen=True)
class RuntimeDeps:
    """Runtime dependencies assembled by the orchestrator factory.

    Bundles all protocol implementations and service objects produced by
    `_build_dependencies`. Pipeline wiring and the orchestrator both consume
    this named structure instead of a positional tuple.

    Attributes:
        evidence_check: GateChecker for quality gate validation.
        code_reviewer: CodeReviewer for post-commit code reviews.
        beads: IssueProvider for issue tracking operations.
        event_sink: MalaEventSink for run lifecycle logging.
        agent_provider: AgentProvider for the chosen coder backend; threaded
            into AgentSessionRunner and FixerService so all SDK-style calls
            go through one selection point.
        command_runner: CommandRunnerPort for executing shell commands.
        env_config: EnvConfigPort for environment configuration.
        lock_manager: LockManagerPort for file locking coordination.
        mala_config: MalaConfig for orchestrator configuration.
        evidence_provider: EvidenceProvider for session log access.
        telemetry_provider: TelemetryProvider for tracing.
        epic_verifier: Optional EpicVerifierProtocol; None when the chosen
            verifier backend is unavailable.
    """

    evidence_check: GateChecker
    code_reviewer: CodeReviewer
    beads: IssueProvider
    event_sink: MalaEventSink
    agent_provider: AgentProvider
    command_runner: CommandRunnerPort
    env_config: EnvConfigPort
    lock_manager: LockManagerPort
    mala_config: MalaConfig
    evidence_provider: EvidenceProvider
    telemetry_provider: TelemetryProvider
    epic_verifier: EpicVerifierProtocol | None = None
