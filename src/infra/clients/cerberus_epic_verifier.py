"""Cerberus epic verification adapter for mala orchestrator.

This module provides CerberusEpicVerifier implementing the EpicVerificationModel
protocol using the Cerberus CLI spawn/wait flow.

It handles:
- CLI orchestration (spawn-epic-verify + wait)
- Epic file creation for Cerberus input
- JSON response parsing to EpicVerdict
- Error classification (timeout, execution, parse)

Subprocess management is delegated to CerberusCLI.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.constants import DEFAULT_CERBERUS_REVIEW_TIMEOUT_SECONDS
from src.core.models import EpicVerdict, UnmetCriterion
from src.infra.clients.cerberus_cli import CerberusCLI
from src.infra.clients.cerberus_iteration_findings import read_findings
from src.infra.clients.cerberus_output_parser import parse_gate_state
from src.infra.tools.command_runner import CommandRunner

if TYPE_CHECKING:
    from src.infra.clients.cerberus_output_parser import GateState, ReviewIssue

logger = logging.getLogger(__name__)

PASSING_VERDICTS = {"pass", "needs_work"}


class VerificationTimeoutError(Exception):
    """Raised when epic verification times out.

    This error is retryable per R6 specification.
    """


class VerificationExecutionError(Exception):
    """Raised when subprocess returns non-zero exit code.

    This error is retryable per R6 specification.
    """


class VerificationParseError(Exception):
    """Raised when JSON response cannot be parsed.

    This error is retryable per R6 specification.
    """


@dataclass
class CerberusEpicVerifier:
    """Epic verifier implementation using the Cerberus CLI.

    This class conforms to the EpicVerificationModel protocol and provides
    epic verification using the Cerberus CLI spawn-wait pattern.

    The verifier spawns `spawn-epic-verify <epic-file> [diff args]` followed by
    `wait`, then parses the JSON response to EpicVerdict.

    Attributes:
        repo_path: Working directory for subprocess execution.
        state_root: Optional Cerberus state root. Defaults to .mala/cerberus.
        project_key: Optional Cerberus project key. User env may override it.
        timeout: Timeout in seconds for the verification subprocess.
        spawn_args: Additional arguments for spawn-epic-verify.
        wait_args: Additional arguments for wait.
        env: Additional environment variables to merge with os.environ.
    """

    repo_path: Path
    state_root: Path | None = None
    project_key: str | None = None
    timeout: int = DEFAULT_CERBERUS_REVIEW_TIMEOUT_SECONDS
    spawn_args: tuple[str, ...] = ()
    wait_args: tuple[str, ...] = ()
    env: dict[str, str] | None = None

    @staticmethod
    def _compute_criterion_hash(criterion: str) -> str:
        """Compute SHA256 hash of criterion text for deduplication."""
        return hashlib.sha256(criterion.encode()).hexdigest()

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a CLAUDE_SESSION_ID for epic verification."""
        return f"epic-{secrets.token_hex(6)}"

    def _write_epic_file(self, epic_text: str) -> Path:
        """Write epic content to a temporary markdown file.

        Args:
            epic_text: Epic content text (contains acceptance criteria).

        Returns:
            Path to the temporary epic markdown file.
        """
        fd, path = tempfile.mkstemp(
            suffix=".md", prefix="epic-verify-", dir=self.repo_path
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(epic_text)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
        return Path(path)

    def _map_findings_to_unmet_criteria(
        self, findings: list[ReviewIssue]
    ) -> list[UnmetCriterion]:
        """Map Cerberus reviewer findings to epic unmet criteria."""
        unmet_criteria: list[UnmetCriterion] = []
        for finding in findings:
            title = finding.title.strip()
            body = finding.body.strip()
            criterion = title or body or finding.file or "Unspecified criterion"
            priority = 1 if finding.priority is None else finding.priority
            priority = max(0, min(3, priority))
            unmet_criteria.append(
                UnmetCriterion(
                    criterion=criterion,
                    evidence=body,
                    priority=priority,
                    criterion_hash=self._compute_criterion_hash(criterion),
                )
            )
        return unmet_criteria

    def _requires_decision_criterion(self, gate_state: GateState) -> UnmetCriterion:
        """Create an unmet criterion for a Cerberus requires_decision verdict."""
        criterion = "Cerberus reviewers reached no consensus"
        evidence = "Gate verdict=requires_decision. Human decision or re-run required."
        if gate_state.resolution_reason:
            evidence = f"{evidence} Resolution reason: {gate_state.resolution_reason}"
        return UnmetCriterion(
            criterion=criterion,
            evidence=evidence,
            priority=1,
            criterion_hash=self._compute_criterion_hash(criterion),
        )

    def _parse_wait_output(
        self,
        stdout: str,
        *,
        returncode: int,
        stderr: str,
        state_root: Path,
        project_key: str,
        run_key: str,
    ) -> EpicVerdict:
        """Parse Cerberus v2 gate-state and iteration artifacts into EpicVerdict."""
        if returncode != 0:
            detail = stderr.strip() or f"cerberus wait failed with exit {returncode}"
            raise VerificationExecutionError(detail)

        try:
            gate_state = parse_gate_state(stdout)
        except ValueError as e:
            preview = stdout[:500] if stdout else "(empty)"
            raise VerificationParseError(
                f"Invalid Cerberus gate-state: {e}. Output preview: {preview}"
            ) from e

        findings, parse_errors = read_findings(
            state_root=state_root,
            project_key=project_key,
            run_key=run_key,
        )
        if parse_errors:
            raise VerificationParseError(
                "Could not read Cerberus iteration findings: "
                + "; ".join(parse_errors[:3])
            )

        unmet_criteria = self._map_findings_to_unmet_criteria(findings)
        if gate_state.verdict == "requires_decision":
            unmet_criteria.append(self._requires_decision_criterion(gate_state))
        has_blocking_findings = any(
            criterion.priority <= 1 for criterion in unmet_criteria
        )

        return EpicVerdict(
            passed=gate_state.verdict in PASSING_VERDICTS and not has_blocking_findings,
            unmet_criteria=unmet_criteria,
            reasoning=f"Cerberus epic verification verdict {gate_state.verdict}.",
        )

    async def verify(
        self,
        epic_context: str,
    ) -> EpicVerdict:
        """Verify if the commit scope satisfies the epic's acceptance criteria.

        Implements the EpicVerificationModel protocol using Cerberus spawn-wait
        pattern per R4 specification.

        Args:
            epic_context: The epic context text containing acceptance criteria.

        Returns:
            EpicVerdict with pass/fail and unmet criteria details.

        Raises:
            VerificationTimeoutError: If subprocess times out.
            VerificationExecutionError: If subprocess returns non-zero exit.
            VerificationParseError: If response JSON cannot be parsed.
        """
        epic_path = self._write_epic_file(epic_context)

        try:
            cli = CerberusCLI(
                repo_path=self.repo_path,
                env=self.env or {},
            )

            validation_error = cli.validate_binary()
            if validation_error is not None:
                raise VerificationExecutionError(validation_error)

            runner = CommandRunner(cwd=self.repo_path)
            session_id = self._generate_session_id()
            run_key = f"mala-{session_id}"
            state_root = self.state_root or (self.repo_path / ".mala" / "cerberus")
            env = cli.build_env(
                run_key=run_key,
                state_root=state_root,
                project_key=self.project_key,
                claude_session_id=session_id,
            )
            project_key = env["CERBERUS_PROJECT_KEY"]

            # Step 1: Spawn epic verification
            logger.info("Spawning epic verification via cerberus")
            spawn_result = await cli.spawn_epic_verify(
                runner=runner,
                env=env,
                timeout=self.timeout,
                epic_path=epic_path,
                spawn_args=self.spawn_args,
            )

            if spawn_result.timed_out:
                raise VerificationTimeoutError(
                    f"spawn-epic-verify timed out after {self.timeout}s"
                )

            if not spawn_result.success:
                raise VerificationExecutionError(
                    f"spawn failed: {spawn_result.error_detail}"
                )

            # Step 2: Wait for review completion
            user_timeout = CerberusCLI.extract_wait_timeout(self.wait_args)
            wait_result = await cli.wait_for_review(
                run_key=run_key,
                runner=runner,
                env=env,
                cli_timeout=self.timeout,
                wait_args=self.wait_args,
                user_timeout=user_timeout,
            )

            if wait_result.timed_out or wait_result.returncode == 3:
                raise VerificationTimeoutError(f"wait timed out after {self.timeout}s")

            # Step 3: Parse JSON response
            return self._parse_wait_output(
                wait_result.stdout,
                returncode=wait_result.returncode,
                stderr=wait_result.stderr,
                state_root=state_root,
                project_key=project_key,
                run_key=run_key,
            )

        finally:
            epic_path.unlink(missing_ok=True)
