"""Cerberus epic verification adapter for mala orchestrator.

This module provides CerberusEpicVerifier implementing the EpicVerificationModel
protocol using the review-gate CLI spawn/wait flow.

It handles:
- CLI orchestration (spawn-epic-verify + wait)
- Epic file creation for review-gate input
- JSON response parsing to EpicVerdict
- Error classification (timeout, execution, parse)

Subprocess management is delegated to CerberusGateCLI patterns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from src.core.models import EpicVerdict, UnmetCriterion
from src.infra.clients.cerberus_gate_cli import CerberusGateCLI
from src.infra.tools.command_runner import CommandRunner

logger = logging.getLogger(__name__)


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
    """Epic verifier implementation using Cerberus review-gate CLI.

    This class conforms to the EpicVerificationModel protocol and provides
    epic verification using the Cerberus CLI spawn-wait pattern.

    The verifier spawns `spawn-epic-verify <epic-file> [diff args]` followed by
    `wait`, then parses the JSON response to EpicVerdict.

    Attributes:
        repo_path: Working directory for subprocess execution.
        bin_path: Optional path to directory containing review-gate binary.
            If None, uses bare "review-gate" from PATH.
        timeout: Timeout in seconds for the verification subprocess.
        spawn_args: Additional arguments for spawn-epic-verify.
        wait_args: Additional arguments for wait.
        env: Additional environment variables to merge with os.environ.
    """

    repo_path: Path
    bin_path: Path | None = None
    timeout: int = 300
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

    def _parse_wait_output(self, stdout: str) -> EpicVerdict:
        """Parse review-gate wait JSON into EpicVerdict.

        Args:
            stdout: Raw JSON output from the wait command.

        Returns:
            Parsed EpicVerdict.

        Raises:
            VerificationParseError: If JSON is invalid or missing required fields.
            VerificationExecutionError: If review-gate returns error status.
            VerificationTimeoutError: If review-gate reports a timeout status.
        """
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            preview = stdout[:500] if stdout else "(empty)"
            raise VerificationParseError(
                f"Invalid JSON response: {e}. Output preview: {preview}"
            ) from e

        if not isinstance(data, dict):
            preview = stdout[:500] if stdout else "(empty)"
            raise VerificationParseError(
                f"Expected JSON object, got {type(data).__name__}. Output preview: {preview}"
            )

        status = data.get("status", "")
        consensus = data.get("consensus_verdict", "")
        parse_errors = data.get("parse_errors", [])

        if status == "timeout":
            raise VerificationTimeoutError("wait returned timeout")

        if status in ("error", "no_reviewers"):
            detail = "review-gate wait returned error status"
            if isinstance(parse_errors, list) and parse_errors:
                detail = f"{detail}: {parse_errors[:3]}"
            raise VerificationExecutionError(detail)

        if not consensus:
            raise VerificationExecutionError(
                "review-gate wait returned no consensus verdict"
            )

        if consensus == "ERROR":
            detail = "review-gate consensus ERROR"
            if isinstance(parse_errors, list) and parse_errors:
                detail = f"{detail}: {parse_errors[:3]}"
            raise VerificationExecutionError(detail)

        findings = data.get("aggregated_findings", [])
        if not isinstance(findings, list):
            raise VerificationParseError(
                f"'aggregated_findings' must be array, got {type(findings).__name__}"
            )

        unmet_criteria: list[UnmetCriterion] = []
        for i, item in enumerate(findings):
            if not isinstance(item, dict):
                raise VerificationParseError(
                    f"aggregated_findings[{i}] must be object, got {type(item).__name__}"
                )
            title = str(item.get("title", "")).strip()
            body = str(item.get("body", "")).strip()
            criterion = title or body or "Unspecified criterion"
            priority_val = item.get("priority", 1)
            try:
                priority = int(priority_val)
            except (TypeError, ValueError):
                priority = 1
            priority = max(0, min(3, priority))
            unmet_criteria.append(
                UnmetCriterion(
                    criterion=criterion,
                    evidence=body,
                    priority=priority,
                    criterion_hash=self._compute_criterion_hash(criterion),
                )
            )

        passed = consensus == "PASS"
        reasoning = f"Cerberus epic verification consensus {consensus}."
        if isinstance(parse_errors, list) and parse_errors:
            reasoning = f"{reasoning} Parse errors present for some reviewers."

        return EpicVerdict(
            passed=passed,
            unmet_criteria=unmet_criteria,
            reasoning=reasoning,
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
            cli = CerberusGateCLI(
                repo_path=self.repo_path,
                bin_path=self.bin_path,
                env=self.env or {},
            )

            validation_error = cli.validate_binary()
            if validation_error is not None:
                raise VerificationExecutionError(validation_error)

            runner = CommandRunner(cwd=self.repo_path)
            session_id = self._generate_session_id()
            env = cli.build_env(claude_session_id=session_id)

            # Step 1: Spawn epic verification
            logger.info("Spawning epic verification via review-gate")
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
                if spawn_result.already_active:
                    resolve_result = await cli.resolve_gate(runner, env)
                    if resolve_result.success:
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
                    else:
                        raise VerificationExecutionError(
                            "spawn failed: "
                            f"{spawn_result.error_detail} "
                            f"(auto-resolve failed: {resolve_result.error_detail})"
                        )
                else:
                    raise VerificationExecutionError(
                        f"spawn failed: {spawn_result.error_detail}"
                    )

            # Step 2: Wait for review completion
            user_timeout = CerberusGateCLI.extract_wait_timeout(self.wait_args)
            wait_result = await cli.wait_for_review(
                session_id=session_id,
                runner=runner,
                env=env,
                cli_timeout=self.timeout,
                wait_args=self.wait_args,
                user_timeout=user_timeout,
            )

            if wait_result.timed_out or wait_result.returncode == 3:
                raise VerificationTimeoutError(f"wait timed out after {self.timeout}s")

            # Step 3: Parse JSON response
            return self._parse_wait_output(wait_result.stdout)

        finally:
            epic_path.unlink(missing_ok=True)
