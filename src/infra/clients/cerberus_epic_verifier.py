"""Cerberus epic verification adapter for mala orchestrator.

This module provides CerberusEpicVerifier implementing the EpicVerificationModel
protocol using the spawn-wait pattern with the review-gate CLI.

It handles:
- CLI orchestration (spawn-epic-review + wait)
- Context JSON file creation
- JSON response parsing to EpicVerdict
- Error classification (timeout, execution, parse)

Subprocess management is delegated to CerberusGateCLI patterns.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.models import EpicVerdict, UnmetCriterion
from src.infra.tools.command_runner import CommandRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Schema version for context JSON format
CONTEXT_SCHEMA_VERSION = "1"


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

    The verifier spawns `spawn-epic-review --context-json <path>` followed by
    `wait`, then parses the JSON response to EpicVerdict.

    Attributes:
        repo_path: Working directory for subprocess execution.
        bin_path: Optional path to directory containing review-gate binary.
            If None, uses bare "review-gate" from PATH.
        timeout: Timeout in seconds for the verification subprocess.
        env: Additional environment variables to merge with os.environ.
    """

    repo_path: Path
    bin_path: Path | None = None
    timeout: int = 300
    env: dict[str, str] | None = None

    def _review_gate_bin(self) -> str:
        """Get the path to the review-gate binary."""
        if self.bin_path is not None:
            return str(self.bin_path / "review-gate")
        return "review-gate"

    def _write_context_json(
        self,
        epic_id: str,
        acceptance_criteria: str,
        spec_content: str | None,
        commit_range: str,
        commit_list: Sequence[str],
    ) -> Path:
        """Write context JSON to a temporary file.

        Args:
            epic_id: The epic identifier.
            acceptance_criteria: The epic's acceptance criteria text.
            spec_content: Optional content of linked spec file.
            commit_range: Commit range hint covering child issue commits.
            commit_list: List of commit SHAs to inspect.

        Returns:
            Path to the temporary context JSON file.
        """
        context = {
            "epic_id": epic_id,
            "acceptance_criteria": acceptance_criteria,
            "spec_content": spec_content or "",
            "commit_range": commit_range,
            "commit_list": list(commit_list),
            "schema_version": CONTEXT_SCHEMA_VERSION,
        }

        # Create temp file in repo directory for locality
        fd, path = tempfile.mkstemp(
            suffix=".json", prefix=f"epic-verify-{epic_id}-", dir=self.repo_path
        )
        try:
            # Use os.fdopen to convert fd to file object (fd is int, not path)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(context, f, indent=2)
        except Exception:
            # Clean up on error
            Path(path).unlink(missing_ok=True)
            raise

        return Path(path)

    def _parse_verdict_json(self, stdout: str) -> EpicVerdict:
        """Parse JSON response into EpicVerdict.

        Args:
            stdout: Raw JSON output from the wait command.

        Returns:
            Parsed EpicVerdict.

        Raises:
            VerificationParseError: If JSON is invalid or missing required fields.
        """
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            # Include first 500 bytes for debugging per R4 spec
            preview = stdout[:500] if stdout else "(empty)"
            raise VerificationParseError(
                f"Invalid JSON response: {e}. Output preview: {preview}"
            ) from e

        if not isinstance(data, dict):
            preview = stdout[:500] if stdout else "(empty)"
            raise VerificationParseError(
                f"Expected JSON object, got {type(data).__name__}. Output preview: {preview}"
            )

        # Extract required fields
        try:
            passed = data["passed"]
            if not isinstance(passed, bool):
                raise VerificationParseError(
                    f"'passed' must be boolean, got {type(passed).__name__}"
                )

            confidence = data.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)):
                raise VerificationParseError(
                    f"'confidence' must be number, got {type(confidence).__name__}"
                )

            reasoning = data.get("reasoning", "")
            if not isinstance(reasoning, str):
                raise VerificationParseError(
                    f"'reasoning' must be string, got {type(reasoning).__name__}"
                )

            unmet_criteria_raw = data.get("unmet_criteria", [])
            if not isinstance(unmet_criteria_raw, list):
                raise VerificationParseError(
                    f"'unmet_criteria' must be array, got {type(unmet_criteria_raw).__name__}"
                )

            unmet_criteria: list[UnmetCriterion] = []
            for i, item in enumerate(unmet_criteria_raw):
                if not isinstance(item, dict):
                    raise VerificationParseError(
                        f"unmet_criteria[{i}] must be object, got {type(item).__name__}"
                    )
                unmet_criteria.append(
                    UnmetCriterion(
                        criterion=str(item.get("criterion", "")),
                        evidence=str(item.get("evidence", "")),
                        priority=int(item.get("priority", 1)),
                        criterion_hash=str(item.get("criterion_hash", "")),
                    )
                )

            return EpicVerdict(
                passed=passed,
                unmet_criteria=unmet_criteria,
                confidence=float(confidence),
                reasoning=reasoning,
            )

        except KeyError as e:
            raise VerificationParseError(f"Missing required field: {e}") from e
        except (ValueError, TypeError) as e:
            raise VerificationParseError(f"Invalid field value: {e}") from e

    async def verify(
        self,
        epic_criteria: str,
        commit_range: str,
        commit_list: str,
        spec_content: str | None,
        *,
        epic_id: str = "",
    ) -> EpicVerdict:
        """Verify if the commit scope satisfies the epic's acceptance criteria.

        Implements the EpicVerificationModel protocol using Cerberus spawn-wait
        pattern per R4 specification.

        Args:
            epic_criteria: The epic's acceptance criteria text.
            commit_range: Commit range hint covering child issue commits.
            commit_list: Comma-separated list of commit SHAs to inspect.
            spec_content: Optional content of linked spec file.
            epic_id: Epic identifier for context (optional).

        Returns:
            EpicVerdict with pass/fail and unmet criteria details.

        Raises:
            VerificationTimeoutError: If subprocess times out.
            VerificationExecutionError: If subprocess returns non-zero exit.
            VerificationParseError: If response JSON cannot be parsed.
        """
        # Parse commit list from comma-separated string
        commits = [c.strip() for c in commit_list.split(",") if c.strip()]

        # Write context JSON to temp file
        context_path = self._write_context_json(
            epic_id=epic_id,
            acceptance_criteria=epic_criteria,
            spec_content=spec_content,
            commit_range=commit_range,
            commit_list=commits,
        )

        try:
            runner = CommandRunner(cwd=self.repo_path)
            # Merge self.env with os.environ to preserve PATH and other required vars
            env = dict(os.environ)
            if self.env:
                env.update(self.env)

            # Step 1: Spawn epic review
            spawn_cmd = [
                self._review_gate_bin(),
                "spawn-epic-review",
                "--context-json",
                str(context_path),
            ]

            logger.info("Spawning epic review: %s", " ".join(spawn_cmd))
            spawn_result = await runner.run_async(
                spawn_cmd, env=env, timeout=self.timeout
            )

            if spawn_result.timed_out:
                raise VerificationTimeoutError(
                    f"spawn-epic-review timed out after {self.timeout}s"
                )

            if spawn_result.returncode != 0:
                stderr = spawn_result.stderr_tail()
                stdout = spawn_result.stdout_tail()
                detail = stderr or stdout or "spawn failed"
                raise VerificationExecutionError(
                    f"spawn-epic-review failed (exit {spawn_result.returncode}): {detail}"
                )

            # Step 2: Wait for review completion
            wait_cmd = [
                self._review_gate_bin(),
                "wait",
                "--json",
                "--timeout",
                str(self.timeout),
            ]

            logger.info("Waiting for epic review: %s", " ".join(wait_cmd))
            # Use timeout + grace period for subprocess
            wait_result = await runner.run_async(
                wait_cmd, env=env, timeout=self.timeout + 30
            )

            if wait_result.timed_out:
                raise VerificationTimeoutError(
                    f"wait timed out after {self.timeout + 30}s"
                )

            if wait_result.returncode != 0:
                stderr = wait_result.stderr_tail()
                stdout = wait_result.stdout_tail()
                detail = stderr or stdout or "wait failed"
                raise VerificationExecutionError(
                    f"wait failed (exit {wait_result.returncode}): {detail}"
                )

            # Step 3: Parse JSON response
            return self._parse_verdict_json(wait_result.stdout)

        finally:
            # Clean up temp file
            context_path.unlink(missing_ok=True)
