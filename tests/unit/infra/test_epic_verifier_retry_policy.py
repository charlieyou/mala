"""Unit tests for EpicVerifier per-category retry policy (R6).

Tests the retry behavior in EpicVerifier._verify_with_category_retries() including:
1. Per-category retry limits (timeout, execution, parse)
2. Safe handling of invalid retry policy values (P1 regression)
3. asyncio.TimeoutError handling for Agent SDK (P1 regression)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infra.clients.cerberus_epic_verifier import (
    VerificationExecutionError,
    VerificationParseError,
    VerificationTimeoutError,
)
from src.infra.epic_verifier import EpicVerifier


@dataclass
class MockRetryPolicy:
    """Mock retry policy for testing."""

    timeout_retries: int = 3
    execution_retries: int = 2
    parse_retries: int = 1


class TestRetryPolicyValidation:
    """Tests for retry policy value validation (P1 fix)."""

    @pytest.mark.asyncio
    async def test_negative_retry_values_clamped_to_zero(self) -> None:
        """Negative retry values should be clamped to 0 (no retries)."""
        # Create mock model that always times out
        mock_model = MagicMock()
        mock_model.verify = AsyncMock(side_effect=VerificationTimeoutError("timeout"))

        # Policy with negative values - should be treated as 0
        policy = MockRetryPolicy(
            timeout_retries=-1, execution_retries=-5, parse_retries=-10
        )

        verifier = EpicVerifier(
            beads=MagicMock(),
            model=mock_model,
            repo_path=Path("/tmp"),
            command_runner=MagicMock(),
            retry_policy=policy,  # type: ignore[arg-type]
        )

        # Should fail immediately (0 retries, not infinite loop)
        verdict = await verifier._verify_with_category_retries("epic context")

        assert not verdict.passed
        assert "timeout" in verdict.reasoning.lower()
        # Should only call verify once (initial attempt, no retries)
        assert mock_model.verify.call_count == 1

    @pytest.mark.asyncio
    async def test_non_integer_retry_values_use_defaults(self) -> None:
        """Non-integer retry values should fall back to defaults."""

        @dataclass
        class BadPolicy:
            timeout_retries: str = "not an int"  # type: ignore[assignment]
            execution_retries: float = 2.5  # type: ignore[assignment]
            parse_retries: list[object] | None = None

            def __post_init__(self) -> None:
                self.parse_retries = []

        mock_model = MagicMock()
        call_count = 0

        async def verify_side_effect(*args: object, **kwargs: object) -> NoReturn:
            nonlocal call_count
            call_count += 1
            raise VerificationTimeoutError("timeout")

        mock_model.verify = AsyncMock(side_effect=verify_side_effect)

        verifier = EpicVerifier(
            beads=MagicMock(),
            model=mock_model,
            repo_path=Path("/tmp"),
            command_runner=MagicMock(),
            retry_policy=BadPolicy(),  # type: ignore[arg-type]
        )

        verdict = await verifier._verify_with_category_retries("epic context")

        assert not verdict.passed
        # With default timeout_retries=3, we get 1 initial + 3 retries = 4 calls
        assert call_count == 4


class TestAsyncioTimeoutHandling:
    """Tests for asyncio.TimeoutError handling (P1 fix for Agent SDK)."""

    @pytest.mark.asyncio
    async def test_asyncio_timeout_uses_timeout_retries(self) -> None:
        """asyncio.TimeoutError should use timeout_retries limit."""
        mock_model = MagicMock()
        call_count = 0

        async def verify_side_effect(*args: object, **kwargs: object) -> NoReturn:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("agent sdk timeout")

        mock_model.verify = AsyncMock(side_effect=verify_side_effect)

        policy = MockRetryPolicy(
            timeout_retries=2, execution_retries=0, parse_retries=0
        )

        verifier = EpicVerifier(
            beads=MagicMock(),
            model=mock_model,
            repo_path=Path("/tmp"),
            command_runner=MagicMock(),
            retry_policy=policy,  # type: ignore[arg-type]
        )

        verdict = await verifier._verify_with_category_retries("epic context")

        assert not verdict.passed
        # 1 initial + 2 retries = 3 attempts
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_asyncio_timeout_eventually_succeeds(self) -> None:
        """asyncio.TimeoutError should allow retries that eventually succeed."""
        mock_model = MagicMock()
        call_count = 0

        @dataclass
        class MockUnmetCriterion:
            criterion: str = "test"
            evidence: str = "evidence"
            priority: str = "P1"
            criterion_hash: str = "hash123"

        @dataclass
        class MockVerifyResult:
            passed: bool = True
            unmet_criteria: list[object] | None = None
            confidence: float = 0.95
            reasoning: str = "Success"

            def __post_init__(self) -> None:
                self.unmet_criteria = []

        async def verify_side_effect(
            *args: object, **kwargs: object
        ) -> MockVerifyResult:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("transient timeout")
            return MockVerifyResult()

        mock_model.verify = AsyncMock(side_effect=verify_side_effect)

        policy = MockRetryPolicy(timeout_retries=5)

        verifier = EpicVerifier(
            beads=MagicMock(),
            model=mock_model,
            repo_path=Path("/tmp"),
            command_runner=MagicMock(),
            retry_policy=policy,  # type: ignore[arg-type]
        )

        verdict = await verifier._verify_with_category_retries("epic context")

        assert verdict.passed
        assert call_count == 3


class TestPerCategoryRetries:
    """Tests for per-category retry behavior."""

    @pytest.mark.asyncio
    async def test_timeout_retries_respected(self) -> None:
        """Timeout errors should use timeout_retries limit."""
        mock_model = MagicMock()
        call_count = 0

        async def verify_side_effect(*args: object, **kwargs: object) -> NoReturn:
            nonlocal call_count
            call_count += 1
            raise VerificationTimeoutError("cerberus timeout")

        mock_model.verify = AsyncMock(side_effect=verify_side_effect)

        policy = MockRetryPolicy(
            timeout_retries=2, execution_retries=5, parse_retries=5
        )

        verifier = EpicVerifier(
            beads=MagicMock(),
            model=mock_model,
            repo_path=Path("/tmp"),
            command_runner=MagicMock(),
            retry_policy=policy,  # type: ignore[arg-type]
        )

        verdict = await verifier._verify_with_category_retries("epic context")

        assert not verdict.passed
        # 1 initial + 2 retries = 3 attempts
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execution_retries_respected(self) -> None:
        """Execution errors should use execution_retries limit."""
        mock_model = MagicMock()
        call_count = 0

        async def verify_side_effect(*args: object, **kwargs: object) -> NoReturn:
            nonlocal call_count
            call_count += 1
            raise VerificationExecutionError("execution failed")

        mock_model.verify = AsyncMock(side_effect=verify_side_effect)

        policy = MockRetryPolicy(
            timeout_retries=5, execution_retries=1, parse_retries=5
        )

        verifier = EpicVerifier(
            beads=MagicMock(),
            model=mock_model,
            repo_path=Path("/tmp"),
            command_runner=MagicMock(),
            retry_policy=policy,  # type: ignore[arg-type]
        )

        verdict = await verifier._verify_with_category_retries("epic context")

        assert not verdict.passed
        # 1 initial + 1 retry = 2 attempts
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_parse_retries_respected(self) -> None:
        """Parse errors should use parse_retries limit."""
        mock_model = MagicMock()
        call_count = 0

        async def verify_side_effect(*args: object, **kwargs: object) -> NoReturn:
            nonlocal call_count
            call_count += 1
            raise VerificationParseError("parse failed")

        mock_model.verify = AsyncMock(side_effect=verify_side_effect)

        policy = MockRetryPolicy(
            timeout_retries=5, execution_retries=5, parse_retries=0
        )

        verifier = EpicVerifier(
            beads=MagicMock(),
            model=mock_model,
            repo_path=Path("/tmp"),
            command_runner=MagicMock(),
            retry_policy=policy,  # type: ignore[arg-type]
        )

        verdict = await verifier._verify_with_category_retries("epic context")

        assert not verdict.passed
        # 1 initial + 0 retries = 1 attempt
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_unknown_exception_no_retry(self) -> None:
        """Unknown exceptions should not trigger retry."""
        mock_model = MagicMock()
        call_count = 0

        async def verify_side_effect(*args: object, **kwargs: object) -> NoReturn:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("unexpected error")

        mock_model.verify = AsyncMock(side_effect=verify_side_effect)

        policy = MockRetryPolicy(
            timeout_retries=5, execution_retries=5, parse_retries=5
        )

        verifier = EpicVerifier(
            beads=MagicMock(),
            model=mock_model,
            repo_path=Path("/tmp"),
            command_runner=MagicMock(),
            retry_policy=policy,  # type: ignore[arg-type]
        )

        verdict = await verifier._verify_with_category_retries("epic context")

        assert not verdict.passed
        assert "unexpected error" in verdict.reasoning
        # Should only call once - no retries for unknown errors
        assert call_count == 1
