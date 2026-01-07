"""Unit tests for orchestration factory functions.

Tests for:
- _check_review_availability: Review availability checking by reviewer_type
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.infra.io.config import MalaConfig
from src.orchestration.factory import _check_review_availability


class TestCheckReviewAvailability:
    """Tests for _check_review_availability function."""

    @pytest.fixture
    def mala_config(self) -> MalaConfig:
        """Create a minimal MalaConfig for testing."""
        return MalaConfig(
            runs_dir=Path("/tmp/runs"),
            lock_dir=Path("/tmp/locks"),
        )

    def test_agent_sdk_reviewer_is_available(self, mala_config: MalaConfig) -> None:
        """agent_sdk reviewer should always be available (no external deps)."""
        result = _check_review_availability(
            mala_config=mala_config,
            disabled_validations=set(),
            reviewer_type="agent_sdk",
        )
        assert result is None

    def test_explicitly_disabled_review_returns_none(
        self, mala_config: MalaConfig
    ) -> None:
        """Explicitly disabled review should return None (no warning needed)."""
        result = _check_review_availability(
            mala_config=mala_config,
            disabled_validations={"review"},
            reviewer_type="agent_sdk",
        )
        assert result is None

    def test_unknown_reviewer_type_returns_reason(
        self, mala_config: MalaConfig
    ) -> None:
        """Unknown reviewer_type should disable review with warning."""
        result = _check_review_availability(
            mala_config=mala_config,
            disabled_validations=set(),
            reviewer_type="unknown_type",
        )
        assert result is not None
        assert "unknown reviewer_type" in result
        assert "unknown_type" in result

    def test_cerberus_without_binary_returns_reason(
        self, mala_config: MalaConfig
    ) -> None:
        """cerberus reviewer without binary should disable review."""
        # Patch shutil.which to return None (no binary found)
        with patch("shutil.which", return_value=None):
            result = _check_review_availability(
                mala_config=mala_config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result is not None
        assert "review-gate" in result

    def test_cerberus_with_binary_is_available(self, mala_config: MalaConfig) -> None:
        """cerberus reviewer with binary available should return None."""
        # Patch shutil.which to return a path (binary found)
        with patch("shutil.which", return_value="/usr/bin/review-gate"):
            result = _check_review_availability(
                mala_config=mala_config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result is None

    def test_cerberus_with_explicit_bin_path_existing(self) -> None:
        """cerberus reviewer with explicit bin_path to existing binary is available."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir)
            # Create the review-gate binary
            review_gate = bin_path / "review-gate"
            review_gate.touch()
            review_gate.chmod(0o755)

            config = MalaConfig(
                runs_dir=Path("/tmp/runs"),
                lock_dir=Path("/tmp/locks"),
                cerberus_bin_path=bin_path,
            )
            result = _check_review_availability(
                mala_config=config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result is None

    def test_cerberus_with_explicit_bin_path_missing_binary(self) -> None:
        """cerberus reviewer with explicit bin_path but missing binary is disabled."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir)
            # Do NOT create the review-gate binary

            config = MalaConfig(
                runs_dir=Path("/tmp/runs"),
                lock_dir=Path("/tmp/locks"),
                cerberus_bin_path=bin_path,
            )
            result = _check_review_availability(
                mala_config=config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result is not None
        assert "review-gate" in result
