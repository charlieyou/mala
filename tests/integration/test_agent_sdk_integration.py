"""Integration tests for factory wiring and ReviewRunner with AgentSDKReviewer.

These tests verify end-to-end wiring: factory creates correct reviewer type,
ReviewRunner can call it. Tests traverse the full factory → reviewer creation path.

Tests use real factory functions but mock SDK client (no network calls).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.infra.clients.agent_sdk_review import AgentSDKReviewer
from src.infra.clients.cerberus_review import DefaultReviewer
from src.orchestration.factory import _create_code_reviewer, _get_reviewer_config
from src.pipeline.review_runner import ReviewInput, ReviewRunner, ReviewRunnerConfig
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.sdk_client import FakeSDKClientFactory


def _make_review_json(
    verdict: str = "PASS",
    issues: list[dict[str, Any]] | None = None,
) -> str:
    """Create a valid review JSON response."""
    return json.dumps(
        {
            "consensus_verdict": verdict,
            "aggregated_findings": issues or [],
        }
    )


class TestFactoryCreatesAgentSDKReviewerByDefault:
    """Test that factory creates AgentSDKReviewer when no config specified."""

    def test_factory_creates_agent_sdk_reviewer_by_default(
        self, tmp_path: Path
    ) -> None:
        """Verify factory creates AgentSDKReviewer when no config present."""
        # Create minimal repo structure (no mala.yaml)
        (tmp_path / ".git").mkdir()

        # Get reviewer config (should return defaults since no mala.yaml)
        reviewer_config = _get_reviewer_config(tmp_path)

        # Verify default reviewer_type is agent_sdk
        assert reviewer_config.reviewer_type == "agent_sdk"

        # Create minimal MalaConfig mock
        mala_config = MagicMock()
        mala_config.cerberus_bin_path = None
        mala_config.cerberus_spawn_args = ()
        mala_config.cerberus_wait_args = ()
        mala_config.cerberus_env = {}

        event_sink = FakeEventSink()

        # Patch SDKClientFactory and prompts loading to avoid external dependencies
        with (
            patch(
                "src.infra.sdk_adapter.SDKClientFactory",
                return_value=FakeSDKClientFactory(),
            ),
            patch("src.domain.prompts.load_prompts") as mock_load_prompts,
        ):
            mock_prompts = MagicMock()
            mock_prompts.review_agent_prompt = "Review the code"
            mock_load_prompts.return_value = mock_prompts

            # Create reviewer via factory path
            reviewer = _create_code_reviewer(
                repo_path=tmp_path,
                mala_config=mala_config,
                event_sink=event_sink,
                reviewer_config=reviewer_config,
            )

            # Verify correct type was created
            assert isinstance(reviewer, AgentSDKReviewer)


class TestFactoryCreatesCerberusReviewerWhenConfigured:
    """Test that factory creates DefaultReviewer when reviewer_type=cerberus."""

    def test_factory_creates_cerberus_reviewer_when_configured(
        self, tmp_path: Path
    ) -> None:
        """Verify factory creates DefaultReviewer with reviewer_type=cerberus."""
        from src.orchestration.factory import _ReviewerConfig

        # Create reviewer config with cerberus type
        reviewer_config = _ReviewerConfig(reviewer_type="cerberus")

        # Create minimal MalaConfig mock with cerberus settings
        mala_config = MagicMock()
        mala_config.cerberus_bin_path = Path("/usr/bin/review-gate")
        mala_config.cerberus_spawn_args = ("--spawn",)
        mala_config.cerberus_wait_args = ("--wait",)
        mala_config.cerberus_env = {"CERBERUS_MODE": "test"}

        event_sink = FakeEventSink()

        # Create reviewer via factory path
        reviewer = _create_code_reviewer(
            repo_path=tmp_path,
            mala_config=mala_config,
            event_sink=event_sink,
            reviewer_config=reviewer_config,
        )

        # Verify correct type was created
        assert isinstance(reviewer, DefaultReviewer)
        # Verify settings were passed through
        assert reviewer.repo_path == tmp_path
        assert reviewer.bin_path == Path("/usr/bin/review-gate")
        assert reviewer.spawn_args == ("--spawn",)
        assert reviewer.wait_args == ("--wait",)
        assert reviewer.env == {"CERBERUS_MODE": "test"}


class TestFactoryCreatesAgentSDKReviewerWhenConfigured:
    """Test that factory creates AgentSDKReviewer when reviewer_type=agent_sdk."""

    def test_factory_creates_agent_sdk_reviewer_when_configured(
        self, tmp_path: Path
    ) -> None:
        """Verify factory creates AgentSDKReviewer with reviewer_type=agent_sdk."""
        from src.orchestration.factory import _ReviewerConfig

        # Create reviewer config with explicit agent_sdk type
        reviewer_config = _ReviewerConfig(
            reviewer_type="agent_sdk",
            agent_sdk_review_timeout=900,
            agent_sdk_reviewer_model="opus",
        )

        # Create minimal MalaConfig mock
        mala_config = MagicMock()
        mala_config.cerberus_bin_path = None
        mala_config.cerberus_spawn_args = ()
        mala_config.cerberus_wait_args = ()
        mala_config.cerberus_env = {}

        event_sink = FakeEventSink()

        # Patch SDKClientFactory and prompts loading
        with (
            patch(
                "src.infra.sdk_adapter.SDKClientFactory",
                return_value=FakeSDKClientFactory(),
            ),
            patch("src.domain.prompts.load_prompts") as mock_load_prompts,
        ):
            mock_prompts = MagicMock()
            mock_prompts.review_agent_prompt = "Custom review prompt"
            mock_load_prompts.return_value = mock_prompts

            # Create reviewer via factory path
            reviewer = _create_code_reviewer(
                repo_path=tmp_path,
                mala_config=mala_config,
                event_sink=event_sink,
                reviewer_config=reviewer_config,
            )

            # Verify correct type was created
            assert isinstance(reviewer, AgentSDKReviewer)
            # Verify settings were passed through
            assert reviewer.repo_path == tmp_path
            assert reviewer.default_timeout == 900
            assert reviewer.model == "opus"


class TestReviewRunnerIntegration:
    """Test ReviewRunner → AgentSDKReviewer → ReviewResult integration."""

    @pytest.mark.asyncio
    async def test_review_runner_integration(self, tmp_path: Path) -> None:
        """Verify ReviewRunner → AgentSDKReviewer → ReviewResult flow with mocked SDK."""
        from src.orchestration.factory import _ReviewerConfig

        # Set up a git repo to satisfy git operations
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        # Create reviewer config with agent_sdk type
        reviewer_config = _ReviewerConfig(
            reviewer_type="agent_sdk",
            agent_sdk_review_timeout=60,
            agent_sdk_reviewer_model="sonnet",
        )

        # Create minimal MalaConfig mock
        mala_config = MagicMock()
        mala_config.cerberus_bin_path = None
        mala_config.cerberus_spawn_args = ()
        mala_config.cerberus_wait_args = ()
        mala_config.cerberus_env = {}

        event_sink = FakeEventSink()

        # Create FakeSDKClientFactory with configured response
        fake_sdk_factory = FakeSDKClientFactory()
        fake_sdk_factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": _make_review_json("PASS")}],
                )
            ]
        )

        # Patch SDKClientFactory, prompts loading, and CommandRunner
        with (
            patch(
                "src.infra.sdk_adapter.SDKClientFactory",
                return_value=fake_sdk_factory,
            ),
            patch("src.domain.prompts.load_prompts") as mock_load_prompts,
            patch(
                "src.infra.clients.agent_sdk_review.CommandRunner"
            ) as mock_runner_class,
        ):
            mock_prompts = MagicMock()
            mock_prompts.review_agent_prompt = "Review the code changes"
            mock_load_prompts.return_value = mock_prompts

            # Mock git diff --stat to return non-empty (changes exist)
            async def run_async_mock(*args: object, **kwargs: object) -> MagicMock:
                return MagicMock(
                    returncode=0,
                    stdout=" 1 file changed, 10 insertions(+)",
                    stderr="",
                )

            mock_runner = MagicMock()
            mock_runner.run_async = run_async_mock
            mock_runner_class.return_value = mock_runner

            # Create reviewer via factory path
            reviewer = _create_code_reviewer(
                repo_path=tmp_path,
                mala_config=mala_config,
                event_sink=event_sink,
                reviewer_config=reviewer_config,
            )

            # Verify it's the right type
            assert isinstance(reviewer, AgentSDKReviewer)

            # Create ReviewRunner with the factory-created reviewer
            runner = ReviewRunner(
                code_reviewer=reviewer,
                config=ReviewRunnerConfig(
                    max_review_retries=3,
                    review_timeout=60,
                ),
            )

            # Create review input
            review_input = ReviewInput(
                issue_id="test-issue-1",
                repo_path=tmp_path,
                commit_sha="abc123",
                issue_description="Test issue for integration",
                baseline_commit="def456",
                claude_session_id="integration-test-session",
            )

            # Run the review through the full pipeline
            output = await runner.run_review(review_input)

            # Verify the result
            assert output.result.passed is True
            assert output.result.issues == []
            assert output.result.parse_error is None
            assert output.result.fatal_error is False

            # Verify SDK client was created and called
            assert len(fake_sdk_factory.clients) == 1
            client = fake_sdk_factory.clients[0]
            assert len(client.queries) == 1
            # Verify query included the diff range
            query_text = client.queries[0][0]
            assert "def456..abc123" in query_text
