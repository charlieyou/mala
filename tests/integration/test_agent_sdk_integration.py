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

from src.core.models import ReviewInput
from src.infra.clients.agent_sdk_review import AgentSDKReviewer
from src.infra.clients.cerberus_review import DefaultReviewer
from src.orchestration.factory import _create_code_reviewer, _get_reviewer_config
from src.pipeline.review_runner import ReviewRunner, ReviewRunnerConfig
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

    def test_factory_creates_cerberus_reviewer_with_yaml_config(
        self, tmp_path: Path
    ) -> None:
        """Verify factory uses cerberus settings from mala.yaml code_review.cerberus."""
        # Create mala.yaml with cerberus settings in code_review.cerberus
        (tmp_path / "mala.yaml").write_text(
            "preset: python-uv\n"
            "validation_triggers:\n"
            "  session_end:\n"
            "    failure_mode: continue\n"
            "    code_review:\n"
            "      enabled: true\n"
            "      reviewer_type: cerberus\n"
            "      cerberus:\n"
            "        spawn_args: ['--yaml-spawn']\n"
            "        wait_args: ['--yaml-wait', '--timeout', '600']\n"
            "        env:\n"
            "          YAML_VAR: yaml_value\n"
        )

        # Load reviewer config via factory path (validates mala.yaml parsing)
        reviewer_config = _get_reviewer_config(tmp_path)
        assert reviewer_config.reviewer_type == "cerberus"
        assert reviewer_config.cerberus_config is not None
        assert reviewer_config.cerberus_config.spawn_args == ("--yaml-spawn",)

        # Create MalaConfig mock with DIFFERENT values (should be ignored)
        mala_config = MagicMock()
        mala_config.cerberus_bin_path = Path("/usr/bin")
        mala_config.cerberus_spawn_args = ("--env-spawn",)
        mala_config.cerberus_wait_args = ("--env-wait",)
        mala_config.cerberus_env = {"ENV_VAR": "env_value"}

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
        # Verify settings come from YAML config, not env vars
        assert reviewer.repo_path == tmp_path
        assert reviewer.bin_path == Path("/usr/bin")  # bin_path still from mala_config
        assert reviewer.spawn_args == ("--yaml-spawn",)
        assert reviewer.wait_args == ("--yaml-wait", "--timeout", "600")
        assert reviewer.env == {"YAML_VAR": "yaml_value"}

    def test_factory_creates_cerberus_reviewer_fallback_to_env_vars(
        self, tmp_path: Path
    ) -> None:
        """Verify factory falls back to mala_config when no cerberus config in yaml."""
        # Create mala.yaml WITHOUT cerberus section (backward compat)
        (tmp_path / "mala.yaml").write_text(
            "preset: python-uv\n"
            "validation_triggers:\n"
            "  session_end:\n"
            "    failure_mode: continue\n"
            "    code_review:\n"
            "      enabled: true\n"
            "      reviewer_type: cerberus\n"
        )

        # Load reviewer config via factory path (validates mala.yaml parsing)
        reviewer_config = _get_reviewer_config(tmp_path)
        assert reviewer_config.reviewer_type == "cerberus"
        assert reviewer_config.cerberus_config is None  # No cerberus section in yaml

        # Create minimal MalaConfig mock with cerberus settings
        # cerberus_bin_path is a directory containing review-gate binary
        mala_config = MagicMock()
        mala_config.cerberus_bin_path = Path("/usr/bin")
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
        # Verify settings were passed through from env vars
        assert reviewer.repo_path == tmp_path
        assert reviewer.bin_path == Path("/usr/bin")
        assert reviewer.spawn_args == ("--spawn",)
        assert reviewer.wait_args == ("--wait",)
        assert reviewer.env == {"CERBERUS_MODE": "test"}

    def test_factory_injects_cerberus_timeout_into_wait_args(
        self, tmp_path: Path
    ) -> None:
        """Verify factory injects cerberus.timeout into wait_args when not specified."""
        # Create mala.yaml with cerberus.timeout but no --timeout in wait_args
        (tmp_path / "mala.yaml").write_text(
            "preset: python-uv\n"
            "validation_triggers:\n"
            "  session_end:\n"
            "    failure_mode: continue\n"
            "    code_review:\n"
            "      enabled: true\n"
            "      reviewer_type: cerberus\n"
            "      cerberus:\n"
            "        timeout: 600\n"
            "        wait_args: ['--other-flag']\n"
        )

        reviewer_config = _get_reviewer_config(tmp_path)
        assert reviewer_config.cerberus_config is not None
        assert reviewer_config.cerberus_config.timeout == 600

        mala_config = MagicMock()
        mala_config.cerberus_bin_path = Path("/usr/bin")

        event_sink = FakeEventSink()

        reviewer = _create_code_reviewer(
            repo_path=tmp_path,
            mala_config=mala_config,
            event_sink=event_sink,
            reviewer_config=reviewer_config,
        )

        assert isinstance(reviewer, DefaultReviewer)
        # timeout should be injected at the front of wait_args
        assert reviewer.wait_args == ("--timeout", "600", "--other-flag")

    def test_factory_does_not_duplicate_timeout_in_wait_args(
        self, tmp_path: Path
    ) -> None:
        """Verify factory does not inject timeout when already in wait_args."""
        # Create mala.yaml with both cerberus.timeout and --timeout in wait_args
        (tmp_path / "mala.yaml").write_text(
            "preset: python-uv\n"
            "validation_triggers:\n"
            "  session_end:\n"
            "    failure_mode: continue\n"
            "    code_review:\n"
            "      enabled: true\n"
            "      reviewer_type: cerberus\n"
            "      cerberus:\n"
            "        timeout: 600\n"
            "        wait_args: ['--timeout', '120']\n"
        )

        reviewer_config = _get_reviewer_config(tmp_path)
        mala_config = MagicMock()
        mala_config.cerberus_bin_path = Path("/usr/bin")
        event_sink = FakeEventSink()

        reviewer = _create_code_reviewer(
            repo_path=tmp_path,
            mala_config=mala_config,
            event_sink=event_sink,
            reviewer_config=reviewer_config,
        )

        assert isinstance(reviewer, DefaultReviewer)
        # Should NOT duplicate --timeout; explicit wait_args value wins
        assert reviewer.wait_args == ("--timeout", "120")


class TestFactoryCreatesAgentSDKReviewerWhenConfigured:
    """Test that factory creates AgentSDKReviewer when reviewer_type=agent_sdk."""

    def test_factory_creates_agent_sdk_reviewer_when_configured(
        self, tmp_path: Path
    ) -> None:
        """Verify factory creates AgentSDKReviewer with custom config."""
        # Create mala.yaml with agent_sdk config including custom timeout/model
        (tmp_path / "mala.yaml").write_text(
            "preset: python-uv\n"
            "validation_triggers:\n"
            "  session_end:\n"
            "    failure_mode: continue\n"
            "    code_review:\n"
            "      enabled: true\n"
            "      reviewer_type: agent_sdk\n"
            "      agent_sdk_timeout: 900\n"
            "      agent_sdk_model: opus\n"
        )

        # Load reviewer config via factory path (validates mala.yaml parsing)
        reviewer_config = _get_reviewer_config(tmp_path)
        assert reviewer_config.reviewer_type == "agent_sdk"
        assert reviewer_config.agent_sdk_review_timeout == 900
        assert reviewer_config.agent_sdk_reviewer_model == "opus"

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
            # Verify custom settings were passed through
            assert reviewer.repo_path == tmp_path
            assert reviewer.default_timeout == 900  # Custom timeout
            assert reviewer.model == "opus"  # Custom model


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

        # Patch SDKClientFactory and prompts loading
        with (
            patch(
                "src.infra.sdk_adapter.SDKClientFactory",
                return_value=fake_sdk_factory,
            ),
            patch("src.domain.prompts.load_prompts") as mock_load_prompts,
        ):
            mock_prompts = MagicMock()
            mock_prompts.review_agent_prompt = "Review the code changes"
            mock_load_prompts.return_value = mock_prompts

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
                issue_description="Test issue for integration",
                commit_shas=["abc123"],
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
            # Verify query included the commit list
            query_text = client.queries[0][0]
            assert "abc123" in query_text
