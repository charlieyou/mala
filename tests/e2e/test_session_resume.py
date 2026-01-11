"""E2E test for session resumption using ClaudeAgentOptions(resume=...).

This test verifies that mala correctly uses the SDK's session resume mechanism
when retrying after idle timeouts or review failures. The SDK's resume feature
loads prior conversation context, allowing Claude to continue where it left off.

The bug (fixed in SDK 2.0.22): session_id on query() only tags outgoing messages
for multiplexing concurrent conversations. To actually RESUME a session with
prior context, you must pass resume=session_id in ClaudeAgentOptions.

This test validates:
1. First query establishes a session with a unique "secret"
2. Disconnect client, create NEW client with resume=session_id
3. Second query without secret context - model should recall the secret
4. If model recalls secret: resume worked (context was loaded)
5. If model cannot recall: resume failed (started fresh session)

See GitHub issue #5012 for SDK bug details.
"""

import uuid
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from tests.e2e.claude_auth import has_valid_oauth_credentials, is_claude_cli_available

pytestmark = [pytest.mark.e2e, pytest.mark.flaky_sdk]


@pytest.fixture(autouse=True)
def require_claude_cli_auth() -> None:
    """Skip tests if Claude Code CLI is not available or OAuth credentials missing."""
    if not is_claude_cli_available():
        pytest.skip("Claude Code CLI not installed")
    if not has_valid_oauth_credentials():
        pytest.skip(
            "Claude Code CLI not logged in or token expired - run `claude` and login"
        )


@pytest.fixture(autouse=True)
def clean_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean environment for tests - use CLI auth."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


async def _collect_session_id(client: ClaudeSDKClient) -> str | None:
    session_id: str | None = None
    response: object = client.receive_response()
    try:
        async for message in response:
            if hasattr(message, "session_id"):
                session_id = getattr(message, "session_id")
    finally:
        aclose: Any = getattr(response, "aclose", None)
        if aclose is not None:
            await aclose()
        await _close_query_streams(client)
    return session_id


async def _collect_response_text(client: ClaudeSDKClient) -> str:
    parts: list[str] = []
    response: object = client.receive_response()
    try:
        async for message in response:
            if hasattr(message, "content"):
                for block in getattr(message, "content", []):
                    if hasattr(block, "text"):
                        parts.append(getattr(block, "text", ""))
    finally:
        aclose: Any = getattr(response, "aclose", None)
        if aclose is not None:
            await aclose()
        await _close_query_streams(client)
    return "".join(parts)


async def _close_query_streams(client: ClaudeSDKClient) -> None:
    query = getattr(client, "_query", None)
    if query is None:
        return
    for name in ("_message_receive", "_message_send"):
        stream = getattr(query, name, None)
        if stream is None:
            continue
        aclose: Any = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()


class TestSessionResume:
    """Test that session resume works correctly using ClaudeAgentOptions(resume=...)."""

    @pytest.mark.asyncio
    async def test_resume_preserves_conversation_context(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Session resume should preserve conversation context from prior session.

        This test demonstrates the session resume mechanism:
        1. Establish a session and tell Claude a unique secret
        2. Disconnect and create a NEW client with resume=session_id
        3. Ask Claude to recall the secret WITHOUT providing it again
        4. If resume works: Claude recalls the secret (prior context loaded)
        5. If resume fails: Claude cannot recall (fresh session with no context)

        The test FAILS if resume isn't working because Claude won't know the secret.
        """
        import tempfile

        work_dir = tempfile.mkdtemp()
        secret = f"SECRET_{uuid.uuid4().hex[:8]}"

        # Session 1: Tell Claude the secret and get session_id
        options1 = ClaudeAgentOptions(
            cwd=work_dir,
            permission_mode="bypassPermissions",
            model="haiku",
            max_turns=1,
        )

        session_id: str | None = None
        async with ClaudeSDKClient(options=options1) as client:
            await client.query(
                f"Remember this secret exactly: {secret}. "
                "Say only 'Acknowledged' - nothing else."
            )
            session_id = await _collect_session_id(client)

        assert session_id is not None, "Expected session_id from first query"

        # Session 2: Resume with new client and ask for the secret
        # THIS IS THE KEY - we pass resume=session_id to ClaudeAgentOptions
        options2 = ClaudeAgentOptions(
            cwd=work_dir,
            permission_mode="bypassPermissions",
            model="haiku",
            max_turns=1,
            resume=session_id,  # <-- This is the correct way to resume
        )

        response_text = ""
        async with ClaudeSDKClient(options=options2) as client:
            await client.query(
                "What was the secret I just told you? "
                "Reply with ONLY the secret value, nothing else."
            )
            response_text = await _collect_response_text(client)

        # If resume worked, Claude should know the secret
        assert secret in response_text, (
            f"Session resume failed - Claude did not recall secret.\n"
            f"Expected '{secret}' in response but got: '{response_text}'\n"
            f"This indicates resume=session_id was not used correctly."
        )

    @pytest.mark.asyncio
    async def test_session_id_on_query_does_not_resume(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Passing session_id to query() does NOT resume prior context.

        This test demonstrates the BUG: passing session_id to client.query()
        only tags messages for multiplexing, it does NOT load prior context.

        The test shows that Claude CANNOT recall the secret when we use the
        incorrect session_id=... on query() instead of resume=... on options.
        """
        import tempfile

        work_dir = tempfile.mkdtemp()
        secret = f"SECRET_{uuid.uuid4().hex[:8]}"

        # Session 1: Tell Claude the secret
        options1 = ClaudeAgentOptions(
            cwd=work_dir,
            permission_mode="bypassPermissions",
            model="haiku",
            max_turns=1,
        )

        session_id: str | None = None
        async with ClaudeSDKClient(options=options1) as client:
            await client.query(
                f"Remember this secret exactly: {secret}. "
                "Say only 'Acknowledged' - nothing else."
            )
            session_id = await _collect_session_id(client)

        assert session_id is not None, "Expected session_id from first query"

        # Session 2: Use session_id on query() - THIS IS THE BUG
        # This creates a NEW client WITHOUT resume, then passes session_id to query
        options2 = ClaudeAgentOptions(
            cwd=work_dir,
            permission_mode="bypassPermissions",
            model="haiku",
            max_turns=1,
            # NOTE: No resume= parameter - this is the bug!
        )

        response_text = ""
        async with ClaudeSDKClient(options=options2) as client:
            # This is how mala currently does it (incorrectly)
            await client.query(
                "What was the secret I just told you? "
                "Reply with ONLY the secret value, nothing else.",
                session_id=session_id,  # <-- Bug: this doesn't resume context
            )
            response_text = await _collect_response_text(client)

        # Without resume, Claude should NOT know the secret
        # (This test passes if the bug exists - Claude can't recall)
        assert secret not in response_text, (
            "Unexpected: Claude recalled secret without resume.\n"
            "This suggests session_id on query() now works for resume, "
            "which would mean the SDK behavior changed."
        )
