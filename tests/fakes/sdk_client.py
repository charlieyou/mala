"""Canonical FakeSDKClient and FakeSDKClientFactory for testing.

Provides in-memory implementations of SDKClientProtocol and SDKClientFactoryProtocol
that enable behavior-based testing without SDK/API dependencies.

Observable state:
- FakeSDKClient.queries: list of (prompt, session_id) tuples sent via query()
- FakeSDKClient.disconnect_called: whether disconnect() was invoked
- FakeSDKClientFactory.clients: list of clients created
- FakeSDKClientFactory.create_calls: list of options passed to create()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from src.core.protocols.sdk import SDKClientFactoryProtocol, SDKClientProtocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType


def _make_default_result_message() -> object:
    """Create a default ResultMessage for completing SDK response iteration.

    Imports claude_agent_sdk lazily to avoid import errors in environments
    where the SDK is not installed.
    """
    from claude_agent_sdk import ResultMessage

    return ResultMessage(
        subtype="result",
        duration_ms=0,
        duration_api_ms=0,
        is_error=False,
        num_turns=1,
        session_id="fake-session",
        result=None,
    )


# Sentinel to distinguish "no result message configured" from "explicitly set to None"
_NO_RESULT_MESSAGE = object()


@dataclass
class FakeSDKClient(SDKClientProtocol):
    """In-memory fake SDK client for testing.

    Allows tests to configure response messages and errors. Tracks all
    queries and disconnect calls for verification.

    By default, yields a minimal ResultMessage to complete iteration. Set
    result_message=None explicitly to suppress this (simulates end-of-stream
    without the expected ResultMessage).

    Attributes:
        messages: Messages to yield from receive_response() before result.
        result_message: Final message to yield (typically ResultMessage).
            Defaults to a minimal ResultMessage. Set to None to suppress.
        query_error: If set, raise this on query().
        queries: List of (prompt, session_id) tuples from query() calls.
        disconnect_called: Whether disconnect() was called.
        disconnect_delay: Optional delay in seconds before disconnect completes.
    """

    messages: list[Any] = field(default_factory=list)
    result_message: Any = field(default=_NO_RESULT_MESSAGE)
    query_error: Exception | None = None
    queries: list[tuple[str, str | None]] = field(default_factory=list)
    disconnect_called: bool = False
    disconnect_delay: float = 0

    # Aliases for backward compatibility with existing tests
    @property
    def _query_calls(self) -> list[tuple[str, str | None]]:
        """Alias for queries (backward compatibility)."""
        return self.queries

    @property
    def _disconnect_called(self) -> bool:
        """Alias for disconnect_called (backward compatibility)."""
        return self.disconnect_called

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        """Record query and optionally raise configured error."""
        self.queries.append((prompt, session_id))
        if self.query_error:
            raise self.query_error

    async def receive_response(self) -> AsyncIterator[Any]:
        """Yield configured messages, then result_message.

        If result_message was not configured, yields a default ResultMessage.
        If result_message was explicitly set to None, yields nothing after messages.
        """
        for msg in self.messages:
            yield msg
        if self.result_message is _NO_RESULT_MESSAGE:
            # Default: yield a minimal ResultMessage to complete iteration
            yield _make_default_result_message()
        elif self.result_message is not None:
            # Explicitly configured result message
            yield self.result_message
        # If result_message is None, end without ResultMessage (protocol edge case)

    async def disconnect(self) -> None:
        """Mark client as disconnected, optionally with delay."""
        if self.disconnect_delay > 0:
            await asyncio.sleep(self.disconnect_delay)
        self.disconnect_called = True


class FakeSDKClientFactory(SDKClientFactoryProtocol):
    """In-memory fake SDK client factory for testing.

    Can be initialized with a single client or a sequence of clients.
    Tracks all create() calls and options for verification.

    Attributes:
        client: Optional single client to always return (for simple test cases).
        clients: List of clients created (or pre-configured).
        create_calls: List of options passed to create().
        created_options: List of options dicts returned by create_options().
        created_matchers: List of (name, matcher, hooks) tuples from create_hook_matcher().
        with_resume_calls: List of (options, resume) tuples passed to with_resume().
        _client_queue: Internal queue of clients to return.
    """

    def __init__(self, client: FakeSDKClient | None = None) -> None:
        """Initialize the factory.

        Args:
            client: Optional single client to always return. If provided, this
                client is returned on every create() call.
        """
        self.client = client
        self.clients: list[FakeSDKClient] = []
        self.create_calls: list[object] = []
        self.created_options: list[dict[str, Any]] = []
        self.created_matchers: list[tuple[str, object | None, list[object]]] = []
        self.with_resume_calls: list[tuple[object, str | None]] = []
        self._client_queue: list[FakeSDKClient] = []

    def configure_next_client(
        self,
        messages: list[Any] | None = None,
        responses: list[Any] | None = None,
        result_message: object = _NO_RESULT_MESSAGE,
        query_error: Exception | None = None,
        raise_on_receive: Exception | None = None,
    ) -> FakeSDKClient:
        """Configure and queue the next client to be returned by create().

        Args:
            messages: Messages to yield (preferred name).
            responses: Alias for messages (for compatibility).
            result_message: Final message to yield. Defaults to the sentinel
                that triggers a default ResultMessage. Pass None explicitly
                to suppress the ResultMessage (protocol edge case testing).
            query_error: Exception to raise on query().
            raise_on_receive: Alias for query_error (for compatibility).

        Returns the created client for further configuration if needed.
        """
        # Support both 'messages' and 'responses' as parameter names
        msg_list = messages if messages is not None else (responses or [])
        error = query_error if query_error is not None else raise_on_receive
        client = FakeSDKClient(
            messages=msg_list,
            result_message=result_message,
            query_error=error,
        )
        self._client_queue.append(client)
        return client

    def create(self, options: object) -> SDKClientProtocol:
        """Return next queued client, fixed client, or create a new one."""
        self.create_calls.append(options)
        if self.client is not None:
            # Always return the same client if one was provided
            # Track in clients list for test assertions
            if self.client not in self.clients:
                self.clients.append(self.client)
            return self.client
        if self._client_queue:
            client = self._client_queue.pop(0)
        else:
            client = FakeSDKClient()
        self.clients.append(client)
        return client

    def create_options(
        self,
        *,
        cwd: str,
        permission_mode: str = "bypassPermissions",
        model: str = "opus",
        system_prompt: dict[str, str] | None = None,
        output_format: object | None = None,
        settings: str | None = None,
        setting_sources: list[str] | None = None,
        mcp_servers: object | None = None,
        disallowed_tools: list[str] | None = None,
        env: dict[str, str] | None = None,
        hooks: dict[str, list[object]] | None = None,
        resume: str | None = None,
    ) -> dict[str, Any]:
        """Return a dict of options and track the call."""
        opts: dict[str, Any] = {
            "cwd": cwd,
            "permission_mode": permission_mode,
            "model": model,
            "system_prompt": system_prompt,
            "output_format": output_format,
            "settings": settings,
            "setting_sources": setting_sources,
            "mcp_servers": mcp_servers,
            "disallowed_tools": disallowed_tools,
            "env": env,
            "hooks": hooks,
            "resume": resume,
        }
        self.created_options.append(opts)
        return opts

    def with_resume(self, options: object, resume: str | None) -> dict[str, Any]:
        """Create a copy of options with a different resume session ID.

        For testing, this simply copies the dict and updates the resume field.
        Tracks the call in with_resume_calls for test verification.
        """
        self.with_resume_calls.append((options, resume))
        if isinstance(options, dict):
            new_opts = dict(options)
        else:
            new_opts = {}
        new_opts["resume"] = resume
        return new_opts

    def create_hook_matcher(
        self,
        matcher: object | None,
        hooks: list[object],
    ) -> tuple[str, object | None, list[object]]:
        """Return a simple tuple representing the hook matcher and track the call."""
        result = ("matcher", matcher, hooks)
        self.created_matchers.append(result)
        return result
