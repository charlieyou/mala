"""SDK adapter for Claude Agent SDK.

This module provides a Claude-private factory for creating Claude SDK
clients, isolating SDK imports to the infra layer. The pipeline layer
uses ``SDKClientProtocol`` and the slim cross-coder
``SDKClientFactoryProtocol`` from ``core.protocols``; Claude-specific
knobs (``create_options``, ``create_hook_matcher``) live on the
concrete :class:`SDKClientFactory` defined here and never leak onto the
cross-coder protocol.

Design principles:
- All SDK imports are local (inside methods, not at module level)
- Factory pattern enables dependency injection and testing
- TYPE_CHECKING imports for SDK types avoid runtime dependency

The Claude pipeline path returns a :class:`_ClaudeAgentEventClient`
wrapper whose ``receive_response()`` emits coder-agnostic
:class:`AgentEvent` values directly, mirroring
:meth:`AmpClient.receive_response` and removing the per-call translation
that previously lived in the pipeline. Claude-private callers that pass
bare ``ClaudeAgentOptions`` (e.g. :class:`AgentSDKReviewer`, which
consumes Anthropic ``ResultMessage`` fields like ``structured_output``
directly) still receive the unwrapped SDK client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType
    from typing import Self

    from src.core.protocols.agent_event import AgentEventValue
    from src.core.protocols.sdk import SDKClientProtocol


class _ClaudeAgentEventClient:
    """Wrap :class:`ClaudeSDKClient` so ``receive_response()`` emits ``AgentEvent``s.

    The pipeline only iterates ``receive_response()``; every other
    method delegates verbatim to the inner SDK client so context-manager
    entry/exit, ``query``, and ``disconnect`` retain their original
    Anthropic-SDK semantics.
    """

    def __init__(self, inner: SDKClientProtocol) -> None:
        self._inner = inner

    async def __aenter__(self) -> Self:
        await self._inner.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._inner.__aexit__(exc_type, exc_val, exc_tb)

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        await self._inner.query(prompt, session_id)

    def receive_response(self) -> AsyncIterator[AgentEventValue]:
        from src.core.protocols.agent_event import to_agent_events

        return to_agent_events(self._inner.receive_response())

    async def disconnect(self) -> None:
        await self._inner.disconnect()


class SDKClientFactory:
    """Claude-private factory for creating Claude SDK clients.

    Conforms to the slim cross-coder ``SDKClientFactoryProtocol``
    (``create`` / ``with_resume``) and additionally exposes the
    Claude-only knobs ``create_options`` and ``create_hook_matcher``
    used by Claude-specific wiring code (``ClaudeAgentRuntimeBuilder``,
    ``AgentSDKReviewer``). The Claude knobs intentionally do not appear
    on the cross-coder protocol so Amp / Codex backends are not forced
    to provide ``NotImplementedError`` walls for them.

    Usage:
        factory = SDKClientFactory()
        client = factory.create(runtime)
        async with client:
            await client.query(prompt)
            async for msg in client.receive_response():
                ...
    """

    def create(self, runtime: object) -> SDKClientProtocol:
        """Create a new SDK client for the given Claude runtime.

        Args:
            runtime: Opaque Claude-shaped runtime. Either an
                :class:`AgentRuntime` produced by
                :class:`ClaudeAgentRuntimeBuilder.build`, in which case its
                ``options`` field carries the ``ClaudeAgentOptions``, or a
                bare ``ClaudeAgentOptions`` for Claude-private callers
                (e.g. :class:`AgentSDKReviewer`) that build options
                themselves. The pipeline only goes through the runtime
                form; the bare-options shape is a Claude-private
                convenience preserved here.

        Returns:
            SDKClientProtocol. For pipeline (``AgentRuntime``) callers,
            this is a :class:`_ClaudeAgentEventClient` wrapping a
            ``ClaudeSDKClient`` so ``receive_response()`` yields
            ``AgentEvent`` values directly. For Claude-private bare
            ``ClaudeAgentOptions`` callers, the underlying
            ``ClaudeSDKClient`` is returned unwrapped so they can
            consume Anthropic-shaped ``ResultMessage`` fields
            (``structured_output``, etc.) that have no AgentEvent
            counterpart.
        """
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # noqa: TC002
        from src.infra.agent_runtime import AgentRuntime
        from src.infra.sdk_transport import ensure_sigint_isolated_cli_transport

        options = getattr(runtime, "options", runtime)

        ensure_sigint_isolated_cli_transport()
        raw_client = cast(
            "SDKClientProtocol",
            ClaudeSDKClient(options=cast("ClaudeAgentOptions", options)),
        )
        if isinstance(runtime, AgentRuntime):
            return cast("SDKClientProtocol", _ClaudeAgentEventClient(raw_client))
        return raw_client

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
        effort: str | None = None,
    ) -> object:
        """Create SDK options without requiring SDK import in caller.

        This method encapsulates the ClaudeAgentOptions construction,
        allowing callers to build options without SDK imports.

        Args:
            cwd: Working directory for the agent.
            permission_mode: Permission mode (default "bypassPermissions").
            model: Model to use (default "opus").
            system_prompt: System prompt configuration.
            output_format: Structured output format configuration.
            settings: JSON settings string passed to ClaudeAgentOptions.
            setting_sources: List of setting sources (default ["local", "project"]).
            mcp_servers: List of MCP server configurations.
            disallowed_tools: List of tools to disallow.
            env: Environment variables for the agent.
            hooks: Hook configurations keyed by event type.
            resume: Session ID to resume from. When set, the SDK loads
                the prior conversation context before processing the query.
                This is different from session_id on query() which only
                tags messages for multiplexing.
            effort: Optional reasoning effort forwarded verbatim to
                ``ClaudeAgentOptions.effort``. ``None`` leaves the SDK
                default in place. The SDK statically types the field as
                ``Literal["low", "medium", "high", "max"]``; ``"xhigh"`` is
                accepted by the underlying CLI but produces a static-type
                warning here, hence the ``type: ignore`` below.

        Returns:
            ClaudeAgentOptions instance.
        """
        from claude_agent_sdk import ClaudeAgentOptions

        effective_env = env or {}

        effective_sources = (
            ["local", "project"] if setting_sources is None else setting_sources
        )
        return ClaudeAgentOptions(
            cwd=cwd,
            permission_mode=permission_mode,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            model=model,
            system_prompt=system_prompt or {"type": "preset", "preset": "claude_code"},  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            output_format=output_format,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            settings=settings,  # type: ignore[arg-type]
            setting_sources=effective_sources,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            mcp_servers=mcp_servers,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            disallowed_tools=disallowed_tools,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            env=effective_env,  # type: ignore[arg-type]
            hooks=hooks,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            resume=resume,  # type: ignore[arg-type]
            effort=effort,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        )

    def create_hook_matcher(
        self,
        matcher: object | None,
        hooks: list[object],
    ) -> object:
        """Create a HookMatcher for SDK hook registration.

        Args:
            matcher: Optional matcher configuration.
            hooks: List of hook callables.

        Returns:
            HookMatcher instance.
        """
        from claude_agent_sdk.types import HookMatcher

        return HookMatcher(matcher=matcher, hooks=hooks)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    def with_resume(self, runtime: object, resume: str | None) -> object:
        """Create a copy of the runtime with a different resume session ID.

        This is used to resume a prior session when retrying after idle timeout
        or review failures. The SDK's resume feature loads the prior conversation
        context before processing the next query.

        Args:
            runtime: Existing Claude-shaped runtime. Either an
                :class:`AgentRuntime` (with an ``options`` field carrying
                the ``ClaudeAgentOptions``) or a bare ``ClaudeAgentOptions``
                for Claude-private callers.
            resume: Session ID to resume from, or None to start fresh.

        Returns:
            Same shape as ``runtime`` with its ``ClaudeAgentOptions``
            ``resume`` field updated.
        """
        from claude_agent_sdk import ClaudeAgentOptions
        from dataclasses import fields, replace

        from src.infra.agent_runtime import AgentRuntime

        if isinstance(runtime, AgentRuntime):
            opts = runtime.options
        else:
            opts = runtime

        if not isinstance(opts, ClaudeAgentOptions):
            raise TypeError(f"Expected ClaudeAgentOptions, got {type(opts)}")

        # Build kwargs from existing options, overriding resume
        kwargs = {f.name: getattr(opts, f.name) for f in fields(opts)}
        kwargs["resume"] = resume
        if "settings" in kwargs and kwargs["settings"] is None:
            kwargs["settings"] = '{"autoCompactEnabled": true}'

        new_opts = ClaudeAgentOptions(**kwargs)  # type: ignore[arg-type]
        if isinstance(runtime, AgentRuntime):
            return replace(runtime, options=new_opts)
        return new_opts
