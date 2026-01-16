"""SDK adapter for Claude Agent SDK.

This module provides a factory for creating Claude SDK clients, isolating
SDK imports to the infra layer. The pipeline layer uses SDKClientProtocol
from core.protocols instead of importing SDK types directly.

Design principles:
- All SDK imports are local (inside methods, not at module level)
- Factory pattern enables dependency injection and testing
- TYPE_CHECKING imports for SDK types avoid runtime dependency
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from src.core.protocols.sdk import SDKClientProtocol


class SDKClientFactory:
    """Factory for creating Claude SDK clients.

    This factory encapsulates SDK imports and client creation, allowing
    the pipeline layer to use SDK clients without importing SDK directly.

    Usage:
        factory = SDKClientFactory()
        client = factory.create(options)
        async with client:
            await client.query(prompt)
            async for msg in client.receive_response():
                ...
    """

    def create(self, options: object) -> SDKClientProtocol:
        """Create a new SDK client with the given options.

        Args:
            options: ClaudeAgentOptions (or compatible) for the client.
                The type is `object` to avoid requiring SDK import in
                the caller's type annotations.

        Returns:
            SDKClientProtocol wrapping a ClaudeSDKClient.
        """
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # noqa: TC002
        from src.infra.sdk_transport import ensure_sigint_isolated_cli_transport

        ensure_sigint_isolated_cli_transport()
        return cast(
            "SDKClientProtocol",
            ClaudeSDKClient(options=cast("ClaudeAgentOptions", options)),
        )

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
            permission_mode=permission_mode,  # type: ignore[arg-type]
            model=model,
            system_prompt=system_prompt or {"type": "preset", "preset": "claude_code"},  # type: ignore[arg-type]
            output_format=output_format,  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            setting_sources=effective_sources,  # type: ignore[arg-type]
            mcp_servers=mcp_servers,  # type: ignore[arg-type]
            disallowed_tools=disallowed_tools,  # type: ignore[arg-type]
            env=effective_env,  # type: ignore[arg-type]
            hooks=hooks,  # type: ignore[arg-type]
            resume=resume,  # type: ignore[arg-type]
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

        return HookMatcher(matcher=matcher, hooks=hooks)  # type: ignore[arg-type]

    def with_resume(self, options: object, resume: str | None) -> object:
        """Create a copy of options with a different resume session ID.

        This is used to resume a prior session when retrying after idle timeout
        or review failures. The SDK's resume feature loads the prior conversation
        context before processing the next query.

        Args:
            options: Existing ClaudeAgentOptions to clone.
            resume: Session ID to resume from, or None to start fresh.

        Returns:
            New ClaudeAgentOptions with the resume field set.
        """
        from claude_agent_sdk import ClaudeAgentOptions
        from dataclasses import fields

        # ClaudeAgentOptions is a dataclass - extract all field values
        opts = options
        if not isinstance(opts, ClaudeAgentOptions):
            raise TypeError(f"Expected ClaudeAgentOptions, got {type(opts)}")

        # Build kwargs from existing options, overriding resume
        kwargs = {f.name: getattr(opts, f.name) for f in fields(opts)}
        kwargs["resume"] = resume
        if "settings" in kwargs and kwargs["settings"] is None:
            kwargs["settings"] = '{"autoCompactEnabled": true}'

        return ClaudeAgentOptions(**kwargs)  # type: ignore[arg-type]
