"""Shared Anthropic client factory for mala orchestrator.

This module provides a centralized way to create Anthropic clients with:
- Consistent configuration from MalaConfig (api_key, base_url)

Usage:
    from src.infra.clients.anthropic_client import create_anthropic_client
    from src.infra.io.config import MalaConfig

    config = MalaConfig.from_env()
    client = create_anthropic_client(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        timeout=120.0,
    )

    # Client is ready to use
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

from typing import Any


def create_anthropic_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> Any:  # noqa: ANN401 - Return type is dynamic (Anthropic client)
    """Create an Anthropic client with consistent configuration.

    This factory function centralizes Anthropic client creation to ensure:
    1. Configuration is applied consistently (api_key, base_url)
    2. Error handling works uniformly across components

    Args:
        api_key: Anthropic API key. If not provided, the client will use
            the ANTHROPIC_API_KEY environment variable.
        base_url: Optional base URL for API requests. Use this to route
            requests through proxies.
        timeout: Optional timeout in seconds for API requests.

    Returns:
        An Anthropic client instance.

    Raises:
        RuntimeError: If the anthropic package is not installed.

    Example:
        from src.infra.clients.anthropic_client import create_anthropic_client

        # Basic usage with env var for API key
        client = create_anthropic_client()

        # With explicit configuration
        client = create_anthropic_client(
            api_key="sk-...",
            base_url="https://proxy.example.com/v1",
            timeout=60.0,
        )
    """
    try:
        from anthropic import Anthropic  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "anthropic package is required. Install with: uv add anthropic"
        ) from e

    # Build client kwargs, only including non-None values
    client_kwargs: dict[str, object] = {}
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    if timeout is not None:
        client_kwargs["timeout"] = timeout

    # Create and return the client
    return Anthropic(**client_kwargs)
