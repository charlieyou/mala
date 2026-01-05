"""Contract tests for fake-vs-real implementation parity.

Contract tests validate that fake implementations behave identically to their
real counterparts. Each test is parametrized to run against both the fake
and real implementation, ensuring fakes don't silently diverge.

Real provider tests skip gracefully when prerequisites are missing:
- bd CLI not in PATH
- BEADS_TEST_WORKSPACE env var not set

Run contract tests:
    uv run pytest tests/contracts/ -v

Run with real providers (requires prerequisites):
    BEADS_TEST_WORKSPACE=/path/to/workspace uv run pytest tests/contracts/ -v
"""
