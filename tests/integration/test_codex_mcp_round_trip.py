"""Codex MCP launcher round-trip (Phase G T016).

Exercises the bundled ``mala-codex-mcp-locking`` console script — the new
sibling of ``mala-amp-mcp-locking`` shipped by this issue — by spawning
it as a stdio MCP subprocess and round-tripping ``lock_acquire`` /
``lock_release`` through the real MCP wire protocol. This is the same
launcher Codex's bundled plugin (``plugins/codex/mala-safety/.codex-plugin/.mcp.json``)
references; verifying the launcher works end-to-end via stdio MCP proves
the Codex side of plan G1+G2 without requiring a real Codex install.

The on-disk assertions mirror ``tests/integration/test_lock_mcp_via_amp.py``:
after ``lock_acquire`` a ``<hash>.lock`` file lands in ``MALA_LOCK_DIR``;
after ``lock_release`` it is gone. Both tests share the same launcher
module (``src.infra.tools.locking_mcp_stdio``) — the round-trip here
proves the new ``mala-codex-mcp-locking`` console-script entry point
binds correctly without changing the locking semantics Amp already
verifies through its own round-trip path (plan ``L80``: "different
entry-point name, same module").

Gating
------
Skipif the ``mala-codex-mcp-locking`` console script is not on PATH
(typical when ``uv sync`` hasn't been run since the entry-point was
added). The skip reason names the fix so a missing install does not
look like a launcher regression.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _resolve_codex_launcher_path() -> str | None:
    """Return the absolute path to ``mala-codex-mcp-locking``.

    Resolution order mirrors :func:`tests.integration.test_lock_mcp_via_amp._resolve_locking_launcher_path`:
    sibling-of-``sys.executable`` (where ``[project.scripts]`` lands console
    scripts after ``uv sync`` / ``pip install -e .``) first, then ``$PATH``
    for system-wide installs (``pipx install mala-agent``).
    """
    sibling = Path(sys.executable).parent / "mala-codex-mcp-locking"
    if sibling.is_file():
        return str(sibling)
    on_path = shutil.which("mala-codex-mcp-locking")
    if on_path is not None:
        return on_path
    return None


_LAUNCHER_PATH = _resolve_codex_launcher_path()

_LAUNCHER_REASON = (
    "mala-codex-mcp-locking console script not registered. The launcher is "
    "defined in src.infra.tools.locking_mcp_stdio and exposed via "
    "[project.scripts] in pyproject.toml; after `uv sync` it lives at "
    f"{Path(sys.executable).parent}/mala-codex-mcp-locking. Run `uv sync` "
    "and re-run this test."
)


@pytest.mark.skipif(_LAUNCHER_PATH is None, reason=_LAUNCHER_REASON)
async def test_codex_launcher_lock_acquire_release_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full MCP stdio round-trip through the new ``mala-codex-mcp-locking``.

    Spawns the launcher subprocess (the same way Codex's bundled plugin
    does via ``.mcp.json``) and drives it through ``initialize`` →
    ``tools/list`` → ``tools/call lock_acquire`` → ``tools/call
    lock_release``.

    Asserts the on-disk lock fixture appears after acquire and is gone
    after release — same observable end state as the Amp round-trip
    test, which the plan calls out as the parity contract.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    assert _LAUNCHER_PATH is not None  # gate enforces this

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    target = repo_path / "locked.py"
    target.write_text("# original\n")

    monkeypatch.setenv("MALA_LOCK_DIR", str(lock_dir))

    agent_id = "agent-codex-roundtrip"
    server_params = StdioServerParameters(
        command=_LAUNCHER_PATH,
        args=[
            "--agent-id",
            agent_id,
            "--repo-namespace",
            str(repo_path),
        ],
        env={
            "MALA_AGENT_ID": agent_id,
            "MALA_REPO_NAMESPACE": str(repo_path),
            "MALA_LOCK_DIR": str(lock_dir),
            "PATH": __import__("os").environ.get("PATH", ""),
        },
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_response = await session.list_tools()
            tool_names = {t.name for t in tools_response.tools}
            assert tool_names == {"lock_acquire", "lock_release"}, (
                f"Unexpected tool surface from mala-codex-mcp-locking: "
                f"{tool_names!r}; the new Codex console-script must expose the "
                f"same {{lock_acquire, lock_release}} pair as mala-amp-mcp-locking "
                f"because both back the same locking_mcp_stdio module (plan G2)."
            )

            acquire_result = await session.call_tool(
                "lock_acquire",
                arguments={"filepaths": [str(target)], "timeout_seconds": 0},
            )
            # Tool returned content; precise body shape is the same JSON
            # the in-process locking_mcp uses (locking_mcp_stdio reuses
            # the locking primitives) — we assert the on-disk effect
            # rather than re-parsing the JSON, mirroring the Amp
            # round-trip test's posture.
            assert not acquire_result.isError, (
                f"lock_acquire returned isError=True: {acquire_result.content!r}"
            )
            lock_files = list(lock_dir.glob("*.lock"))
            assert lock_files, (
                "lock_acquire did not produce a <hash>.lock file in "
                f"{lock_dir}; the new mala-codex-mcp-locking entry point may "
                "have failed silently or diverged from locking_mcp_stdio.main."
            )

            release_result = await session.call_tool(
                "lock_release",
                arguments={"filepaths": [str(target)]},
            )
            assert not release_result.isError, (
                f"lock_release returned isError=True: {release_result.content!r}"
            )
            remaining = list(lock_dir.glob("*.lock"))
            assert not remaining, (
                f"lock_release did not remove the <hash>.lock fixture; "
                f"remaining files: {remaining!r}. The new console-script "
                "must round-trip release identically to the Amp launcher "
                "(plan G2 'same locking server Amp uses')."
            )
