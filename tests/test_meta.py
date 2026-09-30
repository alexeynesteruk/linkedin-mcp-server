"""linkedin_health, linkedin_ping, and the opt-in linkedin_* aliases."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from linkedin_mcp_server.bootstrap import SetupState, get_bootstrap_state
from linkedin_mcp_server.profile_lease import get_profile_lease
from linkedin_mcp_server.sequential_tool_middleware import (
    LOCK_FREE_TOOL_NAMES,
    SequentialToolExecutionMiddleware,
)
from linkedin_mcp_server.server import create_mcp_server
from linkedin_mcp_server.session_state import portable_cookie_path, source_state_path
from linkedin_mcp_server.tools.meta import (
    META_TOOL_NAMES,
    TOOL_ALIASES_ENV,
    _session_auth_ready,
    build_health_payload,
    build_ping_payload,
    register_meta_tools,
    register_tool_aliases,
)

_WORKER = Path(__file__).parent / "helpers" / "profile_lease_worker.py"


def _hold_profile(auth_root: Path, seconds: float) -> subprocess.Popen[str]:
    """Spawn a process that owns *auth_root*'s lease, and wait until it does."""
    process = subprocess.Popen(
        [sys.executable, str(_WORKER), "hold", str(auth_root), str(seconds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if "HELD" in line:
            return process
        if line == "":
            process.kill()
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(f"lease worker exited without holding: {stderr}")


def _call_context(tool_name: str) -> MagicMock:
    context = MagicMock()
    context.message.name = tool_name
    context.fastmcp_context = None
    return context


def _write_session_files(profile_dir: Path) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "marker").write_text("profile")
    portable_cookie_path(profile_dir).write_text("[]")
    source_state_path(profile_dir).write_text('{"version": 1}')


def _local_tool_names(mcp: FastMCP) -> set[str]:
    return {component.name for component in mcp.local_provider._components.values()}


class TestHealthPayload:
    def test_warns_when_the_browser_is_missing(self, monkeypatch):
        monkeypatch.setattr(
            "linkedin_mcp_server.tools.meta.browser_setup_ready", lambda: False
        )

        payload = build_health_payload()

        assert payload["bootstrap"]["browser_ready"] is False
        assert any("Patchright Chromium" in w for w in payload["warnings"])

    def test_warns_and_is_not_ok_without_a_session(self):
        payload = build_health_payload()

        assert payload["ok"] is False
        assert payload["bootstrap"]["auth_ready"] is False
        assert any("--login" in w for w in payload["warnings"])

    def test_reports_the_last_bootstrap_error(self):
        get_bootstrap_state().last_error = "install failed"

        payload = build_health_payload()

        assert any("install failed" in w for w in payload["warnings"])

    def test_ok_when_session_and_browser_are_ready(self, profile_dir, monkeypatch):
        _write_session_files(profile_dir)
        monkeypatch.setattr(
            "linkedin_mcp_server.tools.meta.get_authentication_source", lambda: True
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.tools.meta.browser_setup_ready", lambda: True
        )
        get_bootstrap_state().setup_state = SetupState.READY

        payload = build_health_payload()

        assert payload["ok"] is True
        assert payload["bootstrap"]["auth_ready"] is True
        assert "warnings" not in payload
        assert payload["storage"]["profile_dir"] == str(profile_dir.resolve())

    def test_session_is_not_ready_when_the_source_lookup_raises(
        self, profile_dir, monkeypatch
    ):
        _write_session_files(profile_dir)

        def broken() -> bool:
            raise RuntimeError("bad metadata")

        monkeypatch.setattr(
            "linkedin_mcp_server.tools.meta.get_authentication_source", broken
        )

        assert _session_auth_ready(profile_dir) is False

    def test_session_is_not_ready_when_the_source_lookup_says_no(
        self, profile_dir, monkeypatch
    ):
        _write_session_files(profile_dir)
        monkeypatch.setattr(
            "linkedin_mcp_server.tools.meta.get_authentication_source", lambda: False
        )

        assert _session_auth_ready(profile_dir) is False


class TestHealthHasNoSideEffects:
    def test_health_writes_nothing_under_the_auth_root(self, profile_dir, tmp_path):
        # It runs past the lease, so anything it wrote could race a process
        # that holds the profile: a claim, a lease file, a browser cache.
        _write_session_files(profile_dir)
        before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

        build_health_payload()
        build_health_payload()

        after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
        assert after == before


class TestPingPayload:
    async def test_lists_tools_and_their_aliases(self):
        mcp = FastMCP("test")
        register_meta_tools(mcp)

        @mcp.tool
        async def get_person_profile() -> dict[str, bool]:
            return {"ok": True}

        register_tool_aliases(mcp)

        payload = await build_ping_payload(mcp)

        names = {tool["name"] for tool in payload["tools"]}
        assert {"get_person_profile", "linkedin_get_person_profile"} <= names
        assert payload["capabilities"]["tool_aliases"] == [
            "linkedin_get_person_profile"
        ]
        assert payload["capabilities"]["meta_tools"] == sorted(META_TOOL_NAMES)
        assert payload["tool_count"] == len(payload["tools"])


class TestToolAliases:
    def test_aliases_every_local_tool_but_the_prefixed_ones(self):
        mcp = FastMCP("test")
        register_meta_tools(mcp)

        @mcp.tool
        async def get_inbox() -> dict[str, bool]:
            return {"ok": True}

        added = register_tool_aliases(mcp)

        assert added == ["linkedin_get_inbox"]
        assert "linkedin_linkedin_health" not in _local_tool_names(mcp)

    def test_leaves_a_taken_alias_alone(self):
        mcp = FastMCP("test")

        @mcp.tool
        async def get_inbox() -> dict[str, bool]:
            return {"ok": True}

        @mcp.tool(name="linkedin_get_inbox")
        async def custom() -> dict[str, bool]:
            return {"custom": True}

        assert register_tool_aliases(mcp) == []

    def test_one_failing_alias_does_not_stop_the_rest(self, caplog):
        mcp = FastMCP("test")

        @mcp.tool
        async def get_feed() -> dict[str, bool]:
            return {"ok": True}

        @mcp.tool
        async def get_inbox() -> dict[str, bool]:
            return {"ok": True}

        original = mcp.add_tool

        def flaky(tool: Any) -> Any:
            if tool.name == "linkedin_get_feed":
                raise RuntimeError("boom")
            return original(tool)

        mcp.add_tool = cast(Any, flaky)
        with caplog.at_level(logging.ERROR):
            added = register_tool_aliases(mcp)

        assert added == ["linkedin_get_inbox"]
        assert "Failed to register linkedin_* alias for get_feed" in caplog.text

    def test_a_missing_registry_skips_instead_of_raising(self, caplog):
        mcp = FastMCP("test")
        mcp.local_provider._components = cast(Any, None)

        with caplog.at_level(logging.WARNING):
            assert register_tool_aliases(mcp) == []

        assert "No local tool registry available" in caplog.text


class TestServerRegistration:
    async def test_meta_tools_are_served_and_aliases_are_off_by_default(self):
        names = {tool.name for tool in await create_mcp_server().list_tools()}

        assert META_TOOL_NAMES <= names
        assert "linkedin_get_person_profile" not in names

    async def test_the_environment_switch_aliases_every_tool(self, monkeypatch):
        monkeypatch.setenv(TOOL_ALIASES_ENV, "true")

        names = {tool.name for tool in await create_mcp_server().list_tools()}

        plain = {name for name in names if not name.startswith("linkedin_")}
        assert plain, "the server registered no tools"
        assert {f"linkedin_{name}" for name in plain} <= names

    async def test_linkedin_health_answers_through_the_server(self):
        result = await create_mcp_server().call_tool("linkedin_health", {})

        payload = result.structured_content
        assert payload is not None
        assert payload["server"] == "linkedin-mcp"
        assert {"storage", "bootstrap", "session"} <= set(payload)


class TestMetaToolsSkipTheScraperLock:
    def test_the_lock_free_names_are_the_meta_tools(self):
        assert LOCK_FREE_TOOL_NAMES == META_TOOL_NAMES

    async def test_health_answers_while_another_process_holds_the_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BROWSER_WAIT", "0.5")
        holder = _hold_profile(tmp_path, 30)
        try:
            lease = get_profile_lease(tmp_path / "profile")
            with patch(
                "linkedin_mcp_server.sequential_tool_middleware.get_profile_lease",
                return_value=lease,
            ):
                middleware = SequentialToolExecutionMiddleware()

                health = AsyncMock(return_value="health")
                assert (
                    await middleware.on_call_tool(
                        _call_context("linkedin_health"), health
                    )
                    == "health"
                )
                health.assert_awaited_once()

                # The same holder makes a scraping tool report busy, which is
                # what shows the health call really went around the lease.
                scrape = AsyncMock()
                with pytest.raises(ToolError, match="using the browser"):
                    await middleware.on_call_tool(
                        _call_context("get_person_profile"), scrape
                    )
                scrape.assert_not_awaited()
        finally:
            holder.kill()
            holder.wait(timeout=10)

    async def test_health_does_not_queue_behind_a_call_in_this_process(self):
        middleware = SequentialToolExecutionMiddleware()
        call_next = AsyncMock(return_value="health")

        # Bounded, so a regression fails here instead of deadlocking the run.
        async with middleware._lock:
            result = await asyncio.wait_for(
                middleware.on_call_tool(_call_context("linkedin_ping"), call_next),
                timeout=5,
            )

        assert result == "health"
