"""MCP meta tools: health, ping, and optional linkedin_* aliases.

Neither meta tool drives Chromium. ``SequentialToolExecutionMiddleware`` lets
them through without the scraper lock or the profile lease, so a health probe
answers while another client holds the browser instead of queueing behind it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools.base import Tool

from linkedin_mcp_server import __version__
from linkedin_mcp_server.authentication import get_authentication_source
from linkedin_mcp_server.bootstrap import (
    SetupState,
    browser_setup_ready,
    browsers_path,
    get_bootstrap_state,
    get_runtime_policy,
    initialize_bootstrap,
)
from linkedin_mcp_server.config import get_config
from linkedin_mcp_server.sequential_tool_middleware import LOCK_FREE_TOOL_NAMES
from linkedin_mcp_server.drivers.browser import get_profile_dir, profile_exists
from linkedin_mcp_server.session_state import (
    get_runtime_id,
    portable_cookie_path,
    runtime_profiles_root,
    source_state_path,
)

logger = logging.getLogger(__name__)

META_PING_TIMEOUT_SECONDS = 10.0
META_HEALTH_TIMEOUT_SECONDS = 30.0
META_TOOL_NAMES: frozenset[str] = LOCK_FREE_TOOL_NAMES

ALIAS_PREFIX = "linkedin_"
# Opt-in, because every alias is a second copy of a tool schema in the client's
# context. Clients that load every tool eagerly pay for all of them twice.
TOOL_ALIASES_ENV = "LINKEDIN_MCP_TOOL_ALIASES"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def tool_aliases_enabled() -> bool:
    """Whether ``LINKEDIN_MCP_TOOL_ALIASES`` asks for linkedin_* aliases."""
    return os.environ.get(TOOL_ALIASES_ENV, "").strip().lower() in _TRUTHY


def _session_auth_ready(profile_dir: Path) -> bool:
    if not (
        profile_exists(profile_dir)
        and portable_cookie_path(profile_dir).exists()
        and source_state_path(profile_dir).exists()
    ):
        return False
    try:
        return bool(get_authentication_source())
    except Exception:
        return False


def _build_storage_paths(profile_dir: Path) -> dict[str, str]:
    resolved = profile_dir.expanduser().resolve()
    return {
        "auth_root": str(resolved.parent),
        "profile_dir": str(resolved),
        "cookies_json": str(portable_cookie_path(profile_dir)),
        "source_state_json": str(source_state_path(profile_dir)),
        "runtime_profiles_dir": str(runtime_profiles_root(profile_dir)),
        "patchright_browsers_dir": str(browsers_path()),
    }


def build_health_payload() -> dict[str, Any]:
    """Return server health without launching the browser."""
    initialize_bootstrap()
    config = get_config()
    profile_dir = get_profile_dir()
    bootstrap = get_bootstrap_state()
    auth_ready = _session_auth_ready(profile_dir)
    browser_ready = browser_setup_ready()

    warnings: list[str] = []
    if not browser_ready:
        warnings.append(
            "Patchright Chromium is not installed or its metadata is stale. "
            "The first scraping tool call installs it in the background."
        )
    if not auth_ready:
        warnings.append(
            "No valid LinkedIn session. Run `mcp-server-linkedin --login` on the host."
        )
    if bootstrap.last_error:
        warnings.append(f"Bootstrap last error: {bootstrap.last_error}")

    payload: dict[str, Any] = {
        "ok": auth_ready
        and (browser_ready or bootstrap.setup_state is SetupState.READY),
        "server": "linkedin-mcp",
        "version": __version__,
        "transport": config.server.transport,
        "runtime_policy": get_runtime_policy().value,
        "runtime_id": get_runtime_id(),
        "bootstrap": {
            "setup_state": bootstrap.setup_state.value,
            "auth_state": bootstrap.auth_state.value,
            "browser_ready": browser_ready,
            "auth_ready": auth_ready,
        },
        "session": {
            "profile_exists": profile_exists(profile_dir),
            "cookies_present": portable_cookie_path(profile_dir).exists(),
            "source_state_present": source_state_path(profile_dir).exists(),
            "auth_ready": auth_ready,
        },
        "storage": _build_storage_paths(profile_dir),
        "timeouts_seconds": {
            "tool_default": config.server.tool_timeout_seconds,
            "browser_page_default_ms": config.browser.default_timeout,
        },
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


async def build_ping_payload(mcp: FastMCP) -> dict[str, Any]:
    """Return the capability-discovery payload for linkedin_ping."""
    config = get_config()
    tools = await mcp.list_tools(run_middleware=False)
    tool_list = [
        {"name": tool.name, "description": tool.description or ""}
        for tool in sorted(tools, key=lambda t: t.name)
    ]
    names = {tool["name"] for tool in tool_list}
    aliases = sorted(
        name
        for name in names
        if name.startswith(ALIAS_PREFIX)
        and name not in META_TOOL_NAMES
        and name.removeprefix(ALIAS_PREFIX) in names
    )
    return {
        "ok": True,
        "pong": True,
        "server": "linkedin-mcp",
        "version": __version__,
        "transport": config.server.transport,
        "tools": tool_list,
        "tool_count": len(tool_list),
        "capabilities": {
            "sequential_tool_execution": True,
            "meta_tools": sorted(META_TOOL_NAMES),
            "tool_aliases": aliases,
            "default_tool_timeout_seconds": config.server.tool_timeout_seconds,
        },
        "storage": _build_storage_paths(get_profile_dir()),
    }


def _local_tools(mcp: FastMCP) -> dict[str, Tool]:
    """Registered local tools by name, read without the async list API.

    ``list_tools`` is async and alias registration runs during synchronous
    server setup, so this reads the local provider's registry. If a FastMCP
    release moves it, the map comes back empty and aliases are skipped rather
    than stopping the server.
    """
    provider = getattr(mcp, "local_provider", None)
    components = getattr(provider, "_components", None)
    if not isinstance(components, dict):
        return {}
    return {
        component.name: component
        for component in components.values()
        if isinstance(component, Tool)
    }


def register_tool_aliases(mcp: FastMCP) -> list[str]:
    """Register a ``linkedin_<name>`` copy of every local tool.

    Call it after every other registration. Meta tools already carry the
    prefix, and a name that is taken is left alone. Returns the aliases added.
    """
    tools = _local_tools(mcp)
    if not tools:
        logger.warning("No local tool registry available; skipping linkedin_* aliases")
        return []

    added: list[str] = []
    for name, tool in sorted(tools.items()):
        if name.startswith(ALIAS_PREFIX):
            continue
        alias = f"{ALIAS_PREFIX}{name}"
        if alias in tools:
            continue
        try:
            mcp.add_tool(Tool.from_tool(tool, name=alias))
        except Exception:
            logger.exception("Failed to register linkedin_* alias for %s", name)
            continue
        added.append(alias)
    return added


def register_meta_tools(mcp: FastMCP) -> None:
    """Register the linkedin_health and linkedin_ping meta tools."""

    @mcp.tool(
        name="linkedin_health",
        timeout=META_HEALTH_TIMEOUT_SECONDS,
        title="LinkedIn MCP Health",
        annotations={"readOnlyHint": True},
        tags={"meta"},
    )
    async def linkedin_health() -> dict[str, Any]:
        """Report server version, storage paths, and browser/session readiness.

        Never opens the browser and never waits for the scraper lock, so it
        answers while another tool call or another MCP client is scraping.
        """
        return build_health_payload()

    @mcp.tool(
        name="linkedin_ping",
        timeout=META_PING_TIMEOUT_SECONDS,
        title="LinkedIn MCP Ping",
        annotations={"readOnlyHint": True},
        tags={"meta"},
    )
    async def linkedin_ping() -> dict[str, Any]:
        """List the registered tools and server capabilities."""
        return await build_ping_payload(mcp)
