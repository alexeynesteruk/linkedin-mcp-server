"""Tests for Claude Code / client shim argument stripping."""

from __future__ import annotations

from unittest.mock import AsyncMock

import mcp.types as mt
import pytest
from fastmcp import FastMCP
from fastmcp.server.middleware import MiddlewareContext

from linkedin_mcp_server.client_compat_middleware import (
    StripClientShimArgsMiddleware,
    strip_client_shim_arguments,
)
from linkedin_mcp_server.server import create_mcp_server


class TestStripClientShimArguments:
    def test_returns_same_mapping_when_empty(self):
        assert strip_client_shim_arguments(None) is None
        assert strip_client_shim_arguments({}) == {}

    def test_returns_same_mapping_when_no_shim(self):
        args = {"limit": 20}
        assert strip_client_shim_arguments(args) is args

    def test_strips_underscore_shim_and_keeps_real_args(self):
        cleaned = strip_client_shim_arguments({"_": True, "limit": 20})
        assert cleaned == {"limit": 20}

    def test_strips_underscore_when_it_is_the_only_arg(self):
        cleaned = strip_client_shim_arguments({"_": False})
        assert cleaned == {}


class TestStripClientShimArgsMiddleware:
    async def test_middleware_forwards_cleaned_arguments(self):
        middleware = StripClientShimArgsMiddleware()
        call_next = AsyncMock(return_value={"ok": True})
        message = mt.CallToolRequestParams(
            name="get_inbox",
            arguments={"_": True, "limit": 5},
        )
        context = MiddlewareContext(
            message=message,
            source="client",
            type="request",
            method="tools/call",
        )

        result = await middleware.on_call_tool(context, call_next)

        assert result == {"ok": True}
        call_next.assert_awaited_once()
        forwarded = call_next.await_args.args[0]
        assert forwarded.message.arguments == {"limit": 5}
        assert forwarded.message.name == "get_inbox"

    async def test_middleware_skips_copy_when_no_shim(self):
        middleware = StripClientShimArgsMiddleware()
        call_next = AsyncMock(return_value={"ok": True})
        message = mt.CallToolRequestParams(
            name="get_inbox",
            arguments={"limit": 5},
        )
        context = MiddlewareContext(
            message=message,
            source="client",
            type="request",
            method="tools/call",
        )

        await middleware.on_call_tool(context, call_next)

        forwarded = call_next.await_args.args[0]
        assert forwarded is context

    async def test_create_mcp_server_registers_shim_middleware(self):
        mcp = create_mcp_server()
        assert any(
            isinstance(middleware, StripClientShimArgsMiddleware)
            for middleware in mcp.middleware
        )

    async def test_linkedin_health_accepts_claude_code_underscore_shim(self):
        """Claude Code requires '_' on tools with no required fields."""
        mcp = create_mcp_server()
        # Must not raise ValidationError for unexpected keyword '_'.
        result = await mcp.call_tool("linkedin_health", {"_": True})
        assert result.structured_content is not None
        assert result.structured_content["server"] == "linkedin-mcp"
        # ok depends on local browser/session state; only require a full payload.
        assert "bootstrap" in result.structured_content
        assert "session" in result.structured_content

    async def test_get_inbox_schema_path_accepts_underscore_shim(self):
        """Validation must not reject Claude Code's dummy '_' before scraping."""
        mcp = FastMCP("test")
        mcp.add_middleware(StripClientShimArgsMiddleware())

        @mcp.tool
        async def get_inbox(limit: int = 20) -> dict[str, int]:
            return {"limit": limit}

        result = await mcp.call_tool("get_inbox", {"_": True, "limit": 7})
        assert result.structured_content == {"limit": 7}

    async def test_alias_tool_accepts_underscore_shim(self):
        mcp = FastMCP("test")
        mcp.add_middleware(StripClientShimArgsMiddleware())

        @mcp.tool(name="get_inbox")
        async def get_inbox(limit: int = 20) -> dict[str, int]:
            return {"limit": limit}

        from fastmcp.tools.base import Tool

        tools = await mcp.list_tools(run_middleware=False)
        parent = next(t for t in tools if t.name == "get_inbox")
        mcp.add_tool(Tool.from_tool(parent, name="linkedin_get_inbox"))

        result = await mcp.call_tool("linkedin_get_inbox", {"_": False, "limit": 3})
        assert result.structured_content == {"limit": 3}

    async def test_still_rejects_unknown_non_shim_arguments(self):
        mcp = FastMCP("test")
        mcp.add_middleware(StripClientShimArgsMiddleware())

        @mcp.tool
        async def linkedin_health() -> dict[str, bool]:
            return {"ok": True}

        with pytest.raises(Exception):
            await mcp.call_tool(
                "linkedin_health",
                {"reason": "should still fail validation"},
            )
