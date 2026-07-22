"""Middleware that tolerates known MCP client argument shims.

Claude Code (and some other MCP hosts) rewrite tool input schemas that have
no required fields so the model always sends a dummy boolean parameter named
``_``. FastMCP tools use ``additionalProperties: false``, so that dummy key
fails validation with ``Unexpected keyword argument`` even though the real
tool parameters are fine.

This middleware strips only the well-known ``_`` shim key before the call
reaches FastMCP validation. Real tool parameters are never named ``_``.
"""

from __future__ import annotations

import logging
from typing import Any

import mcp.types as mt

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult

logger = logging.getLogger(__name__)

# Claude Code injects this required boolean when a tool schema has no required
# fields. Keep the set tiny and explicit so we never swallow real args.
_CLIENT_SHIM_ARG_KEYS: frozenset[str] = frozenset({"_"})


def strip_client_shim_arguments(
    arguments: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return arguments without known client shim keys.

    Returns the original mapping when nothing needs stripping so callers can
    skip rebuilding the middleware context.
    """
    if not arguments:
        return arguments

    if _CLIENT_SHIM_ARG_KEYS.isdisjoint(arguments):
        return arguments

    cleaned = {
        key: value
        for key, value in arguments.items()
        if key not in _CLIENT_SHIM_ARG_KEYS
    }
    removed = sorted(_CLIENT_SHIM_ARG_KEYS.intersection(arguments))
    logger.debug("Stripped client shim argument(s) from tool call: %s", removed)
    return cleaned


class StripClientShimArgsMiddleware(Middleware):
    """Drop client-injected dummy args that break strict FastMCP schemas."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        original = context.message.arguments
        cleaned = strip_client_shim_arguments(original)
        if cleaned is original:
            return await call_next(context)

        message = context.message.model_copy(update={"arguments": cleaned})
        return await call_next(context.copy(message=message))
