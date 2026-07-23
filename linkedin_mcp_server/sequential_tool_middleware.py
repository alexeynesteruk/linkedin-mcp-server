"""Middleware that serializes MCP tool execution within one server process."""

from __future__ import annotations

import asyncio
import logging
import time

import mcp.types as mt

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult

from linkedin_mcp_server.tools.meta import META_TOOL_NAMES

logger = logging.getLogger(__name__)

# Patchright/Playwright phrases emitted when the browser/page/context dies.
# Matched as case-insensitive substrings so recovery works across Playwright
# versions, transports, and exception wrapping (ToolError, ExceptionGroup).
_BROWSER_CLOSED_MARKERS: tuple[str, ...] = (
    "target page, context or browser has been closed",
    "target closed",
    "browser has been closed",
    "context has been closed",
    "page has been closed",
    "connection closed while reading from the driver",
    "browser.new_context: target page, context or browser has been closed",
)


def _exception_text_chain(exc: BaseException, *, depth: int = 0) -> str:
    """Flatten exception message + cause/context/group members for matching."""
    if depth > 6:
        return ""
    parts = [str(exc)]
    cause = exc.__cause__
    if isinstance(cause, BaseException):
        parts.append(_exception_text_chain(cause, depth=depth + 1))
    ctx = exc.__context__
    if isinstance(ctx, BaseException) and ctx is not cause:
        parts.append(_exception_text_chain(ctx, depth=depth + 1))
    # Python 3.11+ ExceptionGroup
    exceptions = getattr(exc, "exceptions", None)
    if isinstance(exceptions, tuple):
        for nested in exceptions:
            if isinstance(nested, BaseException):
                parts.append(_exception_text_chain(nested, depth=depth + 1))
    return "\n".join(parts)


def _is_browser_context_closed(exc: BaseException) -> bool:
    """Return True if *exc* indicates the Patchright browser context has died."""
    text = _exception_text_chain(exc).lower()
    return any(marker in text for marker in _BROWSER_CLOSED_MARKERS)


class SequentialToolExecutionMiddleware(Middleware):
    """Ensure only one MCP tool call executes at a time per server process."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def _report_progress(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        *,
        message: str,
    ) -> None:
        fastmcp_context = context.fastmcp_context
        if fastmcp_context is None or fastmcp_context.request_context is None:
            return

        await fastmcp_context.report_progress(
            progress=0,
            total=100,
            message=message,
        )

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        tool_name = context.message.name
        if tool_name in META_TOOL_NAMES:
            return await call_next(context)

        wait_started = time.perf_counter()
        logger.debug("Waiting for scraper lock for tool '%s'", tool_name)
        await self._report_progress(
            context,
            message="Queued waiting for scraper lock",
        )

        async with self._lock:
            wait_seconds = time.perf_counter() - wait_started
            logger.debug(
                "Acquired scraper lock for tool '%s' after %.3fs",
                tool_name,
                wait_seconds,
            )
            await self._report_progress(
                context,
                message="Scraper lock acquired, starting tool",
            )
            hold_started = time.perf_counter()
            try:
                return await call_next(context)
            except Exception as exc:
                if _is_browser_context_closed(exc):
                    # The Patchright browser context died mid-operation.
                    # Reset the browser singleton so the next tool call gets a
                    # fresh context (cookies are safe on disk — no re-login needed).
                    logger.warning(
                        "Browser context closed during tool '%s' — resetting for next call",
                        tool_name,
                    )
                    try:
                        # Lazy import avoids a circular dependency at module load time.
                        from linkedin_mcp_server.drivers.browser import close_browser

                        await close_browser()
                    except Exception:
                        logger.debug(
                            "close_browser() failed during crash recovery",
                            exc_info=True,
                        )
                    raise ToolError(
                        "The browser context crashed mid-operation. "
                        "The browser has been reset — please retry this tool. "
                        "Your LinkedIn session is still active."
                    ) from exc
                raise
            finally:
                hold_seconds = time.perf_counter() - hold_started
                logger.debug(
                    "Released scraper lock for tool '%s' after %.3fs",
                    tool_name,
                    hold_seconds,
                )
