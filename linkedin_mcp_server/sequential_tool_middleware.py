"""Middleware that serializes MCP tool execution across server processes."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import mcp.types as mt

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult

from linkedin_mcp_server.config import get_config
from linkedin_mcp_server.daemon_liveness import (
    UNKNOWN_OUTCOME_STATUS,
    abandoned_before_browser_work,
    abandoned_call_error,
    browser_work_begins,
)
from linkedin_mcp_server.exceptions import BrowserBusyError
from linkedin_mcp_server.profile_lease import ProfileLease, get_profile_lease

logger = logging.getLogger(__name__)


async def _reset_quietly(
    reset: Callable[..., Awaitable[str | None]], *args: BaseException
) -> str | None:
    """Run one of the driver's lost-browser resets; a failure is not a reset.

    The close it performs settles the profile in a ``finally`` and swallows its
    own teardown errors, so this is only a backstop. It still matters where it
    sits: on the failure path the tool's own error is what the client should get
    if the reset itself cannot run.
    """
    try:
        return await reset(*args)
    except Exception:
        logger.warning("Could not reset a lost browser", exc_info=True)
        return None


def _lost_browser_message(
    tool_name: str,
    reason: str,
    *,
    recovered: bool,
    could_change_something: bool,
    discarded_result: bool = False,
) -> str:
    """What a client is told when the browser died under its call."""
    lead = (
        f"The LinkedIn browser session was lost while {tool_name} was running "
        f"({reason})"
    )
    if not recovered:
        # The close could not prove Chromium exited, so the profile is kept and
        # nothing can launch on it until the process restarts.
        lead += (
            ", and the browser did not shut down cleanly, so it may still be "
            "running on the profile. Restart the server to recover."
        )
    elif discarded_result:
        lead += (
            ", so what the call read cannot be trusted and was discarded. The "
            "browser has been reset and the saved login was kept."
        )
    else:
        lead += ". The browser has been reset and the saved login was kept."
    if could_change_something:
        return (
            f"{lead} Whether the action reached LinkedIn is unknown. Check "
            "LinkedIn before calling again, because a repeat may perform the "
            "action a second time."
        )
    if not recovered:
        return lead
    return f"{lead} Retry the call; the next one starts a fresh browser."


#: Tools that never touch the browser and so skip both serialization layers.
#: ``tools/meta.py`` registers exactly these.
LOCK_FREE_TOOL_NAMES: frozenset[str] = frozenset({"linkedin_health", "linkedin_ping"})


class SequentialToolExecutionMiddleware(Middleware):
    """Ensure only one tool call at a time drives the shared LinkedIn browser.

    Two layers, because one is not enough:

    * an ``asyncio.Lock`` serializes calls inside this process, where several MCP
      sessions can share one server;
    * the profile lease serializes calls across processes, where each MCP client
      instance spawns its own server against the same Chromium profile.

    Without the second layer two processes open that profile simultaneously and
    the last one to close silently overwrites the other's cookies.
    """

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
        if tool_name in LOCK_FREE_TOOL_NAMES:
            # Meta tools read local state only. Queueing them behind a scrape,
            # or behind another process's lease, would make a health probe
            # report "busy" exactly when it is needed.
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
            return await self._run_owning_the_profile(context, call_next, tool_name)

    async def _run_owning_the_profile(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
        tool_name: str,
    ) -> ToolResult:
        """Run the tool while this process owns the browser profile."""
        # Imported here so the module stays importable without the driver.
        from linkedin_mcp_server.drivers.browser import (
            note_activity,
            note_call_started,
            release_profile_if_idle_or_requested,
            reset_browser_if_lost,
            reset_browser_lost_in,
        )

        lease = get_profile_lease()
        acquired = lease.try_acquire()
        if not acquired:
            await self._report_progress(
                context,
                message=(
                    "Another LinkedIn MCP client is using the browser; "
                    "waiting for it to hand over"
                ),
            )
            budget = get_config().browser.browser_wait_seconds
            acquired = await lease.acquire(timeout=budget)

        if not acquired:
            # Raised as a ToolError here, not via error_handler: an exception
            # thrown in middleware does not pass through raise_tool_error, and
            # mask_error_details would otherwise hide the explanation.
            logger.info("Tool '%s' gave up waiting for the shared browser", tool_name)
            raise ToolError(str(BrowserBusyError()))

        # After every wait this call can spend queued, and before the browser is
        # touched: the last point at which an owner can still decline work its
        # client has given up on. Nothing awaits between here and the body.
        # Only the reference taken above is returned, and the browser-call count
        # is left alone, because the body never began.
        if abandoned_before_browser_work():
            lease.release()
            logger.info("Tool '%s' was abandoned before it began", tool_name)
            raise abandoned_call_error()
        # From here on the call may act, so an owner that has to cut it off
        # reports its outcome as unknown rather than as a call that never ran.
        browser_work_begins()

        hold_started = time.perf_counter()
        try:
            # Marks the browser as in use so the background handoff poll cannot
            # close it out from under this call. Inside the try so the finally
            # always balances it, including if the call is cancelled.
            note_call_started()
            try:
                result = await call_next(context)
            except Exception as exc:
                # A browser that died under this call would fail every call after
                # it, so it is closed here, before the lease is released. Never
                # replayed: the call may have acted before the browser went.
                lost = await _reset_quietly(reset_browser_lost_in, exc)
                if lost is None:
                    raise
                return await self._answer_a_lost_browser(
                    context, tool_name, lost, lease, cause=exc
                )
            # A call can also *return* over a dead browser: a helper that
            # swallows its errors reads an empty page as an answer. Asked of
            # Patchright's state only, so a healthy browser costs nothing here.
            lost = await _reset_quietly(reset_browser_if_lost)
            if lost is not None:
                return await self._answer_a_lost_browser(
                    context, tool_name, lost, lease, returned=result
                )
            return result
        finally:
            hold_seconds = time.perf_counter() - hold_started
            logger.debug(
                "Released scraper lock for tool '%s' after %.3fs",
                tool_name,
                hold_seconds,
            )
            note_activity()
            lease.release()
            # Hand the browser over now if someone is waiting, rather than
            # holding it for the rest of this process's lifetime.
            try:
                await release_profile_if_idle_or_requested()
            except Exception:
                logger.debug("Profile handoff check failed", exc_info=True)

    async def _answer_a_lost_browser(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        tool_name: str,
        reason: str,
        lease: ProfileLease,
        *,
        cause: BaseException | None = None,
        returned: ToolResult | None = None,
    ) -> ToolResult:
        """Answer a call whose browser was lost, after it has been closed.

        Which answer depends on whether running the tool again could repeat an
        effect, read from its annotations the way every replay decision here is
        (``daemon_auth.a_repeat_could_change_something``):

        * a read is safe to repeat, so it fails with an error that says so. A
          result it *returned* is discarded too: it was read from a page that
          died mid-read, and an empty page is exactly what that looks like.
        * anything else may have acted before the browser went, so the client is
          told the outcome is unknown, in the ``outcome_unknown`` shape the
          daemon uses for the same uncertainty. A result such a tool returned is
          its own account of what it saw, and is kept.

        *recovered* comes from the lease rather than from the close: a close
        that could not prove Chromium exited keeps the profile marked open, and
        then no call can succeed until the process restarts.
        """
        from linkedin_mcp_server.daemon_auth import a_repeat_could_change_something

        could_change_something = await a_repeat_could_change_something(context)
        if returned is not None and could_change_something:
            return returned
        recovered = not lease.browser_open
        message = _lost_browser_message(
            tool_name,
            reason,
            recovered=recovered,
            could_change_something=could_change_something,
            discarded_result=returned is not None,
        )
        logger.warning("Tool '%s' lost its browser: %s", tool_name, reason)
        if could_change_something:
            answer = {
                "status": UNKNOWN_OUTCOME_STATUS,
                "message": message,
                "retry_safe": False,
            }
            # Is-error for the same reason the daemon's own unknown outcome is:
            # a client reading no structured content still sees a failure, and
            # an error result is not validated against the tool's output schema.
            return ToolResult(
                content=[mt.TextContent(type="text", text=message)],
                structured_content=answer,
                is_error=True,
            )
        raise ToolError(message) from cause
