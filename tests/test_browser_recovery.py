"""A browser that died is replaced, and the call that met it says so.

Three layers, each with doubles here and together against real Chromium in
``test_browser_recovery_chromium.py``:

* ``BrowserManager`` notices a closed, crashed or disconnected page;
* ``get_or_create_browser`` never hands such a browser out again;
* the serializing middleware closes it under the call that met it, answers that
  call without replaying it, and leaves the profile lease balanced.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from patchright._impl._errors import TargetClosedError

import linkedin_mcp_server.drivers.browser as drv
from linkedin_mcp_server.core.browser import BrowserManager
from linkedin_mcp_server.core.exceptions import NetworkError
from linkedin_mcp_server.exceptions import BrowserBusyError
from linkedin_mcp_server.profile_lease import get_profile_lease
from linkedin_mcp_server.sequential_tool_middleware import (
    SequentialToolExecutionMiddleware,
)

READ = {"readOnlyHint": True}
WRITE = {"destructiveHint": True}


class _Emitter:
    """The listener half of a Patchright object: ``on`` and an ``emit``."""

    def __init__(self) -> None:
        self.listeners: dict[str, list[Any]] = {}

    def on(self, event: str, handler: Any) -> None:
        self.listeners.setdefault(event, []).append(handler)

    def emit(self, event: str) -> None:
        for handler in self.listeners.get(event, []):
            handler(self)


class _Page(_Emitter):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed


class _PatchrightBrowser(_Emitter):
    def __init__(self) -> None:
        super().__init__()
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected


class _Context(_Emitter):
    def __init__(self) -> None:
        super().__init__()
        self.browser = _PatchrightBrowser()


def _watched_manager() -> tuple[BrowserManager, _Context, _Page]:
    """A manager holding a page the way ``start()`` leaves it."""
    manager = BrowserManager()
    context, page = _Context(), _Page()
    manager._context = context  # ty: ignore[invalid-assignment]
    manager._page = page  # ty: ignore[invalid-assignment]
    manager._watch_for_loss(context, page)  # ty: ignore[invalid-argument-type]
    return manager, context, page


class TestTheManagerNoticesALoss:
    def test_a_live_page_is_not_lost(self):
        manager, _, _ = _watched_manager()
        assert manager.lost_reason() is None

    def test_a_crash_is_noticed_although_the_page_reports_open(self):
        # The case only the event can see: a crashed renderer leaves the page
        # "open", and every call on it fails.
        manager, _, page = _watched_manager()
        page.emit("crash")
        assert page.is_closed() is False
        assert manager.lost_reason() == "the page crashed"

    def test_a_closed_page_is_noticed(self):
        manager, _, page = _watched_manager()
        page.emit("close")
        assert manager.lost_reason() == "the page was closed"

    def test_a_closed_context_is_noticed(self):
        manager, context, _ = _watched_manager()
        context.emit("close")
        assert manager.lost_reason() == "the browser context was closed"

    def test_a_disconnected_browser_is_noticed(self):
        manager, context, _ = _watched_manager()
        context.browser.emit("disconnected")
        assert manager.lost_reason() == "the browser disconnected"

    def test_the_state_answers_when_no_event_arrived(self):
        manager, context, page = _watched_manager()
        page.closed = True
        assert manager.lost_reason() == "the page was closed"
        page.closed = False
        context.browser.connected = False
        assert manager.lost_reason() == "the browser disconnected"

    def test_the_first_reason_is_kept(self):
        manager, context, page = _watched_manager()
        page.emit("crash")
        context.emit("close")
        assert manager.lost_reason() == "the page crashed"

    def test_its_own_teardown_is_not_a_loss(self):
        # `close()` clears the page before it closes anything, so the close
        # events that teardown produces answer for nobody.
        manager, context, page = _watched_manager()
        manager._page = None
        page.emit("close")
        context.emit("close")
        assert manager._lost is None

    def test_a_new_launch_forgets_the_previous_loss(self):
        manager, _, page = _watched_manager()
        page.emit("crash")
        manager._close_proven = True
        manager._begin_a_launch()
        assert manager._lost is None

    def test_a_listener_that_cannot_be_attached_does_not_refuse_the_browser(self):
        manager = BrowserManager()
        page = MagicMock(spec=["is_closed"])  # no `on`
        manager._watch_for_loss(MagicMock(), page)  # does not raise


class _Singleton:
    """A driver-level browser double: how it reports itself, how its close went."""

    def __init__(self, *, lost: str | None = None, close_proves: bool = True):
        self.lost = lost
        self.close_proves = close_proves
        self.closes = 0
        self.is_authenticated = True
        self.page = MagicMock(name="page")

    def lost_reason(self) -> str | None:
        return self.lost

    async def close(self) -> bool:
        self.closes += 1
        return self.close_proves

    async def export_cookies(self, _path: Any) -> bool:
        return False


@pytest.fixture
def released(monkeypatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(drv, "release_browser_guardian", lambda: calls.append("go"))
    return calls


def _install(browser: _Singleton):
    """Make *browser* the singleton the way ``_create_browser`` leaves one."""
    lease = get_profile_lease()
    assert lease.try_acquire()
    lease.mark_browser_open()
    drv._browser = browser  # ty: ignore[invalid-assignment]
    drv._browser_lease = lease
    return lease


def _replacement(monkeypatch, seen: dict[str, Any]) -> _Singleton:
    """Stand in for the launch, recording what the old browser left behind."""
    fresh = _Singleton()

    async def create() -> Any:
        lease = get_profile_lease()
        seen["old_singleton"] = drv._browser
        seen["browser_open"] = lease.browser_open
        drv._browser = fresh  # ty: ignore[invalid-assignment]
        return fresh

    monkeypatch.setattr(drv, "_create_browser", create)
    return fresh


class TestGetOrCreateBrowser:
    async def test_a_live_browser_is_handed_out(self, monkeypatch, released):
        live = _Singleton()
        _install(live)
        seen: dict[str, Any] = {}
        _replacement(monkeypatch, seen)

        assert await drv.get_or_create_browser() is live
        assert live.closes == 0
        assert seen == {}

    async def test_a_lost_browser_is_closed_and_replaced(self, monkeypatch, released):
        dead = _Singleton(lost="the page crashed")
        lease = _install(dead)
        seen: dict[str, Any] = {}
        fresh = _replacement(monkeypatch, seen)

        assert await drv.get_or_create_browser() is fresh

        assert dead.closes == 1
        # Settled by the ordinary close before anything launched: nothing was
        # left on the profile for the new browser to share it with.
        assert seen == {"old_singleton": None, "browser_open": False}
        assert released == ["go"]
        assert drv._browser_lease is None
        assert lease.held is False

    async def test_an_unproved_close_refuses_the_relaunch(self, monkeypatch, released):
        # The real launch path, which is what refuses: Chromium may still be on
        # the profile, and a second one there is the corruption the lease exists
        # to prevent.
        dead = _Singleton(lost="the page crashed", close_proves=False)
        lease = _install(dead)

        with pytest.raises(BrowserBusyError, match="Restart the server"):
            await drv.get_or_create_browser()

        assert dead.closes == 1
        assert lease.browser_open is True
        assert lease.held is True
        assert released == []

    async def test_a_browser_that_cannot_answer_is_kept(self, monkeypatch, released):
        # Not knowing is not evidence: a relaunch costs a LinkedIn request.
        live = _Singleton()
        live.lost_reason = MagicMock(side_effect=RuntimeError("no answer"))
        _install(live)

        assert await drv.get_or_create_browser() is live
        assert live.closes == 0


class TestTheResets:
    async def test_an_error_reporting_a_loss_closes_the_browser(self, released):
        browser = _Singleton()
        _install(browser)

        reason = await drv.reset_browser_lost_in(TargetClosedError())

        assert reason == "the page, context or browser was closed"
        assert browser.closes == 1
        assert drv._browser is None

    async def test_an_ordinary_error_closes_nothing(self, released):
        # Even over a browser that reports itself dead: the error is what the
        # call is answered with, and it is not about the browser.
        browser = _Singleton(lost="the page crashed")
        _install(browser)

        assert await drv.reset_browser_lost_in(RuntimeError("selector moved")) is None
        assert browser.closes == 0

    async def test_a_launch_failure_is_not_a_lost_session(self, released):
        # No browser was ever handed out, so the same text means Chromium could
        # not start, and that is not something a reset repairs.
        failed = NetworkError(
            "Failed to start browser: Target page, context or browser has been closed"
        )
        assert await drv.reset_browser_lost_in(failed) is None

    async def test_a_browser_reporting_itself_lost_is_closed(self, released):
        browser = _Singleton(lost="the page was closed")
        _install(browser)

        assert await drv.reset_browser_if_lost() == "the page was closed"
        assert browser.closes == 1

    async def test_a_live_browser_is_left_alone(self, released):
        browser = _Singleton()
        _install(browser)

        assert await drv.reset_browser_if_lost() is None
        assert browser.closes == 0


class TestTheMiddleware:
    """Through a real server and client, so masking and annotations are real."""

    @staticmethod
    def _server() -> FastMCP:
        mcp = FastMCP("test")
        mcp.add_middleware(SequentialToolExecutionMiddleware())
        return mcp

    async def test_a_read_that_meets_a_closed_page_asks_for_a_retry(self, released):
        browser = _Singleton()
        lease = _install(browser)
        mcp = self._server()
        runs: list[str] = []

        @mcp.tool(annotations=READ)
        async def get_feed() -> dict[str, Any]:
            runs.append("ran")
            raise TargetClosedError()

        async with Client(mcp) as client:
            with pytest.raises(ToolError) as raised:
                await client.call_tool("get_feed", {})

        message = str(raised.value)
        assert "was lost while get_feed was running" in message
        assert "Retry the call" in message
        assert runs == ["ran"], "the call was replayed"
        assert browser.closes == 1
        assert drv._browser is None
        assert lease.browser_open is False
        assert lease.held is False, "the profile was not released after the call"

    async def test_the_next_call_gets_a_fresh_browser(self, monkeypatch, released):
        _install(_Singleton())
        mcp = self._server()
        seen: dict[str, Any] = {}

        @mcp.tool(annotations=READ)
        async def get_feed() -> dict[str, Any]:
            browser = await drv.get_or_create_browser()
            if not seen:
                seen["first"] = browser
                raise TargetClosedError()
            return {"fresh": browser is not seen["first"]}

        _replacement(monkeypatch, {})
        async with Client(mcp) as client:
            with pytest.raises(ToolError):
                await client.call_tool("get_feed", {})
            again = await client.call_tool("get_feed", {})

        assert again.structured_content == {"fresh": True}

    async def test_a_write_that_meets_a_dead_browser_reports_an_unknown_outcome(
        self, released
    ):
        browser = _Singleton()
        _install(browser)
        mcp = self._server()
        runs: list[str] = []

        @mcp.tool(annotations=WRITE)
        async def send_message() -> dict[str, Any]:
            runs.append("ran")
            raise TargetClosedError()

        async with Client(mcp) as client:
            result = await client.call_tool("send_message", {}, raise_on_error=False)

        assert result.is_error is True
        assert result.structured_content["status"] == "outcome_unknown"
        assert result.structured_content["retry_safe"] is False
        assert (
            "Check LinkedIn before calling again"
            in (result.structured_content["message"])
        )
        assert "Retry the call" not in result.structured_content["message"]
        assert runs == ["ran"], "a write was replayed"
        assert browser.closes == 1

    async def test_a_read_that_returns_over_a_dead_page_is_discarded(self, released):
        # A helper that swallows its errors reads a dead page as an empty one.
        browser = _Singleton()
        _install(browser)
        mcp = self._server()

        @mcp.tool(annotations=READ)
        async def get_pending_invitations() -> dict[str, Any]:
            browser.lost = "the page crashed"
            return {"url": "x", "sections": {}}

        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="discarded"):
                await client.call_tool("get_pending_invitations", {})

        assert browser.closes == 1

    async def test_a_write_that_returns_over_a_dead_page_keeps_its_answer(
        self, released
    ):
        # Its result is its own account of what it saw, possibly "sent".
        browser = _Singleton()
        _install(browser)
        mcp = self._server()

        @mcp.tool(annotations=WRITE)
        async def send_message() -> dict[str, Any]:
            browser.lost = "the page was closed"
            return {"status": "send_unconfirmed", "retry_safe": False}

        async with Client(mcp) as client:
            result = await client.call_tool("send_message", {})

        assert result.structured_content == {
            "status": "send_unconfirmed",
            "retry_safe": False,
        }
        assert browser.closes == 1

    async def test_an_unrelated_failure_keeps_its_own_answer(self, released):
        browser = _Singleton(lost="the page crashed")
        _install(browser)
        mcp = self._server()

        @mcp.tool(annotations=READ)
        async def get_feed() -> dict[str, Any]:
            raise ToolError("LinkedIn login in progress")

        async with Client(mcp) as client:
            with pytest.raises(ToolError) as raised:
                await client.call_tool("get_feed", {})

        assert str(raised.value) == "LinkedIn login in progress"
        assert browser.closes == 0

    async def test_an_unproved_close_asks_for_a_restart(self, released):
        browser = _Singleton(close_proves=False)
        lease = _install(browser)
        mcp = self._server()

        @mcp.tool(annotations=READ)
        async def get_feed() -> dict[str, Any]:
            raise TargetClosedError()

        async with Client(mcp) as client:
            with pytest.raises(ToolError) as raised:
                await client.call_tool("get_feed", {})

        assert "Restart the server" in str(raised.value)
        assert "Retry the call" not in str(raised.value)
        assert lease.browser_open is True

    async def test_a_healthy_call_is_untouched(self, released):
        browser = _Singleton()
        _install(browser)
        mcp = self._server()

        @mcp.tool(annotations=READ)
        async def get_feed() -> dict[str, Any]:
            return {"sections": {"feed": "text"}}

        async with Client(mcp) as client:
            result = await client.call_tool("get_feed", {})

        assert result.structured_content == {"sections": {"feed": "text"}}
        assert browser.closes == 0

    async def test_a_cancelled_call_is_not_taken_for_a_lost_browser(self, released):
        # Cancellation is how the owner cuts a call off and how an abandoned
        # one ends; it has to reach the liveness layer as itself.
        browser = _Singleton(lost="the page crashed")
        _install(browser)
        context = MagicMock()
        context.message.name = "get_feed"
        context.fastmcp_context = None

        async def call_next(_context: Any) -> Any:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await SequentialToolExecutionMiddleware().on_call_tool(context, call_next)

        assert browser.closes == 0


class TestOnADaemonOwner:
    async def test_a_dead_browser_is_an_unknown_outcome_and_the_books_balance(
        self, monkeypatch, released
    ):
        """The owner answers through its liveness layer without replaying.

        No tool annotations are readable here, which is the cautious reading:
        the frontend is told the outcome is unknown, never to repeat blindly.
        """
        from linkedin_mcp_server import daemon_liveness
        from linkedin_mcp_server.daemon_liveness import (
            CALL_HEADER,
            OwnerCallLivenessMiddleware,
            new_call_id,
        )

        marker = new_call_id()
        monkeypatch.setattr(
            "fastmcp.server.dependencies.get_http_headers",
            lambda **_kw: {CALL_HEADER: marker},
        )
        browser = _Singleton()
        lease = _install(browser)
        sequential = SequentialToolExecutionMiddleware()
        runs: list[str] = []

        async def tool(_context: Any) -> Any:
            runs.append("ran")
            raise TargetClosedError()

        async def call_next(context: Any) -> Any:
            return await sequential.on_call_tool(context, tool)

        context = MagicMock()
        context.message.name = "connect_with_person"
        context.fastmcp_context = None

        result = await OwnerCallLivenessMiddleware().on_call_tool(context, call_next)

        assert result.is_error is True
        assert result.structured_content["status"] == "outcome_unknown"
        assert result.structured_content["retry_safe"] is False
        assert runs == ["ran"]
        liveness = daemon_liveness.get_liveness()
        assert liveness.calls_in_flight() == 0
        assert liveness._waiting == {}
        assert lease.held is False
        assert browser.closes == 1
