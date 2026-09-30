"""A real Chromium dies under the server, and the next call gets a new one.

The doubles in ``test_browser_recovery.py`` prove each layer's decision. This
file proves the premise they rest on, which no double can: that Patchright
really reports each of these losses the way the code reads them, and that the
ordinary close and relaunch really bring back a working page on the same
profile.

The browser is the product's own (``BrowserManager`` through
``get_or_create_browser``, with the launch options the server uses), on a
temporary profile. LinkedIn is never contacted: the /feed/ check is replaced,
and every page here is ``about:blank``.

Marked ``browser_dom`` for the browser cache that marker is given
(``conftest.installed_browser_for_dom_tests``), and skipped where no Chromium is
installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

import linkedin_mcp_server.drivers.browser as drv
from linkedin_mcp_server.profile_lease import get_profile_lease
from linkedin_mcp_server.sequential_tool_middleware import (
    SequentialToolExecutionMiddleware,
)
from linkedin_mcp_server.session_state import portable_cookie_path, write_source_state

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]


@pytest.fixture
async def real_browser(profile_dir, monkeypatch):
    """The server's browser on a temporary profile, closed however the test ends."""
    write_source_state(profile_dir)
    portable_cookie_path(profile_dir).write_text(json.dumps([]))

    async def feed_is_fine(*_args: Any, **_kwargs: Any) -> bool:
        return True

    # The only step that would reach LinkedIn. Everything else is the product.
    monkeypatch.setattr(drv, "_feed_auth_succeeds", feed_is_fine)
    # The crash guardian is a detached process; these tests are about the page.
    monkeypatch.setattr(drv, "start_browser_guardian", lambda _fd: None)
    monkeypatch.setattr(drv, "release_browser_guardian", lambda: None)

    try:
        first = await drv.get_or_create_browser()
    except Exception as exc:  # browser binary missing
        pytest.skip(f"chromium unavailable: {exc}")
    try:
        yield first
    finally:
        await drv.close_browser()


async def _works(browser: Any) -> bool:
    return await browser.page.evaluate("() => 1 + 1") == 2


async def _until(predicate, *, seconds: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + seconds
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("the condition never became true")
        await asyncio.sleep(0.05)


async def _assert_replaced(dead: Any) -> None:
    fresh = await drv.get_or_create_browser()
    assert fresh is not dead
    assert await _works(fresh)
    assert fresh.lost_reason() is None
    # The dead one went through the ordinary close and proved Chromium gone,
    # which is the only thing that let the new one open the same profile.
    assert dead.close_confirmed or dead._close_proven
    lease = get_profile_lease()
    assert lease.browser_open is True
    assert lease.held is True


class TestTheNextCallRecovers:
    async def test_after_the_page_is_closed(self, real_browser):
        # What a user closing a headed window does.
        assert await _works(real_browser)
        await real_browser.page.close()
        assert real_browser.lost_reason() == "the page was closed"

        await _assert_replaced(real_browser)

    async def test_after_the_renderer_crashes(self, real_browser):
        # An out-of-memory kill, or a Mac that slept. The page stays "open".
        page = real_browser.page
        # How Playwright's own crash tests do it. The navigation is rejected by
        # the crash itself ("page crashed"), so it ends rather than hangs; a raw
        # CDP `Page.crash` never answers once its target is gone.
        with contextlib.suppress(Exception):
            await page.goto("chrome://crash", timeout=10_000)
        await _until(lambda: real_browser.lost_reason() is not None)

        assert real_browser.lost_reason() == "the page crashed"
        assert page.is_closed() is False, "only the crash event can see this"
        await _assert_replaced(real_browser)

    async def test_after_the_context_is_closed(self, real_browser):
        await real_browser.context.close()
        assert real_browser.lost_reason() is not None

        await _assert_replaced(real_browser)


class TestThroughTheMiddleware:
    async def test_a_context_closed_mid_call_resets_and_the_retry_works(
        self, real_browser
    ):
        mcp = FastMCP("test")
        mcp.add_middleware(SequentialToolExecutionMiddleware())
        pages: list[Any] = []

        @mcp.tool(annotations={"readOnlyHint": True})
        async def read_page() -> dict[str, Any]:
            browser = await drv.get_or_create_browser()
            pages.append(browser)
            if len(pages) == 1:
                # The browser dies between two steps of the same call.
                await browser.context.close()
            return {"sum": await browser.page.evaluate("() => 1 + 1")}

        async with Client(mcp) as client:
            with pytest.raises(ToolError) as raised:
                await client.call_tool("read_page", {})
            assert "Retry the call" in str(raised.value)
            assert drv._browser is None, "the dead browser was kept"

            again = await client.call_tool("read_page", {})

        assert again.structured_content == {"sum": 2}
        assert pages[1] is not pages[0]
        # Only the live browser's own reference is left: the middleware's was
        # released after each call, the dead browser's by its close.
        lease = get_profile_lease()
        assert drv._browser_lease is lease
        assert lease._refs == 1
