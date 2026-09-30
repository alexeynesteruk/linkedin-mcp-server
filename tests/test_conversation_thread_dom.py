"""Browser-DOM tests for the wait that lets a conversation thread render.

On a ``/messaging/thread/<id>/`` page the inbox sidebar fills ``<main>`` with
text before the thread pane has fetched a single message, so the text-length
wait passes on a thread that is still empty and the read returns nothing or
the first turn only (reproduced live on the fork, 2026-07-10: one turn
without a wait, seven with one). These cases run the real wait program in
headless chromium against a sidebar that is there at once and messages that
mount later, the order LinkedIn renders them in.

Skipped automatically when chromium is not installed; run locally after
``uv run patchright install chromium --no-shell``.
"""

from __future__ import annotations

from typing import Any

import time

import pytest
from patchright.async_api import Page, async_playwright

from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.conversations import ConversationReader
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.profile_page import ProfilePageReader
from linkedin_mcp_server.scraping.session import ScrapingSession

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

SIDEBAR = """
<div class="sidebar">
  <ul>
    <li>Ada Lovelace: Are you free on Thursday for the engine review?</li>
    <li>Grace Hopper: Thanks for the notes from yesterday's session.</li>
    <li>Load more conversations</li>
  </ul>
</div>
"""


def _thread_page(*, message_delay_ms: int | None) -> str:
    """A thread page whose messages mount ``message_delay_ms`` after load.

    ``None`` renders a thread that never gets a message.
    """
    mount = (
        ""
        if message_delay_ms is None
        else f"""
<script>
  setTimeout(() => {{
    const pane = document.getElementById('thread');
    for (const text of ['Hello Ada!', 'Hi, see you Thursday.']) {{
      const item = document.createElement('div');
      item.setAttribute('data-view-name', 'message-list-item');
      item.textContent = text;
      pane.appendChild(item);
    }}
  }}, {message_delay_ms});
</script>
"""
    )
    return (
        f"<!DOCTYPE html><html><body><main>{SIDEBAR}"
        f'<section id="thread"></section></main>{mount}</body></html>'
    )


async def _no_message_target() -> Any:
    raise AssertionError("the conversation reader never reads a message target")


def _reader(page: Page) -> ConversationReader:
    session = ScrapingSession(page)
    return ConversationReader(
        session,
        PageNavigator(session),
        PageContentReader(session),
        ProfilePageReader(session, _no_message_target),
    )


@pytest.fixture
async def dom_page():
    """Real chromium page, or skip when no browser is installed."""
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(channel="chromium", headless=True)
            page = await browser.new_page()
        except Exception as exc:  # browser binary missing
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            yield page
        finally:
            await browser.close()


async def _main_text(page: Page) -> str:
    return await page.evaluate("() => document.querySelector('main').innerText")


async def test_the_sidebar_alone_satisfies_the_text_length_wait(dom_page):
    """Why the wait exists: the old gate passes before any message is there."""
    await dom_page.set_content(_thread_page(message_delay_ms=800))
    reader = _reader(dom_page)

    await reader._wait_for_main_text(log_context="Conversation")

    assert "Hello Ada!" not in await _main_text(dom_page)


async def test_the_wait_returns_once_the_thread_has_a_message(dom_page):
    await dom_page.set_content(_thread_page(message_delay_ms=800))
    reader = _reader(dom_page)

    await reader._wait_for_main_text(log_context="Conversation")
    await reader._wait_for_thread_message()

    text = await _main_text(dom_page)
    assert "Hello Ada!" in text
    assert "Hi, see you Thursday." in text


async def test_sidebar_rows_are_not_mistaken_for_messages(dom_page):
    """A thread that never renders a message holds the wait to its timeout,
    however much sidebar text is on the page, and then lets the read go on."""
    await dom_page.set_content(_thread_page(message_delay_ms=None))
    reader = _reader(dom_page)

    started = time.monotonic()
    await reader._wait_for_thread_message(timeout=700)
    elapsed = time.monotonic() - started

    assert elapsed >= 0.6
    assert "Load more conversations" in await _main_text(dom_page)
