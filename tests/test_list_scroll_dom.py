"""Browser-DOM tests for the lazy-list scroll used by the skills page.

``page.evaluate`` is a mock in the unit suite, so ``SCROLL_LIST_JS`` never
executes there. These run ``scroll_list_until_stable`` against synthetic lists
that append their next batch when the real scroller nears its end, for each
place LinkedIn has been seen to put that scroller: a container inside
``<main>`` and the document itself.

Skipped automatically when chromium is not installed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from patchright.async_api import async_playwright

from linkedin_mcp_server.scraping.list_scroll import scroll_list_until_stable
from linkedin_mcp_server.scraping.session import ScrapingSession

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

TOTAL = 40
BATCH = 10

# Appends BATCH rows whenever the chosen scroller is within 200px of its end.
_LAZY_LIST = """
<style>
  body {{ margin: 0; }}
  main {{ {main_style} }}
  li {{ height: 100px; margin: 0; }}
</style>
<main><ul id="list"></ul>
  <button id="decoy">Show all 3 details</button></main>
<script>
  const TOTAL = {total}, BATCH = {batch};
  let count = 0;
  const list = document.getElementById('list');
  function addBatch() {{
    for (let i = 0; i < BATCH && count < TOTAL; i++, count++) {{
      const li = document.createElement('li');
      li.textContent = 'Skill ' + count;
      list.appendChild(li);
    }}
  }}
  addBatch();
  const scroller = {scroller};
  const target = scroller === document ? window : scroller;
  target.addEventListener('scroll', () => {{
    const el = scroller === document ? document.documentElement : scroller;
    if (el.scrollTop + el.clientHeight > el.scrollHeight - 200) setTimeout(addBatch, 50);
  }});
    document.getElementById('decoy').addEventListener('click', () => document.body.dataset.clicked = '1');
</script>
"""

_CONTAINER = _LAZY_LIST.format(
    main_style="display:block;height:400px;overflow-y:auto;",
    total=TOTAL,
    batch=BATCH,
    scroller="document.querySelector('main')",
)
_DOCUMENT = _LAZY_LIST.format(
    main_style="display:block;", total=TOTAL, batch=BATCH, scroller="document"
)


# The scroller sits in the left 100px, so a wheel over the viewport centre never
# reaches it: only scrolling the element itself loads the list.
_OFF_CENTRE = _CONTAINER.replace(
    "display:block;height:400px;overflow-y:auto;",
    "display:block;height:400px;width:100px;overflow-y:auto;",
)

# An SDUI-style list that ignores scrollTop and appends only on a real wheel
# event over it; the wheel is the only thing that loads it.
_WHEEL_ONLY = _DOCUMENT.replace(
    "target.addEventListener('scroll', () => {",
    "window.addEventListener('wheel', () => setTimeout(addBatch, 50));\n"
    "  target.addEventListener('nomatch', () => {",
)
assert "'wheel'" in _WHEEL_ONLY


@pytest.fixture
async def dom_page():
    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(
                channel="chromium", headless=True
            )
            page = await browser.new_page()
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            yield page
        finally:
            await browser.close()


async def _real_delay(_self: ScrapingSession, seconds: float) -> None:
    import asyncio

    await asyncio.sleep(min(seconds, 0.15))


async def _run(page, rounds: int | None = None) -> int:
    with patch.object(ScrapingSession, "delay", _real_delay):
        await scroll_list_until_stable(ScrapingSession(page), rounds)
    return await page.evaluate("document.querySelectorAll('li').length")


@pytest.mark.parametrize(
    "html",
    [_CONTAINER, _DOCUMENT, _OFF_CENTRE, _WHEEL_ONLY],
    ids=["container", "document", "scroll-only", "wheel-only"],
)
async def test_the_whole_list_loads_whichever_element_scrolls(dom_page, html):
    await dom_page.set_content(html)
    assert await dom_page.evaluate("document.querySelectorAll('li').length") == BATCH

    assert await _run(dom_page) == TOTAL


async def test_the_round_budget_bounds_a_list_that_keeps_growing(dom_page):
    await dom_page.set_content(_CONTAINER)

    loaded = await _run(dom_page, 2)

    assert BATCH < loaded < TOTAL


async def test_scrolling_never_clicks_a_button(dom_page):
    await dom_page.set_content(_CONTAINER)

    await _run(dom_page)

    assert await dom_page.evaluate("document.body.dataset.clicked") is None
