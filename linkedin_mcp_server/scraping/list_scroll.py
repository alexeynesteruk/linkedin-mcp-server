"""Scrolling for lazy lists whose scroller is not always the document."""

from __future__ import annotations

import logging

from linkedin_mcp_server.scraping.comment_thread import MAIN_TEXT_LENGTH_JS
from linkedin_mcp_server.scraping.session import ScrapingSession

logger = logging.getLogger(__name__)

# The invitation list lazy-loads as its scroller nears the bottom, and that
# scroller is <main>, not the document: measured on the sent manager
# (2026-08-25), window.scrollTo moved nothing and scrolling main's scrollTop
# loaded the rest. The largest scrollable element in main is scrolled, the
# way the inbox is, and the document too in case a layout scrolls it instead.
SCROLL_LIST_JS = r"""
() => {
  const main = document.querySelector('main');
  if (main) {
    const isScrollable = element => {
      const style = window.getComputedStyle(element);
      return (
        (style.overflowY === 'auto' || style.overflowY === 'scroll') &&
        element.scrollHeight > element.clientHeight + 20
      );
    };
    const candidates = [main, ...main.querySelectorAll('*')].filter(isScrollable);
    const target = candidates.sort(
      (left, right) => right.scrollHeight - left.scrollHeight
    )[0];
    if (target) target.scrollTop = target.scrollHeight;
  }
  window.scrollTo(0, document.body.scrollHeight);
  return true;
}
"""

DEFAULT_LIST_ROUNDS = 25
_STALE_ROUNDS = 3
_WHEEL_DELTA = 2500
_PAUSE = 1.3


async def _main_text_length(session: ScrapingSession) -> int | None:
    try:
        return int(await session.page.evaluate(MAIN_TEXT_LENGTH_JS))
    except Exception:
        logger.debug("List length probe failed", exc_info=True)
        return None


async def scroll_list_until_stable(
    session: ScrapingSession, max_rounds: int | None = None
) -> None:
    """Scroll a lazy list until main's text stops growing.

    Each round scrolls the largest scrollable element in ``<main>`` and the
    document, and also wheels over the viewport centre: LinkedIn's SDUI lists
    (the skills page among them) append items only on a real wheel event, and
    which element scrolls differs by layout. The text length is the
    locale-independent progress signal; three rounds without growth, a failed
    probe or the round budget end the loop.
    """
    rounds = max_rounds if max_rounds is not None else DEFAULT_LIST_ROUNDS
    if rounds <= 0:
        return
    page = session.page
    viewport = page.viewport_size or {"width": 1280, "height": 720}
    can_wheel = True
    try:
        await page.mouse.move(viewport["width"] // 2, viewport["height"] // 2)
    except Exception:
        logger.debug("Mouse move over list failed", exc_info=True)
        can_wheel = False

    previous: int | None = None
    stale = 0
    for round_number in range(rounds):
        length = await _main_text_length(session)
        if length is None:
            break
        if length == previous:
            stale += 1
            if stale >= _STALE_ROUNDS:
                logger.debug("List stopped growing after %d rounds", round_number)
                break
        else:
            stale = 0
        previous = length
        try:
            await page.evaluate(SCROLL_LIST_JS)
        except Exception:
            logger.debug("List scroll failed", exc_info=True)
            break
        if can_wheel:
            try:
                await page.mouse.wheel(0, _WHEEL_DELTA)
            except Exception:
                logger.debug("List wheel scroll failed", exc_info=True)
                can_wheel = False
        await session.delay(_PAUSE)
