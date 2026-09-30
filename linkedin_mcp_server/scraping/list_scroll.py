"""Scrolling for lazy, virtualized lists whose scroller is not always the document."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from difflib import SequenceMatcher
from typing import Any

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

# One step down a virtualized list: about a screen, so consecutive reads
# overlap and no mounted window is skipped. Jumping to the end, as
# SCROLL_LIST_JS does, would unmount every row between the two windows before
# anything read them. Reports whether anything moved, so a list that is still
# being travelled is not mistaken for one that stopped growing.
SCROLL_LIST_STEP_JS = r"""
() => {
  const main = document.querySelector('main');
  let moved = false;
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
    if (target) {
      const before = target.scrollTop;
      target.scrollTop = before + Math.max(100, target.clientHeight * 0.8);
      moved = target.scrollTop > before;
    }
  }
  const beforeWindow = window.scrollY;
  window.scrollBy(0, Math.max(100, window.innerHeight * 0.8));
  return moved || window.scrollY > beforeWindow;
}
"""

DEFAULT_LIST_ROUNDS = 40
_STALE_ROUNDS = 3
_PAUSE = 1.3


def weave_lines(merged: list[str], snapshot: list[str]) -> list[str]:
    """Merge *snapshot* into *merged*, keeping the page's order.

    A virtualized list keeps only a window of its items mounted, so each read
    is the same header, a different slice of items, and the same sidebar and
    footer. The two reads are aligned as sequences: the shared header, the
    overlap of the slices and the shared sidebar match as blocks, and every
    run only one read holds is kept where it sits, so items the earlier read
    had and items the later one added both land between the header and the
    sidebar, in order. Popular lines ("8 endorsements") are not treated as
    junk, which would stop them anchoring the alignment.
    """
    if not merged:
        return list(snapshot)
    woven: list[str] = []
    matcher = SequenceMatcher(a=merged, b=snapshot, autojunk=False)
    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if tag == "equal":
            woven.extend(merged[a_start:a_end])
            continue
        # Only one read holds these lines: keep what was read before, then
        # what this read added in the same place.
        woven.extend(merged[a_start:a_end])
        woven.extend(snapshot[b_start:b_end])
    return woven


def merge_root_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """One root read from several reads of the same scrolled list.

    Text is woven line by line (see ``weave_lines``); references keep their
    first-seen order and each href once.
    """
    lines: list[str] = []
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    source: Any = None
    for snapshot in snapshots:
        text = snapshot.get("text") or ""
        if text:
            lines = weave_lines(lines, text.split("\n"))
        for reference in snapshot.get("references") or []:
            href = reference.get("href") or ""
            if href in seen:
                continue
            seen.add(href)
            references.append(reference)
        source = snapshot.get("source", source)
    return {"source": source, "text": "\n".join(lines), "references": references}


RootRead = Callable[[], Awaitable[dict[str, Any]]]


async def scroll_list_collecting(
    session: ScrapingSession, read_root: RootRead, max_rounds: int | None = None
) -> dict[str, Any]:
    """Scroll a virtualized list to its end, reading it at every step.

    Returns the merged read (``merge_root_snapshots``). LinkedIn's skills list
    unmounts the items scrolled past (measured live 2026-09-30: three reads of
    the same page returned 335, 1,123 and 2,642 characters), so one read after
    scrolling holds whichever slice happened to be mounted, and main's text
    length stays flat while the slice moves. So it steps about a screen at a
    time (``SCROLL_LIST_STEP_JS``) and reads after each step. Three steps that
    neither moved the scroller nor added a line, or the round budget, end it.
    """
    rounds = max_rounds if max_rounds is not None else DEFAULT_LIST_ROUNDS
    page = session.page
    viewport = page.viewport_size or {"width": 1280, "height": 720}
    can_wheel = True
    try:
        await page.mouse.move(viewport["width"] // 2, viewport["height"] // 2)
    except Exception:
        logger.debug("Mouse move over list failed", exc_info=True)
        can_wheel = False

    wheel_delta = int(viewport["height"] * 0.8)
    snapshots = [await read_root()]
    merged = merge_root_snapshots(snapshots)
    stale = 0
    for round_number in range(max(0, rounds)):
        try:
            moved = bool(await page.evaluate(SCROLL_LIST_STEP_JS))
        except Exception:
            logger.debug("List scroll failed", exc_info=True)
            break
        # The wheel only when the step moved nothing: both move the same
        # scroller, and together they would overshoot a mounted window. A list
        # that loads its next batch only on a real wheel event (the fork
        # measured this on the skills page) is at its end by then, which is
        # exactly when the wheel is needed.
        if can_wheel and not moved:
            try:
                await page.mouse.wheel(0, wheel_delta)
            except Exception:
                logger.debug("List wheel scroll failed", exc_info=True)
                can_wheel = False
        await session.delay(_PAUSE)
        snapshots.append(await read_root())
        grown = merge_root_snapshots(snapshots)
        unchanged = len(grown["text"]) == len(merged["text"]) and len(
            grown["references"]
        ) == len(merged["references"])
        # A step that moved is still travelling the list; only a step that
        # neither moved nor added anything counts toward stopping. A wheel-only
        # list never reports movement, so its end is still found by the text.
        if unchanged and not moved:
            stale += 1
            if stale >= _STALE_ROUNDS:
                logger.debug("List stopped growing after %d rounds", round_number + 1)
                merged = grown
                break
        else:
            stale = 0
        merged = grown
    return merged
