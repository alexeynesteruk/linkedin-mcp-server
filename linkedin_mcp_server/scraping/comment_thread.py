"""Expand the comment thread on a post permalink page.

Comments arrive in two ways, depending on which layout LinkedIn serves: a
pagination button under the thread, and lazy batches that attach when the
post's own scroll container scrolls. Both are driven here, and neither is
recognised by its label.

The pagination button is a plain ``<button>`` with no ``href`` and no
distinguishing attribute, so nothing names it. What is structural, and holds in
every locale, is where it sits and what it lacks:

- it is below the comment composer (a ``contenteditable`` ``role="textbox"``),
  which is rendered on every post that takes comments;
- it is inside the smallest block around that composer that also holds a
  commenter's profile link, so the post header, the sidebar and any
  "people to follow" module are out of reach;
- it carries no ``aria-*`` attribute at all, which every reaction, reply, share,
  follow and menu control does (an icon-only control has no other name), and it
  is not a submit button, not disabled, not inside a form, menu or dialog;
- its text is non-empty and contains no digit, which excludes "3 replies" and
  reaction counts.

A wrong click is the cost of a wrong guess here, so the program narrows rather
than widens. When it finds nothing the loop falls back to wheel-scrolling, and
the page still returns whatever comments the layout renders unprompted.
"""

from __future__ import annotations

import logging

from linkedin_mcp_server.scraping.session import ScrapingSession

logger = logging.getLogger(__name__)

# Present on a post only when commenting is enabled. Attribute presence, no text.
COMPOSER_SELECTOR = 'main [role="textbox"][contenteditable="true"]'

# Clicks at most one pagination button and reports whether it did. Clicked
# buttons are marked so a control that survives its own click is not clicked
# twice in a row; the marks are cleared when the last click made the thread grow
# (``reset``), because LinkedIn reuses one button for every batch.
CLICK_MORE_COMMENTS_JS = r"""
(reset) => {
  const composer = document.querySelector(
    'main [role="textbox"][contenteditable="true"]'
  );
  if (!composer) return false;
  if (reset) {
    for (const marked of document.querySelectorAll('[data-linkedin-mcp-more]')) {
      delete marked.dataset.linkedinMcpMore;
    }
  }
  const follows = node =>
    Boolean(composer.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING);
  let scope = composer.parentElement;
  while (scope && scope.tagName !== 'MAIN') {
    const commenter = Array.from(scope.querySelectorAll('a[href*="/in/"]'))
      .some(anchor => follows(anchor) && !composer.contains(anchor));
    if (commenter) break;
    scope = scope.parentElement;
  }
  if (!scope || scope.tagName === 'MAIN') return false;

  for (const button of scope.querySelectorAll('button')) {
    if (button.dataset.linkedinMcpMore) continue;
    if (!follows(button) || composer.contains(button)) continue;
    if (button.disabled || button.getAttribute('type') === 'submit') continue;
    if (button.getAttributeNames().some(name => name.startsWith('aria-'))) continue;
    if (button.dataset.testid === 'expandable-text-button') continue;
    if (button.closest('form, a, dialog, [role="menu"], [role="dialog"]')) continue;
    const rect = button.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;
    const text = (button.innerText || button.textContent || '').trim();
    if (!text || /\d/.test(text)) continue;
    button.dataset.linkedinMcpMore = '1';
    button.scrollIntoView({ block: 'center' });
    button.click();
    return true;
  }
  return false;
}
"""

# Progress signal: the length of main's text, which grows whenever a batch
# lands, in any locale.
MAIN_TEXT_LENGTH_JS = "() => document.querySelector('main')?.innerText.length || 0"

_DEFAULT_ROUNDS = 5
_STALE_ROUNDS = 2
_WHEEL_DELTA = 2000


async def _main_text_length(session: ScrapingSession) -> int:
    try:
        return int(await session.page.evaluate(MAIN_TEXT_LENGTH_JS))
    except Exception:
        return 0


async def expand_comment_thread(
    session: ScrapingSession, max_rounds: int | None = None
) -> None:
    """Load more of the thread, one click or wheel scroll per round.

    Stops after two rounds without growth or when the budget is spent.
    """
    rounds = max_rounds if max_rounds is not None else _DEFAULT_ROUNDS
    page = session.page
    viewport = page.viewport_size or {"width": 1280, "height": 720}
    try:
        await page.mouse.move(viewport["width"] // 2, viewport["height"] // 2)
    except Exception:
        logger.debug("Mouse move over post failed", exc_info=True)
        return

    stale = 0
    last_length = -1
    grew = False
    for round_number in range(rounds):
        clicked = False
        try:
            clicked = bool(await page.evaluate(CLICK_MORE_COMMENTS_JS, grew))
        except Exception:
            logger.debug("Comment pagination click failed", exc_info=True)
        if not clicked:
            try:
                await page.mouse.wheel(0, _WHEEL_DELTA)
            except Exception:
                logger.debug("Wheel scroll over post failed", exc_info=True)
                break
        await session.delay(1.0)

        length = await _main_text_length(session)
        grew = length > last_length
        if grew:
            stale = 0
            last_length = length
        else:
            stale += 1
            if stale >= _STALE_ROUNDS:
                logger.debug(
                    "Comment thread stopped growing after %d rounds", round_number + 1
                )
                break
