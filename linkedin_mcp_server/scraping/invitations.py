"""Pending network invitations from the invitation manager."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse
from typing import Any, Literal

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import rate_limited_section_error
from linkedin_mcp_server.scraping.link_metadata import (
    RawReference,
    Reference,
    build_references,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession
from linkedin_mcp_server.scraping.text import (
    filter_linkedin_noise_lines,
    truncate_linkedin_noise,
)

logger = logging.getLogger(__name__)

InvitationKind = Literal["received", "sent"]
INVITATION_KINDS: tuple[InvitationKind, ...] = ("received", "sent")

# The list lazy-loads roughly this many cards per screenful.
_CARDS_PER_SCROLL = 10

# Invitation notes longer than a few lines are truncated behind an inline
# toggle. ``data-testid`` is the locale-independent handle; the verb on it
# ("see more", "voir plus") is not. The toggle ships with pointer-events:none,
# so a scoped style re-enables it and a bubbling click reaches React. Clicked
# toggles are marked so the second pass (for cards the first click loaded)
# never re-clicks one: the collapse control has the same testid.
EXPAND_NOTES_JS = r"""
() => {
  if (!document.getElementById('__linkedin_mcp_expand_style')) {
    const style = document.createElement('style');
    style.id = '__linkedin_mcp_expand_style';
    style.textContent = `
      [data-testid="expandable-text-button"],
      [data-testid="expandable-text-box"] { pointer-events: auto !important; }
    `;
    document.head.appendChild(style);
  }
  const buttons = document.querySelectorAll(
    'main [data-testid="expandable-text-button"]'
    + ':not([aria-expanded="true"]):not([data-linkedin-mcp-clicked])'
  );
  let count = 0;
  for (const button of buttons) {
    button.dataset.linkedinMcpClicked = '1';
    button.dispatchEvent(
      new MouseEvent('click', { bubbles: true, cancelable: true, view: window })
    );
    count += 1;
  }
  return count;
}
"""

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

# The received manager renders "people you may know" cards under an empty
# state, and their profile links are not invitations. The selected tab is
# found structurally (aria-current plus the stable /received/ALL route); the
# count it carries is a number in parentheses in any locale.
RECEIVED_COUNT_IS_ZERO_JS = r"""
() => {
  const tab = document.querySelector(
    'main a[aria-current="true"][href*="/mynetwork/invitation-manager/received/ALL"]'
  );
  if (!tab) return false;
  const match = (tab.innerText || tab.textContent || '').match(/\((\d+)\)/);
  return match ? Number(match[1]) === 0 : false;
}
"""


# The sent manager links each invitee only through the avatar, an <a> with no
# text, and prints the name as plain text beside it; build_references drops a
# person link without a label. Each profile link is therefore given the first
# line of its card: the largest block around it that links no other profile.
# Structural, so it holds in any locale (measured live 2026-09-30).
CARD_LABELS_JS = r"""
() => {
  const main = document.querySelector('main');
  if (!main) return {};
  const pathOf = anchor => {
    try {
      const path = new URL(anchor.href, location.href).pathname;
      const match = path.match(/^\/in\/[^/]+/);
      return match ? match[0] + '/' : null;
    } catch (error) {
      return null;
    }
  };
  const labels = {};
  for (const anchor of main.querySelectorAll('a[href*="/in/"]')) {
    const path = pathOf(anchor);
    if (!path || labels[path]) continue;
    let card = null;
    for (let el = anchor.parentElement; el && el !== main; el = el.parentElement) {
      const paths = new Set(
        Array.from(el.querySelectorAll('a[href*="/in/"]')).map(pathOf).filter(Boolean)
      );
      if (paths.size > 1) break;
      card = el;
    }
    if (!card) continue;
    const line = (card.innerText || '')
      .split('\n')
      .map(part => part.trim())
      .find(Boolean);
    if (line) labels[path] = line.slice(0, 200);
  }
  return labels;
}
"""


def label_unlabeled_profiles(
    raw_references: list[RawReference], labels: dict[str, str]
) -> list[RawReference]:
    """Give each text-less /in/ link its card label, leaving the rest alone."""
    if not labels:
        return raw_references
    labeled: list[RawReference] = []
    for raw in raw_references:
        href = raw.get("href") or ""
        if (raw.get("text") or "").strip() or "/in/" not in href:
            labeled.append(raw)
            continue
        match = re.match(r"/in/[^/]+", urlparse(href).path)
        label = labels.get(f"{match.group(0)}/") if match else None
        if label:
            raw = RawReference(**raw)
            raw["text"] = label
        labeled.append(raw)
    return labeled


def invitations_url(kind: InvitationKind) -> str:
    """The invitation-manager page for *kind*."""
    if kind not in INVITATION_KINDS:
        raise ValueError(f"kind must be one of {INVITATION_KINDS!r}, got {kind!r}")
    return f"https://www.linkedin.com/mynetwork/invitation-manager/{kind}/"


def trim_to_limit(text: str, references: list[Reference], limit: int) -> str:
    """Cut *text* where the first invitation past *limit* begins.

    LinkedIn renders a screenful of cards whatever the limit, so the text is
    cut at the first omitted card's profile label when references locate it,
    and otherwise at the blank lines LinkedIn puts between cards.
    """
    if limit < 1 or not text:
        return text

    search_from = 0
    for reference in references[:limit]:
        label = reference.get("text")
        if not label:
            continue
        index = text.find(label, search_from)
        if index >= 0:
            search_from = index + len(label)

    if len(references) > limit:
        next_label = references[limit].get("text")
        if next_label:
            next_index = text.find(next_label, search_from)
            if next_index > 0:
                return text[:next_index].rstrip()

    first_label = next((r.get("text") for r in references if r.get("text")), None)
    if not first_label:
        return text
    first_index = text.find(first_label)
    if first_index < 0:
        return text
    prefix = text[:first_index]
    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", text[first_index:].strip())
        if block.strip()
    ]
    if len(blocks) <= limit:
        return text
    return (prefix + "\n\n".join(blocks[:limit])).rstrip()


class InvitationReader:
    """Read the received or sent invitation manager. Never accepts or ignores."""

    def __init__(
        self,
        session: ScrapingSession,
        navigator: PageNavigator,
        content: PageContentReader,
    ):
        self._session = session
        self._navigator = navigator
        self._content = content

    async def _scroll_list(self, limit: int) -> None:
        """Scroll the list far enough for *limit* cards to have loaded."""
        for _ in range(max(1, limit // _CARDS_PER_SCROLL)):
            try:
                await self._session.page.evaluate(SCROLL_LIST_JS)
            except Exception:
                logger.debug("Invitation list scroll failed", exc_info=True)
                return
            await self._session.delay(0.5)

    async def _expand_notes(self) -> None:
        for _ in range(2):
            try:
                clicked = await self._session.page.evaluate(EXPAND_NOTES_JS)
            except Exception:
                logger.debug("Invitation note expansion failed", exc_info=True)
                return
            if not clicked:
                return
            await self._session.delay(0.3)

    async def _card_labels(self) -> dict[str, str]:
        try:
            labels = await self._session.page.evaluate(CARD_LABELS_JS)
        except Exception:
            logger.debug("Could not read invitation card labels", exc_info=True)
            return {}
        if not isinstance(labels, dict):
            return {}
        return {
            str(path): str(label)
            for path, label in labels.items()
            if isinstance(label, str) and label
        }

    async def _received_count_is_zero(self) -> bool:
        try:
            return bool(await self._session.page.evaluate(RECEIVED_COUNT_IS_ZERO_JS))
        except Exception:
            logger.debug("Could not read the received invitation count", exc_info=True)
            return False

    async def get_pending_invitations(
        self,
        limit: int = 20,
        kind: InvitationKind = "received",
    ) -> dict[str, Any]:
        """List pending invitations of *kind*, up to *limit*.

        Returns:
            {url, sections: {invitations: text}} plus optional
            ``references["invitations"]`` (inviter or invitee profiles, capped
            at *limit*) and ``section_errors``.
        """
        url = invitations_url(kind)
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()
        try:
            await self._session.page.wait_for_selector("main")
        except PlaywrightTimeoutError:
            logger.debug("No <main> element found on %s", url)
        await self._session.dismiss_modal()
        try:
            await self._session.page.wait_for_function(
                """() => {
                    const main = document.querySelector('main');
                    return !!main && main.innerText.length > 100;
                }""",
                timeout=10000,
            )
        except PlaywrightTimeoutError:
            logger.debug("Invitation list did not appear on %s", url)

        await self._scroll_list(limit)
        await self._expand_notes()

        result: dict[str, Any] = {"url": url, "sections": {}}
        if kind == "received" and await self._received_count_is_zero():
            return result

        raw_result = await self._content._extract_root_content(["main"])
        raw = raw_result["text"]
        if not raw:
            return result
        truncated = truncate_linkedin_noise(raw)
        if not truncated and raw.strip():
            logger.warning("Invitations page %s returned only LinkedIn chrome", url)
            result["section_errors"] = {"invitations": rate_limited_section_error()}
            return result
        cleaned = filter_linkedin_noise_lines(truncated)

        raw_references = label_unlabeled_profiles(
            raw_result["references"], await self._card_labels()
        )
        references = build_references(raw_references, "invitations")
        result["sections"]["invitations"] = trim_to_limit(cleaned, references, limit)
        if references[:limit]:
            result["references"] = {"invitations": references[:limit]}
        return result
