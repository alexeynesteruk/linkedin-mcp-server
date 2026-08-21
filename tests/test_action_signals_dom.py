# tests/test_action_signals_dom.py
"""Browser-DOM tests for the incoming-request action-row fingerprint,
the withdraw-anchor click, and the multi-dialog scoping helpers.

The unit suite mocks ``page.evaluate``, so the JS constants tested here
never actually execute there. These tests run the real JS against
synthetic HTML in headless chromium. Fixtures use German labels
throughout: the fingerprint must classify without reading any label
text. Skipped automatically when chromium is not installed (CI installs no
browser; run locally after ``uv run patchright install chromium``).

Fixture structure mirrors the live DOM dumps of two incoming-request
profiles (2026-06-11): three buttons sharing one parent, Accept and Ignore
carrying aria-label, More carrying aria-expanded without aria-label, plus
sidebar cards with labeled compose anchors and other-user invite anchors.

Also covers two live incidents found 2026-08-21: #629 (a creator-mode
profile's [Follow][Save in Sales Navigator][More] top card is structurally
identical to a genuine incoming-request row) and a multi-dialog button
collision found while building ``withdraw_invitation`` (a page-wide "last
button in document order" selector can resolve to an unrelated, hidden
button from a different [role="dialog"] container entirely).
"""

from __future__ import annotations

import pytest
from patchright.async_api import async_playwright

from linkedin_mcp_server.scraping.extractor import (
    _ACTION_SIGNALS_JS,
    _CLICK_INCOMING_ACCEPT_JS,
    _CLICK_LAST_DIALOG_BUTTON_JS,
    _CLICK_WITHDRAW_ANCHOR_JS,
    _DIALOG_BUTTON_COUNT_JS,
    _OPEN_INCOMING_ROW_MORE_BUTTON_JS,
)

pytestmark = pytest.mark.browser_dom


# Each constant is a full <section>. The top card is always the first
# section of <main>; the fingerprint is scoped there, so sidebar and feed
# widgets live in later sections and must never match.

INCOMING_ACTION_ROW = """
  <div class="actions">
    <button type="button" aria-label="Kontaktanfrage von Eric Langlouis annehmen"
      onclick="document.body.setAttribute('data-clicked','accept')">Annehmen</button>
    <button type="button" aria-label="Kontaktanfrage von Eric Langlouis ignorieren"
      onclick="document.body.setAttribute('data-clicked','ignore')">Ignorieren</button>
    <button type="button" aria-expanded="false">Mehr</button>
  </div>
"""

INCOMING_TOP_CARD = f"""
<section class="topcard">
  <h1>Eric Langlouis</h1>
  {INCOMING_ACTION_ROW}
</section>
"""

VIDEO_PLAYER_BAR = """
  <div class="player">
    <button type="button" aria-label="Abspielen">▶</button>
    <button type="button" aria-label="Stummschalten">🔇</button>
    <button type="button" aria-label="Untertitel">CC</button>
    <button type="button" aria-label="Vollbild">⛶</button>
    <button type="button" aria-expanded="false" aria-label="Einstellungen">⚙</button>
  </div>
"""

# Cover-video profile: the player's expander renders before the action row
# within the same top card. The scan must skip it and still find the row.
INCOMING_TOP_CARD_WITH_COVER = f"""
<section class="topcard">
  <h1>Eric Langlouis</h1>
  {VIDEO_PLAYER_BAR}
  {INCOMING_ACTION_ROW}
</section>
"""

SIDEBAR_SECTION = """
<section class="sidebar">
  <div class="card">
    <a href="https://www.linkedin.com/in/julien-f/">Julien</a>
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3AAAA"
      aria-label="Nachricht an Julien senden">Nachricht</a>
  </div>
  <div class="card">
    <a href="https://www.linkedin.com/in/rahul-g/">Rahul</a>
    <a href="/preload/custom-invite/?vanityName=rahul-g"
      aria-label="Rahul als Kontakt einladen">Vernetzen</a>
  </div>
  <button type="button">Mehr anzeigen</button>
</section>
"""

# A widget elsewhere in main with the exact incoming-row shape (two labeled
# buttons + one unlabeled expander). It must NOT match because it lives in a
# later section, outside the scoped top card.
UNRELATED_MATCHING_WIDGET = """
<section class="feed">
  <div class="actions">
    <button type="button" aria-label="Gefällt mir">A</button>
    <button type="button" aria-label="Kommentieren">B</button>
    <button type="button" aria-expanded="false">Mehr</button>
  </div>
</section>
"""

CONNECTED_TOP_CARD = """
<section class="topcard">
  <h1>Fadi Al Eliwi</h1>
  <div class="actions">
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3ABBB"
      aria-disabled="false">Nachricht</a>
    <button type="button" aria-expanded="false">Mehr</button>
  </div>
</section>
"""

FOLLOW_ONLY_TOP_CARD = """
<section class="topcard">
  <h1>Verena</h1>
  <div class="actions">
    <button type="button" aria-label="Verena folgen">Folgen</button>
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3ACCC">Nachricht</a>
    <button type="button" aria-expanded="false">Mehr</button>
  </div>
</section>
"""

PENDING_TOP_CARD = """
<section class="topcard">
  <h1>Florian</h1>
  <div class="actions">
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3ADDD">Nachricht</a>
    <a href="https://www.linkedin.com/in/florian/"
      aria-label="Ausstehend, klicken zum Zurückziehen">Ausstehend</a>
    <button type="button" aria-expanded="false">Mehr</button>
  </div>
</section>
"""

EXPANDER_FIRST_BAR = """
<section class="hostile">
  <button type="button" aria-expanded="false">⚙</button>
  <button type="button" aria-label="Aktion A">A</button>
  <button type="button" aria-label="Aktion B">B</button>
</section>
"""

EXTRA_BUTTON_ROW = """
<section class="hostile">
  <button type="button" aria-label="Aktion A">A</button>
  <button type="button" aria-label="Aktion B">B</button>
  <button type="button" aria-expanded="false">Mehr</button>
  <button type="button">Extra</button>
</section>
"""

# Creator-mode / high-follower profile (#629): top card renders
# [Follow][Save in Sales Navigator][More] with NO Message button/anchor
# anywhere - a structural match for the incoming-request fingerprint,
# even though this is not an incoming request at all. The More button's
# onclick simulates LinkedIn lazily mounting the Connect option into the
# DOM only once the menu is actually opened (it is not present, hidden or
# otherwise, before the click).
CREATOR_MODE_TOP_CARD = """
<section class="topcard">
  <h1>Marc</h1>
  <div class="actions">
    <button type="button" aria-label="Marc folgen">Folgen</button>
    <button type="button" aria-label="In Sales Navigator speichern">Sales Navigator</button>
    <button type="button" aria-expanded="false" onclick="
      var m = document.createElement('div');
      m.setAttribute('role', 'menu');
      var a = document.createElement('a');
      a.setAttribute('href', '/preload/custom-invite/?vanityName=testuser');
      a.setAttribute('aria-label', 'Marc als Kontakt einladen');
      a.textContent = 'Vernetzen';
      m.appendChild(a);
      document.body.appendChild(m);
    ">Mehr</button>
  </div>
</section>
"""

# Same pending-invite shape as PENDING_TOP_CARD, but the anchor has a
# navigation-preventing onclick so clicking it in a real browser doesn't
# actually try to load https://www.linkedin.com/in/florian/.
PENDING_TOP_CARD_CLICKABLE = """
<section class="topcard">
  <h1>Florian</h1>
  <div class="actions">
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3ADDD">Nachricht</a>
    <a href="https://www.linkedin.com/in/florian/"
      aria-label="Ausstehend, klicken zum Zurückziehen"
      onclick="event.preventDefault(); document.body.setAttribute('data-clicked', 'withdraw');"
      >Ausstehend</a>
    <button type="button" aria-expanded="false">Mehr</button>
  </div>
</section>
"""

# Ambiguous variant of the pending row: two labeled anchors in the action
# root. _CLICK_WITHDRAW_ANCHOR_JS must refuse to guess which one is Pending.
AMBIGUOUS_PENDING_TOP_CARD = """
<section class="topcard">
  <h1>Florian</h1>
  <div class="actions">
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3ADDD">Nachricht</a>
    <a href="https://www.linkedin.com/in/florian/"
      aria-label="Ausstehend, klicken zum Zurückziehen">Ausstehend</a>
    <a href="https://www.linkedin.com/in/florian/other" aria-label="Ein weiterer Link"
      >Weiterer Link</a>
    <button type="button" aria-expanded="false">Mehr</button>
  </div>
</section>
"""

# Reproduces the real multi-dialog collision found live 2026-08-21 on
# /mynetwork/invitation-manager/: a native open dialog (the real
# withdraw-confirmation modal) coexists with an unrelated, hidden
# [role="dialog"] container elsewhere in the document (LinkedIn's nav
# search overlay in the wild). A page-wide "last button in document
# order" selector would resolve to the hidden decoy's button instead of
# the real dialog's primary action.
MULTI_DIALOG_PAGE = """
<html><body>
  <dialog open>
    <button type="button" aria-label="Dismiss">X</button>
    <button type="button">Cancel</button>
    <button type="button" aria-label="Withdraw invitation sent to Roshni S."
      onclick="document.body.setAttribute('data-clicked', 'withdraw')">Withdraw</button>
  </dialog>
  <div role="dialog" style="display:none">
    <button type="button" disabled
      onclick="document.body.setAttribute('data-clicked', 'decoy')">Hidden decoy</button>
  </div>
</body></html>
"""

# No native <dialog> present at all - only a visible [role="dialog"]
# alongside a hidden one, exercising _FIND_OPEN_DIALOG_FN_JS's fallback.
ROLE_DIALOG_ONLY_PAGE = """
<html><body>
  <div role="dialog" style="display:none">
    <button type="button" onclick="document.body.setAttribute('data-clicked', 'decoy')">Hidden</button>
  </div>
  <div role="dialog">
    <button type="button">Cancel</button>
    <button type="button"
      onclick="document.body.setAttribute('data-clicked', 'confirm')">Confirm</button>
  </div>
</body></html>
"""


def _page_html(*sections: str) -> str:
    return f"<html><body><main>{''.join(sections)}</main></body></html>"


@pytest.fixture
async def dom_page():
    """Real chromium page, or skip when no browser is installed.

    Only launch/setup is guarded by the skip — the ``yield`` is outside it
    so an assertion failure or JS error in a test body is never swallowed
    into a skip.
    """
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
        except Exception as exc:  # browser binary missing
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            yield page
        finally:
            await browser.close()


async def _signals(page, html: str) -> dict:
    await page.set_content(html)
    return await page.evaluate(_ACTION_SIGNALS_JS, "testuser")


class TestIncomingActionRowFingerprint:
    async def test_incoming_row_detected_next_to_sidebar_cards(self, dom_page):
        data = await _signals(dom_page, _page_html(INCOMING_TOP_CARD, SIDEBAR_SECTION))
        assert data["hasIncomingActionRow"] is True

    async def test_video_player_bar_not_detected(self, dom_page):
        data = await _signals(
            dom_page, _page_html(CONNECTED_TOP_CARD, VIDEO_PLAYER_BAR)
        )
        assert data["hasIncomingActionRow"] is False

    async def test_expander_first_order_guard(self, dom_page):
        data = await _signals(dom_page, _page_html(EXPANDER_FIRST_BAR))
        assert data["hasIncomingActionRow"] is False

    async def test_preceding_nonmatching_expander_does_not_abort_scan(self, dom_page):
        # Cover-video layout: the player's expander renders before the
        # action row inside the same top card; the scan must continue past
        # it and still find the row.
        data = await _signals(dom_page, _page_html(INCOMING_TOP_CARD_WITH_COVER))
        assert data["hasIncomingActionRow"] is True

    async def test_matching_widget_outside_top_card_not_detected(self, dom_page):
        # F1 regression: a widget with the exact incoming-row shape in a
        # later section must not match — the scan is scoped to the top card.
        data = await _signals(
            dom_page, _page_html(CONNECTED_TOP_CARD, UNRELATED_MATCHING_WIDGET)
        )
        assert data["hasIncomingActionRow"] is False

    async def test_extra_unlabeled_button_fails_count_guard(self, dom_page):
        data = await _signals(dom_page, _page_html(EXTRA_BUTTON_ROW))
        assert data["hasIncomingActionRow"] is False

    async def test_follow_only_row_not_detected(self, dom_page):
        data = await _signals(dom_page, _page_html(FOLLOW_ONLY_TOP_CARD))
        assert data["hasIncomingActionRow"] is False

    async def test_pending_row_not_detected(self, dom_page):
        data = await _signals(dom_page, _page_html(PENDING_TOP_CARD))
        assert data["hasIncomingActionRow"] is False

    async def test_connected_row_not_detected(self, dom_page):
        data = await _signals(dom_page, _page_html(CONNECTED_TOP_CARD, SIDEBAR_SECTION))
        assert data["hasIncomingActionRow"] is False


class TestClickIncomingAccept:
    async def test_clicks_first_labeled_button_only(self, dom_page):
        await dom_page.set_content(_page_html(INCOMING_TOP_CARD, SIDEBAR_SECTION))
        clicked = await dom_page.evaluate(_CLICK_INCOMING_ACCEPT_JS)
        assert clicked is True
        # Patchright evaluates in an isolated world; page-world variables
        # are invisible there, but the DOM is shared — the inline onclick
        # records the click as a body attribute.
        recorded = await dom_page.evaluate("document.body.getAttribute('data-clicked')")
        assert recorded == "accept"

    async def test_no_click_without_fingerprint_match(self, dom_page):
        await dom_page.set_content(_page_html(FOLLOW_ONLY_TOP_CARD, VIDEO_PLAYER_BAR))
        clicked = await dom_page.evaluate(_CLICK_INCOMING_ACCEPT_JS)
        assert clicked is False


class TestCreatorModeFalsePositive:
    """Regression fixtures for #629: a creator-mode/high-follower profile's
    top card - [Follow][Save in Sales Navigator][More], no Message button -
    is structurally identical to a genuine incoming-request row. These
    document that ambiguity at the fingerprint level (matching
    TestDetectConnectionState.test_incoming_request_signals_are_ambiguous_with_creator_mode
    on the Python side), then prove the open-More disprove
    (_OPEN_INCOMING_ROW_MORE_BUTTON_JS) correctly reveals the reachable
    Connect option once the menu is actually opened.
    """

    async def test_creator_mode_row_matches_incoming_fingerprint(self, dom_page):
        data = await _signals(dom_page, _page_html(CREATOR_MODE_TOP_CARD))
        assert data["hasIncomingActionRow"] is True
        assert data["hasInvite"] is False
        assert data["hasComposeInActionRoot"] is False

    async def test_open_more_reveals_invite_anchor(self, dom_page):
        await dom_page.set_content(_page_html(CREATOR_MODE_TOP_CARD))
        opened = await dom_page.evaluate(_OPEN_INCOMING_ROW_MORE_BUTTON_JS)
        assert opened is True
        data = await dom_page.evaluate(_ACTION_SIGNALS_JS, "testuser")
        assert data["hasInvite"] is True

    async def test_open_more_reveals_nothing_on_genuine_incoming_row(self, dom_page):
        # The disprove must not misfire on a real incoming-request row:
        # its (non-functional here) More button reveals no invite anchor.
        await dom_page.set_content(_page_html(INCOMING_TOP_CARD))
        opened = await dom_page.evaluate(_OPEN_INCOMING_ROW_MORE_BUTTON_JS)
        assert opened is True
        data = await dom_page.evaluate(_ACTION_SIGNALS_JS, "testuser")
        assert data["hasInvite"] is False


class TestClickWithdrawAnchor:
    async def test_clicks_the_pending_anchor(self, dom_page):
        await dom_page.set_content(_page_html(PENDING_TOP_CARD_CLICKABLE))
        clicked = await dom_page.evaluate(_CLICK_WITHDRAW_ANCHOR_JS)
        assert clicked is True
        recorded = await dom_page.evaluate("document.body.getAttribute('data-clicked')")
        assert recorded == "withdraw"

    async def test_no_click_when_no_labeled_anchor(self, dom_page):
        await dom_page.set_content(_page_html(CONNECTED_TOP_CARD))
        clicked = await dom_page.evaluate(_CLICK_WITHDRAW_ANCHOR_JS)
        assert clicked is False

    async def test_no_click_when_ambiguous_multiple_labeled_anchors(self, dom_page):
        await dom_page.set_content(_page_html(AMBIGUOUS_PENDING_TOP_CARD))
        clicked = await dom_page.evaluate(_CLICK_WITHDRAW_ANCHOR_JS)
        assert clicked is False


class TestDialogScoping:
    """Regression fixtures for the multi-dialog collision found live
    2026-08-21 while building withdraw_invitation: LinkedIn pages can
    carry several [role="dialog"] containers at once, most hidden (nav
    search overlays, preloaded modals) - a page-wide "last button in
    document order" selector resolved to an unrelated, hidden, disabled
    button from a different container entirely, instead of the real open
    dialog's primary action.
    """

    async def test_button_count_scoped_to_open_dialog_ignores_hidden_decoy(
        self, dom_page
    ):
        await dom_page.set_content(MULTI_DIALOG_PAGE)
        count = await dom_page.evaluate(_DIALOG_BUTTON_COUNT_JS)
        assert count == 3

    async def test_click_last_button_ignores_hidden_decoy(self, dom_page):
        await dom_page.set_content(MULTI_DIALOG_PAGE)
        clicked = await dom_page.evaluate(_CLICK_LAST_DIALOG_BUTTON_JS)
        assert clicked is True
        recorded = await dom_page.evaluate("document.body.getAttribute('data-clicked')")
        assert recorded == "withdraw"

    async def test_falls_back_to_visible_role_dialog_when_no_native_dialog(
        self, dom_page
    ):
        await dom_page.set_content(ROLE_DIALOG_ONLY_PAGE)
        count = await dom_page.evaluate(_DIALOG_BUTTON_COUNT_JS)
        assert count == 2
        clicked = await dom_page.evaluate(_CLICK_LAST_DIALOG_BUTTON_JS)
        assert clicked is True
        recorded = await dom_page.evaluate("document.body.getAttribute('data-clicked')")
        assert recorded == "confirm"
