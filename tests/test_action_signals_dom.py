# tests/test_action_signals_dom.py
"""Browser-DOM tests for the locale independence of the action-area reads.

The unit suite mocks ``page.evaluate``, so the programs in
``scraping/connection_actions.py`` never execute there. These tests run the
real ones against synthetic HTML in headless chromium.

Every fixture is built from one set of templates, three sets of words and
one set whose ARIA values are empty: English, German, opaque tokens carrying
no verb, and present-but-empty attributes. The structure is identical across
all four, and each case
asserts the *same* answer for all of them in one assertion that names the
locales. A decision that differs between two of them is a decision that read
a word, which the AGENTS.md Scraping Rules forbid; the opaque set is the
control, because a label with no verb in it cannot be matched by one.

Skipped automatically when chromium is not installed; run locally after
``uv run patchright install chromium --no-shell``. CI installs it, so these
run there, but a skip is still the right answer for a missing browser here:
this file checks extraction JS rather than the browser, and
``test_browser_identity.py`` is the one that has to fail instead.

Fixture structure mirrors the live DOM dumps of two incoming-request
profiles (2026-06-11): three buttons sharing one parent, Accept and Ignore
carrying aria-label, More carrying aria-expanded without aria-label, plus
sidebar cards with labeled compose anchors and other-user invite anchors.
Every control row sits in a container of its own inside its section, the way
LinkedIn renders them, so the walk that fingerprints the row actually
reaches the guards: a row that *is* the scope stops the walk before any of
them, which is a pass for the wrong reason.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

import pytest
from patchright.async_api import Page, async_playwright

from linkedin_mcp_server.scraping.connection import (
    ConnectionState,
    detect_connection_state,
)
from linkedin_mcp_server.scraping.connection_actions import (
    ACTION_SIGNALS_JS,
    CLICK_CONFIRM_DIALOG_PRIMARY_JS,
    CLICK_INCOMING_ACCEPT_JS,
    CLICK_WITHDRAW_ANCHOR_JS,
    CONFIRM_DIALOG_BUTTON_COUNT_JS,
    OPEN_MORE_BUTTON_JS,
    ConnectionActions,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

#: CI uses ``--dist loadgroup``. Keep every test that launches Chromium on one
#: worker so browser startups cannot compete with the DOM cases' wall-clock
#: timers.
#: Without that distribution mode the group mark is inert.
pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

USER = "testuser"


@dataclass(frozen=True, slots=True)
class Labels:
    """Every visible string and aria-label value one locale contributes.

    Only these values change between the four fixture sets, so nothing else
    can explain a decision that changes with them.
    """

    locale: str
    accept: str
    ignore: str
    more: str
    message: str
    connect: str
    follow: str
    pending: str
    edit: str
    play: str
    mute: str
    captions: str
    fullscreen: str
    settings: str
    show_all: str
    like: str
    comment: str
    dismiss: str
    cancel: str
    withdraw: str
    upsell: str


ENGLISH = Labels(
    locale="en",
    accept="Accept Eric Langlouis' invitation to connect",
    ignore="Ignore Eric Langlouis' invitation",
    more="More",
    message="Message Julien",
    connect="Invite Rahul to connect",
    follow="Follow Verena",
    pending="Pending, click to withdraw the invitation sent to Florian",
    edit="Edit intro",
    play="Play",
    mute="Mute",
    captions="Captions",
    fullscreen="Full screen",
    settings="Settings",
    show_all="Show all",
    like="Like",
    comment="Comment",
    dismiss="Dismiss",
    cancel="Cancel",
    withdraw="Withdraw",
    upsell="Try Premium for free",
)

GERMAN = Labels(
    locale="de",
    accept="Kontaktanfrage von Eric Langlouis annehmen",
    ignore="Kontaktanfrage von Eric Langlouis ignorieren",
    more="Mehr",
    message="Nachricht an Julien senden",
    connect="Rahul als Kontakt einladen",
    follow="Verena folgen",
    pending="Ausstehend, klicken zum Zurückziehen",
    edit="Intro bearbeiten",
    play="Abspielen",
    mute="Stummschalten",
    captions="Untertitel",
    fullscreen="Vollbild",
    settings="Einstellungen",
    show_all="Mehr anzeigen",
    like="Gefällt mir",
    comment="Kommentieren",
    dismiss="Verwerfen",
    cancel="Abbrechen",
    withdraw="Zurückziehen",
    upsell="Premium gratis testen",
)

# No verb anywhere, in any language: these labels are identifiers. Whatever
# still classifies correctly here is reading structure, and this set is the
# one a locale table could not rescue.
OPAQUE = Labels(
    locale="opaque",
    accept="a7f3c1",
    ignore="b2e9d4",
    more="c8a0b5",
    message="d1f6e2",
    connect="e4b7a9",
    follow="f0c3d8",
    pending="a9e2f7",
    edit="b5d8c0",
    play="c2f4a6",
    mute="d7b1e3",
    captions="e9a5b8",
    fullscreen="f3c7d1",
    settings="a0b4e6",
    show_all="b8f2c9",
    like="c6d0a3",
    comment="d4e8b7",
    dismiss="e1b6c4",
    cancel="f5d9a2",
    withdraw="a3c8e0",
    upsell="b7e1f5",
)

# Attribute presence and attribute truthiness are different contracts. This
# set keeps every aria-label attribute in the markup while making its value
# empty, so `hasAttribute` survives and `getAttribute(...)` truthiness does not.
EMPTY_ARIA = Labels(
    locale="empty-aria",
    accept="",
    ignore="",
    more="",
    message="",
    connect="",
    follow="",
    pending="",
    edit="",
    play="",
    mute="",
    captions="",
    fullscreen="",
    settings="",
    show_all="",
    like="",
    comment="",
    dismiss="",
    cancel="",
    withdraw="",
    upsell="",
)

LOCALES = (ENGLISH, GERMAN, OPAQUE, EMPTY_ARIA)

Build = Callable[[Labels], str]


# Each builder returns one full <section>. The top card is always the first
# section of <main>; the incoming fingerprint is scoped there, so sidebar and
# feed widgets live in later sections and must never match.


def incoming_action_row(labels: Labels) -> str:
    return f"""
  <div class="actions">
    <button type="button" aria-label="{labels.accept}"
      onclick="document.body.setAttribute('data-clicked','first-labeled')"
      >{labels.accept}</button>
    <button type="button" aria-label="{labels.ignore}"
      onclick="document.body.setAttribute('data-clicked','second-labeled')"
      >{labels.ignore}</button>
    <button type="button" aria-expanded="false">{labels.more}</button>
  </div>
"""


def incoming_top_card(labels: Labels) -> str:
    return f"""
<section class="topcard">
  <h1>Eric Langlouis</h1>
  {incoming_action_row(labels)}
</section>
"""


def video_player_bar(labels: Labels) -> str:
    """Every control carries aria-label here, the settings one included."""
    return f"""
  <div class="player">
    <button type="button" aria-label="{labels.play}">&#9654;</button>
    <button type="button" aria-label="{labels.mute}">&#128264;</button>
    <button type="button" aria-label="{labels.captions}">CC</button>
    <button type="button" aria-label="{labels.fullscreen}">&#9727;</button>
    <button type="button" aria-expanded="false"
      aria-label="{labels.settings}">&#9881;</button>
  </div>
"""


def incoming_top_card_with_cover(labels: Labels) -> str:
    """Cover-video profile: the player's expander precedes the action row."""
    return f"""
<section class="topcard">
  <h1>Eric Langlouis</h1>
  {video_player_bar(labels)}
  {incoming_action_row(labels)}
</section>
"""


def sidebar_section(labels: Labels) -> str:
    """Mutual-connection cards: labeled compose and *other-user* invite anchors."""
    return f"""
<section class="sidebar">
  <div class="card">
    <a href="https://www.linkedin.com/in/julien-f/">Julien</a>
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3AAAA"
      aria-label="{labels.message}">{labels.message}</a>
  </div>
  <div class="card">
    <a href="https://www.linkedin.com/in/rahul-g/">Rahul</a>
    <a href="/preload/custom-invite/?vanityName=rahul-g"
      aria-label="{labels.connect}">{labels.connect}</a>
  </div>
  <button type="button">{labels.show_all}</button>
</section>
"""


def unrelated_matching_widget(labels: Labels) -> str:
    """A later-section widget with the exact incoming-row shape."""
    return f"""
<section class="feed">
  <div class="actions">
    <button type="button" aria-label="{labels.like}">A</button>
    <button type="button" aria-label="{labels.comment}">B</button>
    <button type="button" aria-expanded="false">{labels.more}</button>
  </div>
</section>
"""


def connected_top_card(labels: Labels) -> str:
    """1st degree: a compose anchor carrying only aria-disabled, and More."""
    return f"""
<section class="topcard">
  <h1>Fadi Al Eliwi</h1>
  <div class="actions">
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3ABBB"
      aria-disabled="false">{labels.message}</a>
    <button type="button" aria-expanded="false">{labels.more}</button>
  </div>
</section>
"""


def follow_only_top_card(labels: Labels) -> str:
    """Creator-mode profile: a labeled primary button, no invite anchor."""
    return f"""
<section class="topcard">
  <h1>Verena</h1>
  <div class="actions">
    <button type="button" aria-label="{labels.follow}"
      onclick="document.body.setAttribute('data-clicked','primary-labeled')"
      >{labels.follow}</button>
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3ACCC"
      >{labels.message}</a>
    <button type="button" aria-expanded="false"
      onclick="document.body.setAttribute('data-clicked','expander')"
      >{labels.more}</button>
  </div>
</section>
"""


def pending_top_card(labels: Labels) -> str:
    """Awaiting response: the Pending control is a labeled <a>, not a button."""
    return f"""
<section class="topcard">
  <h1>Florian</h1>
  <div class="actions">
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3ADDD"
      >{labels.message}</a>
    <a href="https://www.linkedin.com/in/florian/"
      aria-label="{labels.pending}">{labels.pending}</a>
    <button type="button" aria-expanded="false">{labels.more}</button>
  </div>
</section>
"""


def connectable_top_card(labels: Labels) -> str:
    """The vanityName invite anchor for *this* user, which is the write gate."""
    return f"""
<section class="topcard">
  <h1>Jane</h1>
  <div class="actions">
    <a href="/preload/custom-invite/?vanityName={USER}"
      aria-label="{labels.connect}">{labels.connect}</a>
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3AEEE"
      >{labels.message}</a>
    <button type="button" aria-expanded="false">{labels.more}</button>
  </div>
</section>
"""


def self_top_card(labels: Labels) -> str:
    """Own profile: the edit-intro URL, and no compose action at all."""
    return f"""
<section class="topcard">
  <h1>Daniel</h1>
  <div class="actions">
    <a href="/in/{USER}/edit/intro/" aria-label="{labels.edit}">{labels.edit}</a>
    <button type="button" aria-expanded="false">{labels.more}</button>
  </div>
</section>
"""


def restricted_top_card(labels: Labels) -> str:
    """Out-of-network profile: nothing but the More menu to act on."""
    return f"""
<section class="topcard">
  <h1>Unknown</h1>
  <div class="actions">
    <button type="button" aria-expanded="false">{labels.more}</button>
  </div>
</section>
"""


def expander_first_bar(labels: Labels) -> str:
    """DOM-order guard: the expander leads, so this is not an action row."""
    return f"""
<section class="hostile">
  <div class="bar">
    <button type="button" aria-expanded="false">{labels.settings}</button>
    <button type="button" aria-label="{labels.like}">A</button>
    <button type="button" aria-label="{labels.comment}">B</button>
  </div>
</section>
"""


def extra_button_row(labels: Labels) -> str:
    """Count guard: a fourth button, unlabeled, after the expander."""
    return f"""
<section class="hostile">
  <div class="bar">
    <button type="button" aria-label="{labels.like}">A</button>
    <button type="button" aria-label="{labels.comment}">B</button>
    <button type="button" aria-expanded="false">{labels.more}</button>
    <button type="button">{labels.show_all}</button>
  </div>
</section>
"""


def _page_html(*sections: str) -> str:
    return f"<html><body><main>{''.join(sections)}</main></body></html>"


def _both(first: Build, second: Build) -> Build:
    """One page carrying two of the sections above, in that order."""
    return lambda labels: first(labels) + second(labels)


@pytest.fixture
async def dom_page():
    """Real chromium page, or skip when no browser is installed.

    Only launch/setup is guarded by the skip — the ``yield`` is outside it
    so an assertion failure or JS error in a test body is never swallowed
    into a skip.

    ``channel="chromium"`` names the browser this project installs. Without
    it Playwright picks the *binary* from the ``headless`` flag alone and
    asks for ``chromium-headless-shell``, which nothing here installs since
    the setup moved to ``--no-shell``: the launch would fail and every case
    in this file would skip itself, silently, wherever the real browser is.
    """
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


def _actions(page) -> ConnectionActions:
    """The owner wired the way the facade does, over a real browser page.

    The main-profile read is never reached: every case here stops at the
    signal read, so the borrow is a callable that refuses to be called.
    """

    async def unreachable(_username: str) -> dict[str, Any]:
        raise AssertionError("the DOM cases never read a profile")

    session = ScrapingSession(cast(Page, page))
    return ConnectionActions(session, PageNavigator(session), unreachable)


async def _signals(page, html: str) -> dict:
    await page.set_content(_page_html(html))
    return await page.evaluate(ACTION_SIGNALS_JS, USER)


async def _fingerprint(page, html: str) -> bool:
    return bool((await _signals(page, html))["hasIncomingActionRow"])


async def _state(page, html: str) -> ConnectionState:
    """Drive the production probe and classifier over one rendered page."""
    await page.set_content(_page_html(html))
    signals = await _actions(page)._read_action_signals(USER)
    return detect_connection_state(signals)


async def _click(page, html: str, program: str) -> tuple[bool, str | None]:
    """Run one click program: what it reported, and what it actually hit.

    Patchright evaluates in an isolated world, so page-world variables are
    invisible there, but the DOM is shared — the inline onclick records the
    click as a body attribute.
    """
    await page.set_content(_page_html(html))
    clicked = bool(await page.evaluate(program))
    recorded = await page.evaluate("document.body.getAttribute('data-clicked')")
    return (clicked, recorded)


async def _in_every_locale(
    page,
    build: Build,
    expected: Any,
    read: Callable[[Any, str], Awaitable[Any]],
) -> None:
    """Assert one answer for the same structure in all four label sets.

    The assertion carries the whole mapping rather than one locale at a
    time, so a divergence names the locale that diverged instead of failing
    on whichever case ran first.
    """
    answers = {labels.locale: await read(page, build(labels)) for labels in LOCALES}
    assert answers == {labels.locale: expected for labels in LOCALES}


STATE_CASES: tuple[tuple[str, Build, ConnectionState], ...] = (
    ("self_profile", self_top_card, "self_profile"),
    ("connectable", connectable_top_card, "connectable"),
    (
        "incoming_request",
        _both(incoming_top_card, sidebar_section),
        "incoming_request",
    ),
    ("pending", pending_top_card, "pending"),
    ("already_connected", connected_top_card, "already_connected"),
    ("follow_only", follow_only_top_card, "follow_only"),
    ("unavailable", restricted_top_card, "unavailable"),
)

FINGERPRINT_CASES: tuple[tuple[str, Build, bool], ...] = (
    ("incoming-next-to-sidebar-cards", _both(incoming_top_card, sidebar_section), True),
    ("cover-video-expander-first-in-card", incoming_top_card_with_cover, True),
    ("video-player-bar", _both(connected_top_card, video_player_bar), False),
    ("expander-first-order-guard", expander_first_bar, False),
    (
        "matching-widget-outside-top-card",
        _both(connected_top_card, unrelated_matching_widget),
        False,
    ),
    ("extra-unlabeled-button", extra_button_row, False),
    ("follow-only-row", follow_only_top_card, False),
    ("pending-row", pending_top_card, False),
    ("connected-row", _both(connected_top_card, sidebar_section), False),
)


class TestConnectionStateIsStructural:
    """Every state the classifier can reach, decided in all four label sets.

    Each case runs the real probe and the real classifier, so it covers the
    whole path from rendered markup to the decision the write gate reads.
    """

    @pytest.mark.parametrize(
        ("build", "expected"),
        [case[1:] for case in STATE_CASES],
        ids=[case[0] for case in STATE_CASES],
    )
    async def test_state_is_the_same_in_every_locale(self, dom_page, build, expected):
        await _in_every_locale(dom_page, build, expected, _state)

    async def test_a_sidebar_invite_for_another_user_is_not_connectable(self, dom_page):
        # The invite anchor is vanityName-scoped, so a mutual-connection card
        # offering Connect for somebody else must not open the write gate.
        await _in_every_locale(
            dom_page,
            _both(connected_top_card, sidebar_section),
            "already_connected",
            _state,
        )


class TestIncomingActionRowFingerprint:
    """The structural fingerprint, positive and negative, in all four label sets."""

    @pytest.mark.parametrize(
        ("build", "expected"),
        [case[1:] for case in FINGERPRINT_CASES],
        ids=[case[0] for case in FINGERPRINT_CASES],
    )
    async def test_fingerprint_answers_the_same_in_every_locale(
        self, dom_page, build, expected
    ):
        await _in_every_locale(dom_page, build, expected, _fingerprint)


class TestActionChoiceIsStructural:
    """Which control each write-side program picks, in all four label sets."""

    async def test_accept_clicks_the_first_labeled_button_only(self, dom_page):
        # Clicking the second labeled button would silently and irreversibly
        # Ignore the request, and the difference between the two is nothing
        # but their words.
        await _in_every_locale(
            dom_page,
            _both(incoming_top_card, sidebar_section),
            (True, "first-labeled"),
            lambda page, html: _click(page, html, CLICK_INCOMING_ACCEPT_JS),
        )

    async def test_accept_does_not_click_without_a_fingerprint_match(self, dom_page):
        await _in_every_locale(
            dom_page,
            _both(follow_only_top_card, video_player_bar),
            (False, None),
            lambda page, html: _click(page, html, CLICK_INCOMING_ACCEPT_JS),
        )

    async def test_the_more_opener_is_the_expander_not_the_primary(self, dom_page):
        # The More button is the one control in the action row *without* an
        # aria-label, and the Follow button beside it is the one with one. A
        # text match cannot pick that control consistently across these
        # fixtures; the attribute picks the same one in all of them.
        await _in_every_locale(
            dom_page,
            follow_only_top_card,
            (True, "expander"),
            lambda page, html: _click(page, html, OPEN_MORE_BUTTON_JS),
        )

    async def test_no_more_opener_outside_an_action_root(self, dom_page):
        # No compose anchor means no action root, so there is no More button
        # to find even though the page renders one.
        await _in_every_locale(
            dom_page,
            self_top_card,
            (False, None),
            lambda page, html: _click(page, html, OPEN_MORE_BUTTON_JS),
        )


# Withdraw. The Pending control and the confirmation dialog it opens are the
# two things a withdrawal clicks; every other control in these fixtures records
# a click too, so a wrong pick shows up as the wrong name rather than as a
# silent pass.


def pending_top_card_clickable(labels: Labels) -> str:
    """Pending, with every control recording what a click would have hit."""
    return f"""
<section class="topcard">
  <h1>Florian</h1>
  <div class="actions">
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3ADDD"
      onclick="event.preventDefault();
        document.body.setAttribute('data-clicked','message')"
      >{labels.message}</a>
    <a href="https://www.linkedin.com/in/florian/" aria-label="{labels.pending}"
      onclick="event.preventDefault();
        document.body.setAttribute('data-clicked','pending')"
      >{labels.pending}</a>
    <button type="button" aria-expanded="false"
      onclick="document.body.setAttribute('data-clicked','more')"
      >{labels.more}</button>
  </div>
</section>
"""


def two_labeled_anchors_top_card(labels: Labels) -> str:
    """Two labeled anchors in the action root: which one is Pending is a guess."""
    return f"""
<section class="topcard">
  <h1>Florian</h1>
  <div class="actions">
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3ADDD"
      >{labels.message}</a>
    <a href="https://www.linkedin.com/in/florian/" aria-label="{labels.pending}"
      onclick="event.preventDefault();
        document.body.setAttribute('data-clicked','pending')"
      >{labels.pending}</a>
    <a href="https://www.linkedin.com/in/florian/follow/" aria-label="{labels.follow}"
      onclick="event.preventDefault();
        document.body.setAttribute('data-clicked','other-labeled')"
      >{labels.follow}</a>
    <button type="button" aria-expanded="false">{labels.more}</button>
  </div>
</section>
"""


def withdraw_dialog(labels: Labels, *, native: bool = True, extra: str = "") -> str:
    """The confirmation: Dismiss first, then Cancel, then the primary Withdraw."""
    buttons = f"""
    <button type="button" aria-label="{labels.dismiss}"
      onclick="document.body.setAttribute('data-clicked','dismiss')">X</button>
    <button type="button"
      onclick="document.body.setAttribute('data-clicked','cancel')"
      >{labels.cancel}</button>
    <button type="button" aria-label="{labels.withdraw}"
      onclick="document.body.setAttribute('data-clicked','withdraw')"
      >{labels.withdraw}</button>
"""
    if native:
        return f'<dialog open id="withdraw" {extra}>{buttons}</dialog>'
    return f'<div role="dialog" id="withdraw" {extra}>{buttons}</div>'


def spinner_dialog(labels: Labels) -> str:
    """What the withdraw dialog shows before its actions mount: Dismiss only."""
    return f"""
<dialog open><div class="spinner"></div>
  <button type="button" aria-label="{labels.dismiss}">X</button>
</dialog>
"""


def chat_overlay(labels: Labels) -> str:
    """The conversation LinkedIn keeps open across pages: a dialog with a composer."""
    return f"""
<div role="dialog" class="chat">
  <form>
    <div role="textbox" contenteditable="true"
      style="display:block;width:200px;height:30px"></div>
    <button type="submit" disabled>{labels.message}</button>
    <button type="button"
      onclick="document.body.setAttribute('data-clicked','chat')"
      >{labels.more}</button>
  </form>
</div>
"""


def hidden_preloaded_dialog(labels: Labels) -> str:
    """A preloaded container: hidden, with a disabled submit as its last button."""
    return f"""
<div role="dialog" style="display:none">
  <button type="submit" disabled
    onclick="document.body.setAttribute('data-clicked','decoy')"
    >{labels.withdraw}</button>
</div>
"""


def _body(*parts: str) -> str:
    return f"<html><body>{''.join(parts)}</body></html>"


async def _confirm(page, html: str) -> tuple[int, bool, str | None]:
    """The count the settle wait reads, then the confirm click and what it hit."""
    await page.set_content(html)
    count = await page.evaluate(CONFIRM_DIALOG_BUTTON_COUNT_JS)
    clicked = bool(await page.evaluate(CLICK_CONFIRM_DIALOG_PRIMARY_JS))
    recorded = await page.evaluate("document.body.getAttribute('data-clicked')")
    return (count, clicked, recorded)


class TestWithdrawAnchorIsStructural:
    """The Pending click, in all four label sets."""

    async def test_the_clickable_fixture_is_a_pending_profile(self, dom_page):
        await _in_every_locale(dom_page, pending_top_card_clickable, "pending", _state)

    async def test_withdraw_clicks_the_pending_anchor_only(self, dom_page):
        # Message is an anchor as well and sits first; More is the other
        # control in the row. Only the labeled anchor is Pending.
        await _in_every_locale(
            dom_page,
            pending_top_card_clickable,
            (True, "pending"),
            lambda page, html: _click(page, html, CLICK_WITHDRAW_ANCHOR_JS),
        )

    async def test_two_labeled_anchors_are_not_guessed_between(self, dom_page):
        await _in_every_locale(
            dom_page,
            two_labeled_anchors_top_card,
            (False, None),
            lambda page, html: _click(page, html, CLICK_WITHDRAW_ANCHOR_JS),
        )

    @pytest.mark.parametrize(
        "build",
        [connected_top_card, follow_only_top_card, self_top_card, restricted_top_card],
        ids=["connected", "follow-only", "own-profile", "restricted"],
    )
    async def test_nothing_is_clicked_without_a_pending_anchor(self, dom_page, build):
        await _in_every_locale(
            dom_page,
            build,
            (False, None),
            lambda page, html: _click(page, html, CLICK_WITHDRAW_ANCHOR_JS),
        )


class TestConfirmDialogChoice:
    """Which dialog the withdraw confirmation counts and clicks.

    The primary action is the last button of the one open dialog. Everything
    else a LinkedIn page can hold at the same time is in these fixtures: a
    conversation overlay with a composer, and hidden preloaded containers whose
    last button is disabled.
    """

    @pytest.mark.parametrize(
        "native", [True, False], ids=["native-dialog", "role-dialog"]
    )
    @pytest.mark.parametrize("decoy_first", [False, True], ids=["after", "before"])
    async def test_the_open_dialog_wins_over_a_hidden_preloaded_one(
        self, dom_page, native, decoy_first
    ):
        def build(labels: Labels) -> str:
            parts = [
                withdraw_dialog(labels, native=native),
                hidden_preloaded_dialog(labels),
            ]
            return _body(*(reversed(parts) if decoy_first else parts))

        await _in_every_locale(dom_page, build, (3, True, "withdraw"), _confirm)

    @pytest.mark.parametrize("chat_first", [False, True], ids=["after", "before"])
    async def test_a_chat_overlay_is_never_the_confirmation(self, dom_page, chat_first):
        def build(labels: Labels) -> str:
            parts = [withdraw_dialog(labels, native=False), chat_overlay(labels)]
            return _body(*(reversed(parts) if chat_first else parts))

        await _in_every_locale(dom_page, build, (3, True, "withdraw"), _confirm)

    @pytest.mark.parametrize(
        "build",
        [chat_overlay, hidden_preloaded_dialog, lambda _labels: ""],
        ids=["chat-overlay-alone", "hidden-dialog-alone", "no-dialog"],
    )
    async def test_nothing_is_confirmed_without_an_open_dialog(self, dom_page, build):
        await _in_every_locale(
            dom_page,
            lambda labels: _body(build(labels)),
            (-1, False, None),
            _confirm,
        )

    async def test_a_modal_dialog_is_seen(self, dom_page):
        # showModal() puts the dialog in the top layer with position: fixed,
        # where offsetParent is null although the dialog covers the page.
        async def confirm_modal(page, html: str) -> tuple[int, bool, str | None]:
            await page.set_content(html)
            await page.evaluate("document.getElementById('withdraw').showModal()")
            count = await page.evaluate(CONFIRM_DIALOG_BUTTON_COUNT_JS)
            clicked = bool(await page.evaluate(CLICK_CONFIRM_DIALOG_PRIMARY_JS))
            recorded = await page.evaluate("document.body.getAttribute('data-clicked')")
            return (count, clicked, recorded)

        await _in_every_locale(
            dom_page,
            lambda labels: _body(
                withdraw_dialog(labels).replace("<dialog open", "<dialog"),
                hidden_preloaded_dialog(labels),
            ),
            (3, True, "withdraw"),
            confirm_modal,
        )

    async def test_a_fixed_position_dialog_is_seen(self, dom_page):
        await _in_every_locale(
            dom_page,
            lambda labels: _body(
                withdraw_dialog(
                    labels, native=False, extra='style="position:fixed;top:20%"'
                ),
                hidden_preloaded_dialog(labels),
            ),
            (3, True, "withdraw"),
            _confirm,
        )

    async def test_the_spinner_state_reads_as_one_button(self, dom_page):
        # The settle wait holds the click until a second button mounts; this
        # is the count it waits past.
        async def count(page, html: str) -> int:
            await page.set_content(html)
            return await page.evaluate(CONFIRM_DIALOG_BUTTON_COUNT_JS)

        await _in_every_locale(
            dom_page, lambda labels: _body(spinner_dialog(labels)), 1, count
        )


def _actions_reading(page, text: str) -> ConnectionActions:
    """The owner over a real page, with a profile read that never navigates.

    Every read answers the same text, so the verdict comes from the signals
    the real probe reads off the page as the flow leaves it.
    """

    async def read(_username: str) -> dict[str, Any]:
        return {
            "url": f"https://www.linkedin.com/in/{USER}/",
            "sections": {"main_profile": text},
        }

    session = ScrapingSession(cast(Page, page))
    return ConnectionActions(session, PageNavigator(session), read)


# The Pending anchor opens a modal the way LinkedIn's does: the shell at once,
# with only its Dismiss control while it loads, and Cancel/Withdraw a moment
# later. Withdraw closes it and puts Connect back in the top card.
_WITHDRAW_FLOW = f"""
<html><body>
<main>
<section class="topcard">
  <h1>Florian</h1>
  <div class="actions">
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3ADDD">Message</a>
    <a id="pending" href="https://www.linkedin.com/in/{USER}/"
      aria-label="Pending, click to withdraw the invitation sent to Florian"
      onclick="event.preventDefault(); openWithdraw();">Pending</a>
    <button type="button" aria-expanded="false">More</button>
  </div>
</section>
</main>
<dialog id="confirm"><button type="button" aria-label="Dismiss">X</button></dialog>
<script>
  function openWithdraw() {{
    const dialog = document.getElementById('confirm');
    dialog.showModal();
    setTimeout(() => {{
      dialog.insertAdjacentHTML('beforeend',
        '<button type="button">Cancel</button>' +
        '<button type="button" onclick="confirmWithdraw()">Withdraw</button>');
    }}, 400);
  }}
  function confirmWithdraw() {{
    document.body.dataset.withdrawn = 'yes';
    const dialog = document.getElementById('confirm');
    dialog.close();
    dialog.remove();
    const invite = document.createElement('a');
    invite.href = '/preload/custom-invite/?vanityName={USER}';
    invite.setAttribute('aria-label', 'Invite Florian to connect');
    invite.textContent = 'Connect';
    document.getElementById('pending').replaceWith(invite);
  }}
</script>
</body></html>
"""


class TestWithdrawFlow:
    """The whole withdrawal over a real page: click, settle, confirm, re-read."""

    async def test_a_pending_invitation_is_withdrawn_through_its_modal(self, dom_page):
        await dom_page.set_content(_WITHDRAW_FLOW)

        result = await _actions_reading(dom_page, "Florian").withdraw_invitation(USER)

        assert await dom_page.evaluate("document.body.dataset.withdrawn") == "yes"
        assert result["status"] == "withdrawn"
        assert "connectable" in result["message"]
