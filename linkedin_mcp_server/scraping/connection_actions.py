"""Invitation actions taken on a loaded person page.

``connection.py`` answers what a profile's action area *says* about the
relationship and stays browser-free; this module is everything that reads
or touches that area: the structural signal probe, the More menu, the
incoming-request Accept click, the invite dialog, the non-submitting
note-quota probe and the verification re-read after a write.

Per the AGENTS.md Scraping Rules every decision here rests on a URL pattern
(``/preload/custom-invite/?vanityName=USER``, ``/in/USER/edit/intro/``,
``/messaging/compose/``), on the *presence* of an ARIA attribute
(``aria-label`` on a button versus an anchor, ``aria-expanded`` on the menu
opener) or on a structural count. No label value is read anywhere, so a
German or an opaquely labelled page classifies exactly as an English one;
``tests/test_action_signals_dom.py`` holds that line against a real DOM in
all four label sets.

The write gate is the reason the order of the checks below matters: the
invite deeplink fires only after ``has_invite_anchor`` is true, and the only
other thing that may open it is the note-quota probe, which never clicks a
primary button.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import asyncio
import logging

from patchright.async_api import ElementHandle, JSHandle
from patchright.async_api import TimeoutError as PlaywrightTimeoutError

import linkedin_mcp_server.scraping.connection as connection
from linkedin_mcp_server.scraping.connection import ActionSignals
from linkedin_mcp_server.scraping.identifiers import (
    normalize_person_identifier,
    person_profile_url,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

logger = logging.getLogger(__name__)

# A messaging overlay (a minimised chat bubble LinkedIn keeps open across
# pages) is also a dialog. Its composer never belongs to an invite, and its
# buttons would otherwise join the positional picks below: measured live in
# September 2026, the last one was the chat's "Open send options" toggle.
_NOT_MESSAGING = ':not(:has([contenteditable="true"]))'
_DIALOG_SELECTOR = f'dialog[open]{_NOT_MESSAGING}, [role="dialog"]{_NOT_MESSAGING}'
_DIALOG_PREMIUM_LINK_SELECTOR = (
    'dialog[open] a[href*="/premium/"], [role="dialog"] a[href*="/premium/"]'
)
# Every visible non-messaging dialog. A wait for this to be hidden ends only
# once none is left: without the filter the wait judges the first match alone,
# and a hidden preloaded dialog earlier in the document answered "closed" at
# once (measured in Chromium) while the invite, or the upsell that replaced
# it, was still on screen.
_VISIBLE_DIALOG_SELECTOR = f"{_DIALOG_SELECTOR} >> visible=true"

# The longest note any account can send: LinkedIn takes 300 characters with
# Premium and 200 without. Which of the two applies is only known once the
# invite dialog is open (its note field's maxlength, or what the field kept),
# so the read-back after the fill is the guard and this only refuses early a
# note no account could send. Counted in UTF-16 code units, as Chromium counts
# maxlength: measured 2026-09-30, an emoji takes two of them and a CR LF pair
# is stored as one LF.
MAX_INVITE_NOTE_LENGTH = 300

# How long the invite dialog's primary must stay disabled over an empty note
# field, with no note to give, before the call reads it as asking for one.
NOTE_REQUIRED_SETTLE_SECONDS = 1.5
_INVITE_POLL_SECONDS = 0.25

# Shared JS function that walks up from any /messaging/compose/ anchor
# inside <main> to find the smallest ancestor that satisfies the
# action-root predicate (>=2 interactive children, >=1 button). This is
# the top-card action row regardless of LinkedIn's class names.
#
# Inlined into both ACTION_SIGNALS_JS and OPEN_MORE_BUTTON_JS so a
# single change to the heuristic propagates to both call sites.
_FIND_ACTION_ROOT_FN_JS = r"""
function findActionRoot(main) {
  const composeAnchors = main.querySelectorAll('a[href*="/messaging/compose/"]');
  for (const a of composeAnchors) {
    let el = a.parentElement;
    while (el && el !== main) {
      const interactive = el.querySelectorAll('button, a').length;
      const buttons = el.querySelectorAll('button').length;
      if (interactive >= 2 && buttons >= 1) {
        return el;
      }
      el = el.parentElement;
    }
  }
  return null;
}
"""

# Shared JS function that fingerprints the incoming-request action row.
# Incoming-request profiles render no Message button in the top card, so
# findActionRoot (compose-anchor walk) cannot locate their action row and
# would mis-anchor on sidebar mutual-connection cards instead. This walk
# anchors on button[aria-expanded] (the More button) and validates the
# smallest multi-button ancestor against the fingerprint verified live
# 2026-06-11 on two German-locale incoming-request profiles:
#
#   [button aria-label (Accept)] [button aria-label (Ignore)]
#   [button aria-expanded, no aria-label (More)]
#
# All checks are attribute presence and structural counts per the
# AGENTS.md Scraping Rules — no label values are read. Every guard kills
# a known false positive: total-button-count === 3 and labeled === 2
# exclude video-player control bars (play/mute/captions all carry
# aria-label); the unlabeled-expander check excludes player settings
# expanders (the profile More button never carries aria-label); the
# DOM-order guard excludes bars with trailing labeled buttons; the
# compose/invite/labeled-anchor exclusions kill follow_only, pending,
# connected top cards and sidebar cards. The scan continues over ALL
# expander candidates because cover-video profiles render the player's
# expander before the top-card row in DOM order.
#
# The search is scoped to the top card — the first <section> of <main>
# (falling back to main's first child, then main). Profile pages render
# the action row in the top card; feed, "people also viewed", and other
# widgets live in later sections. Without the scope an unrelated widget
# elsewhere in main with the same button shape could be misclassified and
# its first labeled button clicked.
#
# Inlined into ACTION_SIGNALS_JS and CLICK_INCOMING_ACCEPT_JS so a
# single change to the fingerprint propagates to both call sites.
_FIND_INCOMING_ACTION_ROW_FN_JS = r"""
function findIncomingActionRow(main) {
  const scope = main.querySelector('section') || main.firstElementChild || main;
  const matches = [];
  for (const expander of scope.querySelectorAll('button[aria-expanded]')) {
    let el = expander.parentElement;
    while (el && el !== scope && el !== main) {
      if (el.querySelectorAll('button').length >= 2) {
        const buttons = el.querySelectorAll('button');
        const labeled = el.querySelectorAll('button[aria-label]');
        const expanders = el.querySelectorAll('button[aria-expanded]');
        if (
          buttons.length === 3 &&
          labeled.length === 2 &&
          expanders.length === 1 &&
          !expanders[0].hasAttribute('aria-label') &&
          expanders[0].compareDocumentPosition(labeled[1]) &
            Node.DOCUMENT_POSITION_PRECEDING &&
          !el.querySelector('a[href*="/messaging/compose/"]') &&
          !el.querySelector('a[href*="/preload/custom-invite/"]') &&
          !el.querySelector('a[aria-label]')
        ) {
          matches.push(el);
        }
        break;
      }
      el = el.parentElement;
    }
  }
  // Require a unique match: a profile's top card has exactly one action
  // row. Ambiguity (two rows matching the shape) is treated as no match so
  // the irreversible Accept click never fires on a guessed control.
  return matches.length === 1 ? matches[0] : null;
}
"""

# Locale-independent connection-state probe. Returns four booleans;
# per AGENTS.md Scraping Rules, every signal is based on URL patterns
# or ARIA-attribute *presence* — never on label text values.
#
# - hasInvite: vanityName-scoped invite anchor anywhere in document.
#   Searches document (not main) so a post-More-menu reread sees
#   portal-rendered menu items. The vanityName parameter is unique to
#   the target user, so document-wide search has no false-positive risk.
# - hasComposeInActionRoot: any /messaging/compose/ anchor exists inside
#   the action root. Scoped to main (not document) to avoid the More
#   menu's "Send profile in a message" anchor, which is a compose URL
#   but lives outside the action area.
# - hasEditIntro: edit-intro URL exists, only rendered on own profile.
# - hasLabeledActionButton: at least one <button[aria-label]> inside the
#   action root. Primary action buttons (Follow / Connect /
#   Save in Sales Navigator) carry aria-label for screen readers; the
#   profile More button uses aria-expanded instead and is not counted.
# - hasLabeledActionAnchor: at least one <a[aria-label]> inside the
#   action root. LinkedIn renders the Pending state as an anchor (linking
#   back to the profile URL) carrying aria-label like "Pending, click to
#   withdraw…". The Message anchor has only aria-disabled, so a labeled
#   anchor is the locale-independent Pending signal.
# - hasIncomingActionRow: the incoming-request fingerprint matched (see
#   _FIND_INCOMING_ACTION_ROW_FN_JS). Computed independently of
#   findActionRoot, which cannot locate the top-card row on incoming
#   profiles (no compose anchor there) and would mis-anchor on sidebar
#   cards.
#
# The username is CSS-escaped before interpolation into attribute
# selectors to defend against malformed inputs containing characters
# that would otherwise break the selector syntax (quotes, brackets).
ACTION_SIGNALS_JS = (
    r"""
((username) => {
"""
    + _FIND_ACTION_ROOT_FN_JS
    + _FIND_INCOMING_ACTION_ROW_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return null;

  const safe = CSS.escape(username);
  const inviteSel = `a[href*="/preload/custom-invite/?vanityName=${safe}"]`;
  const editSel = `a[href*="/in/${safe}/edit/intro/"]`;

  const hasInvite = !!document.querySelector(inviteSel);
  const hasEditIntro = !!main.querySelector(editSel);

  const actionRoot = findActionRoot(main);

  let hasComposeInActionRoot = false;
  let hasLabeledActionButton = false;
  let hasLabeledActionAnchor = false;
  if (actionRoot) {
    hasComposeInActionRoot =
      !!actionRoot.querySelector('a[href*="/messaging/compose/"]');
    for (const b of actionRoot.querySelectorAll('button')) {
      if (b.hasAttribute('aria-label')) {
        hasLabeledActionButton = true;
        break;
      }
    }
    for (const a of actionRoot.querySelectorAll('a')) {
      if (a.hasAttribute('aria-label')) {
        hasLabeledActionAnchor = true;
        break;
      }
    }
  }

  return {
    hasInvite,
    hasComposeInActionRoot,
    hasEditIntro,
    hasLabeledActionButton,
    hasLabeledActionAnchor,
    hasIncomingActionRow: !!findIncomingActionRow(main),
  };
})
"""
)

# Open the profile's More button, located inside the action root via the
# aria-expanded attribute. The aria-expanded attribute uniquely identifies
# the menu opener without text labels (the More button has no aria-label,
# while Follow/Connect/Pending buttons do — the inverse pattern). Returns
# true iff the click landed; the caller waits for [role='menu'] visibility
# before re-scanning signals.
OPEN_MORE_BUTTON_JS = (
    r"""
(() => {
"""
    + _FIND_ACTION_ROOT_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return false;
  const actionRoot = findActionRoot(main);
  if (!actionRoot) return false;
  const moreBtn = actionRoot.querySelector('button[aria-expanded]');
  if (!moreBtn) return false;
  moreBtn.click();
  return true;
})
"""
)

# Click Accept on an incoming-request profile. Accept is the FIRST labeled
# button in the fingerprinted row — primary actions render first in
# top-card action rows (Connect/Message lead on other profile states; the
# inverse of dialogs, where the primary button renders last). Clicking the
# second button would silently and irreversibly Ignore the request, so the
# click only fires when the full fingerprint matched.
CLICK_INCOMING_ACCEPT_JS = (
    r"""
(() => {
"""
    + _FIND_INCOMING_ACTION_ROW_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return false;
  const row = findIncomingActionRow(main);
  if (!row) return false;
  row.querySelectorAll('button[aria-label]')[0].click();
  return true;
})
"""
)

# Open the More menu of the fingerprinted incoming-request row, the disprove
# step that runs before Accept. A creator-mode profile renders
# [Follow][Save in Sales Navigator][More] with no Message action, which is
# the incoming fingerprint exactly, with Connect demoted into that More menu
# (issue #629). LinkedIn mounts the menu's invite anchor only once More is
# clicked, so nothing on the closed page tells the two rows apart, and
# Accept on the wrong one clicks Follow. The row is re-derived here rather
# than trusted from an earlier read, so the click can only land on that
# row's own expander.
OPEN_INCOMING_ROW_MORE_JS = (
    r"""
(() => {
"""
    + _FIND_INCOMING_ACTION_ROW_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return false;
  const row = findIncomingActionRow(main);
  if (!row) return false;
  const opener = row.querySelector('button[aria-expanded]');
  if (!opener) return false;
  opener.click();
  return true;
})
"""
)


# Click the Pending control that withdraws a sent invitation. LinkedIn renders
# Pending as the one labeled <a> in the top-card action root (the signal
# ``hasLabeledActionAnchor`` reads), so the click fires only when exactly one
# such anchor exists; zero or several is a page this cannot reason about.
#
# The root must also sit in the top card, the scope the Accept row is held to.
# ``findActionRoot`` walks from the first compose anchor in <main>, and on a
# profile whose top card offers no Message that is a sidebar card's: the root
# is then the sidebar, and its one labeled anchor (a Message link for another
# member) reads as Pending. The state read cannot tell; the click can refuse.
CLICK_WITHDRAW_ANCHOR_JS = (
    r"""
(() => {
"""
    + _FIND_ACTION_ROOT_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return false;
  const actionRoot = findActionRoot(main);
  if (!actionRoot) return false;
  const topCard = main.querySelector('section') || main.firstElementChild || main;
  if (!topCard.contains(actionRoot)) return false;
  const anchors = actionRoot.querySelectorAll('a[aria-label]');
  if (anchors.length !== 1) return false;
  anchors[0].click();
  return true;
})
"""
)

# The ONE open confirmation dialog: a visible native dialog[open] if there is
# one, else a visible [role="dialog"], and never a messaging overlay (same
# exclusion as ``_DIALOG_SELECTOR``). A page-wide "last button" can land in a
# hidden preloaded container instead: measured live 2026-08-21 on the
# invitation manager, the page-wide last match was a hidden, disabled submit
# button.
#
# One, not the first: with two open (a popup of LinkedIn's own beside the one
# the click opened) nothing structural says which is the confirmation, and the
# last button of the other is somebody else's action. Ambiguity is no dialog,
# as it is for the Accept row. A dialog inside the open one is the same one.
#
# Visible means a rendered box, the test ``message_sender`` uses too, and not
# ``offsetParent``: that is null for anything position: fixed, which is what
# showModal() makes a dialog, so the modal on screen read as hidden.
_FIND_CONFIRM_DIALOG_FN_JS = r"""
function findConfirmDialog() {
  const visible = el =>
    !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length) &&
    getComputedStyle(el).visibility !== 'hidden';
  const usable = el =>
    visible(el) && !el.querySelector('[contenteditable="true"]');
  const open = selector => {
    const found = [...document.querySelectorAll(selector)].filter(usable);
    return found.filter(el => !found.some(other => other !== el && other.contains(el)));
  };
  let dialogs = open('dialog[open]');
  if (dialogs.length === 0) dialogs = open('[role="dialog"]');
  return dialogs.length === 1 ? dialogs[0] : null;
}
"""

CONFIRM_DIALOG_BUTTON_COUNT_JS = (
    r"""
(() => {
"""
    + _FIND_CONFIRM_DIALOG_FN_JS
    + r"""
  const dialog = findConfirmDialog();
  return dialog ? dialog.querySelectorAll('button').length : -1;
})
"""
)

# The primary action renders last in LinkedIn dialogs, the convention the
# invite dialog's submit relies on too.
CLICK_CONFIRM_DIALOG_PRIMARY_JS = (
    r"""
(() => {
"""
    + _FIND_CONFIRM_DIALOG_FN_JS
    + r"""
  const dialog = findConfirmDialog();
  if (!dialog) return false;
  const buttons = dialog.querySelectorAll('button');
  if (buttons.length === 0) return false;
  buttons[buttons.length - 1].click();
  return true;
})
"""
)

# The ONE open invite dialog, which everything that reads, fills or clicks the
# invite goes through. Visible and never a messaging overlay, by the same tests
# as ``findConfirmDialog``, and a dialog inside another one is that one. Unlike
# the confirmation, native dialog[open] and [role="dialog"] count together and
# neither is preferred: the submit clicks by position, and it is the one write
# here whose result reaches a member, so a second open dialog is a refusal
# rather than a tie to break.
#
# Page-wide picks reached past the invite. Measured in Chromium on upstream
# v4.26.1: a hidden preloaded [role=dialog] holding a textarea took the note
# fill, and one after the invite gave the page its last, disabled button, so
# both calls ended connect_unavailable for a dialog that was fine. A native
# <dialog open> without a role had no button under the old selector list,
# whose first entry then matched the dialog itself as the last "button": the
# click landed on the container and the call reported a submit that sent
# nothing.
#
# The note field is the dialog's first visible textarea, else its first one:
# a hidden one is still a field the upsell left behind, which the quota
# checks below rely on telling apart from a missing one.
_FIND_INVITE_DIALOG_FN_JS = r"""
function isShown(el) {
  return !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length) &&
    getComputedStyle(el).visibility !== 'hidden';
}
function openInviteDialogs() {
  const found = [...document.querySelectorAll('dialog[open], [role="dialog"]')]
    .filter(el => isShown(el) && !el.querySelector('[contenteditable="true"]'));
  return found.filter(el => !found.some(other => other !== el && other.contains(el)));
}
function inviteButtons(dialog) {
  return [...dialog.querySelectorAll('button, [role="button"]')];
}
function inviteNoteField(dialog) {
  const fields = [...dialog.querySelectorAll('textarea')];
  return fields.find(isShown) || fields[0] || null;
}
"""

# One read of the invite dialog. The primary is its last button, the
# convention every LinkedIn dialog here follows. Disabled is the property or
# ``aria-disabled="true"``, an attribute value and no label.
INVITE_DIALOG_STATE_JS = (
    r"""
(() => {
"""
    + _FIND_INVITE_DIALOG_FN_JS
    + r"""
  const dialogs = openInviteDialogs();
  if (dialogs.length !== 1) return {inviteDialogs: dialogs.length};
  const buttons = inviteButtons(dialogs[0]);
  const primary = buttons[buttons.length - 1];
  const note = inviteNoteField(dialogs[0]);
  return {
    inviteDialogs: 1,
    buttons: buttons.length,
    primaryDisabled: !!primary &&
      (primary.disabled === true || primary.getAttribute('aria-disabled') === 'true'),
    noteField: note ? (isShown(note) ? 'visible' : 'hidden') : 'none',
    noteValue: note ? note.value : '',
    noteMaxLength: note && note.maxLength > 0 ? note.maxLength : null,
  };
})
"""
)

# The element an action targets, resolved in the same call that finds the
# dialog: its note field, or the button ``fromEnd`` places before its last.
# Null unless exactly one dialog is open. The click and fill themselves go
# through Playwright on the returned handle, so they are real input with
# actionability checks rather than a scripted ``click()``.
INVITE_DIALOG_ELEMENT_JS = (
    r"""
((target) => {
"""
    + _FIND_INVITE_DIALOG_FN_JS
    + r"""
  const dialogs = openInviteDialogs();
  if (dialogs.length !== 1) return null;
  if (target.part === 'note') return inviteNoteField(dialogs[0]);
  const buttons = inviteButtons(dialogs[0]);
  const index = buttons.length - 1 - target.fromEnd;
  return index >= 0 ? buttons[index] : null;
})
"""
)

# Pause between reads of a withdrawal LinkedIn propagates asynchronously.
WITHDRAW_SETTLE_SECONDS = 3.0


class InviteNotSent(Exception):
    """The invite dialog showed the invitation cannot go out as asked.

    Raised by ``_submit_invite_dialog`` before anything was submitted, after
    it dismissed the dialog. ``status`` and ``message`` are what
    ``connect_with_person`` returns; ``note_limit`` is the note length the
    dialog's field takes, when it said.
    """

    def __init__(self, status: str, message: str, *, note_limit: int | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.note_limit = note_limit


@dataclass(frozen=True)
class _InviteDialogState:
    """One read of ``INVITE_DIALOG_STATE_JS``.

    Every field after ``dialogs`` describes the one open dialog and keeps its
    default unless exactly one was open.
    """

    dialogs: int
    buttons: int = 0
    primary_disabled: bool = False
    note_field: str = "none"
    note_value: str = ""
    note_max_length: int | None = None

    @property
    def single(self) -> bool:
        return self.dialogs == 1


def _stored_note(note: str) -> str:
    """The note as a textarea stores it: CR LF and a lone CR become LF."""
    return note.replace("\r\n", "\n").replace("\r", "\n")


def _note_length(note: str) -> int:
    """Length in UTF-16 code units, the unit maxlength counts in Chromium."""
    return len(_stored_note(note).encode("utf-16-le")) // 2


def _note_kept(note: str, kept: str) -> bool:
    """Whether the field holds the note as given.

    Whitespace at either end carries nothing a member reads, so a field that
    trimmed it still holds the note.
    """
    return kept.strip() == _stored_note(note).strip()


def _note_refusal(note: str, state: _InviteDialogState | None) -> InviteNotSent:
    """Why a filled note field that does not hold the note stops the send."""
    if state is None or not state.single:
        return InviteNotSent(
            "connect_unavailable",
            "The note field could not be read back after the note was typed, "
            "so the invitation was not sent.",
        )
    length = _note_length(note)
    limit = state.note_max_length
    if limit is None or length <= limit:
        kept = state.note_value
        limit = (
            _note_length(kept) if kept and _stored_note(note).startswith(kept) else None
        )
    if limit is None:
        return InviteNotSent(
            "connect_unavailable",
            "LinkedIn's note field did not keep the note as written, so the "
            "invitation was not sent.",
        )
    return InviteNotSent(
        "note_too_long",
        f"The note is {length} characters and LinkedIn's note field takes "
        f"{limit} for this account, so the invitation was not sent. Shorten "
        f"the note to at most {limit} characters and call again.",
        note_limit=limit,
    )


async def _release(handle: JSHandle) -> None:
    try:
        await handle.dispose()
    except Exception:
        logger.debug("Could not release an invite dialog handle", exc_info=True)


def _withdraw_result(
    url: str,
    status: str,
    message: str,
    *,
    profile: str = "",
) -> dict[str, Any]:
    """Build a structured response for a withdraw-invitation attempt."""
    result: dict[str, Any] = {"url": url, "status": status, "message": message}
    if profile:
        result["profile"] = profile
    return result


def _shows_withdrawn(text: str, state: str) -> bool:
    """Whether a re-read after a withdrawal is evidence the invitation is gone.

    Only a page that was read and classified counts. An empty read and an
    action area the probe could not locate (``unavailable``) are "not
    pending" as well, and neither says anything about the invitation:
    reporting ``withdrawn`` from one reports a write that may never have
    happened.
    """
    return bool(text) and state not in ("pending", "unavailable")


def _connection_result(
    url: str,
    status: str,
    message: str,
    *,
    note_sent: bool = False,
    note_limit: int | None = None,
    profile: str = "",
) -> dict[str, Any]:
    """Build a structured response for a profile connection attempt."""
    result: dict[str, Any] = {
        "url": url,
        "status": status,
        "message": message,
        "note_sent": note_sent,
    }
    if note_limit is not None:
        result["note_limit"] = note_limit
    if profile:
        result["profile"] = profile
    return result


# One main-profile read of one member, by username. The person workflow owns
# that read and this one only ever needs its ``main_profile`` text, so the
# borrow is a callable rather than the scraper: this module never learns what
# the facade is, and the day the read moves again only the wiring does.
ReadMainProfile = Callable[[str], Awaitable[dict[str, Any]]]


class ConnectionActions:
    """Send, accept and probe invitations for one LinkedIn member."""

    def __init__(
        self,
        session: ScrapingSession,
        navigator: PageNavigator,
        read_main_profile: ReadMainProfile,
    ):
        self._session = session
        self._navigator = navigator
        self._read_main_profile = read_main_profile

    async def _dialog_is_open(self, *, timeout: int = 1000) -> bool:
        """Return whether a dialog is open, waiting up to ``timeout`` ms for one.

        The wait is for a *visible* dialog to appear. Counting matches first
        answered "no dialog" at once whenever the dialog had not mounted yet,
        whatever the timeout said: the invite deeplink returns at
        DOMContentLoaded, and a dialog mounted a moment later was reported as
        LinkedIn opening none. Visible, so a hidden preloaded ``[role=dialog]``
        earlier in the document is not the one waited on.
        """
        locator = self._session.page.locator(
            f"{_DIALOG_SELECTOR} >> visible=true"
        ).first
        try:
            await locator.wait_for(state="visible", timeout=timeout)
            return True
        except Exception:
            return False

    async def _invite_dialog_state(self) -> _InviteDialogState | None:
        """Read the one open invite dialog; None when the read proves nothing."""
        try:
            data = await self._session.page.evaluate(INVITE_DIALOG_STATE_JS)
        except Exception:
            logger.debug("Invite dialog read failed", exc_info=True)
            return None
        if not isinstance(data, dict):
            return None
        dialogs = data.get("inviteDialogs")
        if not isinstance(dialogs, int) or isinstance(dialogs, bool):
            return None
        if dialogs != 1:
            return _InviteDialogState(dialogs=dialogs)
        buttons = data.get("buttons")
        note_field = data.get("noteField")
        note_value = data.get("noteValue")
        max_length = data.get("noteMaxLength")
        return _InviteDialogState(
            dialogs=1,
            buttons=buttons if isinstance(buttons, int) else 0,
            primary_disabled=data.get("primaryDisabled") is True,
            note_field=note_field if note_field in ("visible", "hidden") else "none",
            note_value=note_value if isinstance(note_value, str) else "",
            note_max_length=(
                max_length
                if isinstance(max_length, int)
                and not isinstance(max_length, bool)
                and max_length > 0
                else None
            ),
        )

    async def _settled_invite_dialog(
        self, *, timeout: float = 1.0
    ) -> _InviteDialogState | None:
        """Read the invite dialog, giving a second open one a moment to go.

        A read that proves nothing, or exactly one dialog, answers at once.
        """
        attempts = int(timeout / _INVITE_POLL_SECONDS) + 1
        state = None
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(_INVITE_POLL_SECONDS)
            state = await self._invite_dialog_state()
            if state is None or state.dialogs <= 1:
                return state
        return state

    async def _invite_dialog_element(
        self, part: str, *, from_end: int = 0
    ) -> ElementHandle | None:
        """Resolve the one invite dialog's note field or a button in it.

        ``from_end`` counts buttons back from the last one, the primary.
        None unless exactly one dialog is open; the caller releases the
        handle.
        """
        try:
            handle = await self._session.page.evaluate_handle(
                INVITE_DIALOG_ELEMENT_JS, {"part": part, "fromEnd": from_end}
            )
        except Exception:
            logger.debug("Invite dialog element lookup failed", exc_info=True)
            return None
        element = handle.as_element()
        if element is None:
            await _release(handle)
        return element

    async def _click_invite_dialog_button(
        self, from_end: int, *, timeout: int = 5000
    ) -> bool:
        """Click a button of the one invite dialog, counted back from its last.

        0 is the primary and 1 the secondary beside it. Returns False rather
        than raising when no single dialog is open, or when the click is
        intercepted or times out (a disabled button is never clicked: the
        click waits for it to be enabled).
        """
        button = await self._invite_dialog_element("button", from_end=from_end)
        if button is None:
            return False
        try:
            await button.click(timeout=timeout)
            return True
        except Exception:
            logger.debug("Invite dialog button click failed", exc_info=True)
            return False
        finally:
            await _release(button)

    async def _click_dialog_primary_button(self, *, timeout: int = 5000) -> bool:
        """Click the last (primary/Send) button of the one invite dialog.

        LinkedIn consistently places the primary action as the last button.
        Returns False (rather than raising) when the click is intercepted or
        times out, so callers can fall back to a keyboard submit.
        """
        return await self._click_invite_dialog_button(0, timeout=timeout)

    async def _press_enter_on_primary(self) -> bool:
        """Keyboard fallback: focus the primary, press Enter, see the dialog go.

        Focus first, so Enter reaches the button rather than the note field,
        where it would only insert a newline.
        """
        button = await self._invite_dialog_element("button", from_end=0)
        if button is None:
            return False
        try:
            await button.focus()
            await self._session.page.keyboard.press("Enter")
        except Exception:
            logger.debug("Keyboard submit fallback failed", exc_info=True)
            return False
        finally:
            await _release(button)
        return not await self._dialog_is_open(timeout=2000)

    async def _fill_dialog_textarea(self, value: str, *, timeout: int = 5000) -> bool:
        """Fill the note field of the one invite dialog (structural)."""
        field = await self._invite_dialog_element("note")
        if field is None:
            return False
        try:
            await field.fill(value, timeout=timeout)
            return True
        except Exception:
            logger.debug("Invite note fill failed", exc_info=True)
            return False
        finally:
            await _release(field)

    async def _wait_for_invite_note_field(self, *, timeout: float = 3.0) -> bool:
        """Wait for the one invite dialog to show a note field."""
        attempts = int(timeout / _INVITE_POLL_SECONDS) + 1
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(_INVITE_POLL_SECONDS)
            state = await self._invite_dialog_state()
            if state is not None and state.single and state.note_field == "visible":
                return True
        return False

    async def _send_waits_for_a_note(self) -> bool:
        """Whether the invite's Send stays disabled over an empty note field.

        That is how a profile that only takes an invitation with a note shows
        it (upstream issue #407): the dialog opens with its note field, and
        Send stays disabled until something is typed. Every read over
        ``NOTE_REQUIRED_SETTLE_SECONDS`` has to show one dialog with its
        primary disabled, so a Send that is only disabled while the dialog
        hydrates is clicked as before. A disabled Send without an empty,
        visible note field is some other gate and is not claimed as this one.
        """
        attempts = int(NOTE_REQUIRED_SETTLE_SECONDS / _INVITE_POLL_SECONDS) + 1
        state = None
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(_INVITE_POLL_SECONDS)
            state = await self._invite_dialog_state()
            if state is None or not state.single or not state.primary_disabled:
                return False
        return (
            state is not None
            and state.note_field == "visible"
            and not state.note_value.strip()
        )

    async def _dismiss_dialog(self) -> None:
        """Dismiss any open dialog via Escape key (structural)."""
        await self._session.page.keyboard.press("Escape")
        try:
            await self._session.page.wait_for_selector(
                _DIALOG_SELECTOR, state="hidden", timeout=3000
            )
        except PlaywrightTimeoutError:
            pass

    async def _get_premium_upsell_message(self, *, timeout: int = 2500) -> str | None:
        """Return the raw LinkedIn Premium upsell dialog text when visible.

        LinkedIn intercepts invite-with-note flows with an upsell modal when
        the free personalized-note quota is exhausted. The detector itself is
        locale-independent: the modal links to ``/premium/...``. The returned
        message is the dialog text as rendered by LinkedIn, not a synthesized
        explanation.
        """
        locator = self._session.page.locator(_DIALOG_PREMIUM_LINK_SELECTOR).first
        try:
            await locator.wait_for(state="visible", timeout=timeout)
        except PlaywrightTimeoutError:
            return None
        except Exception:
            try:
                if not await locator.is_visible():
                    return None
            except Exception:
                return None

        try:
            message = await self._session.page.evaluate(
                """() => {
                    const link = document.querySelector(
                        'dialog[open] a[href*="/premium/"], [role="dialog"] a[href*="/premium/"]'
                    );
                    const dialog = link?.closest('dialog,[role="dialog"]');
                    return dialog?.innerText || dialog?.textContent || link?.innerText || '';
                }"""
            )
            if isinstance(message, str) and message.strip():
                return message.strip()
        except Exception:
            logger.debug("Could not read Premium upsell dialog text", exc_info=True)

        try:
            link_text = await locator.inner_text()
            if link_text.strip():
                return link_text.strip()
        except Exception:
            pass
        return "LinkedIn Premium upsell modal detected."

    async def _open_more_menu(self) -> bool:
        """Open the profile's More (three-dot) menu in a locale-independent way.

        Locates the More button structurally as ``actionRoot
        button[aria-expanded]`` — the action-root walk discriminates the
        profile More button from any other More-labelled buttons elsewhere
        on the page (notably the video-player More on profiles with
        background videos), and ``aria-expanded`` distinguishes the menu
        opener from primary action buttons (which carry ``aria-label``
        instead). Returns True iff the click landed and a ``[role='menu']``
        became visible. The caller is expected to follow up with
        ``_read_action_signals`` to scan the now-rendered menu items for
        the vanityName invite anchor; this helper does not classify menu
        contents itself.
        """
        try:
            clicked = await self._session.page.evaluate(OPEN_MORE_BUTTON_JS)
        except Exception:
            logger.debug("More button click via JS failed", exc_info=True)
            return False
        if not clicked:
            return False
        try:
            await self._session.page.wait_for_selector("[role='menu']", timeout=3000)
            return True
        except PlaywrightTimeoutError:
            logger.debug("More menu did not appear after click")
            return False

    async def _open_incoming_row_more_menu(self) -> bool:
        """Open the fingerprinted incoming row's own More menu.

        The disprove step before Accept (issue #629): the caller re-reads
        the signals while the menu is open, and an invite anchor there means
        the row was a creator-mode top card, not an incoming request.
        Returns True iff the click landed and a ``[role='menu']`` became
        visible; on False the caller must not click Accept either.
        """
        try:
            clicked = await self._session.page.evaluate(OPEN_INCOMING_ROW_MORE_JS)
        except Exception:
            logger.debug("Incoming-row More click via JS failed", exc_info=True)
            return False
        if clicked is not True:
            return False
        try:
            await self._session.page.wait_for_selector("[role='menu']", timeout=3000)
            return True
        except PlaywrightTimeoutError:
            logger.debug("Incoming-row More menu did not appear after click")
            return False

    async def _click_incoming_accept(self) -> bool:
        """Click Accept on an incoming-request profile, locale-independently.

        Delegates to ``CLICK_INCOMING_ACCEPT_JS``: the click fires only
        when the full incoming-row fingerprint matches, and it targets the
        FIRST labeled button (Accept renders before Ignore — primary
        actions lead in top-card rows). Clicking the second button would
        silently and irreversibly Ignore the request; the strict
        fingerprint plus the caller's verify-after-click are the
        mitigations. Returns True iff the click landed.
        """
        try:
            return bool(await self._session.page.evaluate(CLICK_INCOMING_ACCEPT_JS))
        except Exception:
            logger.debug("Incoming accept click via JS failed", exc_info=True)
            return False

    async def _read_action_signals(self, username: str) -> ActionSignals:
        """Read locale-independent structural signals for a profile's
        relationship state.

        Detection uses URL patterns and ARIA attribute presence only — never
        text values — per the AGENTS.md Scraping Rules. The vanityName invite
        anchor is searched document-wide because LinkedIn renders the More
        menu's contents in a portal-mounted ``[role='menu']`` outside ``<main>``;
        the URL is uniquely scoped to the target user, so document-wide
        search introduces no false positives. The compose anchor used for
        action-root discovery is scoped to ``<main>`` to avoid the
        portal-rendered "Send profile in a message" anchor that appears
        inside the More menu after click.
        """
        data = await self._session.page.evaluate(ACTION_SIGNALS_JS, username)
        if not isinstance(data, dict):
            return ActionSignals(
                has_invite_anchor=False,
                has_compose_anchor_in_action_root=False,
                has_edit_intro_anchor=False,
                has_labeled_action_button=False,
                has_labeled_action_anchor=False,
                has_incoming_action_row=False,
            )
        return ActionSignals(
            has_invite_anchor=bool(data.get("hasInvite")),
            has_compose_anchor_in_action_root=bool(data.get("hasComposeInActionRoot")),
            has_edit_intro_anchor=bool(data.get("hasEditIntro")),
            has_labeled_action_button=bool(data.get("hasLabeledActionButton")),
            has_labeled_action_anchor=bool(data.get("hasLabeledActionAnchor")),
            has_incoming_action_row=bool(data.get("hasIncomingActionRow")),
        )

    async def _submit_invite_dialog(
        self, note: str | None
    ) -> tuple[bool, bool, str | None]:
        """Submit the invite dialog opened by the custom-invite deeplink.

        Returns ``(submitted, note_sent, note_limit_message)``.

        ``note_sent`` reports *delivery*, not textarea fill — it stays
        False on any failure path, including the Premium upsell that
        LinkedIn shows when the free personalized-note quota is exhausted.
        ``note_limit_message`` is the raw LinkedIn Premium dialog text when
        the upsell was detected; in that case ``submitted`` is False, the
        dialog is dismissed, and callers should surface that text directly.

        Raises ``InviteNotSent``, before anything is submitted, when the
        dialog shows the invitation cannot go out as asked: more than one
        dialog is open, the note field did not keep the note whole
        (``note_too_long`` when it cut it), or, with no note, Send stays
        disabled over an empty note field (``note_required``).

        Every read, fill and click is scoped to the one open invite dialog
        (``_FIND_INVITE_DIALOG_FN_JS``) and uses structural selectors and
        positional indexing — no localized text matching. Owns dialog
        cleanup: the dialog is dismissed on every failure path, callers must
        not dismiss again.
        """
        if not await self._dialog_is_open(timeout=5000):
            return False, False, None

        state = await self._settled_invite_dialog()
        if state is not None and state.dialogs > 1:
            logger.info("%d dialogs open on the invite page", state.dialogs)
            await self._dismiss_dialog()
            raise InviteNotSent(
                "connect_unavailable",
                "More than one dialog was open on the invite page and nothing "
                "tells which one is the invitation, so nothing was clicked and "
                "the invitation was not sent.",
            )

        note_filled = False
        if note:
            if state is not None and state.single and state.note_field == "none":
                # Reveal the note textarea via the secondary action.
                # Two layouts are now in the wild and both place "Add a
                # note" at index ``btn_count - 2``:
                #   * Legacy invite dialog (3 buttons): dismiss, secondary
                #     "Add a note", primary "Send" -> nth(1) is secondary.
                #   * "Add a note to your invitation?" gating dialog (2
                #     buttons, rolled out 2026-05): "Add a note",
                #     "Send without a note" -> nth(0) is the only path
                #     that mounts the textarea. See issue #455.
                # If LinkedIn ever serves a 2-button dismiss/primary
                # no-note layout, the click below misroutes to dismiss;
                # the textarea-presence recheck via _fill_dialog_textarea
                # then fails and the caller returns connect_unavailable
                # without sending — the same outcome as today. The click
                # resolves the dialog again, so it is never the primary of
                # whatever dialog is open by then.
                if state.buttons >= 2:
                    await self._click_invite_dialog_button(1)
                    textarea_appeared = await self._wait_for_invite_note_field()
                    if not textarea_appeared:
                        logger.debug("Note textarea did not appear")
                    # ponytail: LinkedIn now renders a persistent Premium
                    # nudge banner on this step even when quota is NOT
                    # exhausted (observed: "3 personalized invitations
                    # remaining this month" alongside a live, fillable
                    # textarea). Bailing on banner presence alone false-
                    # positives on every note send. Only treat it as a
                    # real block when the textarea never mounted at all —
                    # the one case where LinkedIn actually replaces the
                    # note UI with the upsell instead of showing both.
                    if not textarea_appeared:
                        note_limit_message = await self._get_premium_upsell_message()
                        if note_limit_message is not None:
                            logger.info(
                                "Premium upsell blocked opening invite note editor"
                            )
                            await self._dismiss_dialog()
                            return False, False, note_limit_message

            note_filled = await self._fill_dialog_textarea(note)
            if not note_filled:
                # Same gate as the reveal step: the Premium nudge banner sits
                # beside a live textarea, so a failed fill is a quota block
                # only once no visible textarea is left. A read that fails,
                # or that finds a second dialog, proves no absence, so it
                # claims no block either: a false block invites the caller to
                # resend without the note.
                after = await self._invite_dialog_state()
                textarea_visible = (
                    after is None or after.dialogs > 1 or after.note_field == "visible"
                )
                if textarea_visible:
                    logger.info(
                        "Invite note fill failed without evidence of a quota block"
                    )
                    await self._dismiss_dialog()
                    return False, False, None
                note_limit_message = await self._get_premium_upsell_message()
                if note_limit_message is not None:
                    logger.info("Premium upsell blocked filling invite note")
                    await self._dismiss_dialog()
                    return False, False, note_limit_message
                await self._dismiss_dialog()
                return False, False, None

            # A field shorter than the note keeps its first ``maxlength``
            # characters and drops the rest without a sound (measured: 250
            # typed into LinkedIn's 200-character field left 200), and the
            # member would receive the cut note. Read it back before Send.
            kept = await self._invite_dialog_state()
            if kept is None or not kept.single or not _note_kept(note, kept.note_value):
                refusal = _note_refusal(note, kept)
                logger.info("Invite note not kept whole: %s", refusal.status)
                await self._dismiss_dialog()
                raise refusal
        elif await self._send_waits_for_a_note():
            logger.info("Invite dialog keeps Send disabled until a note is typed")
            await self._dismiss_dialog()
            raise InviteNotSent(
                "note_required",
                "LinkedIn only takes an invitation with a note for this profile: "
                "the invite dialog kept Send disabled while its note field was "
                "empty. Nothing was sent; call again with a note.",
            )

        sent = await self._click_dialog_primary_button()
        if not sent:
            sent = await self._press_enter_on_primary()
            if not sent:
                # The Send click can also fail because LinkedIn swapped the
                # invite dialog for the Premium upsell at submit time — the
                # original primary button is then detached or pointer-event
                # covered, so the click raises or times out. Check for the
                # upsell here so we surface the raw note-limit message
                # instead of dismissing silently and returning
                # connect_unavailable.
                if note:
                    note_limit_message = await self._get_premium_upsell_message()
                    if note_limit_message is not None:
                        logger.info(
                            "Premium upsell modal intercepted invite submit click"
                        )
                        await self._dismiss_dialog()
                        return False, False, note_limit_message
                await self._dismiss_dialog()
                return False, False, None

        dialog_closed = True
        try:
            await self._session.page.wait_for_selector(
                _VISIBLE_DIALOG_SELECTOR, state="hidden", timeout=5000
            )
        except PlaywrightTimeoutError:
            logger.debug("Invite dialog did not close after submit")
            dialog_closed = False

        # LinkedIn may swap the invite dialog for a Premium upsell when the
        # free note quota is exhausted, instead of closing it after Send —
        # the textarea was filled but the invite was not delivered, so
        # surface LinkedIn's raw dialog text. Gated on the dialog still
        # being open: the same benign nudge banner that can sit alongside a
        # live, fillable textarea (see the reveal-step fix above) can also
        # still be in the DOM for a moment right after a successful Send,
        # before it unmounts with the closing dialog. Checking unconditionally
        # here would report a genuinely delivered invite as blocked. A dialog
        # that failed to close is real evidence something went wrong; one
        # that closed on schedule is not, banner or no banner.
        if note and not dialog_closed:
            note_limit_message = await self._get_premium_upsell_message()
            if note_limit_message is not None:
                logger.info("Premium upsell modal intercepted invite submit")
                await self._dismiss_dialog()
                return False, False, note_limit_message

        return True, note_filled, None

    async def _probe_invite_note_limit(self) -> str | None:
        """Open the note editor only to read a Premium note-quota message.

        This is used when the profile did not expose the normal invite anchor.
        Navigating to the custom-invite deeplink and opening the note editor is
        non-destructive, but submitting would weaken the write gate for
        follow-only/unavailable profiles. Therefore this helper never clicks
        the primary Send button: it returns the raw LinkedIn Premium dialog
        text if LinkedIn shows it while opening the note editor, then
        dismisses the dialog. The one click it may make is resolved against
        the one open invite dialog, and never as its last button.
        """
        if not await self._dialog_is_open(timeout=5000):
            return None
        note_limit_message = await self._get_premium_upsell_message(timeout=500)
        if note_limit_message is not None:
            await self._dismiss_dialog()
            return note_limit_message

        state = await self._invite_dialog_state()
        if state is None or not state.single or state.note_field != "none":
            await self._dismiss_dialog()
            return None

        if state.buttons >= 3:
            if not await self._click_invite_dialog_button(1):
                logger.debug("Could not open invite note editor")
            if not await self._wait_for_invite_note_field():
                logger.debug("Note textarea did not appear during quota probe")

        note_limit_message = await self._get_premium_upsell_message()
        await self._dismiss_dialog()
        return note_limit_message

    async def connect_with_person(
        self,
        username: str,
        *,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Send a LinkedIn connection request or accept an incoming one.

        Detection is locale-independent: classification uses URL patterns
        (vanityName invite anchor, edit-intro anchor) and ARIA-attribute
        presence on top-card buttons (`aria-label` for primary actions,
        `aria-expanded` for the More-menu opener). The deeplink-submit
        path is gated strictly on `has_invite_anchor=True` *after* the
        optional More-menu retry, so Pending and follow-only profiles
        cannot trigger a write. If a note was requested but no invite
        anchor is visible, the custom-invite deeplink may still be opened
        only as a non-submitting note-quota probe. Sending itself uses the
        ``/preload/custom-invite/?vanityName=`` deeplink, which works
        whether the user-visible Connect button is in the action bar
        or buried under the More menu. An incoming-request row is only
        accepted after its own More menu has been opened and shown no invite
        anchor, since a creator-mode top card has the same shape (#629).

        A note no account could send (over ``MAX_INVITE_NOTE_LENGTH``) is
        refused as ``note_too_long`` before any page is opened. The dialog
        itself can still refuse a shorter one (``note_too_long`` with the
        field's ``note_limit``) or ask for a note that was not given
        (``note_required``); nothing is sent in any of these.
        """
        username = normalize_person_identifier(username)
        url = person_profile_url(username, "/")

        if note and _note_length(note) > MAX_INVITE_NOTE_LENGTH:
            return _connection_result(
                url,
                "note_too_long",
                f"The note is {_note_length(note)} characters and LinkedIn takes "
                f"at most {MAX_INVITE_NOTE_LENGTH} (200 without Premium), so no "
                "page was opened and nothing was sent. Shorten the note and "
                "call again.",
            )

        profile = await self._read_main_profile(username)
        page_text = profile.get("sections", {}).get("main_profile", "")
        if not page_text:
            return _connection_result(
                url, "unavailable", "Could not read profile page."
            )

        signals = await self._read_action_signals(username)
        state = connection.detect_connection_state(signals)
        logger.info(
            "Connection signals for %s: state=%s signals=%s", username, state, signals
        )

        if state == "self_profile":
            return _connection_result(
                url,
                "connect_unavailable",
                "Cannot send a connection request to your own profile.",
                profile=page_text,
            )
        if state == "already_connected":
            return _connection_result(
                url,
                "already_connected",
                "You are already connected with this profile.",
                profile=page_text,
            )
        if state == "pending":
            return _connection_result(
                url,
                "pending",
                "A connection request is already pending for this profile.",
                profile=page_text,
            )

        if state == "incoming_request":
            # Disprove before Accept (issue #629). A creator-mode top card,
            # [Follow][Save in Sales Navigator][More] with no Message action,
            # carries the incoming fingerprint exactly, and Accept there
            # clicks Follow. Such a row may keep Connect in its More menu,
            # which LinkedIn only mounts on click, so open that row's own
            # menu and read the signals while it is open. A row whose menu
            # cannot be opened is one nothing here can vouch for.
            opened = await self._open_incoming_row_more_menu()
            if not opened:
                return _connection_result(
                    url,
                    "send_failed",
                    "Could not open the More menu of the incoming-request row "
                    "to rule out a creator-mode profile, so Accept was not "
                    "clicked.",
                    profile=page_text,
                )
            menu_signals = await self._read_action_signals(username)
            try:
                await self._session.page.keyboard.press("Escape")
            except Exception:
                logger.debug(
                    "Escape after incoming-row More read failed", exc_info=True
                )
            logger.info(
                "Incoming-row More signals for %s: signals=%s", username, menu_signals
            )
            if menu_signals.has_invite_anchor:
                # Connect was in the menu all along: this is a creator-mode
                # profile, and the invite anchor opens the same write gate
                # the connectable state does.
                signals = menu_signals
                state = "connectable"
            elif note:
                # Accept takes no note, so a note asks for an invitation,
                # and a creator-mode profile can hide Connect even from its
                # More menu. Nothing tells the two apart any more.
                return _connection_result(
                    url,
                    "incoming_request_ambiguous",
                    "The profile's action row has the shape of an incoming "
                    "connection request, but a note was given and Accept takes "
                    "none; the row may be a creator-mode Follow button, so "
                    "nothing was clicked. Call again without a note to accept.",
                    profile=page_text,
                )

        if state == "incoming_request":
            # Accept clicks the first labeled button in the fingerprinted
            # row. There is deliberately no locale-text fallback: clicking
            # a button matched by exact text anywhere in the page risks
            # hitting the wrong control (or the Ignore button in another
            # locale), and accepting/ignoring is irreversible. When the
            # fingerprint does not match we report send_failed rather than
            # guess.
            clicked = await self._click_incoming_accept()
            if not clicked:
                return _connection_result(
                    url,
                    "send_failed",
                    "Could not find or click the Accept button.",
                    profile=page_text,
                )
            # LinkedIn propagates the accepted state asynchronously; an
            # immediate re-read can still render the old top card and
            # would report send_failed for a successful accept (observed
            # live 2026-06-11). Verify with one settle retry.
            verified_text = ""
            verified_state = None
            for attempt in range(2):
                if attempt:
                    await asyncio.sleep(3.0)
                verified = await self._read_main_profile(username)
                verified_text = verified.get("sections", {}).get("main_profile", "")
                verified_signals = await self._read_action_signals(username)
                verified_state = connection.detect_connection_state(verified_signals)
                if verified_state == "already_connected":
                    break
            if verified_state != "already_connected":
                return _connection_result(
                    url,
                    "send_failed",
                    "Accepted, but the profile did not transition to 1st-degree.",
                    profile=verified_text or page_text,
                )
            return _connection_result(
                url,
                "accepted",
                "Connection request accepted.",
                profile=verified_text,
            )

        # Follow-only profiles may have Connect hidden under the More menu
        # (high-follower / creator-mode profiles). Try opening it and
        # re-reading signals; if the vanityName invite anchor surfaces in
        # the menu, we can proceed with the deeplink. (The
        # has_invite_anchor=False guard is implicit: detect_connection_state
        # only returns "follow_only" after the has_invite_anchor branch
        # has already failed, so reaching this branch already implies it.)
        if state == "follow_only":
            opened = await self._open_more_menu()
            if opened:
                signals = await self._read_action_signals(username)
                # Close the menu before any subsequent navigation so it
                # doesn't intercept the upcoming page transition.
                try:
                    await self._session.page.keyboard.press("Escape")
                except Exception:
                    logger.debug("Escape after More-menu reread failed", exc_info=True)
                logger.info("Post-More signals for %s: signals=%s", username, signals)

        invite_url = (
            "https://www.linkedin.com/preload/custom-invite/"
            f"?vanityName={quote_plus(username)}"
        )

        # Write-gate: submit only when LinkedIn exposed the vanityName invite
        # anchor. When a note is requested without that anchor, open the
        # deeplink only as a non-submitting probe so we can report the Premium
        # note-quota block without accidentally sending from a follow-only or
        # otherwise unavailable profile.
        if not signals.has_invite_anchor:
            if note:
                logger.info(
                    "No visible invite anchor for %s; probing custom-invite deeplink "
                    "because a personalized note was requested",
                    username,
                )
                await self._navigator._navigate_to_page(invite_url)
                note_limit_message = await self._probe_invite_note_limit()
                if note_limit_message is not None:
                    return _connection_result(
                        url,
                        "custom_note_limit_reached",
                        note_limit_message,
                        note_sent=False,
                        profile=page_text,
                    )
            return _connection_result(
                url,
                "connect_unavailable",
                "LinkedIn did not expose a usable Connect action for this profile.",
                profile=page_text,
            )

        await self._navigator._navigate_to_page(invite_url)

        try:
            submitted, note_sent, note_limit_message = await self._submit_invite_dialog(
                note
            )
        except InviteNotSent as refusal:
            return _connection_result(
                url,
                refusal.status,
                refusal.message,
                note_limit=refusal.note_limit,
                profile=page_text,
            )
        if note_limit_message is not None:
            return _connection_result(
                url,
                "custom_note_limit_reached",
                note_limit_message,
                note_sent=False,
                profile=page_text,
            )
        if not submitted:
            return _connection_result(
                url,
                "connect_unavailable",
                "LinkedIn did not open a usable invite dialog for this profile.",
                profile=page_text,
            )

        verified = await self._read_main_profile(username)
        verified_signals = await self._read_action_signals(username)
        if verified_signals.has_invite_anchor:
            # The same settle retry as the accept path: an immediate re-read
            # can still render Connect for an invitation LinkedIn already
            # recorded (observed live 2026-09-26: send_failed, then Pending).
            # Only a pending or already accepted invitation is evidence it
            # landed.
            await asyncio.sleep(3.0)
            retry = await self._read_main_profile(username)
            retry_signals = await self._read_action_signals(username)
            if connection.detect_connection_state(retry_signals) in (
                "pending",
                "already_connected",
            ):
                verified, verified_signals = retry, retry_signals
        verified_text = verified.get("sections", {}).get("main_profile", "")
        verified_state = connection.detect_connection_state(verified_signals)

        if verified_signals.has_invite_anchor:
            return _connection_result(
                url,
                "send_failed",
                "Submitted the invite dialog but the profile still exposes Connect.",
                note_sent=note_sent,
                profile=verified_text or page_text,
            )

        return _connection_result(
            url,
            "connected",
            f"Connection request sent. State after send: {verified_state}.",
            note_sent=note_sent,
            profile=verified_text or page_text,
        )

    async def _click_withdraw_anchor(self) -> bool:
        """Click the Pending control; True iff exactly one was there to click."""
        try:
            return bool(await self._session.page.evaluate(CLICK_WITHDRAW_ANCHOR_JS))
        except Exception:
            logger.debug("Withdraw anchor click via JS failed", exc_info=True)
            return False

    async def _wait_for_confirm_buttons(
        self, *, min_count: int = 2, timeout: float = 5.0
    ) -> bool:
        """Wait until the open confirmation dialog shows *min_count* buttons.

        The withdraw dialog first renders a spinner with only its Dismiss
        control (measured live 2026-08-21). "Last button" is only the primary
        action once the real Cancel/Withdraw pair has mounted.
        """
        attempts = max(1, int(timeout / 0.25))
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(0.25)
            try:
                count = await self._session.page.evaluate(
                    CONFIRM_DIALOG_BUTTON_COUNT_JS
                )
            except Exception:
                continue
            if isinstance(count, int) and count >= min_count:
                return True
        return False

    async def _click_confirm_dialog_primary(self) -> bool:
        """Click the primary (last) button of the open confirmation dialog."""
        try:
            return bool(
                await self._session.page.evaluate(CLICK_CONFIRM_DIALOG_PRIMARY_JS)
            )
        except Exception:
            logger.debug("Confirm dialog click via JS failed", exc_info=True)
            return False

    async def _read_state(self, username: str) -> tuple[str, str]:
        """One main-profile read and the relationship state it shows."""
        profile = await self._read_main_profile(username)
        text = profile.get("sections", {}).get("main_profile", "")
        signals = await self._read_action_signals(username)
        return text, connection.detect_connection_state(signals)

    async def withdraw_invitation(self, username: str) -> dict[str, Any]:
        """Withdraw a previously sent connection request.

        Clicks only after a fresh read of the profile classifies it as
        ``pending``, using the same structural signals as
        ``connect_with_person``. Every other state (not pending, connected,
        own profile, unreadable) is reported without touching the page, so a
        stale idea of who is pending can never withdraw the wrong invite.
        Success is declared only when a re-read that was read and classified
        no longer shows pending.
        """
        username = normalize_person_identifier(username)
        url = person_profile_url(username, "/")

        page_text, state = await self._read_state(username)
        if not page_text:
            return _withdraw_result(url, "unavailable", "Could not read profile page.")
        logger.info("Connection state for %s (withdraw): %s", username, state)

        if state == "self_profile":
            return _withdraw_result(
                url,
                "self_profile",
                "Cannot withdraw an invitation from your own profile.",
                profile=page_text,
            )
        if state != "pending":
            return _withdraw_result(
                url,
                "not_pending",
                f"No pending sent invitation to withdraw (current state: {state}).",
                profile=page_text,
            )

        if not await self._click_withdraw_anchor():
            return _withdraw_result(
                url,
                "withdraw_unavailable",
                "Could not find exactly one Pending control to click.",
                profile=page_text,
            )

        if not await self._dialog_is_open(timeout=3000):
            # A flow without a confirmation step is possible; the page decides.
            verified_text, verified_state = await self._read_state(username)
            if _shows_withdrawn(verified_text, verified_state):
                return _withdraw_result(
                    url,
                    "withdrawn",
                    f"Invitation withdrawn. State after withdrawal: {verified_state}.",
                    profile=verified_text or page_text,
                )
            message = "LinkedIn did not open a confirmation dialog for withdrawal."
            if not (verified_text and verified_state == "pending"):
                message += (
                    " The profile could not be read back to check that nothing"
                    " was withdrawn without one."
                )
            return _withdraw_result(
                url, "withdraw_unavailable", message, profile=page_text
            )

        if not await self._wait_for_confirm_buttons():
            await self._dismiss_dialog()
            return _withdraw_result(
                url,
                "withdraw_failed",
                "The withdrawal dialog never finished rendering its buttons, or "
                "it was not the only dialog open. Nothing was confirmed.",
                profile=page_text,
            )

        if not await self._click_confirm_dialog_primary():
            await self._dismiss_dialog()
            return _withdraw_result(
                url,
                "withdraw_failed",
                "Could not confirm the withdrawal dialog.",
                profile=page_text,
            )

        try:
            await self._session.page.wait_for_selector(
                _DIALOG_SELECTOR, state="hidden", timeout=5000
            )
        except PlaywrightTimeoutError:
            logger.debug("Withdraw confirmation dialog did not close in time")

        verified_text = ""
        verified_state = "pending"
        for attempt in range(2):
            if attempt:
                await asyncio.sleep(WITHDRAW_SETTLE_SECONDS)
            verified_text, verified_state = await self._read_state(username)
            if _shows_withdrawn(verified_text, verified_state):
                return _withdraw_result(
                    url,
                    "withdrawn",
                    f"Invitation withdrawn. State after withdrawal: {verified_state}.",
                    profile=verified_text,
                )

        if verified_text and verified_state == "pending":
            return _withdraw_result(
                url,
                "withdraw_failed",
                "Confirmed the dialog, but the profile still shows a pending "
                "invitation.",
                profile=verified_text,
            )
        return _withdraw_result(
            url,
            "withdraw_failed",
            "Confirmed the dialog, but the profile could not be read back "
            f"(state: {verified_state}), so the withdrawal is unverified. Check "
            "the profile before calling again.",
            profile=verified_text or page_text,
        )
