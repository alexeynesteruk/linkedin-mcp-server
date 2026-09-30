"""Invite-dialog submission against a real DOM with a chat overlay open.

Measured on LinkedIn in September 2026: after a message send, LinkedIn keeps
the conversation open as an overlay dialog on later pages, including the
custom-invite deeplink, so two dialogs are open when the invite renders.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, patch

import json

import pytest
from patchright.async_api import Page, async_playwright

from linkedin_mcp_server.scraping.connection_actions import (
    INVITE_DIALOG_ELEMENT_JS,
    INVITE_DIALOG_STATE_JS,
    ConnectionActions,
    InviteNotSent,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

INVITE_DIALOG = """
  <div role="dialog" id="invite">
    <h2>Add a note to your invitation?</h2>
    <button onclick="document.body.dataset.invite = 'note';
      const note = document.createElement('textarea');
      note.style.display = 'block';
      document.getElementById('invite').insertBefore(note, this);
      this.nextElementSibling.textContent = 'Send'">Add a note</button>
    <button onclick="document.body.dataset.invite = 'sent';
      const note = document.querySelector('#invite textarea');
      document.body.dataset.note = note ? note.value : '';
      document.getElementById('invite').remove()">Send without a note</button>
  </div>
"""

CHAT_OVERLAY = """
  <div role="dialog" id="chat">
    <form class="msg-form">
      <div role="textbox" contenteditable="true"
           style="display:block;width:200px;height:30px"></div>
      <button type="submit" disabled>Send</button>
      <button type="button" class="msg-form__send-toggle"
        onclick="document.body.dataset.chat = 'clicked'">Open send options</button>
    </form>
  </div>
"""


@pytest.fixture
async def dom_page():
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
    async def unreachable(_username: str) -> dict[str, Any]:
        raise AssertionError("the dialog cases never read a profile")

    session = ScrapingSession(cast(Page, page))
    return ConnectionActions(session, PageNavigator(session), unreachable)


@pytest.mark.parametrize(
    "body", [INVITE_DIALOG + CHAT_OVERLAY, CHAT_OVERLAY + INVITE_DIALOG]
)
async def test_invite_is_sent_past_an_open_chat_overlay(dom_page, body):
    await dom_page.set_content(f"<!DOCTYPE html><html><body>{body}</body></html>")

    submitted, note_sent, note_limit = await _actions(dom_page)._submit_invite_dialog(
        None
    )

    assert (submitted, note_sent, note_limit) == (True, False, None)
    assert await dom_page.evaluate("document.body.dataset.invite") == "sent"
    assert await dom_page.evaluate("document.body.dataset.chat") is None


async def test_chat_overlay_alone_is_not_an_invite_dialog(dom_page):
    await dom_page.set_content(
        f"<!DOCTYPE html><html><body>{CHAT_OVERLAY}</body></html>"
    )

    submitted, _, _ = await _actions(dom_page)._submit_invite_dialog(None)

    assert submitted is False
    assert await dom_page.evaluate("document.body.dataset.chat") is None


async def test_invite_note_is_sent_past_an_open_chat_overlay(dom_page):
    await dom_page.set_content(
        f"<!DOCTYPE html><html><body>{INVITE_DIALOG}{CHAT_OVERLAY}</body></html>"
    )

    submitted, note_sent, note_limit = await _actions(dom_page)._submit_invite_dialog(
        "Hello"
    )

    assert (submitted, note_sent, note_limit) == (True, True, None)
    assert await dom_page.evaluate("document.body.dataset.invite") == "sent"
    assert await dom_page.evaluate("document.body.dataset.note") == "Hello"
    assert await dom_page.evaluate("document.body.dataset.chat") is None


LATE_INVITE_DIALOG = """
  <script>
    setTimeout(() => {
      const holder = document.createElement('div');
      holder.innerHTML = %s;
      document.body.appendChild(holder.firstElementChild);
    }, 400);
  </script>
"""

HIDDEN_PRELOADED_DIALOG = """
  <div role="dialog" id="preloaded" style="display:none">
    <button onclick="document.body.dataset.preloaded = 'clicked'">Close</button>
  </div>
"""


async def test_an_invite_dialog_that_mounts_after_load_is_sent(dom_page):
    """The deeplink returns at DOMContentLoaded, and the dialog can mount a
    moment later. Answering "no dialog" before it had the chance reported a
    Connect-able profile as one LinkedIn opened no invite dialog for."""
    script = LATE_INVITE_DIALOG % json.dumps(INVITE_DIALOG.strip())
    await dom_page.set_content(f"<!DOCTYPE html><html><body>{script}</body></html>")

    submitted, note_sent, note_limit = await _actions(dom_page)._submit_invite_dialog(
        None
    )

    assert (submitted, note_sent, note_limit) == (True, False, None)
    assert await dom_page.evaluate("document.body.dataset.invite") == "sent"


async def test_a_hidden_preloaded_dialog_does_not_hide_the_invite(dom_page):
    """A hidden [role=dialog] earlier in the document is not the one waited
    on: the visible invite is found and sent, and the hidden one is not
    clicked."""
    await dom_page.set_content(
        "<!DOCTYPE html><html><body>"
        f"{HIDDEN_PRELOADED_DIALOG}{INVITE_DIALOG}</body></html>"
    )

    submitted, note_sent, note_limit = await _actions(dom_page)._submit_invite_dialog(
        None
    )

    assert (submitted, note_sent, note_limit) == (True, False, None)
    assert await dom_page.evaluate("document.body.dataset.invite") == "sent"
    assert await dom_page.evaluate("document.body.dataset.preloaded") is None


# LinkedIn closes its invite modal on Escape. Modelled so a refusal's
# dismissal does not sit out ``_dismiss_dialog``'s wait; the page state the
# assertions read is recorded before it.
ESCAPE_CLOSES_INVITE = """
  <script>
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      document.getElementById('invite')?.remove();
      document.getElementById('popup')?.remove();
    });
  </script>
"""

# Upstream #407: the dialog opens with its note field, and Send stays
# disabled until something is typed. ``disabled`` is the property, and the
# ARIA form of the same gate leaves the button clickable, so a click on it is
# recorded either way.
NOTE_REQUIRED_DIALOG = """
  <div role="dialog" id="invite">
    <h2>Add a note to your invitation</h2>
    <textarea
      oninput="const send = document.getElementById('send');
        send.disabled = !this.value;
        send.setAttribute('aria-disabled', String(!this.value));"></textarea>
    <button onclick="document.body.dataset.invite = 'dismissed'">Cancel</button>
    <button id="send" disabled aria-disabled="true"
      onclick="document.body.dataset.invite = 'sent';
        document.body.dataset.note = document.querySelector('#invite textarea').value;
        document.getElementById('invite').remove()">Send</button>
  </div>
"""

NOTE_REQUIRED_ARIA_DIALOG = NOTE_REQUIRED_DIALOG.replace(
    'id="send" disabled', 'id="send"'
).replace("send.disabled = !this.value;", "")

# A Send that is disabled only while the dialog hydrates, beside a note field
# that is optional.
HYDRATING_SEND_DIALOG = NOTE_REQUIRED_DIALOG.replace(
    "</div>",
    """</div>
  <script>
    setTimeout(() => {
      const send = document.getElementById('send');
      send.disabled = false;
      send.setAttribute('aria-disabled', 'false');
    }, 300);
  </script>""",
)


def note_dialog(field: str = "<textarea></textarea>") -> str:
    """An invite dialog with its note field already open and Send enabled."""
    return f"""
  <div role="dialog" id="invite">
    <h2>Add a note</h2>
    {field}
    <button onclick="document.body.dataset.invite = 'dismissed'">Cancel</button>
    <button onclick="document.body.dataset.invite = 'sent';
      document.body.dataset.note = document.querySelector('#invite textarea').value;
      document.getElementById('invite').remove()">Send</button>
  </div>
"""


MAXLENGTH_DIALOG = note_dialog('<textarea maxlength="200"></textarea>')
# The same limit enforced by script, the way a controlled field can do it.
SCRIPT_TRIMMED_DIALOG = note_dialog(
    '<textarea oninput="if (this.value.length > 200) '
    'this.value = this.value.slice(0, 200)"></textarea>'
)

HIDDEN_PRELOADED_NOTE_DIALOG = """
  <div role="dialog" id="preloaded" style="display:none">
    <textarea></textarea>
    <button onclick="document.body.dataset.preloaded = 'clicked'">Cancel</button>
    <button disabled
      onclick="document.body.dataset.preloaded = 'clicked'">Submit</button>
  </div>
"""

POPUP_DIALOG = """
  <div role="dialog" id="popup">
    <button onclick="document.body.dataset.popup = 'clicked'">Close</button>
    <button onclick="document.body.dataset.popup = 'clicked'">Try it</button>
  </div>
"""


def native(dialog: str) -> str:
    """The same invite as a native <dialog open> with no role attribute."""
    return (
        dialog.replace('<div role="dialog" id="invite">', '<dialog open id="invite">')
        .strip()
        .removesuffix("</div>")
        + "</dialog>"
    )


def _page(*parts: str) -> str:
    return f"<!DOCTYPE html><html><body>{''.join(parts)}{ESCAPE_CLOSES_INVITE}</body></html>"


async def _dataset(page, key: str) -> str | None:
    return await page.evaluate(f"document.body.dataset.{key} ?? null")


class TestInviteDialogResolution:
    """Which dialogs count as the one invite dialog (the algorithm alone)."""

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (INVITE_DIALOG, 1),
            (native(INVITE_DIALOG), 1),
            (HIDDEN_PRELOADED_NOTE_DIALOG, 0),
            (
                HIDDEN_PRELOADED_NOTE_DIALOG.replace(
                    "display:none", "visibility:hidden"
                ),
                0,
            ),
            (CHAT_OVERLAY, 0),
            (INVITE_DIALOG + CHAT_OVERLAY + HIDDEN_PRELOADED_NOTE_DIALOG, 1),
            (f"<dialog open>{INVITE_DIALOG}</dialog>", 1),
            (INVITE_DIALOG + POPUP_DIALOG, 2),
            (native(INVITE_DIALOG) + POPUP_DIALOG, 2),
        ],
        ids=[
            "role-dialog",
            "native-no-role",
            "hidden-display",
            "hidden-visibility",
            "chat-overlay",
            "invite-beside-chat-and-hidden",
            "nested-counts-once",
            "two-role-dialogs",
            "native-beside-role",
        ],
    )
    async def test_open_invite_dialogs_are_counted(self, dom_page, body, expected):
        await dom_page.set_content(_page(body))

        state = await dom_page.evaluate(INVITE_DIALOG_STATE_JS)

        assert state["inviteDialogs"] == expected

    @pytest.mark.parametrize(
        "target",
        [{"part": "note", "fromEnd": 0}, {"part": "button", "fromEnd": 0}],
        ids=["note-field", "primary"],
    )
    async def test_nothing_is_resolved_to_act_on_beside_a_second_dialog(
        self, dom_page, target
    ):
        # The action-time gate, independent of the refusal up front: a
        # second dialog that opens between the read and the click still
        # leaves nothing to click.
        await dom_page.set_content(_page(note_dialog(), POPUP_DIALOG))

        handle = await dom_page.evaluate_handle(INVITE_DIALOG_ELEMENT_JS, target)

        assert handle.as_element() is None

    async def test_a_modal_dialog_counts(self, dom_page):
        await dom_page.set_content(_page(native(INVITE_DIALOG).replace(" open", "")))
        await dom_page.evaluate("document.getElementById('invite').showModal()")

        state = await dom_page.evaluate(INVITE_DIALOG_STATE_JS)

        assert state["inviteDialogs"] == 1


class TestNoteRequired:
    """Upstream #407: a profile that only takes an invitation with a note."""

    @pytest.mark.parametrize(
        "dialog",
        [NOTE_REQUIRED_DIALOG, NOTE_REQUIRED_ARIA_DIALOG, native(NOTE_REQUIRED_DIALOG)],
        ids=["disabled", "aria-disabled", "native-dialog"],
    )
    async def test_no_note_is_note_required_and_sends_nothing(self, dom_page, dialog):
        await dom_page.set_content(_page(dialog))

        with pytest.raises(InviteNotSent) as refused:
            await _actions(dom_page)._submit_invite_dialog(None)

        assert refused.value.status == "note_required"
        assert await _dataset(dom_page, "invite") is None

    @pytest.mark.parametrize(
        "dialog",
        [NOTE_REQUIRED_DIALOG, NOTE_REQUIRED_ARIA_DIALOG],
        ids=["disabled", "aria-disabled"],
    )
    async def test_a_note_opens_the_same_dialog(self, dom_page, dialog):
        await dom_page.set_content(_page(dialog))

        result = await _actions(dom_page)._submit_invite_dialog("Hello Ada")

        assert result == (True, True, None)
        assert await _dataset(dom_page, "invite") == "sent"
        assert await _dataset(dom_page, "note") == "Hello Ada"

    async def test_a_send_disabled_while_hydrating_is_still_clicked(self, dom_page):
        await dom_page.set_content(_page(HYDRATING_SEND_DIALOG))

        result = await _actions(dom_page)._submit_invite_dialog(None)

        assert result == (True, False, None)
        assert await _dataset(dom_page, "invite") == "sent"

    async def test_a_chat_overlays_disabled_send_is_not_the_invites(self, dom_page):
        # The chat's Send is disabled and its toggle is the page's last
        # button; neither belongs to the invite, whose Send is enabled.
        await dom_page.set_content(_page(INVITE_DIALOG, CHAT_OVERLAY))

        result = await _actions(dom_page)._submit_invite_dialog(None)

        assert result == (True, False, None)
        assert await _dataset(dom_page, "invite") == "sent"
        assert await _dataset(dom_page, "chat") is None

    async def test_note_required_beside_a_chat_overlay(self, dom_page):
        await dom_page.set_content(_page(CHAT_OVERLAY, NOTE_REQUIRED_DIALOG))

        with pytest.raises(InviteNotSent) as refused:
            await _actions(dom_page)._submit_invite_dialog(None)

        assert refused.value.status == "note_required"
        assert await _dataset(dom_page, "invite") is None
        assert await _dataset(dom_page, "chat") is None


class TestNoteReadBack:
    """A note the field cannot hold whole is never sent cut."""

    @pytest.mark.parametrize(
        "dialog",
        [MAXLENGTH_DIALOG, native(MAXLENGTH_DIALOG)],
        ids=["role-dialog", "native-dialog"],
    )
    async def test_a_note_over_maxlength_is_not_sent(self, dom_page, dialog):
        await dom_page.set_content(_page(dialog))

        with pytest.raises(InviteNotSent) as refused:
            await _actions(dom_page)._submit_invite_dialog("n" * 250)

        assert refused.value.status == "note_too_long"
        assert refused.value.note_limit == 200
        assert await _dataset(dom_page, "invite") is None

    async def test_a_note_the_script_trimmed_is_not_sent(self, dom_page):
        await dom_page.set_content(_page(SCRIPT_TRIMMED_DIALOG))

        with pytest.raises(InviteNotSent) as refused:
            await _actions(dom_page)._submit_invite_dialog("s" * 250)

        assert refused.value.status == "note_too_long"
        assert refused.value.note_limit == 200
        assert await _dataset(dom_page, "invite") is None

    async def test_an_emoji_that_would_straddle_the_limit_is_not_sent(self, dom_page):
        # 199 + 2 code units: Chromium drops the whole emoji.
        await dom_page.set_content(_page(MAXLENGTH_DIALOG))

        with pytest.raises(InviteNotSent) as refused:
            await _actions(dom_page)._submit_invite_dialog("e" * 199 + "\U0001f600")

        assert refused.value.status == "note_too_long"
        # The field's own limit, not the 199 units it happened to keep.
        assert refused.value.note_limit == 200
        assert await _dataset(dom_page, "invite") is None

    @pytest.mark.parametrize(
        "note",
        ["m" * 200, "Hi Ada,\r\nlet's talk about the engine."],
        ids=["at-the-limit", "crlf"],
    )
    async def test_a_note_the_field_holds_is_sent_whole(self, dom_page, note):
        await dom_page.set_content(_page(MAXLENGTH_DIALOG))

        result = await _actions(dom_page)._submit_invite_dialog(note)

        assert result == (True, True, None)
        assert await _dataset(dom_page, "note") == note.replace("\r\n", "\n")

    async def test_a_cut_note_beside_a_chat_overlay_is_not_sent(self, dom_page):
        await dom_page.set_content(_page(CHAT_OVERLAY, MAXLENGTH_DIALOG))

        with pytest.raises(InviteNotSent) as refused:
            await _actions(dom_page)._submit_invite_dialog("c" * 201)

        assert refused.value.status == "note_too_long"
        assert await _dataset(dom_page, "invite") is None
        assert await _dataset(dom_page, "chat") is None


class TestOneInviteDialog:
    """Every read, fill and click lands in the one open invite dialog."""

    @pytest.mark.parametrize("note", [None, "Hello"], ids=["no-note", "note"])
    @pytest.mark.parametrize("decoy_first", [True, False], ids=["before", "after"])
    async def test_a_hidden_preloaded_dialog_with_a_textarea_is_ignored(
        self, dom_page, note, decoy_first
    ):
        parts = [HIDDEN_PRELOADED_NOTE_DIALOG, INVITE_DIALOG]
        await dom_page.set_content(_page(*(parts if decoy_first else parts[::-1])))

        result = await _actions(dom_page)._submit_invite_dialog(note)

        assert result == (True, bool(note), None)
        assert await _dataset(dom_page, "invite") == "sent"
        assert await _dataset(dom_page, "note") == (note or "")
        assert await _dataset(dom_page, "preloaded") is None
        assert (
            await dom_page.evaluate(
                "document.querySelector('#preloaded textarea').value"
            )
            == ""
        )

    async def test_the_note_goes_into_the_field_that_is_shown(self, dom_page):
        # A hidden textarea earlier in the same dialog is not its note field.
        await dom_page.set_content(
            _page(
                note_dialog(
                    '<textarea id="template" style="display:none"></textarea>'
                    '<textarea id="note"></textarea>'
                ).replace(
                    "document.querySelector('#invite textarea').value",
                    "document.getElementById('note').value",
                )
            )
        )

        result = await _actions(dom_page)._submit_invite_dialog("Hello")

        assert result == (True, True, None)
        assert await _dataset(dom_page, "note") == "Hello"

    @pytest.mark.parametrize("note", [None, "Hello"], ids=["no-note", "note"])
    async def test_a_native_dialog_without_a_role_is_sent(self, dom_page, note):
        await dom_page.set_content(_page(native(INVITE_DIALOG)))

        result = await _actions(dom_page)._submit_invite_dialog(note)

        assert result == (True, bool(note), None)
        assert await _dataset(dom_page, "invite") == "sent"
        assert await _dataset(dom_page, "note") == (note or "")

    async def test_a_modal_invite_is_sent(self, dom_page):
        await dom_page.set_content(
            _page(
                HIDDEN_PRELOADED_NOTE_DIALOG, native(INVITE_DIALOG).replace(" open", "")
            )
        )
        await dom_page.evaluate("document.getElementById('invite').showModal()")

        result = await _actions(dom_page)._submit_invite_dialog("Hello")

        assert result == (True, True, None)
        assert await _dataset(dom_page, "note") == "Hello"

    @pytest.mark.parametrize("note", [None, "Hello"], ids=["no-note", "note"])
    @pytest.mark.parametrize("popup_first", [True, False], ids=["before", "after"])
    async def test_two_open_dialogs_are_not_guessed_between(
        self, dom_page, note, popup_first
    ):
        parts = [POPUP_DIALOG, INVITE_DIALOG]
        await dom_page.set_content(_page(*(parts if popup_first else parts[::-1])))

        with pytest.raises(InviteNotSent) as refused:
            await _actions(dom_page)._submit_invite_dialog(note)

        assert refused.value.status == "connect_unavailable"
        assert await _dataset(dom_page, "invite") is None
        assert await _dataset(dom_page, "popup") is None

    async def test_an_upsell_swapped_in_at_send_is_seen_past_a_hidden_dialog(
        self, dom_page
    ):
        """The close wait: a hidden preloaded dialog first in the document
        answered "closed" at once, and a note the upsell swallowed read as
        sent."""
        upsell_swap = note_dialog().replace(
            "document.body.dataset.invite = 'sent';",
            "document.body.dataset.invite = 'swapped';"
            " this.parentElement.innerHTML ="
            " '<p>Out of free notes</p><a href=\\'/premium/products/\\'>Try Premium</a>';"
            " return;",
        )
        await dom_page.set_content(_page(HIDDEN_PRELOADED_NOTE_DIALOG, upsell_swap))

        submitted, note_sent, note_limit = await _actions(
            dom_page
        )._submit_invite_dialog("Hello")

        assert await _dataset(dom_page, "invite") == "swapped"
        assert (submitted, note_sent) == (False, False)
        assert note_limit is not None and "Try Premium" in note_limit


# The legacy three-button dialog with the quota spent: "Add a note" shows the
# upsell instead of a note field, and Send would send without one.
QUOTA_SPENT_LEGACY_DIALOG = """
  <div role="dialog" id="invite">
    <button onclick="document.body.dataset.invite = 'dismissed'">X</button>
    <button onclick="document.body.dataset.invite = 'note';
      this.parentElement.insertAdjacentHTML('beforeend',
        '<p>Out of free notes</p><a href=\\'/premium/products/\\'>Try Premium</a>')"
      >Add a note</button>
    <button onclick="document.body.dataset.invite = 'sent'">Send</button>
  </div>
"""


async def test_the_quota_probe_opens_the_note_editor_of_the_one_dialog(dom_page):
    """The probe runs where the write gate said no: it may press "Add a note"
    of the visible dialog and nothing else, whatever else the page holds."""
    decoy = HIDDEN_PRELOADED_NOTE_DIALOG.replace("<textarea></textarea>", "").replace(
        "</div>",
        "<button onclick=\"document.body.dataset.preloaded = 'clicked'\">X</button></div>",
    )
    await dom_page.set_content(_page(QUOTA_SPENT_LEGACY_DIALOG, decoy))

    message = await _actions(dom_page)._probe_invite_note_limit()

    assert message is not None and "Try Premium" in message
    assert await _dataset(dom_page, "invite") == "note"
    assert await _dataset(dom_page, "preloaded") is None


# A connectable top card: the vanityName invite anchor is the write gate.
CONNECTABLE_PROFILE = """
<!DOCTYPE html><html><body><main>
<section class="topcard">
  <h1>Ada</h1>
  <div class="actions">
    <a href="/preload/custom-invite/?vanityName=testuser" aria-label="Invite">Connect</a>
    <a href="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3AEEE">Message</a>
    <button type="button" aria-expanded="false">More</button>
  </div>
</section>
</main></body></html>
"""


def _actions_reading(page) -> ConnectionActions:
    async def read(_username: str) -> dict[str, Any]:
        return {
            "url": "https://www.linkedin.com/in/testuser/",
            "sections": {"main_profile": "Ada"},
        }

    session = ScrapingSession(cast(Page, page))
    return ConnectionActions(session, PageNavigator(session), read)


@pytest.mark.parametrize(
    ("dialog", "note", "status", "note_limit"),
    [
        (NOTE_REQUIRED_DIALOG, None, "note_required", None),
        (MAXLENGTH_DIALOG, "t" * 250, "note_too_long", 200),
    ],
    ids=["note-required", "note-too-long"],
)
async def test_connect_reports_what_the_dialog_refused(
    dom_page, dialog, note, status, note_limit
):
    """The whole flow over real pages, with the deeplink swapped for a local
    invite page: the real signals open the write gate, and the dialog says
    no."""
    await dom_page.set_content(CONNECTABLE_PROFILE)
    visited: list[str] = []

    async def open_invite(url: str) -> None:
        visited.append(url)
        await dom_page.set_content(_page(dialog))

    with patch.object(
        PageNavigator, "_navigate_to_page", new=AsyncMock(side_effect=open_invite)
    ):
        result = await _actions_reading(dom_page).connect_with_person(
            "testuser", note=note
        )

    assert visited == [
        "https://www.linkedin.com/preload/custom-invite/?vanityName=testuser"
    ]
    assert result["status"] == status
    assert result["note_sent"] is False
    assert result.get("note_limit") == note_limit
    assert await _dataset(dom_page, "invite") is None
