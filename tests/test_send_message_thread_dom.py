"""Browser-DOM tests for replying inside an existing thread (issue #483).

The unit suite mocks ``page.evaluate``, so the thread-scoped target only ever
runs here: route pinning to ``/messaging/thread/<id>/``, the composer docked in
that thread's own pane, the pane-header identity check, and the acknowledgement
rules. The page JavaScript runs in headless Chromium against a synthetic full
messaging page. No LinkedIn request or write is made.

The fixture models what #1108 measured on the full messaging page, not a copy
of LinkedIn's markup: the conversation pane holds a participant header that
links the other member's ``/in/<profile URN>/`` outside every message item,
the message list, and a composer ``<form>`` that holds neither; a sender header
links its sender twice; an open thread renders a client-side placeholder for a
submission, then inserts a separate server-URN node and removes the placeholder.
"""

from __future__ import annotations

import os
import time
from unittest.mock import AsyncMock, patch

import pytest
from patchright.async_api import async_playwright

from linkedin_mcp_server.scraping import message_sender as message_sender_module
from linkedin_mcp_server.scraping.message_sender import MessageSender
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

THREAD_ID = "2-cmVjcnVpdGVyLXRocmVhZA=="
THREAD_URL = f"https://www.linkedin.com/messaging/thread/{THREAD_ID}/"
OTHER_THREAD_PATH = "/messaging/thread/2-b3RoZXItdGhyZWFk/"
MESSAGE = "UNDELIVERED SENTINEL"
# Synthetic profile URNs: the recruiter who wrote first, and the signed-in
# member, who replies.
RECRUITER_URN = "ACoAAR"
SELF_URN = "ACoAAS"
EARLIER_TEXT = "Would you be open to a new role?"


def _sender(page) -> MessageSender:
    session = ScrapingSession(page)
    return MessageSender(session, PageNavigator(session))


def history_item(text: str, urn: str, sender: str | None) -> str:
    links = (
        f'<a href="https://www.linkedin.com/in/{sender}/"></a>'
        f'<a href="https://www.linkedin.com/in/{sender}/">Sender</a>'
        if sender
        else ""
    )
    return f"""
      <div class="msg" data-view-name="message-list-item" data-event-urn="{urn}">
        {links}<span class="message-unit">{text}</span>
      </div>
    """


def thread_page(
    send_js: str = "",
    *,
    header: str | None = RECRUITER_URN,
    history: str | None = None,
    form_identity: str = "",
    draft: str = "",
    sidebar: str = "",
) -> str:
    if history is None:
        history = history_item(
            EARLIER_TEXT, "urn:li:msg_message:(self,recruiter-1)", RECRUITER_URN
        )
    header_html = (
        f'<header id="header"><a href="https://www.linkedin.com/in/{header}/">'
        "Recruiter</a></header>"
        if header
        else '<header id="header">Recruiter</header>'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
  <head><meta charset="utf-8"><title>Messaging</title></head>
  <body>
    <main><div id="layout">
      <section id="sidebar">{sidebar}</section>
      <section id="pane">
        {header_html}
        <div id="thread">{history}</div>
        <form id="composer-scope" onsubmit="return false">
          {form_identity}
          <div id="composer" role="textbox" contenteditable="true"
               style="display:block;width:200px;height:30px">{draft}</div>
          <button id="send" type="submit">Send</button>
        </form>
      </section>
    </div></main>
    <script>
      const SELF_URN = '{SELF_URN}';
      const RECRUITER_URN = '{RECRUITER_URN}';
      function messageItem(text, eventUrn, sender) {{
        const entry = document.createElement('div');
        entry.className = 'msg';
        entry.dataset.viewName = 'message-list-item';
        entry.dataset.eventUrn = eventUrn;
        for (const label of sender ? ['', 'Sender'] : []) {{
          const link = document.createElement('a');
          link.href = `https://www.linkedin.com/in/${{sender}}/`;
          link.textContent = label;
          entry.appendChild(link);
        }}
        const unit = document.createElement('span');
        unit.className = 'message-unit';
        unit.textContent = text;
        entry.appendChild(unit);
        return entry;
      }}
      function onSend(handler) {{
        document.getElementById('send').addEventListener('click', event => {{
          event.preventDefault();
          document.body.dataset.clicked = String(
            Number(document.body.dataset.clicked || 0) + 1);
          const composer = document.getElementById('composer');
          const text = composer.innerText;
          handler(text, composer);
        }});
      }}
      {send_js}
    </script>
  </body>
</html>
"""


# The measured open-thread sequence: a client placeholder for the submission,
# then a separate server-URN node headed by the sender, placeholder removed.
PLACEHOLDER_THEN_SERVER_JS = """
  onSend((text, composer) => {
    const placeholder = messageItem(text, 'client-uuid');
    document.getElementById('thread').appendChild(placeholder);
    composer.textContent = '';
    setTimeout(() => {
      document.getElementById('thread').appendChild(
        messageItem(text, 'urn:li:msg_message:(self,server-new)', SELF_URN));
      placeholder.remove();
    }, 50);
  });
"""

# LinkedIn rendered nothing locally for the submit, yet a server node with the
# text and the member's own header arrived.
SERVER_WITHOUT_PLACEHOLDER_JS = """
  onSend((text, composer) => {
    composer.textContent = '';
    setTimeout(() => {
      document.getElementById('thread').appendChild(
        messageItem(text, 'urn:li:msg_message:(self,server-new)', SELF_URN));
    }, 30);
  });
"""

# The placeholder came and went, but the node that arrived is the recruiter's
# own message carrying the same text.
PLACEHOLDER_THEN_RECRUITER_JS = """
  onSend((text, composer) => {
    const placeholder = messageItem(text, 'client-uuid');
    document.getElementById('thread').appendChild(placeholder);
    composer.textContent = '';
    setTimeout(() => {
      document.getElementById('thread').appendChild(messageItem(
        text, 'urn:li:msg_message:(self,recruiter-2)', RECRUITER_URN));
      placeholder.remove();
    }, 50);
  });
"""

# The submit is ignored, and the recruiter happens to send the same text.
INCOMING_SAME_TEXT_JS = """
  onSend((text) => {
    setTimeout(() => {
      document.getElementById('thread').appendChild(messageItem(
        text, 'urn:li:msg_message:(self,recruiter-2)', RECRUITER_URN));
    }, 30);
  });
"""

# The acknowledgement lands after the page moved to a different thread.
MOVES_TO_ANOTHER_THREAD_JS = f"""
  onSend((text, composer) => {{
    const placeholder = messageItem(text, 'client-uuid');
    document.getElementById('thread').appendChild(placeholder);
    composer.textContent = '';
    setTimeout(() => {{
      history.pushState({{}}, '', '{OTHER_THREAD_PATH}');
      document.getElementById('thread').appendChild(
        messageItem(text, 'urn:li:msg_message:(self,server-new)', SELF_URN));
      placeholder.remove();
    }}, 50);
  }});
"""

# A server copy arrives and is re-rendered, and LinkedIn never showed a local
# placeholder: taking the re-render for one would confirm a node the member
# may not have sent.
SERVER_RERENDER_WITHOUT_PLACEHOLDER_JS = """
  onSend((text, composer) => {
    composer.textContent = '';
    const entry = messageItem(
      text, 'urn:li:msg_message:(self,server-new)', SELF_URN);
    document.getElementById('thread').appendChild(entry);
    setTimeout(() => {
      entry.remove();
      document.getElementById('thread').appendChild(entry.cloneNode(true));
    }, 50);
  });
"""

NOOP_SEND_JS = "onSend(() => {});"


@pytest.fixture
async def dom_page():
    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(
                channel="chromium", headless=True
            )
            page = await browser.new_page()
        except Exception as exc:
            if os.environ.get("CI"):
                raise
            pytest.skip(f"chromium unavailable: {exc}")
        # The sender never navigates here, so every wait it runs keeps this
        # short page default unless it names its own timeout.
        page.set_default_timeout(600)
        page.set_default_navigation_timeout(10_000)
        await page.route(
            "https://www.linkedin.com/**",
            lambda route: route.fulfill(
                status=200,
                content_type="text/html",
                body='<!DOCTYPE html><html><head><meta charset="utf-8"></head></html>',
            ),
        )
        try:
            yield page
        finally:
            await browser.close()


def _short_thread_waits():
    """Cap the thread-page ceilings for cases that are meant to time out."""
    return (
        patch.object(message_sender_module, "_THREAD_READY_TIMEOUT_MS", 600),
        patch.object(message_sender_module, "_THREAD_CONFIRMATION_TIMEOUT_MS", 800),
    )


async def reply(
    page,
    html: str,
    *,
    url: str = THREAD_URL,
    message: str = MESSAGE,
    confirm_send: bool = True,
    short_waits: bool = True,
) -> dict:
    await page.goto(url)
    await page.set_content(html)
    sender = _sender(page)
    ready, confirmation = _short_thread_waits()
    with (
        patch.object(
            PageNavigator, "_navigate_to_page", new_callable=AsyncMock
        ) as navigate,
        patch.object(
            sender, "_read_profile_message_target", new_callable=AsyncMock
        ) as profile_target,
    ):
        if short_waits:
            with ready, confirmation:
                result = await sender.send_message(
                    "ignored-username",
                    message,
                    confirm_send=confirm_send,
                    thread_id=THREAD_ID,
                )
        else:
            result = await sender.send_message(
                "ignored-username",
                message,
                confirm_send=confirm_send,
                thread_id=THREAD_ID,
            )
    # A reply goes to its thread or nowhere: the profile flow never runs.
    navigate.assert_awaited_once_with(THREAD_URL)
    profile_target.assert_not_awaited()
    return result


async def clicked(page) -> str | None:
    return await page.evaluate("document.body.dataset.clicked ?? null")


async def composer_text(page) -> str:
    return (await page.locator("#composer").inner_text()).strip()


class TestThreadReplyConfirmationDom:
    async def test_placeholder_replaced_by_server_node_is_confirmed(self, dom_page):
        result = await reply(dom_page, thread_page(PLACEHOLDER_THEN_SERVER_JS))

        assert result["status"] == "sent"
        assert result["sent"] is True
        assert result["retry_safe"] is False
        assert result["recipient_selected"] is True
        assert result["url"] == THREAD_URL
        assert await clicked(dom_page) == "1"
        entries = dom_page.locator("#thread .msg")
        assert await entries.count() == 2
        assert (await entries.last.locator(".message-unit").inner_text()) == MESSAGE

    async def test_server_node_without_local_placeholder_is_not_confirmed(
        self, dom_page
    ):
        """No local rendering of this submit, so the node proves nothing."""
        result = await reply(dom_page, thread_page(SERVER_WITHOUT_PLACEHOLDER_JS))

        assert result["status"] == "send_unconfirmed"
        assert result["sent"] is False
        assert result["retry_safe"] is False
        assert await clicked(dom_page) == "1"

    async def test_node_from_the_header_participant_is_not_confirmed(self, dom_page):
        result = await reply(dom_page, thread_page(PLACEHOLDER_THEN_RECRUITER_JS))

        assert result["status"] == "send_unconfirmed"
        assert result["sent"] is False
        assert result["retry_safe"] is False

    async def test_incoming_message_with_the_same_text_is_not_confirmed(self, dom_page):
        result = await reply(dom_page, thread_page(INCOMING_SAME_TEXT_JS))

        assert result["status"] == "send_unconfirmed"
        assert result["retry_safe"] is False

    async def test_acknowledgement_on_another_thread_is_not_confirmed(self, dom_page):
        result = await reply(dom_page, thread_page(MOVES_TO_ANOTHER_THREAD_JS))

        assert result["status"] == "send_unconfirmed"
        assert result["retry_safe"] is False
        assert result["url"].endswith(OTHER_THREAD_PATH)

    async def test_ignored_submit_is_not_confirmed_and_keeps_the_text(self, dom_page):
        result = await reply(dom_page, thread_page(NOOP_SEND_JS))

        assert result["status"] == "send_unconfirmed"
        assert result["retry_safe"] is False
        # Submitted text is never cleared: it may be on its way.
        assert await composer_text(dom_page) == MESSAGE

    async def test_rerendered_server_node_is_not_a_placeholder(self, dom_page):
        result = await reply(
            dom_page, thread_page(SERVER_RERENDER_WITHOUT_PLACEHOLDER_JS)
        )

        assert result["status"] == "send_unconfirmed"
        assert result["retry_safe"] is False

    async def test_headerless_pane_still_needs_the_placeholder(self, dom_page):
        # With no participant header there is nobody to refuse by sender, so
        # the placeholder is what separates a reply from an incoming copy.
        confirmed = await reply(
            dom_page, thread_page(PLACEHOLDER_THEN_SERVER_JS, header=None)
        )
        unconfirmed = await reply(
            dom_page, thread_page(SERVER_WITHOUT_PLACEHOLDER_JS, header=None)
        )

        assert confirmed["status"] == "sent"
        assert unconfirmed["status"] == "send_unconfirmed"


class TestThreadReplyTargetDom:
    @pytest.mark.parametrize(
        "foreign_route",
        [
            f"https://www.linkedin.com{OTHER_THREAD_PATH}",
            f"{THREAD_URL}?recipient={RECRUITER_URN}",
            f"{THREAD_URL}?profileUrn=urn%3Ali%3Afsd_profile%3A{RECRUITER_URN}",
        ],
        ids=["other-thread", "recipient-query", "profile-urn-query"],
    )
    async def test_owner_is_pinned_only_on_its_own_thread_route(
        self, dom_page, foreign_route
    ):
        target = message_sender_module._thread_message_target(THREAD_ID)
        sender = _sender(dom_page)

        await dom_page.goto(foreign_route)
        await dom_page.set_content(thread_page())
        # Even a caller that agrees with the page cannot pin another thread.
        foreign = await sender._resolve_message_owner(
            target, expected_route=dom_page.url
        )

        await dom_page.goto(THREAD_URL)
        await dom_page.set_content(thread_page())
        own = await sender._resolve_message_owner(target, expected_route=dom_page.url)

        assert foreign is None
        assert own is not None
        await sender._dispose_message_owner(own)

    async def test_dry_run_verifies_the_thread_and_types_nothing(self, dom_page):
        html = thread_page(
            """
              document.getElementById('composer').addEventListener('focus', () => {
                document.body.dataset.focused = 'true';
              });
            """
        )

        result = await reply(dom_page, html, confirm_send=False)

        assert result == {
            "url": THREAD_URL,
            "status": "confirmation_required",
            "message": "Set confirm_send=true to send the message.",
            "recipient_selected": True,
            "sent": False,
            "retry_safe": True,
        }
        assert await dom_page.evaluate("document.body.dataset.focused ?? null") is None
        assert await composer_text(dom_page) == ""

    async def test_landing_on_another_thread_is_thread_unavailable(self, dom_page):
        result = await reply(
            dom_page,
            thread_page(PLACEHOLDER_THEN_SERVER_JS),
            url=f"https://www.linkedin.com{OTHER_THREAD_PATH}",
        )

        assert result["status"] == "thread_unavailable"
        assert result["recipient_selected"] is False
        assert result["retry_safe"] is True
        assert await clicked(dom_page) is None
        assert await composer_text(dom_page) == ""

    async def test_landing_with_a_recipient_query_is_thread_unavailable(self, dom_page):
        result = await reply(
            dom_page,
            thread_page(PLACEHOLDER_THEN_SERVER_JS),
            url=f"{THREAD_URL}?recipient={RECRUITER_URN}",
        )

        assert result["status"] == "thread_unavailable"
        assert await clicked(dom_page) is None

    async def test_overlay_chat_editor_is_never_the_thread_composer(self, dom_page):
        # The route names the thread, whose pane renders its history but no
        # reply box. The only editor on the page is an overlay chat: a dialog
        # holding some other conversation, beside the pane in the same layout,
        # so the nearest ancestor holding a message is shared by both.
        html = (
            thread_page(PLACEHOLDER_THEN_SERVER_JS)
            .replace(
                '<form id="composer-scope" onsubmit="return false">',
                '</section><aside id="overlay" role="dialog">'
                + history_item("Overlay chat", "urn:li:msg_message:(o,1)", "ACoAAO")
                + '<form id="composer-scope" onsubmit="return false">',
            )
            .replace("</form>\n      </section>", "</form>\n      </aside>")
        )

        result = await reply(dom_page, html)

        assert await dom_page.locator("#pane #composer").count() == 0
        assert await dom_page.locator("#overlay #composer").count() == 1
        assert result["status"] == "composer_unavailable"
        assert result["retry_safe"] is True
        assert await clicked(dom_page) is None
        assert await composer_text(dom_page) == ""

    async def test_pane_without_history_never_proves_the_thread(self, dom_page):
        result = await reply(
            dom_page, thread_page(PLACEHOLDER_THEN_SERVER_JS, history="")
        )

        assert result["status"] == "composer_unavailable"
        assert await clicked(dom_page) is None

    async def test_waits_for_a_heavy_thread_past_the_page_default(self, dom_page):
        # The history renders 1.2s after load, twice the page's 600ms default,
        # which is the shape of the fork's "Thread page did not load".
        html = thread_page(
            """
              setTimeout(() => {
                document.getElementById('thread').appendChild(messageItem(
                  'Would you be open to a new role?',
                  'urn:li:msg_message:(self,recruiter-1)', RECRUITER_URN));
              }, 1200);
            """,
            history="",
        )
        started = time.monotonic()

        result = await reply(dom_page, html, confirm_send=False, short_waits=False)

        assert result["status"] == "confirmation_required"
        assert time.monotonic() - started >= 1.1

    async def test_local_identity_the_header_does_not_show_fails_closed(self, dom_page):
        result = await reply(
            dom_page,
            thread_page(
                PLACEHOLDER_THEN_SERVER_JS,
                form_identity='<a href="https://www.linkedin.com/in/ACoAAO/">Bob</a>',
            ),
        )

        assert result["status"] == "composer_unavailable"
        assert await clicked(dom_page) is None
        assert await composer_text(dom_page) == ""

    @pytest.mark.parametrize(
        "form_identity",
        [
            f'<span data-recipient-urn="urn:li:fsd_profile:{RECRUITER_URN}">R</span>',
            f'<a href="https://www.linkedin.com/in/{RECRUITER_URN}/">Recruiter</a>',
        ],
        ids=["recipient-urn", "profile-link"],
    )
    async def test_local_identity_the_header_shows_is_corroboration(
        self, dom_page, form_identity
    ):
        result = await reply(
            dom_page,
            thread_page(PLACEHOLDER_THEN_SERVER_JS, form_identity=form_identity),
        )

        assert result["status"] == "sent"

    async def test_identity_appearing_during_entry_cleans_without_click(self, dom_page):
        html = thread_page(
            PLACEHOLDER_THEN_SERVER_JS
            + """
              document.getElementById('composer').addEventListener('input', () => {
                const form = document.getElementById('composer-scope');
                if (document.getElementById('composer').innerText &&
                    !form.querySelector('#chip')) {
                  form.insertAdjacentHTML('afterbegin',
                    '<a id="chip" href="https://www.linkedin.com/in/ACoAAO/">Bob</a>');
                }
              });
            """
        )

        result = await reply(dom_page, html)

        assert result["status"] == "compose_interact_failed"
        assert result["retry_safe"] is True
        assert await clicked(dom_page) is None
        assert await composer_text(dom_page) == ""

    async def test_local_identity_without_any_header_fails_closed(self, dom_page):
        result = await reply(
            dom_page,
            thread_page(
                PLACEHOLDER_THEN_SERVER_JS,
                header=None,
                form_identity=f'<span data-recipient-urn="{RECRUITER_URN}">R</span>',
            ),
        )

        assert result["status"] == "composer_unavailable"
        assert await clicked(dom_page) is None

    async def test_other_conversations_in_the_sidebar_do_not_interfere(self, dom_page):
        sidebar = (
            '<a href="https://www.linkedin.com/in/ACoAAO/">Bob</a>'
            '<div data-recipient-urn="ACoAAO">Bob preview</div>'
        )

        result = await reply(
            dom_page, thread_page(PLACEHOLDER_THEN_SERVER_JS, sidebar=sidebar)
        )

        assert result["status"] == "sent"

    async def test_existing_draft_is_left_untouched(self, dom_page):
        result = await reply(
            dom_page, thread_page(PLACEHOLDER_THEN_SERVER_JS, draft="Private draft")
        )

        assert result["status"] == "composer_occupied"
        assert result["retry_safe"] is True
        assert await clicked(dom_page) is None
        assert await composer_text(dom_page) == "Private draft"

    async def test_route_change_after_insertion_cleans_owned_text(self, dom_page):
        html = thread_page(
            PLACEHOLDER_THEN_SERVER_JS
            + f"""
              document.getElementById('composer').addEventListener('input', () => {{
                if (document.getElementById('composer').innerText) {{
                  history.replaceState({{}}, '', '{OTHER_THREAD_PATH}');
                }}
              }});
            """
        )

        result = await reply(dom_page, html)

        assert result["status"] == "recipient_resolution_failed"
        assert result["retry_safe"] is True
        assert await clicked(dom_page) is None
        assert await composer_text(dom_page) == ""

    async def test_composer_leaving_its_pane_before_submit_is_not_clicked(
        self, dom_page
    ):
        # Once text is in, a message renders next to the composer inside a
        # wrapper of its own, so the nearest pane is no longer the one that
        # was verified. Focus and the route stay put; only the pane moved.
        html = thread_page(
            PLACEHOLDER_THEN_SERVER_JS
            + """
              const form = document.getElementById('composer-scope');
              const wrap = document.createElement('div');
              wrap.id = 'form-wrap';
              form.replaceWith(wrap);
              wrap.appendChild(form);
              document.getElementById('composer').addEventListener('input', () => {
                if (document.getElementById('composer').innerText &&
                    !document.querySelector('#form-wrap .msg')) {
                  wrap.prepend(messageItem(
                    'Some other conversation', 'urn:li:msg_message:(x,y)',
                    'ACoAAO'));
                }
              });
            """
        )

        result = await reply(dom_page, html)

        assert result["status"] == "recipient_resolution_failed"
        assert result["retry_safe"] is True
        assert await clicked(dom_page) is None
        assert await composer_text(dom_page) == ""

    async def test_enter_to_send_preference_is_reported(self, dom_page):
        html = thread_page().replace(
            '<button id="send" type="submit">Send</button>',
            '<button id="toggle" type="button" class="msg-form__send-toggle">'
            "Open send options</button>",
        )

        result = await reply(dom_page, html)

        assert result["status"] == "enter_to_send_enabled"
        assert result["retry_safe"] is True
        assert await composer_text(dom_page) == ""


# A multi-paragraph recruiter reply (#441): paragraph breaks and plain line
# breaks, so an entry that turns either into the other changes its lines.
MULTILINE = "Hi,\n\nThanks for thinking of me.\nI would be glad to talk.\n\nBest,\nAlex"

# Ways a sent multi-line message can render; see the profile-send DOM tests.
MULTILINE_RENDER_JS = """
  function fillUnit(unit, text, mode) {
    unit.textContent = '';
    if (mode === 'pre-wrap') {
      unit.style.whiteSpace = 'pre-wrap';
      unit.textContent = text;
    } else if (mode === 'br') {
      text.split('\\n').forEach((line, index) => {
        if (index) unit.appendChild(document.createElement('br'));
        unit.appendChild(document.createTextNode(line));
      });
    } else if (mode === 'paragraphs') {
      for (const paragraph of text.split(/\\n{2,}/)) {
        const p = document.createElement('p');
        fillUnit(p, paragraph, 'br');
        unit.appendChild(p);
      }
    } else if (mode === 'line-paragraphs') {
      for (const line of text.split('\\n').filter(Boolean)) {
        const p = document.createElement('p');
        p.textContent = line;
        unit.appendChild(p);
      }
    } else if (mode === 'joined') {
      unit.textContent = text.split(/\\n+/).join(' ');
    }
  }
  function renderedItem(text, eventUrn, sender, mode) {
    const entry = messageItem('', eventUrn, sender);
    const unit = document.createElement('div');
    unit.className = 'message-unit';
    entry.querySelector('.message-unit').replaceWith(unit);
    fillUnit(unit, text, mode);
    return entry;
  }
"""

# The thread's composer sends on Enter through every key path at once, and
# every send renders the measured sequence: placeholder, then server node.
ENTER_SENDS_JS = (
    MULTILINE_RENDER_JS
    + """
  const composer = document.getElementById('composer');
  document.body.dataset.sends = '';
  document.body.dataset.keyEvents = '0';
  function sendComposer(source) {
    document.body.dataset.sends += `${source};`;
    document.body.dataset.sentHtml = composer.innerHTML;
    const text = composer.innerText;
    const placeholder = renderedItem(text, 'client-uuid', undefined, RENDER_MODE);
    document.getElementById('thread').appendChild(placeholder);
    composer.textContent = '';
    setTimeout(() => {
      document.getElementById('thread').appendChild(renderedItem(
        text, 'urn:li:msg_message:(self,server-new)', SELF_URN, RENDER_MODE));
      placeholder.remove();
    }, 50);
  }
  for (const type of ['keydown', 'keypress', 'keyup', 'beforeinput', 'textInput']) {
    composer.addEventListener(type, () => {
      document.body.dataset.keyEvents = String(
        Number(document.body.dataset.keyEvents) + 1);
    });
  }
  for (const type of ['keydown', 'keypress']) {
    composer.addEventListener(type, event => {
      if (event.key === 'Enter') {
        event.preventDefault();
        sendComposer(`enter-${type}`);
      }
    });
  }
  composer.addEventListener('beforeinput', event => {
    if (['insertLineBreak', 'insertParagraph'].includes(event.inputType)) {
      event.preventDefault();
      sendComposer('beforeinput');
    }
  });
  onSend(() => sendComposer('button'));
"""
)


def enter_sends_thread(mode: str = "br", *, extra_js: str = "") -> str:
    return thread_page(f"const RENDER_MODE = '{mode}';" + ENTER_SENDS_JS + extra_js)


async def sends(page) -> str:
    return await page.evaluate("document.body.dataset.sends")


class TestThreadReplyMultilineDom:
    """A multi-line reply goes out whole, in its thread (#441)."""

    async def test_line_breaks_are_entered_without_a_key_even_where_enter_sends(
        self, dom_page
    ):
        result = await reply(dom_page, enter_sends_thread(), message=MULTILINE)

        assert result["status"] == "sent"
        assert result["sent"] is True
        assert result["url"] == THREAD_URL
        assert await sends(dom_page) == "button;"
        assert await dom_page.evaluate("document.body.dataset.keyEvents") == "0"
        sent_html = await dom_page.evaluate("document.body.dataset.sentHtml")
        assert sent_html.count("<br>") == MULTILINE.count("\n")
        assert "<div" not in sent_html and "<p" not in sent_html
        entries = dom_page.locator("#thread .msg")
        assert await entries.count() == 2
        assert await entries.last.locator(".message-unit").inner_text() == MULTILINE

    @pytest.mark.parametrize(
        "mode", ["br", "pre-wrap", "paragraphs", "line-paragraphs"]
    )
    async def test_confirmation_matches_the_lines_however_they_render(
        self, dom_page, mode
    ):
        result = await reply(dom_page, enter_sends_thread(mode), message=MULTILINE)

        assert result["status"] == "sent"
        assert result["retry_safe"] is False

    async def test_the_same_words_on_one_line_are_not_this_reply(self, dom_page):
        result = await reply(dom_page, enter_sends_thread("joined"), message=MULTILINE)

        assert await sends(dom_page) == "button;"
        assert result["status"] == "send_unconfirmed"
        assert result["retry_safe"] is False

    async def test_the_recruiter_sending_the_same_lines_is_not_this_reply(
        self, dom_page
    ):
        # The placeholder came and went, but the node that arrived is the
        # recruiter's, carrying the same lines.
        html = thread_page(
            MULTILINE_RENDER_JS
            + """
          onSend((text, composer) => {
            const placeholder = renderedItem(text, 'client-uuid', undefined, 'br');
            document.getElementById('thread').appendChild(placeholder);
            composer.textContent = '';
            setTimeout(() => {
              document.getElementById('thread').appendChild(renderedItem(
                text, 'urn:li:msg_message:(self,recruiter-2)', RECRUITER_URN,
                'paragraphs'));
              placeholder.remove();
            }, 50);
          });
        """
        )

        result = await reply(dom_page, html, message=MULTILINE)

        assert result["status"] == "send_unconfirmed"
        assert result["retry_safe"] is False

    async def test_disabled_submit_cleans_every_entered_line(self, dom_page):
        html = enter_sends_thread().replace(
            '<button id="send" type="submit">Send</button>',
            '<button id="send" type="submit" disabled>Send</button>',
        )

        result = await reply(dom_page, html, message=MULTILINE)

        assert result["status"] == "send_unavailable"
        assert result["retry_safe"] is True
        assert await sends(dom_page) == ""
        assert await dom_page.locator("#composer").inner_text() == ""

    async def test_route_change_mid_entry_cleans_the_lines_entered(self, dom_page):
        html = enter_sends_thread(
            extra_js=f"""
          composer.addEventListener('input', event => {{
            if (event.inputType === 'insertLineBreak') {{
              history.replaceState({{}}, '', '{OTHER_THREAD_PATH}');
            }}
          }});
        """
        )

        result = await reply(dom_page, html, message=MULTILINE)

        assert result["status"] == "recipient_resolution_failed"
        assert result["retry_safe"] is True
        assert await sends(dom_page) == ""
        # The first line and the break after it were entered before the
        # route moved; both are removed.
        assert await dom_page.locator("#composer").inner_text() == ""

    async def test_a_composer_sending_on_a_line_break_input_stops_the_entry(
        self, dom_page
    ):
        html = enter_sends_thread(
            extra_js="""
          composer.addEventListener('input', event => {
            if (event.inputType === 'insertLineBreak') sendComposer('input');
          });
        """
        )

        result = await reply(dom_page, html, message=MULTILINE)

        assert result["status"] == "send_unconfirmed"
        assert result["retry_safe"] is False
        assert await sends(dom_page) == "input;"
        assert await dom_page.locator("#composer").inner_text() == ""
