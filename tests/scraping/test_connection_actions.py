"""Tests for the invitation-action owner.

The write gate is what most of these hold: the invite deeplink may only be
opened to submit once LinkedIn has exposed the vanityName invite anchor, and
the one other thing allowed to open it — the note-quota probe — never clicks
a primary button. Every case that reaches a decision drives the real
classifier from structural signals; no case reads a label.

``tests/test_action_signals_dom.py`` covers the other half, where the
programs run against a real DOM in four label sets. Here ``page.evaluate`` is
a mock, so the JS never executes and the signals are supplied directly.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping.connection import ActionSignals
from linkedin_mcp_server.scraping.connection_actions import (
    MAX_INVITE_NOTE_LENGTH,
    ConnectionActions,
    InviteNotSent,
    _InviteDialogState,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

PREMIUM_MESSAGE = (
    "Wysyłaj nieograniczoną liczbę spersonalizowanych zaproszeń dzięki Premium"
)


def _actions(page, read_main_profile: Any = None) -> ConnectionActions:
    """Wire the connection owner the way the facade does.

    The facade hands over one main-profile read and nothing else of the
    person workflow, so the borrow is the whole collaborator surface a test
    has to supply. The default refuses the call: a case that never reads a
    profile should not be able to, and one that does says which texts it
    expects.
    """

    async def unread(_username: str) -> dict[str, Any]:
        raise AssertionError("this case does not read a profile")

    session = ScrapingSession(page)
    return ConnectionActions(
        session,
        PageNavigator(session),
        read_main_profile if read_main_profile is not None else unread,
    )


def _reads(*texts: str) -> AsyncMock:
    """Script the main-profile read: one answer per call, in order.

    A single text answers every call, which is what the states that never
    re-read need. Two or more script the verification re-reads an action
    performs, and a further call raises ``StopIteration`` rather than
    quietly repeating the last page.
    """
    pages = [
        {
            "url": "https://www.linkedin.com/in/testuser/",
            "sections": {"main_profile": text} if text else {},
        }
        for text in texts
    ]
    if len(pages) == 1:
        return AsyncMock(return_value=pages[0])
    return AsyncMock(side_effect=pages)


def _signals(
    invite: bool = False,
    compose: bool = False,
    edit: bool = False,
    labeled_action: bool = False,
    labeled_anchor: bool = False,
    incoming_row: bool = False,
) -> ActionSignals:
    return ActionSignals(
        has_invite_anchor=invite,
        has_compose_anchor_in_action_root=compose,
        has_edit_intro_anchor=edit,
        has_labeled_action_button=labeled_action,
        has_labeled_action_anchor=labeled_anchor,
        has_incoming_action_row=incoming_row,
    )


class _FakeInviteDialog:
    """The one open invite dialog, as the two scoped page reads answer.

    ``page.evaluate`` is a mock here, so the dialog programs never run
    (``tests/test_invite_dialog_dom.py`` runs them against a real DOM). This
    stands in for their answers: buttons by position, a note field, and what
    a click or a fill does to them. Clicks are recorded as the index of the
    button in the dialog, so ``buttons - 1`` is the primary.
    """

    def __init__(
        self,
        *,
        buttons: int = 3,
        note_field: str = "none",
        max_length: int | None = None,
        keeps: int | None = None,
        dialogs: int = 1,
        primary_disabled: bool = False,
        enabled_after_reads: int | None = None,
        on_click: dict[int, Any] | None = None,
    ):
        self.buttons = buttons
        self.note_field = note_field
        self.note_value = ""
        self.max_length = max_length
        self.keeps = keeps if keeps is not None else max_length
        self.dialogs = dialogs
        self.primary_disabled = primary_disabled
        self.enabled_after_reads = enabled_after_reads
        self.on_click = on_click or {}
        self.reads = 0
        self.clicks: list[int] = []
        self.lookups: list[tuple[str, int]] = []
        self.fills: list[str] = []

    async def state(self) -> _InviteDialogState:
        self.reads += 1
        if (
            self.enabled_after_reads is not None
            and self.reads > self.enabled_after_reads
        ):
            self.primary_disabled = False
        if self.dialogs != 1:
            return _InviteDialogState(dialogs=self.dialogs)
        return _InviteDialogState(
            dialogs=1,
            buttons=self.buttons,
            primary_disabled=self.primary_disabled,
            note_field=self.note_field,
            note_value=self.note_value,
            note_max_length=self.max_length,
        )

    async def element(self, part: str, *, from_end: int = 0) -> Any:
        self.lookups.append((part, from_end))
        if self.dialogs != 1:
            return None
        handle = MagicMock()
        handle.dispose = AsyncMock()
        if part == "note":
            if self.note_field == "none":
                return None

            async def fill(value: str, **_kwargs: Any) -> None:
                self.fills.append(value)
                stored = value.replace("\r\n", "\n")
                self.note_value = stored[: self.keeps] if self.keeps else stored

            handle.fill = AsyncMock(side_effect=fill)
            return handle
        index = self.buttons - 1 - from_end
        if index < 0:
            return None

        async def click(**_kwargs: Any) -> None:
            self.clicks.append(index)
            hook = self.on_click.get(index)
            if hook is not None:
                hook(self)

        handle.click = AsyncMock(side_effect=click)
        handle.focus = AsyncMock()
        return handle

    @contextmanager
    def installed(self, actions: ConnectionActions):
        with (
            patch.object(actions, "_invite_dialog_state", new=self.state),
            patch.object(actions, "_invite_dialog_element", new=self.element),
        ):
            yield self


def _state(
    note_field: str = "visible",
    *,
    buttons: int = 2,
    value: str = "",
    dialogs: int = 1,
) -> _InviteDialogState:
    """One scripted read of the invite dialog."""
    if dialogs != 1:
        return _InviteDialogState(dialogs=dialogs)
    return _InviteDialogState(
        dialogs=1, buttons=buttons, note_field=note_field, note_value=value
    )


class TestConnectWithPerson:
    async def test_connectable_navigates_deeplink_and_verifies(self, mock_page):
        """Connect via deeplink: dialog opens, submit succeeds, anchor disappears."""
        text = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"
        post_text = "Jane\n\n· 3rd\n\nEngineer\n\nMessage\nPending\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text, post_text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[_signals(invite=True), _signals()],
            ),
            patch.object(
                PageNavigator,
                "_navigate_to_page",
                new_callable=AsyncMock,
            ) as mock_nav,
            patch.object(
                actions,
                "_dialog_is_open",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connected"
        mock_nav.assert_awaited_once()
        await_args = mock_nav.await_args
        assert await_args is not None
        assert "preload/custom-invite" in await_args.args[0]

    async def test_connectable_send_failed_when_anchor_persists(self, mock_page):
        """Profile still exposes Connect after the settle retry → send_failed."""
        text = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text, text, text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    _signals(invite=True),
                    _signals(invite=True),
                    _signals(invite=True),
                ],
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "send_failed"

    @pytest.mark.parametrize(
        ("retry_signals", "state"),
        [
            (_signals(labeled_anchor=True), "pending"),
            (_signals(compose=True), "already_connected"),
        ],
        ids=["pending", "accepted-meanwhile"],
    )
    async def test_connectable_connected_on_settle_retry(
        self, mock_page, retry_signals, state
    ):
        """The first post-send read still renders Connect; the settle retry
        sees the invitation pending (or already accepted) and reports
        connected."""
        text = "Jane\n\n· 2nd\n\nEngineer\n\nConnect\nMore\nAbout\n"
        post = "Jane\n\n· 2nd\n\nEngineer\n\nMessage\nPending\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text, text, post))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    _signals(invite=True),
                    _signals(invite=True),
                    retry_signals,
                ],
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connected"
        assert state in result["message"]
        mock_sleep.assert_awaited_once()

    @pytest.mark.parametrize(
        "retry_signals",
        [_signals(), _signals(compose=True, labeled_action=True)],
        ids=["unreadable", "follow-only"],
    )
    async def test_settle_retry_without_pending_keeps_send_failed(
        self, mock_page, retry_signals
    ):
        """Only a pending invitation on the retry is evidence it landed."""
        text = "Jane\n\n· 2nd\n\nEngineer\n\nConnect\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text, text, ""))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    _signals(invite=True),
                    _signals(invite=True),
                    retry_signals,
                ],
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "send_failed"

    async def test_connectable_no_dialog_returns_connect_unavailable(self, mock_page):
        """Deeplink opened but no dialog appeared → connect_unavailable."""
        text = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(invite=True),
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=False
            ),
            patch.object(actions, "_dismiss_dialog", new_callable=AsyncMock),
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connect_unavailable"

    async def test_returns_already_connected_via_anchor(self, mock_page):
        """1st-degree detected via /messaging/compose anchor."""
        text = "Collin\n\n· 1st\n\nEngineer\n\nMessage\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text))

        with patch.object(
            actions,
            "_read_action_signals",
            new_callable=AsyncMock,
            return_value=_signals(compose=True),
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "already_connected"

    async def test_returns_self_profile_via_edit_intro_anchor(self, mock_page):
        """Editing-your-own-profile anchor blocks connect attempts."""
        actions = _actions(mock_page, _reads("Daniel\n\nEdit profile\n"))

        with patch.object(
            actions,
            "_read_action_signals",
            new_callable=AsyncMock,
            return_value=_signals(edit=True),
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connect_unavailable"
        assert "own profile" in result["message"]

    async def test_connect_via_more_menu(self, mock_page):
        """Follow-primary profile with Connect under More: detection sees
        no invite anchor initially, _open_more_menu surfaces it, deeplink
        fires."""
        # Pre-More: Follow primary, Connect hidden under the More dropdown.
        pre = "Christian\n\n· 2nd\n\nFounder\n\nFollow\nMessage\nMore\n"
        post = "Christian\n\n· 2nd\n\nFounder\n\nMessage\nPending\nMore\n"
        actions = _actions(mock_page, _reads(pre, post))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                # 1st: follow_only (compose+labeled, no invite).
                # 2nd: post-More reread reveals invite anchor.
                # 3rd: post-deeplink verification — invite anchor gone.
                side_effect=[
                    _signals(compose=True, labeled_action=True),
                    _signals(invite=True, compose=True, labeled_action=True),
                    _signals(),
                ],
            ),
            patch.object(
                actions,
                "_open_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_open_more,
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                actions,
                "_dialog_is_open",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connected"
        mock_open_more.assert_awaited_once()
        # Deeplink fired exactly once.
        assert mock_nav.await_count == 1
        await_args = mock_nav.await_args
        assert await_args is not None
        assert "preload/custom-invite" in await_args.args[0]

    async def test_follow_only_after_more_does_not_send(self, mock_page):
        """Pending or genuinely follow-only profile: invite anchor never
        appears even after More-menu open. Critical write-gate guardrail —
        no deeplink fires, no connection request goes out."""
        text = "Public Figure\n\n· 3rd+\n\nCEO\n\nFollow\nMessage\nMore\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                # Both reads (initial + post-More) show no invite anchor.
                side_effect=[
                    _signals(compose=True, labeled_action=True),
                    _signals(compose=True, labeled_action=True),
                ],
            ),
            patch.object(
                actions,
                "_open_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_open_more,
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
                # A successful submit, so a gate that stopped holding reports
                # the deeplink it fired rather than crashing on the mock.
                return_value=(True, False, None),
            ) as mock_submit,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connect_unavailable"
        assert result.get("note_sent") is False or "note_sent" not in result
        mock_open_more.assert_awaited_once()
        # Critical: deeplink must NOT fire and dialog must NOT be submitted.
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()

    async def test_follow_only_with_note_reports_note_limit_from_deeplink_probe(
        self, mock_page
    ):
        """A requested note may reveal Premium quota without submitting."""
        text = "Public Figure\n\n· 3rd+\n\nCEO\n\nFollow\nMessage\nMore\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    _signals(compose=True, labeled_action=True),
                    _signals(compose=True, labeled_action=True),
                ],
            ),
            patch.object(
                actions,
                "_open_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                actions,
                "_probe_invite_note_limit",
                new_callable=AsyncMock,
                return_value=PREMIUM_MESSAGE,
            ) as mock_probe,
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
                # A successful submit, so a gate that stopped holding reports
                # the deeplink it fired rather than crashing on the mock.
                return_value=(True, False, None),
            ) as mock_submit,
        ):
            result = await actions.connect_with_person("testuser", note="Hello")

        assert result["status"] == "custom_note_limit_reached"
        assert result["message"] == PREMIUM_MESSAGE
        assert result["note_sent"] is False
        mock_nav.assert_awaited_once()
        mock_probe.assert_awaited_once()
        mock_submit.assert_not_awaited()

    async def test_more_menu_unavailable_does_not_send(self, mock_page):
        """Action root present but no More button (unusual but possible):
        _open_more_menu returns False, no retry, no deeplink fires."""
        text = "Public Figure\n\n· 3rd+\n\nCEO\n\nFollow\nMessage\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(compose=True, labeled_action=True),
            ),
            patch.object(
                actions,
                "_open_more_menu",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
                # A successful submit, so a gate that stopped holding reports
                # the deeplink it fired rather than crashing on the mock.
                return_value=(True, False, None),
            ) as mock_submit,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connect_unavailable"
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()

    async def test_returns_pending(self, mock_page):
        """Profile with a pending invitation: detected via labeled <a> in
        the action root. Returns status='pending' without firing the
        deeplink (LinkedIn would only show 'already invited' anyway)."""
        text = "Frank\n\n· 3rd\n\nFounder\n\nMessage\nPending\nMore\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(compose=True, labeled_anchor=True),
            ),
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
                # A successful submit, so a gate that stopped holding reports
                # the deeplink it fired rather than crashing on the mock.
                return_value=(True, False, None),
            ) as mock_submit,
            patch.object(
                actions, "_open_more_menu", new_callable=AsyncMock
            ) as mock_open_more,
            patch.object(
                actions, "_click_incoming_accept", new_callable=AsyncMock
            ) as mock_accept,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "pending"
        # No write-path side effects, and no action taken on the invitation
        # already out: withdrawing it is what the Pending control does.
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()
        mock_open_more.assert_not_awaited()
        mock_accept.assert_not_awaited()

    async def test_returns_incoming_request_accepted(self, mock_page):
        """Structural detection + structural accept click, German locale."""
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"
        post = "Eric\n\n· 1.\n\nAachen\n\nNachricht\nMehr\nInfo\n"
        actions = _actions(mock_page, _reads(pre, post))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    _signals(incoming_row=True),
                    # The row's own More menu, open: no invite anchor.
                    _signals(incoming_row=True),
                    _signals(compose=True),
                ],
            ),
            patch.object(
                actions,
                "_open_incoming_row_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_incoming_accept",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_accept,
            patch.object(
                PageNavigator,
                "_navigate_to_page",
                new_callable=AsyncMock,
            ) as mock_nav,
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
            ) as mock_submit,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "accepted"
        mock_accept.assert_awaited_once()
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()

    async def test_incoming_request_send_failed_when_click_fails(self, mock_page):
        """Structural accept click did not land; no locale-text guessing —
        report send_failed without navigating or clicking anything else.

        The owner holds no click-by-text helper to patch, so the claim is
        made against the page: a text fallback would have to build a
        locator, and nothing here builds one.
        """
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"
        actions = _actions(mock_page, _reads(pre))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(incoming_row=True),
            ),
            patch.object(
                actions,
                "_open_incoming_row_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_incoming_accept",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                PageNavigator,
                "_navigate_to_page",
                new_callable=AsyncMock,
            ) as mock_nav,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "send_failed"
        mock_nav.assert_not_awaited()
        mock_page.locator.assert_not_called()

    async def test_incoming_request_send_failed_when_no_first_degree(self, mock_page):
        """Accept clicked but profile never transitions to 1st-degree."""
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"
        actions = _actions(mock_page, _reads(pre, pre, pre))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(incoming_row=True),
            ),
            patch.object(
                actions,
                "_open_incoming_row_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_incoming_accept",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "send_failed"

    async def test_incoming_request_accepted_on_settle_retry(self, mock_page):
        """The first post-click read still renders the old top card;
        the settle retry sees the 1st-degree state and reports accepted."""
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"
        post = "Eric\n\n· 1.\n\nAachen\n\nNachricht\nMehr\nInfo\n"
        actions = _actions(mock_page, _reads(pre, pre, post))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    _signals(incoming_row=True),
                    # The row's own More menu, open: no invite anchor.
                    _signals(incoming_row=True),
                    _signals(incoming_row=True),
                    _signals(compose=True),
                ],
            ),
            patch.object(
                actions,
                "_open_incoming_row_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_incoming_accept",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "accepted"
        mock_sleep.assert_awaited_once()

    async def test_incoming_shape_with_connect_in_its_menu_is_invited(self, mock_page):
        """Issue #629: a creator-mode top card, [Follow][Save in Sales
        Navigator][More], carries the incoming fingerprint. Its own More menu
        holds the invite anchor, so the call invites through the deeplink and
        never clicks the row's first button, which there is Follow."""
        pre = "Marc\n\n· 2nd\n\nCreator\n\nFollow\nSave in Sales Navigator\nMore\n"
        post = "Marc\n\n· 2nd\n\nCreator\n\nFollow\nPending\nMore\n"
        actions = _actions(mock_page, _reads(pre, post))
        mock_page.keyboard.press = AsyncMock()

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    _signals(incoming_row=True),
                    # The row's own More menu, open: Connect is in it.
                    _signals(invite=True, incoming_row=True),
                    _signals(compose=True, labeled_anchor=True),
                ],
            ),
            patch.object(
                actions,
                "_open_incoming_row_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_open_row_menu,
            patch.object(
                actions, "_click_incoming_accept", new_callable=AsyncMock
            ) as mock_accept,
            patch.object(
                actions, "_open_more_menu", new_callable=AsyncMock
            ) as mock_open_more,
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
                return_value=(True, False, None),
            ) as mock_submit,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connected"
        mock_open_row_menu.assert_awaited_once()
        mock_accept.assert_not_awaited()
        mock_open_more.assert_not_awaited()
        mock_submit.assert_awaited_once_with(None)
        mock_nav.assert_awaited_once()
        await_args = mock_nav.await_args
        assert await_args is not None
        assert "preload/custom-invite/?vanityName=testuser" in await_args.args[0]
        # The menu is closed again before the deeplink navigation.
        mock_page.keyboard.press.assert_awaited_once_with("Escape")

    async def test_incoming_row_whose_menu_will_not_open_is_not_accepted(
        self, mock_page
    ):
        """No open menu, no disprove: Accept is not clicked on a guess."""
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"
        actions = _actions(mock_page, _reads(pre))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(incoming_row=True),
            ) as mock_signals,
            patch.object(
                actions,
                "_open_incoming_row_more_menu",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                actions, "_click_incoming_accept", new_callable=AsyncMock
            ) as mock_accept,
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "send_failed"
        mock_accept.assert_not_awaited()
        mock_nav.assert_not_awaited()
        mock_signals.assert_awaited_once()

    async def test_incoming_shape_with_a_note_and_no_menu_connect_is_ambiguous(
        self, mock_page
    ):
        """A note asks for an invitation, and Accept takes none. With no
        Connect in the row's menu either, the row could still be a Follow
        button, so nothing is clicked and nothing is navigated."""
        pre = "Marc\n\n· 2nd\n\nCreator\n\nFollow\nSave in Sales Navigator\nMore\n"
        actions = _actions(mock_page, _reads(pre))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(incoming_row=True),
            ),
            patch.object(
                actions,
                "_open_incoming_row_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions, "_click_incoming_accept", new_callable=AsyncMock
            ) as mock_accept,
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                actions, "_submit_invite_dialog", new_callable=AsyncMock
            ) as mock_submit,
        ):
            result = await actions.connect_with_person("testuser", note="Hi Marc")

        assert result["status"] == "incoming_request_ambiguous"
        assert result["note_sent"] is False
        mock_accept.assert_not_awaited()
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()

    async def test_returns_unavailable_when_no_signals_and_text(self, mock_page):
        """No structural signals, no actionable text → connect_unavailable."""
        text = "Public Figure\n\n· 3rd+\n\nCEO\n\nFollow\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(),
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=False
            ),
            patch.object(actions, "_dismiss_dialog", new_callable=AsyncMock),
        ):
            result = await actions.connect_with_person("testuser")

        # follow_only path goes through deeplink; no dialog opens → unavailable
        assert result["status"] == "connect_unavailable"

    async def test_returns_unavailable_on_empty_page(self, mock_page):
        actions = _actions(mock_page, _reads(""))

        result = await actions.connect_with_person("testuser")

        assert result["status"] == "unavailable"

    async def test_normalizes_before_its_own_downstream_use(self, mock_page):
        """The traversal case cannot see this one.

        The person workflow normalizes too, so removing this workflow's own
        call still raises on "../../feed". A full URL is what separates
        them: the read would succeed while the invite deeplink and the
        action-signal selectors kept receiving the URL where they expect
        the vanity.
        """
        read = _reads("text")
        actions = _actions(mock_page, read)
        seen: list[str] = []

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=lambda username: seen.append(username) or _signals(),
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
        ):
            await actions.connect_with_person(
                "https://de.linkedin.com/in/williamhgates"
            )

        assert seen == ["williamhgates"]
        read.assert_awaited_once_with("williamhgates")


class TestInviteDialog:
    async def test_premium_upsell_message_reads_linkedin_dialog_text(self, mock_page):
        """Premium upsell detection returns LinkedIn's raw dialog text."""
        actions = _actions(mock_page)
        premium_link = MagicMock()
        premium_link.wait_for = AsyncMock(return_value=None)
        premium_link.is_visible = AsyncMock(return_value=True)
        premium_link.inner_text = AsyncMock(return_value="fallback")
        premium_link.first = premium_link
        mock_page.locator.return_value = premium_link
        mock_page.evaluate = AsyncMock(return_value=PREMIUM_MESSAGE)

        result = await actions._get_premium_upsell_message(timeout=1234)

        assert result == PREMIUM_MESSAGE
        mock_page.locator.assert_called_once_with(
            'dialog[open] a[href*="/premium/"], [role="dialog"] a[href*="/premium/"]'
        )
        premium_link.wait_for.assert_awaited_once_with(state="visible", timeout=1234)

    async def test_reports_premium_after_add_note(self, mock_page):
        """Add-note Premium upsell is a note-limit block, not no-dialog."""
        actions = _actions(mock_page)
        # The legacy three-button dialog, whose "Add a note" never mounts a
        # textarea: the upsell took its place.
        dialog = _FakeInviteDialog(buttons=3, note_field="none")

        with (
            dialog.installed(actions),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_wait_for_invite_note_field",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=PREMIUM_MESSAGE,
            ) as mock_message,
            patch.object(
                actions, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            result = await actions._submit_invite_dialog("Hello")

        assert result == (False, False, PREMIUM_MESSAGE)
        # "Add a note" only: the secondary, never the primary.
        assert dialog.clicks == [1]
        mock_message.assert_awaited_once()
        mock_dismiss.assert_awaited_once()

    async def test_note_sent_despite_premium_banner_when_textarea_appears(
        self, mock_page
    ):
        """A Premium nudge banner alongside a live textarea is not a block.

        LinkedIn renders the "N personalized invitations remaining" /
        Activate Premium banner on this step as a persistent upsell nudge,
        not only when the free quota is truly exhausted, and it can still be
        in the DOM for a moment right after a successful Send, before it
        unmounts with the closing dialog. The regression this guards: an
        earlier version bailed the instant that banner was detectable —
        either the moment the textarea appeared, or the moment Send
        succeeded — silently dropping the note on every single send that
        rendered it, independent of actual remaining quota.

        ``_get_premium_upsell_message`` is mocked to always return a message
        (a banner that is genuinely detectable start to finish), so a
        version that still gates either check on banner presence alone
        fails this test; only gating on textarea absence / dialog staying
        open lets it pass.
        """
        actions = _actions(mock_page)

        def mount_textarea(dialog: _FakeInviteDialog) -> None:
            dialog.note_field = "visible"

        # The gating dialog: "Add a note" (index 0) mounts the textarea.
        dialog = _FakeInviteDialog(
            buttons=2, note_field="none", on_click={0: mount_textarea}
        )
        # The dialog closes on schedule after Send — a plain "did not time
        # out" wait, and the Premium banner is present in the DOM throughout
        # regardless.
        mock_page.wait_for_selector = AsyncMock(return_value=None)

        async def fill(note: str) -> bool:
            dialog.note_value = note
            return True

        with (
            dialog.installed(actions),
            patch("linkedin_mcp_server.scraping.connection_actions.asyncio.sleep"),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_fill_dialog_textarea",
                new_callable=AsyncMock,
                side_effect=fill,
            ) as mock_fill,
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=PREMIUM_MESSAGE,
            ) as mock_message,
        ):
            result = await actions._submit_invite_dialog("Hello")

        # Proves the reveal-step bail is gone: the code must reach the fill
        # call rather than returning the instant the banner is detectable.
        mock_fill.assert_awaited_once_with("Hello")
        assert result == (True, True, None)
        # Neither the reveal-step check nor the post-submit check calls
        # _get_premium_upsell_message on this path: the textarea appeared
        # (skips the reveal-step gate) and the dialog closed on schedule
        # (skips the post-submit gate). A version that still gates either
        # check on banner presence alone would call this and return a
        # blocked result instead of (True, True, None) above.
        mock_message.assert_not_called()

    @pytest.mark.parametrize(
        "recount",
        [_state("visible"), None],
        ids=["textarea-mounted", "recount-failed"],
    )
    async def test_failed_fill_beside_a_mounted_textarea_is_not_a_note_limit(
        self, mock_page, recount
    ):
        """A fill that fails while the textarea may still be there sends nothing.

        The Premium nudge banner is detectable throughout, so reading it
        after any failed fill reported ``custom_note_limit_reached`` for an
        account with quota left (observed live: the dialog said three
        personalized invitations remained). A recount that fails proves no
        absence, so it reports no quota either.
        """
        actions = _actions(mock_page)

        with (
            # Mounted when the dialog opens; the recount after the failed
            # fill either still sees it or cannot be read at all.
            patch.object(
                actions,
                "_invite_dialog_state",
                new_callable=AsyncMock,
                side_effect=[_state("visible"), recount],
            ),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_fill_dialog_textarea",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=PREMIUM_MESSAGE,
            ),
            patch.object(
                actions, "_click_dialog_primary_button", new_callable=AsyncMock
            ) as mock_send,
            patch.object(
                actions, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            result = await actions._submit_invite_dialog("Hello")

        assert result == (False, False, None)
        mock_send.assert_not_called()
        mock_dismiss.assert_awaited_once()

    async def test_failed_fill_after_the_upsell_replaced_the_textarea(self, mock_page):
        """The upsell taking the textarea's place is still a note limit."""
        actions = _actions(mock_page)

        with (
            # Mounted when the dialog opens, gone once the fill has failed.
            patch.object(
                actions,
                "_invite_dialog_state",
                new_callable=AsyncMock,
                side_effect=[_state("visible"), _state("none")],
            ),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_fill_dialog_textarea",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=PREMIUM_MESSAGE,
            ),
            patch.object(
                actions, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            result = await actions._submit_invite_dialog("Hello")

        assert result == (False, False, PREMIUM_MESSAGE)
        mock_dismiss.assert_awaited_once()

    async def test_failed_fill_beside_a_hidden_textarea_is_a_note_limit(
        self, mock_page
    ):
        """A textarea the upsell left mounted but hidden is no note field."""
        actions = _actions(mock_page)

        with (
            patch.object(
                actions,
                "_invite_dialog_state",
                new_callable=AsyncMock,
                return_value=_state("hidden"),
            ),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_fill_dialog_textarea",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=PREMIUM_MESSAGE,
            ),
            patch.object(actions, "_dismiss_dialog", new_callable=AsyncMock),
        ):
            result = await actions._submit_invite_dialog("Hello")

        assert result == (False, False, PREMIUM_MESSAGE)

    async def test_reports_premium_after_send_click_failure(self, mock_page):
        """Premium upsell intercepting the Send click is a note-limit block.

        When LinkedIn swaps the invite dialog for the Premium upsell at the
        moment of submit, the original primary button is detached or pointer-
        event covered, so ``_click_dialog_primary_button`` and the keyboard
        fallback both fail. Without the post-click upsell probe the caller
        would dismiss the dialog and report ``connect_unavailable`` even
        though LinkedIn's raw quota message is sitting in the visible modal.
        """
        actions = _actions(mock_page)

        # Textarea already exposed so the reveal/fill branch succeeds and the
        # test focuses on the post-submit failure path.
        dialog = _FakeInviteDialog(buttons=2, note_field="visible")
        dialog.note_value = "Hello"
        mock_page.keyboard = MagicMock()
        mock_page.keyboard.press = AsyncMock()

        message = "You're out of free custom notes. Bypass the limit with Premium..."

        with (
            dialog.installed(actions),
            patch.object(
                actions,
                "_dialog_is_open",
                new_callable=AsyncMock,
                # First call: dialog open at entry. Second call: still open
                # after the keyboard fallback, so sent remains False.
                side_effect=[True, True],
            ),
            patch.object(
                actions,
                "_fill_dialog_textarea",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=message,
            ) as mock_message,
            patch.object(
                actions, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            result = await actions._submit_invite_dialog("Hello")

        assert result == (False, False, message)
        mock_message.assert_awaited_once()
        mock_dismiss.assert_awaited_once()

    async def test_reports_premium_after_an_accepted_send_click(self, mock_page):
        """A Send click that succeeds is not yet a note that was delivered.

        LinkedIn accepts the click and then swaps the invite dialog for the
        quota upsell modal, which matches the same dialog selector as the
        invite dialog it replaced — so the close-wait genuinely times out
        rather than resolving, and that timeout is the real, structural
        signal (no label text read) that distinguishes this from a benign
        nudge banner that closes with the dialog on a real send. Reporting
        this as a send would tell the caller a note reached a member who
        never got one, and the two earlier upsell probes cannot see it: both
        sit on failure paths.
        """
        actions = _actions(mock_page)

        dialog = _FakeInviteDialog(buttons=2, note_field="visible")
        dialog.note_value = "Hello"
        # The close-wait times out: the upsell modal that replaced the
        # invite dialog still matches _DIALOG_SELECTOR, so it never becomes
        # "hidden".
        mock_page.wait_for_selector = AsyncMock(
            side_effect=PlaywrightTimeoutError("dialog still open")
        )

        with (
            dialog.installed(actions),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_fill_dialog_textarea",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=PREMIUM_MESSAGE,
            ) as mock_message,
            patch.object(
                actions, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            result = await actions._submit_invite_dialog("Hello")

        assert result == (False, False, PREMIUM_MESSAGE)
        mock_message.assert_awaited_once()
        mock_dismiss.assert_awaited_once()
        # The close wait is what proves this is the blocked path: it is
        # attempted and times out, which is the real signal gating the
        # premium check below it.
        mock_page.wait_for_selector.assert_awaited_once()

    async def test_the_quota_probe_never_clicks_the_primary_button(self, mock_page):
        """The probe opens the note editor and touches nothing else.

        It runs only on profiles the write gate already refused, so a click
        on the dialog's *last* button would send the invitation this workflow
        decided not to send. The index ``btn_count - 2`` is the whole of that
        guarantee, and nothing else held it: every case that drives the probe
        through ``connect_with_person`` replaces it wholesale.
        """
        actions = _actions(mock_page)
        # The legacy three-button invite dialog: dismiss, "Add a note", Send.
        dialog = _FakeInviteDialog(buttons=3, note_field="none")

        with (
            dialog.installed(actions),
            patch("linkedin_mcp_server.scraping.connection_actions.asyncio.sleep"),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                # Nothing before the note editor opens, the quota block after.
                side_effect=[None, PREMIUM_MESSAGE],
            ),
            patch.object(
                actions, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            message = await actions._probe_invite_note_limit()

        assert message == PREMIUM_MESSAGE
        assert dialog.clicks == [1]
        mock_dismiss.assert_awaited_once()

    async def test_handles_two_button_gating_dialog(self, mock_page):
        """Two-button "Add a note to your invitation?" gating dialog (issue
        #455): nth(0) is "Add a note", nth(1) is "Send without a note".

        Asserts the secondary-button click that reveals the textarea fires
        even with btn_count == 2 (legacy guard required >= 3 and skipped
        the click, leaving the textarea unmounted)."""
        actions = _actions(mock_page)

        def mount_textarea(dialog: _FakeInviteDialog) -> None:
            dialog.note_field = "visible"

        # Two buttons inside the gating dialog: index 0 "Add a note"
        # reveals the textarea, index 1 "Send without a note". Clicks are
        # recorded by index so the "Add a note" path is visible.
        dialog = _FakeInviteDialog(
            buttons=2, note_field="none", on_click={0: mount_textarea}
        )
        mock_page.wait_for_selector = AsyncMock()
        mock_page.keyboard = MagicMock()
        mock_page.keyboard.press = AsyncMock()

        with (
            dialog.installed(actions),
            patch("linkedin_mcp_server.scraping.connection_actions.asyncio.sleep"),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            (
                submitted,
                note_sent,
                note_limit_message,
            ) = await actions._submit_invite_dialog("Hi from a test")

        assert submitted is True
        assert note_sent is True
        assert note_limit_message is None
        # Clicked "Add a note" (index 0) to reveal the textarea, then the
        # primary button (index 1) to send.
        assert dialog.clicks == [0, 1]
        assert dialog.fills == ["Hi from a test"]


_SLEEP = "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep"


@contextmanager
def _submitting(actions: ConnectionActions, dialog: _FakeInviteDialog):
    """Drive ``_submit_invite_dialog`` over the fake: open at once, the close
    after Send on schedule, and the dismissal recorded."""
    with ExitStack() as stack:
        stack.enter_context(dialog.installed(actions))
        stack.enter_context(patch(_SLEEP))
        stack.enter_context(
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            )
        )
        stack.enter_context(
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=None,
            )
        )
        dismiss = stack.enter_context(
            patch.object(actions, "_dismiss_dialog", new_callable=AsyncMock)
        )
        yield dismiss


class TestInviteRefusals:
    """What the one invite dialog can refuse before anything is submitted."""

    async def test_send_disabled_over_an_empty_note_field_is_note_required(
        self, mock_page
    ):
        """Upstream #407: the profile only takes an invitation with a note."""
        actions = _actions(mock_page)
        dialog = _FakeInviteDialog(
            buttons=2, note_field="visible", primary_disabled=True
        )

        with _submitting(actions, dialog) as dismiss:
            with pytest.raises(InviteNotSent) as refused:
                await actions._submit_invite_dialog(None)

        assert refused.value.status == "note_required"
        assert dialog.clicks == []
        dismiss.assert_awaited_once()
        # The whole settle window was read, not a first glance.
        assert dialog.reads >= 7

    async def test_a_send_that_enables_inside_the_window_is_clicked(self, mock_page):
        """Disabled only while the dialog hydrates is not a request for a note."""
        actions = _actions(mock_page)
        mock_page.wait_for_selector = AsyncMock()
        dialog = _FakeInviteDialog(
            buttons=2,
            note_field="visible",
            primary_disabled=True,
            enabled_after_reads=4,
        )

        with _submitting(actions, dialog):
            result = await actions._submit_invite_dialog(None)

        assert result == (True, False, None)
        assert dialog.clicks == [1]

    @pytest.mark.parametrize("note_field", ["none", "hidden"])
    async def test_a_disabled_send_without_a_shown_note_field_is_not_note_required(
        self, mock_page, note_field
    ):
        """Some other gate; the click path (which waits for Send) is kept."""
        actions = _actions(mock_page)
        mock_page.wait_for_selector = AsyncMock()
        dialog = _FakeInviteDialog(
            buttons=3, note_field=note_field, primary_disabled=True
        )

        with _submitting(actions, dialog):
            await actions._submit_invite_dialog(None)

        assert ("button", 0) in dialog.lookups
        assert dialog.clicks == [2]

    async def test_a_note_the_field_cut_is_not_sent(self, mock_page):
        """A 250-character note in a 200-character field, measured live: the
        field keeps 200 without a sound, and Send would deliver them."""
        actions = _actions(mock_page)
        dialog = _FakeInviteDialog(buttons=2, note_field="visible", max_length=200)

        with _submitting(actions, dialog) as dismiss:
            with pytest.raises(InviteNotSent) as refused:
                await actions._submit_invite_dialog("x" * 250)

        assert refused.value.status == "note_too_long"
        assert refused.value.note_limit == 200
        assert "250" in refused.value.message
        assert dialog.fills == ["x" * 250]
        assert dialog.clicks == []
        dismiss.assert_awaited_once()

    async def test_a_cut_without_maxlength_reports_what_was_kept(self, mock_page):
        """A field that trims in script says its limit only by what it kept."""
        actions = _actions(mock_page)
        dialog = _FakeInviteDialog(buttons=2, note_field="visible", keeps=200)

        with _submitting(actions, dialog):
            with pytest.raises(InviteNotSent) as refused:
                await actions._submit_invite_dialog("y" * 230)

        assert refused.value.status == "note_too_long"
        assert refused.value.note_limit == 200
        assert dialog.clicks == []

    async def test_a_note_at_the_limit_is_sent_whole(self, mock_page):
        actions = _actions(mock_page)
        mock_page.wait_for_selector = AsyncMock()
        dialog = _FakeInviteDialog(buttons=2, note_field="visible", max_length=200)

        with _submitting(actions, dialog):
            result = await actions._submit_invite_dialog("z" * 200)

        assert result == (True, True, None)
        assert dialog.clicks == [1]

    async def test_line_endings_the_field_stores_as_lf_are_the_same_note(
        self, mock_page
    ):
        actions = _actions(mock_page)
        mock_page.wait_for_selector = AsyncMock()
        dialog = _FakeInviteDialog(buttons=2, note_field="visible")

        with _submitting(actions, dialog):
            result = await actions._submit_invite_dialog("Hi Ada,\r\nlet's talk.")

        assert result == (True, True, None)
        assert dialog.clicks == [1]

    @pytest.mark.parametrize(
        "kept",
        [_state("visible", value="something else"), None, _state(dialogs=0)],
        ids=["different-text", "unreadable", "dialog-gone"],
    )
    async def test_a_note_that_cannot_be_confirmed_is_not_sent(self, mock_page, kept):
        """The field holds text that is not the note, or cannot be read back."""
        actions = _actions(mock_page)

        with (
            _submitting(actions, _FakeInviteDialog()) as dismiss,
            patch.object(
                actions,
                "_invite_dialog_state",
                new_callable=AsyncMock,
                side_effect=[_state("visible"), kept],
            ),
            patch.object(
                actions,
                "_fill_dialog_textarea",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions, "_click_dialog_primary_button", new_callable=AsyncMock
            ) as mock_send,
        ):
            with pytest.raises(InviteNotSent) as refused:
                await actions._submit_invite_dialog("Hello there")

        assert refused.value.status == "connect_unavailable"
        assert refused.value.note_limit is None
        mock_send.assert_not_awaited()
        dismiss.assert_awaited_once()

    @pytest.mark.parametrize("note", [None, "Hello"], ids=["no-note", "note"])
    async def test_two_open_dialogs_are_not_guessed_between(self, mock_page, note):
        actions = _actions(mock_page)
        dialog = _FakeInviteDialog(dialogs=2, note_field="visible")

        with _submitting(actions, dialog) as dismiss:
            with pytest.raises(InviteNotSent) as refused:
                await actions._submit_invite_dialog(note)

        assert refused.value.status == "connect_unavailable"
        assert "More than one dialog" in refused.value.message
        assert dialog.lookups == []
        assert dialog.fills == []
        dismiss.assert_awaited_once()

    async def test_a_second_dialog_that_closes_in_the_settle_is_waited_out(
        self, mock_page
    ):
        actions = _actions(mock_page)
        mock_page.wait_for_selector = AsyncMock()
        dialog = _FakeInviteDialog(buttons=2)
        reads = [_state(dialogs=2), _state("none", buttons=2)]

        async def settling() -> _InviteDialogState:
            return reads.pop(0) if reads else await dialog.state()

        with (
            _submitting(actions, dialog),
            patch.object(actions, "_invite_dialog_state", new=settling),
        ):
            result = await actions._submit_invite_dialog(None)

        assert result == (True, False, None)
        assert dialog.clicks == [1]

    async def test_the_quota_probe_touches_nothing_beside_a_second_dialog(
        self, mock_page
    ):
        actions = _actions(mock_page)
        dialog = _FakeInviteDialog(dialogs=2)

        with _submitting(actions, dialog) as dismiss:
            message = await actions._probe_invite_note_limit()

        assert message is None
        assert dialog.clicks == []
        dismiss.assert_awaited_once()


class TestInviteRefusalStatuses:
    """How ``connect_with_person`` reports what the dialog refused."""

    _TEXT = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"

    @pytest.mark.parametrize(
        ("refusal", "note"),
        [
            (InviteNotSent("note_required", "needs a note"), None),
            (InviteNotSent("note_too_long", "too long", note_limit=200), "x" * 250),
        ],
        ids=["note-required", "note-too-long"],
    )
    async def test_a_refusal_is_the_status_and_nothing_is_verified(
        self, mock_page, refusal, note
    ):
        # One read: a refusal is final, so no verification re-read follows.
        read = _reads(self._TEXT)
        actions = _actions(mock_page, read)

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(invite=True),
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
                side_effect=refusal,
            ),
        ):
            result = await actions.connect_with_person("testuser", note=note)

        assert result["status"] == refusal.status
        assert result["message"] == refusal.message
        assert result["note_sent"] is False
        assert result.get("note_limit") == refusal.note_limit
        assert result["profile"] == self._TEXT
        read.assert_awaited_once()

    @pytest.mark.parametrize(
        "note",
        [
            "x" * (MAX_INVITE_NOTE_LENGTH + 1),
            # An emoji is two UTF-16 code units, which is what maxlength counts.
            "\U0001f600" * (MAX_INVITE_NOTE_LENGTH // 2 + 1),
        ],
        ids=["ascii", "emoji"],
    )
    async def test_a_note_no_account_could_send_opens_no_page(self, mock_page, note):
        # The default read refuses to be called.
        actions = _actions(mock_page)

        with patch.object(
            PageNavigator, "_navigate_to_page", new_callable=AsyncMock
        ) as mock_nav:
            result = await actions.connect_with_person("testuser", note=note)

        assert result["status"] == "note_too_long"
        assert result["note_sent"] is False
        assert str(MAX_INVITE_NOTE_LENGTH) in result["message"]
        mock_nav.assert_not_awaited()

    @pytest.mark.parametrize(
        "note",
        [
            "x" * MAX_INVITE_NOTE_LENGTH,
            "\U0001f600" * (MAX_INVITE_NOTE_LENGTH // 2),
            # 450 Python characters, 300 once the field stores CR LF as LF.
            "a\r\n" * (MAX_INVITE_NOTE_LENGTH // 2),
        ],
        ids=["ascii", "emoji", "crlf"],
    )
    async def test_a_note_at_the_ceiling_reaches_the_profile(self, mock_page, note):
        read = _reads("")
        actions = _actions(mock_page, read)

        result = await actions.connect_with_person("testuser", note=note)

        # The empty read is what stopped it, so the length check let it by.
        assert result["status"] == "unavailable"
        read.assert_awaited_once()


_PENDING_TEXT = "Frank\n\n· 3rd\n\nFounder\n\nMessage\nPending\nMore\n"
_CONNECT_TEXT = "Frank\n\n· 3rd\n\nFounder\n\nConnect\nMore\n"
_PENDING = _signals(compose=True, labeled_anchor=True)


class TestWithdrawInvitation:
    """The write gate: nothing is clicked unless a fresh read says pending."""

    @pytest.mark.parametrize(
        ("signals", "status"),
        [
            (_signals(compose=True), "not_pending"),
            (_signals(invite=True), "not_pending"),
            (_signals(edit=True), "self_profile"),
        ],
        ids=["connected", "connectable", "own profile"],
    )
    async def test_a_profile_that_is_not_pending_is_never_clicked(
        self, mock_page, signals, status
    ):
        actions = _actions(mock_page, _reads(_PENDING_TEXT))
        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=signals,
            ),
            patch.object(
                actions, "_click_withdraw_anchor", new_callable=AsyncMock
            ) as click,
        ):
            result = await actions.withdraw_invitation("testuser")

        assert result["status"] == status
        click.assert_not_awaited()

    async def test_an_unreadable_profile_is_never_clicked(self, mock_page):
        actions = _actions(mock_page, _reads(""))
        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_PENDING,
            ),
            patch.object(
                actions, "_click_withdraw_anchor", new_callable=AsyncMock
            ) as click,
        ):
            result = await actions.withdraw_invitation("testuser")

        assert result["status"] == "unavailable"
        click.assert_not_awaited()

    async def test_withdraws_and_verifies(self, mock_page):
        actions = _actions(mock_page, _reads(_PENDING_TEXT, _CONNECT_TEXT))
        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[_PENDING, _signals(invite=True)],
            ),
            patch.object(
                actions,
                "_click_withdraw_anchor",
                new_callable=AsyncMock,
                return_value=True,
            ) as click,
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_wait_for_confirm_buttons",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_confirm_dialog_primary",
                new_callable=AsyncMock,
                return_value=True,
            ) as confirm,
        ):
            result = await actions.withdraw_invitation(
                "https://www.linkedin.com/in/testuser/"
            )

        assert result["status"] == "withdrawn"
        assert result["url"] == "https://www.linkedin.com/in/testuser/"
        assert "connectable" in result["message"]
        assert result["profile"] == _CONNECT_TEXT
        click.assert_awaited_once()
        confirm.assert_awaited_once()

    async def test_no_single_pending_control_reports_unavailable(self, mock_page):
        actions = _actions(mock_page, _reads(_PENDING_TEXT))
        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_PENDING,
            ),
            patch.object(
                actions,
                "_click_withdraw_anchor",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(actions, "_dialog_is_open", new_callable=AsyncMock) as dialog,
        ):
            result = await actions.withdraw_invitation("testuser")

        assert result["status"] == "withdraw_unavailable"
        dialog.assert_not_awaited()

    async def test_no_dialog_and_still_pending_is_unavailable(self, mock_page):
        actions = _actions(mock_page, _reads(_PENDING_TEXT, _PENDING_TEXT))
        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_PENDING,
            ),
            patch.object(
                actions,
                "_click_withdraw_anchor",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=False
            ),
            patch.object(
                actions, "_click_confirm_dialog_primary", new_callable=AsyncMock
            ) as confirm,
        ):
            result = await actions.withdraw_invitation("testuser")

        assert result["status"] == "withdraw_unavailable"
        confirm.assert_not_awaited()

    async def test_no_dialog_but_no_longer_pending_is_withdrawn(self, mock_page):
        actions = _actions(mock_page, _reads(_PENDING_TEXT, _CONNECT_TEXT))
        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[_PENDING, _signals(invite=True)],
            ),
            patch.object(
                actions,
                "_click_withdraw_anchor",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=False
            ),
        ):
            result = await actions.withdraw_invitation("testuser")

        assert result["status"] == "withdrawn"

    async def test_an_unsettled_dialog_is_dismissed_unclicked(self, mock_page):
        actions = _actions(mock_page, _reads(_PENDING_TEXT))
        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_PENDING,
            ),
            patch.object(
                actions,
                "_click_withdraw_anchor",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_wait_for_confirm_buttons",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                actions, "_click_confirm_dialog_primary", new_callable=AsyncMock
            ) as confirm,
            patch.object(actions, "_dismiss_dialog", new_callable=AsyncMock) as dismiss,
        ):
            result = await actions.withdraw_invitation("testuser")

        assert result["status"] == "withdraw_failed"
        confirm.assert_not_awaited()
        dismiss.assert_awaited_once()

    async def test_a_failed_confirm_click_dismisses_the_dialog(self, mock_page):
        actions = _actions(mock_page, _reads(_PENDING_TEXT))
        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_PENDING,
            ),
            patch.object(
                actions,
                "_click_withdraw_anchor",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_wait_for_confirm_buttons",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_confirm_dialog_primary",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(actions, "_dismiss_dialog", new_callable=AsyncMock) as dismiss,
        ):
            result = await actions.withdraw_invitation("testuser")

        assert result["status"] == "withdraw_failed"
        dismiss.assert_awaited_once()

    async def test_still_pending_after_the_settle_retry_is_a_failure(self, mock_page):
        actions = _actions(
            mock_page, _reads(_PENDING_TEXT, _PENDING_TEXT, _PENDING_TEXT)
        )
        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_PENDING,
            ),
            patch.object(
                actions,
                "_click_withdraw_anchor",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_wait_for_confirm_buttons",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_confirm_dialog_primary",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
        ):
            result = await actions.withdraw_invitation("testuser")

        assert result["status"] == "withdraw_failed"
        sleep.assert_awaited_once()

    async def test_a_slow_propagation_is_caught_by_the_settle_retry(self, mock_page):
        actions = _actions(
            mock_page, _reads(_PENDING_TEXT, _PENDING_TEXT, _CONNECT_TEXT)
        )
        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[_PENDING, _PENDING, _signals(invite=True)],
            ),
            patch.object(
                actions,
                "_click_withdraw_anchor",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_wait_for_confirm_buttons",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_confirm_dialog_primary",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await actions.withdraw_invitation("testuser")

        assert result["status"] == "withdrawn"


@contextmanager
def _confirmed_withdrawal(actions: ConnectionActions, signals: list[ActionSignals]):
    """Drive a withdrawal through a confirmed dialog; the reads are the test's."""
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=signals,
            )
        )
        for name in (
            "_click_withdraw_anchor",
            "_dialog_is_open",
            "_wait_for_confirm_buttons",
            "_click_confirm_dialog_primary",
        ):
            stack.enter_context(
                patch.object(actions, name, new_callable=AsyncMock, return_value=True)
            )
        stack.enter_context(
            patch(
                "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep",
                new_callable=AsyncMock,
            )
        )
        yield


class TestWithdrawVerification:
    """``withdrawn`` needs a re-read that was read and classified.

    An empty read (a navigation that failed, a page that never rendered) and
    an action area the probe could not locate (``unavailable``) both come back
    "not pending", and neither says anything about the invitation.
    """

    @pytest.mark.parametrize(
        ("texts", "after"),
        [
            (("", ""), _signals()),
            ((_CONNECT_TEXT, _CONNECT_TEXT), _signals()),
            # A re-read whose navigation failed leaves the page as the click
            # left it, and an optimistic Connect there is not LinkedIn's word.
            (("", ""), _signals(invite=True)),
        ],
        ids=["empty-read", "no-action-area", "empty-read-over-a-stale-page"],
    )
    async def test_a_re_read_that_shows_nothing_is_not_a_withdrawal(
        self, mock_page, texts, after
    ):
        actions = _actions(mock_page, _reads(_PENDING_TEXT, *texts))
        with _confirmed_withdrawal(actions, [_PENDING, after, after]):
            result = await actions.withdraw_invitation("testuser")

        assert result["status"] == "withdraw_failed"
        assert "unverified" in result["message"]

    async def test_an_unreadable_first_re_read_still_gets_the_settle_retry(
        self, mock_page
    ):
        actions = _actions(mock_page, _reads(_PENDING_TEXT, "", _CONNECT_TEXT))
        signals = [_PENDING, _signals(), _signals(invite=True)]
        with _confirmed_withdrawal(actions, signals):
            result = await actions.withdraw_invitation("testuser")

        assert result["status"] == "withdrawn"
        assert "connectable" in result["message"]

    async def test_no_dialog_and_an_unreadable_re_read_is_not_a_withdrawal(
        self, mock_page
    ):
        actions = _actions(mock_page, _reads(_PENDING_TEXT, ""))
        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[_PENDING, _signals()],
            ),
            patch.object(
                actions,
                "_click_withdraw_anchor",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=False
            ),
        ):
            result = await actions.withdraw_invitation("testuser")

        assert result["status"] == "withdraw_unavailable"
