"""Tests for the invitation-manager reader.

``page.evaluate`` is a mock here, so the note-expansion and empty-count
programs never execute; ``tests/test_invitations_dom.py`` runs them against a
real DOM.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import rate_limited_section_error
from linkedin_mcp_server.scraping.invitations import (
    EXPAND_NOTES_JS,
    RECEIVED_COUNT_IS_ZERO_JS,
    SCROLL_LIST_JS,
    InvitationReader,
    invitations_url,
    trim_to_limit,
)
from linkedin_mcp_server.scraping.link_metadata import Reference
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession


def _reader(page: Any) -> InvitationReader:
    session = ScrapingSession(page)
    return InvitationReader(session, PageNavigator(session), PageContentReader(session))


def _person(slug: str, text: str) -> Reference:
    return {"kind": "person", "url": f"/in/{slug}/", "text": text}


@pytest.fixture(autouse=True)
def session_boundaries():
    with (
        patch.object(ScrapingSession, "check_rate_limit", new_callable=AsyncMock),
        patch.object(ScrapingSession, "dismiss_modal", new_callable=AsyncMock),
        patch.object(ScrapingSession, "delay", new_callable=AsyncMock),
        patch.object(ScrapingSession, "scroll_body", new_callable=AsyncMock) as scroll,
        patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock) as nav,
    ):
        yield SimpleNamespace(scroll=scroll, navigate=nav)


def _evaluate(*, expanded: int = 0, zero: bool = False) -> AsyncMock:
    """Answer the page programs the reader runs, by identity."""
    passes = iter([expanded, 0])

    async def evaluate(script: str, *args: Any) -> Any:
        if script == SCROLL_LIST_JS:
            return True
        if script == EXPAND_NOTES_JS:
            return next(passes, 0)
        if script == RECEIVED_COUNT_IS_ZERO_JS:
            return zero
        raise AssertionError(f"unexpected script: {script[:60]}")

    return AsyncMock(side_effect=evaluate)


def _root(text: str, references: list[dict[str, Any]] | None = None) -> AsyncMock:
    return AsyncMock(
        return_value={"source": "root", "text": text, "references": references or []}
    )


class TestInvitationsUrl:
    def test_each_kind_has_its_own_page(self):
        assert invitations_url("received").endswith(
            "/mynetwork/invitation-manager/received/"
        )
        assert invitations_url("sent").endswith("/mynetwork/invitation-manager/sent/")

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="kind must be one of"):
            invitations_url("accepted")  # type: ignore[arg-type]


class TestTrimToLimit:
    def test_cuts_where_the_first_omitted_invitation_starts(self):
        text = "Header\n\nAda\nnote a\n\nBob\nnote b\n\nCyd\nnote c"
        refs = [_person("ada", "Ada"), _person("bob", "Bob"), _person("cyd", "Cyd")]

        assert trim_to_limit(text, refs, 2) == "Header\n\nAda\nnote a\n\nBob\nnote b"

    def test_falls_back_to_card_blocks_when_references_run_out(self):
        text = "Header\n\nAda\nnote a\n\nUnlinked\nnote b\n\nOther\nnote c"
        refs = [_person("ada", "Ada")]

        assert (
            trim_to_limit(text, refs, 2) == "Header\n\nAda\nnote a\n\nUnlinked\nnote b"
        )

    def test_leaves_text_within_the_limit_alone(self):
        text = "Ada\nnote a"
        assert trim_to_limit(text, [_person("ada", "Ada")], 5) == text


class TestGetPendingInvitations:
    async def test_references_reach_the_limit_ceiling(self, mock_page):
        mock_page.evaluate = _evaluate()
        reader = _reader(mock_page)
        refs = [
            {"href": f"https://www.linkedin.com/in/p{i}/", "text": f"Person {i}"}
            for i in range(120)
        ]
        text = "\n\n".join(f"Person {i}\nnote" for i in range(120))
        with patch.object(reader._content, "_extract_root_content", _root(text, refs)):
            result = await reader.get_pending_invitations(limit=100, kind="sent")

        assert len(result["references"]["invitations"]) == 100

    async def test_reads_sent_invitations_with_capped_references(
        self, mock_page, session_boundaries
    ):
        mock_page.evaluate = _evaluate(expanded=2)
        reader = _reader(mock_page)
        refs = [
            {"href": "https://www.linkedin.com/in/ada/", "text": "Ada"},
            {"href": "https://www.linkedin.com/in/bob/", "text": "Bob"},
        ]
        with patch.object(
            reader._content, "_extract_root_content", _root("Ada\nhi\n\nBob\nhey", refs)
        ):
            result = await reader.get_pending_invitations(limit=1, kind="sent")

        session_boundaries.navigate.assert_awaited_once_with(invitations_url("sent"))
        assert result["url"] == invitations_url("sent")
        assert result["sections"]["invitations"] == "Ada\nhi"
        assert [r["url"] for r in result["references"]["invitations"]] == ["/in/ada/"]
        # Both expansion passes ran: the second catches cards the first loaded.
        scripts = [call.args[0] for call in mock_page.evaluate.await_args_list]
        assert scripts.count(EXPAND_NOTES_JS) == 2
        # The sent manager has no received counter to consult.
        assert RECEIVED_COUNT_IS_ZERO_JS not in scripts

    async def test_the_limit_sets_the_scroll_budget(self, mock_page):
        mock_page.evaluate = _evaluate()
        reader = _reader(mock_page)
        with patch.object(reader._content, "_extract_root_content", _root("")):
            await reader.get_pending_invitations(limit=45)

        scripts = [call.args[0] for call in mock_page.evaluate.await_args_list]
        assert scripts.count(SCROLL_LIST_JS) == 4

    async def test_a_zero_received_count_skips_the_recommendations_below_it(
        self, mock_page
    ):
        mock_page.evaluate = _evaluate(zero=True)
        reader = _reader(mock_page)
        extract = _root(
            "People you may know\nZed",
            [{"href": "https://www.linkedin.com/in/zed/", "text": "Zed"}],
        )
        with patch.object(reader._content, "_extract_root_content", extract):
            result = await reader.get_pending_invitations(kind="received")

        assert result == {"url": invitations_url("received"), "sections": {}}
        extract.assert_not_awaited()

    async def test_chrome_only_text_is_reported_as_a_rate_limit(self, mock_page):
        mock_page.evaluate = _evaluate()
        reader = _reader(mock_page)
        with patch.object(
            reader._content,
            "_extract_root_content",
            _root("About\nAccessibility\nTalent Solutions"),
        ):
            result = await reader.get_pending_invitations(kind="sent")

        assert result["sections"] == {}
        assert result["section_errors"]["invitations"] == rate_limited_section_error()
