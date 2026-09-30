"""Tests for the post comment-thread owner."""

from __future__ import annotations

from unittest.mock import AsyncMock, call, patch

import pytest

from linkedin_mcp_server.core.exceptions import InvalidReferenceError
from linkedin_mcp_server.scraping.capture import (
    CaptureMode,
    CapturePlan,
    SectionCapture,
)
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    ExtractedSection,
    rate_limited_section_error,
)
from linkedin_mcp_server.scraping.link_metadata import Reference
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.post_comments import PostComments
from linkedin_mcp_server.scraping.session import ScrapingSession

URN = "urn:li:activity:7203847123456789012"
PERMALINK = f"https://www.linkedin.com/feed/update/{URN}/"


def _owner(page) -> PostComments:
    session = ScrapingSession(page)
    return PostComments(
        SectionCapture(session, PageNavigator(session), PageContentReader(session))
    )


def extracted(
    text: str,
    references: list[Reference] | None = None,
    error: dict | None = None,
) -> ExtractedSection:
    return ExtractedSection(text=text, references=references or [], error=error)


def _capturing(owner: PostComments, result: ExtractedSection):
    return patch.object(
        owner._capture, "capture", new_callable=AsyncMock, return_value=result
    )


class TestGetPostComments:
    async def test_the_thread_is_captured_from_the_canonical_permalink(self, mock_page):
        owner = _owner(mock_page)
        with _capturing(owner, extracted("Post body\nComment one")) as capture:
            result = await owner.get_post_comments(URN)

        assert result == {
            "url": PERMALINK,
            "sections": {"post": "Post body\nComment one"},
        }
        assert capture.await_args_list == [
            call(
                PERMALINK,
                section_name="post",
                plan=CapturePlan(CaptureMode.COMMENT_THREAD, None),
            )
        ]

    async def test_max_scrolls_reaches_the_plan(self, mock_page):
        owner = _owner(mock_page)
        with _capturing(owner, extracted("x")) as capture:
            await owner.get_post_comments(URN, max_scrolls=9)

        assert capture.await_args is not None
        assert capture.await_args.kwargs["plan"] == CapturePlan(
            CaptureMode.COMMENT_THREAD, 9
        )

    async def test_references_are_filed_under_post(self, mock_page):
        refs: list[Reference] = [{"kind": "person", "url": "/in/bob/", "text": "Bob"}]
        owner = _owner(mock_page)
        with _capturing(owner, extracted("Body", refs)):
            result = await owner.get_post_comments(URN)

        assert result["references"] == {"post": refs}

    async def test_a_rate_limited_read_is_reported_not_returned_as_text(
        self, mock_page
    ):
        owner = _owner(mock_page)
        with _capturing(owner, extracted(RATE_LIMITED_SECTION_TEXT)):
            result = await owner.get_post_comments(URN)

        assert result["sections"] == {}
        assert result["section_errors"] == {"post": rate_limited_section_error()}

    async def test_a_failed_read_carries_its_diagnostics(self, mock_page):
        error = {"error_type": "TimeoutError", "error_message": "slow"}
        owner = _owner(mock_page)
        with _capturing(owner, extracted("", error=error)):
            result = await owner.get_post_comments(URN)

        assert result["section_errors"] == {"post": error}
        assert "references" not in result

    @pytest.mark.parametrize(
        "value",
        [
            "https://www.linkedin.com/in/alice/",
            "https://evil.example/feed/update/urn:li:activity:1/",
            "",
        ],
    )
    async def test_a_value_that_names_no_post_never_reaches_the_browser(
        self, mock_page, value
    ):
        owner = _owner(mock_page)
        with _capturing(owner, extracted("x")) as capture:
            with pytest.raises(InvalidReferenceError):
                await owner.get_post_comments(value)

        capture.assert_not_awaited()
        mock_page.goto.assert_not_awaited()
