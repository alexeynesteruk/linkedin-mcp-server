"""Tests for the own-analytics scraping owner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.scraping.analytics import AnalyticsScraper
from linkedin_mcp_server.scraping.capture import (
    CaptureMode,
    CapturePlan,
    SectionCapture,
)
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    ExtractedSection,
    FilterValidationError,
)
from linkedin_mcp_server.scraping.fields import (
    ANALYTICS_SECTIONS,
    ANALYTICS_TIME_RANGE_SECTIONS,
    normalize_analytics_time_range,
    parse_analytics_sections,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import NAV_DELAY, ScrapingSession

BASE = "https://www.linkedin.com/analytics"


def _scraper(page) -> AnalyticsScraper:
    session = ScrapingSession(page)
    navigator = PageNavigator(session)
    return AnalyticsScraper(
        session, SectionCapture(session, navigator, PageContentReader(session))
    )


def _section(text: str, error: dict | None = None) -> ExtractedSection:
    return ExtractedSection(text=text, references=[], error=error)


class TestAnalyticsSections:
    def test_section_urls_are_the_five_dashboards(self):
        assert ANALYTICS_SECTIONS == {
            "content": "/creator/content/",
            "audience": "/creator/audience/",
            "top_posts": "/creator/top-posts/",
            "profile_views": "/profile-views/",
            "search_appearances": "/search-appearances/",
        }
        assert ANALYTICS_TIME_RANGE_SECTIONS == {"content", "audience"}

    def test_empty_selects_everything(self):
        for value in (None, ""):
            assert parse_analytics_sections(value) == (set(ANALYTICS_SECTIONS), [])

    def test_names_are_trimmed_and_case_folded(self):
        assert parse_analytics_sections(" Content , AUDIENCE ") == (
            {"content", "audience"},
            [],
        )

    def test_unknown_names_are_returned_and_the_rest_kept(self):
        assert parse_analytics_sections("content,bogus") == ({"content"}, ["bogus"])

    def test_only_unknown_names_fall_back_to_everything(self):
        assert parse_analytics_sections("bogus") == (
            set(ANALYTICS_SECTIONS),
            ["bogus"],
        )


class TestTimeRange:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            ("7d", "past_7_days"),
            (" 28D ", "past_28_days"),
            ("90d", "past_90_days"),
            ("365d", "past_365_days"),
            ("Past_90_Days", "past_90_days"),
        ],
    )
    def test_accepted_values(self, value, expected):
        assert normalize_analytics_time_range(value) == expected

    @pytest.mark.parametrize("value", ["", "1y", "30d", "past_30_days", "7"])
    def test_refused_values(self, value):
        with pytest.raises(FilterValidationError, match="Invalid time_range"):
            normalize_analytics_time_range(value)


class TestGetMyAnalytics:
    async def test_one_capture_per_section_in_declared_order(self, mock_page):
        scraper = _scraper(mock_page)
        with (
            patch.object(
                scraper._capture,
                "capture",
                new_callable=AsyncMock,
                side_effect=lambda url, name, plan: _section(f"{name} text"),
            ) as capture,
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
        ):
            result = await scraper.get_my_analytics(
                {"search_appearances", "content", "top_posts"}
            )

        assert [c.args[0] for c in capture.call_args_list] == [
            f"{BASE}/creator/content/",
            f"{BASE}/creator/top-posts/",
            f"{BASE}/search-appearances/",
        ]
        assert all(
            c.args[2].mode is CaptureMode.STANDARD for c in capture.call_args_list
        )
        assert list(result["sections"]) == [
            "content",
            "top_posts",
            "search_appearances",
        ]
        assert result["url"] == f"{BASE}/"
        assert "section_errors" not in result
        assert sleep.await_args_list == [call(NAV_DELAY)] * 2

    async def test_time_range_only_reaches_sections_that_honour_it(self, mock_page):
        scraper = _scraper(mock_page)
        with (
            patch.object(
                scraper._capture,
                "capture",
                new_callable=AsyncMock,
                return_value=_section("text"),
            ) as capture,
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            await scraper.get_my_analytics(set(ANALYTICS_SECTIONS), time_range="28d")

        urls = {c.args[1]: c.args[0] for c in capture.call_args_list}
        assert urls["content"] == f"{BASE}/creator/content/?timeRange=past_28_days"
        assert urls["audience"] == f"{BASE}/creator/audience/?timeRange=past_28_days"
        for name in ("top_posts", "profile_views", "search_appearances"):
            assert "?" not in urls[name]

    async def test_an_invalid_time_range_is_refused_before_navigating(self, mock_page):
        scraper = _scraper(mock_page)
        callbacks = MagicMock()
        with patch.object(
            scraper._capture, "capture", new_callable=AsyncMock
        ) as capture:
            with pytest.raises(FilterValidationError):
                await scraper.get_my_analytics(
                    {"content"}, time_range="1y", callbacks=callbacks
                )

        capture.assert_not_awaited()
        assert callbacks.method_calls == []

    async def test_max_scrolls_rides_on_the_plan(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=_section("text"),
        ) as capture:
            await scraper.get_my_analytics({"content"}, max_scrolls=7)

        assert capture.await_args.args[2] == CapturePlan(CaptureMode.STANDARD, 7)

    async def test_rate_limit_stops_the_walk_and_keeps_earlier_sections(
        self, mock_page
    ):
        scraper = _scraper(mock_page)
        texts = {"content": "ok", "audience": RATE_LIMITED_SECTION_TEXT}
        with (
            patch.object(
                scraper._capture,
                "capture",
                new_callable=AsyncMock,
                side_effect=lambda url, name, plan: _section(texts.get(name, "later")),
            ) as capture,
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await scraper.get_my_analytics(set(ANALYTICS_SECTIONS))

        assert capture.await_count == 2
        assert result["sections"] == {"content": "ok"}
        assert set(result["section_errors"]) == {"audience"}

    async def test_a_section_failure_is_isolated(self, mock_page):
        scraper = _scraper(mock_page)

        async def capture(url, name, plan):
            if name == "content":
                raise RuntimeError("boom")
            return _section("audience text")

        with (
            patch.object(scraper._capture, "capture", side_effect=capture),
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await scraper.get_my_analytics({"content", "audience"})

        assert result["sections"] == {"audience": "audience text"}
        assert "content" in result["section_errors"]

    async def test_an_empty_section_carries_its_error(self, mock_page):
        scraper = _scraper(mock_page)
        error = {"issue": "empty"}
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=_section("", error=error),
        ):
            result = await scraper.get_my_analytics({"content"})

        assert result["sections"] == {}
        assert result["section_errors"] == {"content": error}

    async def test_scraper_exceptions_notify_callbacks_and_propagate(self, mock_page):
        scraper = _scraper(mock_page)
        callbacks = MagicMock()
        callbacks.on_start = AsyncMock()
        callbacks.on_error = AsyncMock()
        error = AuthenticationError("expired")
        with patch.object(
            scraper._capture, "capture", new_callable=AsyncMock, side_effect=error
        ):
            with pytest.raises(AuthenticationError):
                await scraper.get_my_analytics({"content"}, callbacks=callbacks)

        callbacks.on_error.assert_awaited_once_with(error)

    async def test_progress_is_reported_per_section(self, mock_page):
        scraper = _scraper(mock_page)
        callbacks = MagicMock()
        callbacks.on_start = AsyncMock()
        callbacks.on_progress = AsyncMock()
        callbacks.on_complete = AsyncMock()
        with (
            patch.object(
                scraper._capture,
                "capture",
                new_callable=AsyncMock,
                return_value=_section("text"),
            ),
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            await scraper.get_my_analytics({"content", "audience"}, callbacks=callbacks)

        callbacks.on_start.assert_awaited_once_with("analytics", BASE)
        assert callbacks.on_progress.await_args_list == [
            call("Scraped content (1/2)", 48),
            call("Scraped audience (2/2)", 95),
        ]
        callbacks.on_complete.assert_awaited_once()
