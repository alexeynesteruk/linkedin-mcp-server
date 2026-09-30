"""A dead browser leaves every capture path as an error, never as a section.

Each broad ``except Exception`` that turns a failure into a ``section_errors``
entry, or skips past it, would otherwise report a closed page as a section that
came back empty, and the call as a success. One case per handler, driven through
the owner that holds it, beside the ordinary failure that still becomes a
section error there.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from patchright._impl._errors import TargetClosedError

from linkedin_mcp_server.core.exceptions import BrowserLostError
from linkedin_mcp_server.scraping.analytics import AnalyticsScraper
from linkedin_mcp_server.scraping.capture import SectionCapture
from linkedin_mcp_server.scraping.company import CompanyScraper
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import ExtractedSection
from linkedin_mcp_server.scraping.feed import FeedScraper
from linkedin_mcp_server.scraping.job_pages import JobPageReader
from linkedin_mcp_server.scraping.jobs import JobScraper
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.person import PersonScraper
from linkedin_mcp_server.scraping.profile_page import ProfilePageReader
from linkedin_mcp_server.scraping.session import ScrapingSession


def _parts(page) -> tuple[ScrapingSession, PageNavigator, PageContentReader]:
    session = ScrapingSession(page)
    return session, PageNavigator(session), PageContentReader(session)


def _capture(page) -> SectionCapture:
    session, navigator, content = _parts(page)
    return SectionCapture(session, navigator, content)


def _person(page) -> PersonScraper:
    session, navigator, content = _parts(page)

    async def no_target() -> Any:
        return SimpleNamespace(target=None)

    return PersonScraper(
        session,
        navigator,
        SectionCapture(session, navigator, content),
        ProfilePageReader(session, no_target),
    )


def _text(text: str) -> ExtractedSection:
    return ExtractedSection(text=text, references=[])


@pytest.fixture
def quiet_page(mock_page):
    """No rate limit, no modal, no real sleeps."""
    with (
        patch(
            "linkedin_mcp_server.scraping.session.detect_rate_limit",
            new_callable=AsyncMock,
        ),
        patch(
            "linkedin_mcp_server.scraping.session.handle_modal_close",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "linkedin_mcp_server.scraping.session.asyncio.sleep",
            new_callable=AsyncMock,
        ),
        patch(
            "linkedin_mcp_server.scraping.jobs.asyncio.sleep", new_callable=AsyncMock
        ),
    ):
        yield mock_page


class TestCapture:
    async def test_a_closed_page_propagates_instead_of_reading_as_empty(
        self, quiet_page
    ):
        closed = TargetClosedError()
        quiet_page.goto = AsyncMock(side_effect=closed)

        with pytest.raises(BrowserLostError) as raised:
            await _capture(quiet_page).extract_page(
                "https://www.linkedin.com/in/testuser/", section_name="main_profile"
            )

        assert raised.value.__cause__ is closed

    async def test_a_crashed_renderer_during_the_read_propagates(self, quiet_page):
        quiet_page.evaluate = AsyncMock(
            side_effect=Exception("Page.evaluate: Target crashed ")
        )

        with pytest.raises(BrowserLostError, match="the page crashed"):
            await _capture(quiet_page).extract_page(
                "https://www.linkedin.com/in/testuser/", section_name="main_profile"
            )


class TestPerson:
    async def test_a_loss_in_a_later_section_fails_the_whole_scrape(self, quiet_page):
        scraper = _person(quiet_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            side_effect=[_text("profile text"), TargetClosedError()],
        ):
            with pytest.raises(BrowserLostError):
                await scraper.scrape_person("testuser", {"experience"})

    async def test_an_ordinary_section_failure_is_still_a_section_error(
        self, quiet_page
    ):
        scraper = _person(quiet_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            side_effect=[_text("profile text"), RuntimeError("Simulated failure")],
        ):
            result = await scraper.scrape_person("testuser", {"experience"})

        assert "experience" in result["section_errors"]

    async def test_a_loss_on_a_show_all_page_is_not_skipped(self, quiet_page):
        quiet_page.evaluate = AsyncMock(
            return_value={
                "sections": {"people_you_may_know": ["/in/dave/"]},
                "showAllUrls": {
                    "people_you_may_know": "https://www.linkedin.com/search/results/people/"
                },
            }
        )
        scraper = _person(quiet_page)
        navigations = iter([None, TargetClosedError()])

        async def navigate(_url: str) -> None:
            outcome = next(navigations)
            if outcome is not None:
                raise outcome

        with patch.object(
            scraper._navigator, "_navigate_to_page", side_effect=navigate
        ):
            with pytest.raises(BrowserLostError):
                await scraper.get_sidebar_profiles("testuser")


class TestCompany:
    async def test_a_loss_in_a_section_fails_the_whole_scrape(self, quiet_page):
        session, navigator, content = _parts(quiet_page)
        scraper = CompanyScraper(session, SectionCapture(session, navigator, content))
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            side_effect=[_text("about text"), TargetClosedError()],
        ):
            with pytest.raises(BrowserLostError):
                await scraper.scrape_company("testcorp", {"posts"})


class TestAnalytics:
    async def test_a_loss_in_a_dashboard_fails_the_whole_read(self, quiet_page):
        session, navigator, content = _parts(quiet_page)
        scraper = AnalyticsScraper(session, SectionCapture(session, navigator, content))
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            side_effect=TargetClosedError(),
        ):
            with pytest.raises(BrowserLostError):
                await scraper.get_my_analytics({"content"})


class TestFeed:
    async def test_a_loss_is_not_a_feed_section_error(self, quiet_page):
        scraper = FeedScraper(*_parts(quiet_page))

        async def once(num_posts: int) -> ExtractedSection:
            raise TargetClosedError()

        with patch.object(scraper, "_extract_feed_once", once):
            with pytest.raises(BrowserLostError):
                await scraper.extract_feed(num_posts=5)


def _jobs(page) -> JobScraper:
    session, navigator, content = _parts(page)
    return JobScraper(
        navigator,
        SectionCapture(session, navigator, content),
        JobPageReader(session, navigator, content),
    )


class TestJobs:
    async def test_a_loss_on_a_search_page_fails_the_search(self, quiet_page):
        scraper = _jobs(quiet_page)
        with patch.object(
            scraper._pages, "_extract_search_page", side_effect=TargetClosedError()
        ):
            with pytest.raises(BrowserLostError):
                await scraper.search_jobs("python", max_pages=1)

    async def test_a_loss_on_a_saved_jobs_page_fails_the_list(self, quiet_page):
        scraper = _jobs(quiet_page)
        with patch.object(
            scraper._pages, "_extract_saved_jobs_page", side_effect=TargetClosedError()
        ):
            with pytest.raises(BrowserLostError):
                await scraper.get_saved_jobs(max_pages=1)


class TestJobPages:
    @staticmethod
    def _reader(page) -> JobPageReader:
        session, navigator, content = _parts(page)
        return JobPageReader(session, navigator, content)

    async def test_a_loss_while_reading_a_search_page_propagates(self, quiet_page):
        quiet_page.url = "https://www.linkedin.com/jobs/search/?keywords=test"
        reader = self._reader(quiet_page)
        with (
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.job_pages.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.job_pages.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.job_pages.scroll_job_sidebar",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                PageContentReader,
                "_extract_root_content",
                new_callable=AsyncMock,
                side_effect=TargetClosedError(),
            ),
        ):
            with pytest.raises(BrowserLostError):
                await reader._extract_search_page(
                    "https://www.linkedin.com/jobs/search/?keywords=test",
                    section_name="search_results",
                )

    async def test_a_loss_while_reading_saved_jobs_propagates(self, quiet_page):
        quiet_page.url = "https://www.linkedin.com/jobs-tracker/"
        reader = self._reader(quiet_page)
        with (
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.job_pages.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.job_pages.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.job_pages.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch.object(
                PageContentReader,
                "_extract_root_content",
                new_callable=AsyncMock,
                side_effect=TargetClosedError(),
            ),
        ):
            with pytest.raises(BrowserLostError):
                await reader._extract_saved_jobs_page(
                    "https://www.linkedin.com/jobs-tracker/",
                    section_name="saved_jobs",
                )
