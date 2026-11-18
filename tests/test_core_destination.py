"""Tests for off-LinkedIn / captive-portal landing refusal (#1235)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.exceptions import ToolError

from linkedin_mcp_server.core.destination import (
    is_another_site,
    is_linkedin_landing,
    raise_if_off_linkedin,
    refuse_landing,
)
from linkedin_mcp_server.core.exceptions import (
    AuthenticationError,
    OffLinkedInLandingError,
)
from linkedin_mcp_server.drivers.browser import _feed_auth_succeeds
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession


class TestIsLinkedInLanding:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.linkedin.com/feed/",
            "https://linkedin.com/in/alice/",
            "https://de.linkedin.com/jobs/search/",
            "https://www.linkedin.com:443/messaging/",
        ],
    )
    def test_accepts_https_linkedin_hosts(self, url: str):
        assert is_linkedin_landing(url) is True
        assert is_another_site(url) is False
        raise_if_off_linkedin(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://www.linkedin.com/feed/",
            "http://captive.apple.com/hotspot-detect.html",
            "https://notlinkedin.com/feed/",
            "https://linkedin.com.evil.example/login",
            "https://user@www.linkedin.com/feed/",
            "about:blank",
            "",
        ],
    )
    def test_rejects_non_linkedin_or_non_https_landings(self, url: str):
        assert is_linkedin_landing(url) is False
        with pytest.raises(OffLinkedInLandingError, match="instead of LinkedIn"):
            raise_if_off_linkedin(url)

    def test_refuse_landing_names_host_without_query_or_credentials(self):
        with pytest.raises(OffLinkedInLandingError) as exc_info:
            refuse_landing(
                "https://user:secret@portal.hotel.example:8443/login?token=abc"
            )
        text = str(exc_info.value)
        assert "https://portal.hotel.example:8443" in text
        assert "secret" not in text
        assert "token=abc" not in text


class TestOffLinkedInRefusalInAuthAndNavigation:
    async def test_feed_auth_check_raises_instead_of_returning_false_on_captive_portal(
        self,
    ):
        """A captive portal titled 'LinkedIn Login' must raise OffLinkedInLandingError,
        not return False (which would rotate/retire a valid session)."""
        browser = MagicMock()
        browser.page = MagicMock()
        browser.page.url = "https://portal.hotel.example/login"
        browser.page.goto = AsyncMock()
        browser.page.title = AsyncMock(return_value="LinkedIn Login")

        with pytest.raises(
            OffLinkedInLandingError, match=r"https://portal\.hotel\.example"
        ):
            await _feed_auth_succeeds(browser)

    async def test_navigator_refuses_off_linkedin_landing_before_auth_barrier(self):
        page = MagicMock()
        page.url = "https://portal.hotel.example/login"
        page.goto = AsyncMock()
        navigator = PageNavigator(ScrapingSession(page))

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                side_effect=AssertionError("checked auth barrier on foreign site"),
            ) as mock_barrier,
            pytest.raises(
                OffLinkedInLandingError, match=r"https://portal\.hotel\.example"
            ),
        ):
            await navigator._goto_with_auth_checks("https://www.linkedin.com/feed/")

        mock_barrier.assert_not_awaited()

    def test_error_handler_surfaces_off_linkedin_message_unmasked(self):
        with pytest.raises(OffLinkedInLandingError) as exc_info:
            refuse_landing("https://portal.hotel.example/splash")
        err = exc_info.value
        assert not isinstance(err, AuthenticationError)

        with pytest.raises(ToolError, match=r"https://portal\.hotel\.example"):
            raise_tool_error(err, "get_person_profile")
