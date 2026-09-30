"""The signed-in member's own analytics dashboards."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import logging

from linkedin_mcp_server.core.exceptions import LinkedInScraperException
from linkedin_mcp_server.error_diagnostics import build_issue_diagnostics
from linkedin_mcp_server.scraping.capture import SectionCapture
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    rate_limited_section_error,
)
from linkedin_mcp_server.scraping.fields import (
    ANALYTICS_TIME_RANGE_SECTIONS,
    _analytics_section_specs,
    normalize_analytics_time_range,
)
from linkedin_mcp_server.scraping.link_metadata import Reference
from linkedin_mcp_server.scraping.session import NAV_DELAY, ScrapingSession

if TYPE_CHECKING:
    from linkedin_mcp_server.callbacks import ProgressCallback

logger = logging.getLogger(__name__)

ANALYTICS_BASE_URL = "https://www.linkedin.com/analytics"


class AnalyticsScraper:
    """Own the "Private to you" dashboards under /analytics/."""

    def __init__(self, session: ScrapingSession, capture: SectionCapture):
        self._session = session
        self._capture = capture

    async def get_my_analytics(
        self,
        requested: set[str],
        time_range: str | None = None,
        callbacks: ProgressCallback | None = None,
        max_scrolls: int | None = None,
    ) -> dict[str, Any]:
        """Read one dashboard per requested section, one navigation each.

        ``time_range`` is validated before anything is navigated and is added
        only to the sections that honour it (content, audience); the other
        dashboards use LinkedIn's fixed windows.

        Returns:
            {url, sections: {name: text}, references?, section_errors?}

        Raises:
            FilterValidationError: for an unrecognised ``time_range``.
        """
        normalized_range = normalize_analytics_time_range(time_range)

        sections: dict[str, str] = {}
        references: dict[str, list[Reference]] = {}
        section_errors: dict[str, dict[str, Any]] = {}
        rate_limited = False

        requested_ordered = [
            spec
            for spec in _analytics_section_specs(max_scrolls=max_scrolls)
            if spec.name in requested
        ]
        total = len(requested_ordered)

        if callbacks:
            await callbacks.on_start("analytics", ANALYTICS_BASE_URL)

        try:
            for i, spec in enumerate(requested_ordered):
                if i > 0:
                    await self._session.delay(NAV_DELAY)

                section_name = spec.name
                url = ANALYTICS_BASE_URL + spec.suffix
                if (
                    normalized_range is not None
                    and section_name in ANALYTICS_TIME_RANGE_SECTIONS
                ):
                    url += f"?timeRange={normalized_range}"

                try:
                    extracted = await self._capture.capture(
                        url, section_name, spec.plan
                    )
                    if extracted.text and extracted.text != RATE_LIMITED_SECTION_TEXT:
                        sections[section_name] = extracted.text
                        if extracted.references:
                            references[section_name] = extracted.references
                    elif extracted.text == RATE_LIMITED_SECTION_TEXT:
                        section_errors[section_name] = rate_limited_section_error()
                        # Each remaining dashboard is another navigation and
                        # LinkedIn has just asked for fewer; keep what was read.
                        rate_limited = True
                    elif extracted.error:
                        section_errors[section_name] = extracted.error
                except LinkedInScraperException:
                    raise
                except Exception as e:
                    logger.warning(
                        "Error scraping analytics section %s: %s", section_name, e
                    )
                    section_errors[section_name] = build_issue_diagnostics(
                        e,
                        context="get_my_analytics",
                        target_url=url,
                        section_name=section_name,
                    )

                if callbacks:
                    percent = round((i + 1) / total * 95)
                    await callbacks.on_progress(
                        f"Scraped {section_name} ({i + 1}/{total})", percent
                    )

                if rate_limited:
                    break
        except LinkedInScraperException as e:
            if callbacks:
                await callbacks.on_error(e)
            raise

        result: dict[str, Any] = {"url": f"{ANALYTICS_BASE_URL}/", "sections": sections}
        if references:
            result["references"] = references
        if section_errors:
            result["section_errors"] = section_errors

        if callbacks:
            await callbacks.on_complete("analytics", result)

        return result
