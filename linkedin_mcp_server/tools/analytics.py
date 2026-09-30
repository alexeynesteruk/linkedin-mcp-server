"""
LinkedIn analytics tool.

Reads the signed-in member's own "Private to you" dashboards under
/analytics/ with innerText extraction. There is no per-username variant.
"""

import logging
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from linkedin_mcp_server.callbacks import MCPContextProgressCallback
from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.scraping import parse_analytics_sections
from linkedin_mcp_server.scraping.contracts import FilterValidationError
from linkedin_mcp_server.scraping.fields import normalize_analytics_time_range

logger = logging.getLogger(__name__)


def register_analytics_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register the analytics tool with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get My Analytics",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"analytics", "scraping"},
    )
    async def get_my_analytics(
        ctx: Context,
        sections: str | None = None,
        time_range: str | None = None,
        max_scrolls: Annotated[int, Field(ge=1, le=50)] | None = None,
    ) -> dict[str, Any]:
        """
        Get the signed-in user's own LinkedIn analytics dashboards.

        These are the "Private to you" analytics pages, so they exist only for
        the logged-in account; there is no per-username variant.

        Args:
            ctx: FastMCP context for progress reporting
            sections: Comma-separated list of sections to scrape.
                Available sections: content, audience, top_posts, profile_views, search_appearances
                - content: impressions, members reached, engagement breakdown, top performing posts
                - audience: total followers, follower growth, top demographics
                - top_posts: recent posts ranked by impressions, with per-post counts
                - profile_views: profile viewers over the past 90 days and viewer highlights
                - search_appearances: search appearance counts and where the profile appeared
                Default (None) scrapes ALL five sections (one page navigation each).
            time_range: Reporting window for the content and audience sections:
                "7d", "28d", "90d" or "365d" (past_7_days style values are also
                accepted). Other sections use LinkedIn's fixed windows
                (top_posts 14 days, profile_views 90 days) and ignore it.
                An unrecognised value is refused before any page is opened.
                Default (None) uses LinkedIn's default of 7 days.
            max_scrolls: Maximum scroll-to-bottom iterations per section.
                Default (None) uses the standard limit; the dashboards are short pages.

        Returns:
            Dict with url, sections (name -> raw text), and optional references and section_errors.
            Sections may be absent if extraction yielded no content for that page.
            Includes unknown_sections when unrecognised names are passed.
            The LLM should parse the raw text in each section; chart data appears
            as axis-description text with min and max values.
        """
        try:
            normalize_analytics_time_range(time_range)
        except FilterValidationError as e:
            raise ToolError(str(e)) from e

        try:
            extractor = await get_ready_extractor(ctx, tool_name="get_my_analytics")
            requested, unknown = parse_analytics_sections(sections)

            logger.info(
                "Scraping own analytics (sections=%s, time_range=%s)",
                sections,
                time_range,
            )

            cb = MCPContextProgressCallback(ctx)
            result = await extractor.get_my_analytics(
                requested,
                time_range=time_range,
                callbacks=cb,
                max_scrolls=max_scrolls,
            )

            if unknown:
                result["unknown_sections"] = unknown

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_my_analytics")
        except Exception as e:
            raise_tool_error(e, "get_my_analytics")  # NoReturn
