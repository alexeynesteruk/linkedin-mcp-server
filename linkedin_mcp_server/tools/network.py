"""
LinkedIn network tools.

Provides access to pending network invitations (received or sent) from
``/mynetwork/invitation-manager/``, and withdrawing a sent invitation.
Accept and ignore actions remain intentionally not exposed.
"""

import logging
from typing import Annotated, Any, Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import extractor_depends, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.scrape_guards import annotate_empty_scrape_result
from linkedin_mcp_server.scraping.usernames import normalize_linkedin_username

logger = logging.getLogger(__name__)


def _require_username(value: str | None, *, tool_name: str) -> str:
    """Normalize a LinkedIn vanity or raise ToolError for bad agent input."""
    username = normalize_linkedin_username(value)
    if username is None:
        raise ToolError(
            f"Invalid linkedin_username {value!r}. "
            "Pass a bare vanity (e.g. 'williamhgates') or a full "
            "https://www.linkedin.com/in/... profile URL."
        )
    return username


def register_network_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register all network-related tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Pending Invitations",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"network", "scraping"},
    )
    async def get_pending_invitations(
        ctx: Context,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
        kind: Literal["received", "sent"] = "received",
        extractor: Any = extractor_depends("get_pending_invitations"),
    ) -> dict[str, Any]:
        """
        List pending LinkedIn network invitations (received or sent).

        Reads ``/mynetwork/invitation-manager/{received|sent}/`` and returns
        the page's visible text plus references to inviter/invitee profiles.
        Read-only - accepting or ignoring invitations is not exposed. To
        withdraw a sent invitation, use ``withdraw_invitation``.

        Args:
            ctx: FastMCP context for progress reporting
            limit: Maximum number of invitations to return (1-100, default 20).
                References are capped exactly; readable text is trimmed at the
                first omitted invitation when LinkedIn renders extra cards.
            kind: "received" (default) for incoming invitations, "sent" for
                outgoing ones awaiting the recipient's response.

        Returns:
            Dict with url, sections (invitations -> raw text), and optional
            references.
        """
        try:
            logger.info("Fetching pending invitations (kind=%s, limit=%d)", kind, limit)

            await ctx.report_progress(
                progress=0, total=100, message=f"Loading {kind} invitations"
            )

            result = await extractor.get_pending_invitations(limit=limit, kind=kind)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return annotate_empty_scrape_result(
                result,
                tool_name="get_pending_invitations",
                required_sections=("invitations",),
            )

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_pending_invitations")
        except Exception as e:
            raise_tool_error(e, "get_pending_invitations")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Withdraw Invitation",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"network", "actions"},
    )
    async def withdraw_invitation(
        linkedin_username: str,
        ctx: Context,
        extractor: Any = extractor_depends("withdraw_invitation"),
    ) -> dict[str, Any]:
        """
        Withdraw a previously sent LinkedIn connection request.

        Navigates to the invitee's profile and only clicks Withdraw when a
        fresh read confirms a connection request is actually pending there -
        any other state (already withdrawn, already connected, self
        profile, unavailable) is reported back without touching the page.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel", "williamhgates")
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, status, message, and optional profile.
            Statuses: withdrawn, not_pending, self_profile, unavailable,
            withdraw_unavailable, withdraw_failed.
        """
        try:
            username = _require_username(
                linkedin_username, tool_name="withdraw_invitation"
            )
            logger.info("Withdrawing invitation to: %s", username)

            await ctx.report_progress(
                progress=0,
                total=100,
                message="Starting LinkedIn withdraw-invitation flow",
            )

            result = await extractor.withdraw_invitation(username)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except ToolError:
            raise
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "withdraw_invitation")
        except Exception as e:
            raise_tool_error(e, "withdraw_invitation")  # NoReturn
