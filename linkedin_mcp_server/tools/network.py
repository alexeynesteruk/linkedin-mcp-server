"""
LinkedIn network tools.

Lists pending invitations (received or sent) and withdraws a sent one.
Accepting or ignoring an invitation from the manager is deliberately not
exposed; connect_with_person accepts an incoming request from the profile.
"""

import logging
from typing import Annotated, Any, Literal

from fastmcp import Context, FastMCP
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.scraping.identifiers import normalize_person_identifier

logger = logging.getLogger(__name__)


def register_network_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register the invitation tools with the MCP server."""

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
    ) -> dict[str, Any]:
        """
        List pending LinkedIn network invitations, received or sent.

        Reads /mynetwork/invitation-manager/{received|sent}/ and expands
        truncated invitation notes first, so the text holds each full note.
        Read-only: nothing is accepted, ignored or withdrawn. Use
        withdraw_invitation to take back a sent invitation.

        Args:
            ctx: FastMCP context for progress reporting
            limit: Maximum invitations to return (1-100, default 20).
                References are capped exactly; the text is cut where the first
                omitted invitation begins.
            kind: "received" (default) for incoming invitations, "sent" for
                outgoing ones still awaiting an answer.

        Returns:
            Dict with url, sections (invitations -> raw text), and optional
            references (inviter or invitee profiles) and section_errors. An
            empty sections dict with no error means there are none pending.
        """
        try:
            extractor = await get_ready_extractor(
                ctx, tool_name="get_pending_invitations"
            )
            logger.info("Fetching pending invitations (kind=%s, limit=%d)", kind, limit)

            await ctx.report_progress(
                progress=0, total=100, message=f"Loading {kind} invitations"
            )

            result = await extractor.get_pending_invitations(limit=limit, kind=kind)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

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
    ) -> dict[str, Any]:
        """
        Withdraw a connection request you sent that is still pending.

        Opens the person's profile and clicks Withdraw only when a fresh read
        of that profile shows the request as pending. Any other state is
        reported back without touching the page. LinkedIn may block a new
        invitation to the same person for a while after a withdrawal.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel", "williamhgates"). A full profile URL is accepted too and is reduced to the username.
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, status, message, and optional profile.
            Statuses: withdrawn, not_pending, self_profile, unavailable,
            withdraw_unavailable, withdraw_failed. ``withdrawn`` means a
            re-read no longer shows the request as pending; the message names
            the state read after the withdrawal.

            A status of ``outcome_unknown`` comes from the transport rather
            than the page: the browser process went away with the call in
            flight. Check the profile before calling again.
        """
        try:
            linkedin_username = normalize_person_identifier(linkedin_username)
            extractor = await get_ready_extractor(ctx, tool_name="withdraw_invitation")
            logger.info("Withdrawing invitation to %s", linkedin_username)

            await ctx.report_progress(
                progress=0, total=100, message="Starting LinkedIn withdrawal flow"
            )

            result = await extractor.withdraw_invitation(linkedin_username)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "withdraw_invitation")
        except Exception as e:
            raise_tool_error(e, "withdraw_invitation")  # NoReturn
