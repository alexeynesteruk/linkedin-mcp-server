"""Read one post permalink together with its comment thread."""

from __future__ import annotations

from typing import Any

from linkedin_mcp_server.scraping.capture import (
    CaptureMode,
    CapturePlan,
    SectionCapture,
)
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    rate_limited_section_error,
)
from linkedin_mcp_server.scraping.identifiers import normalize_post_url
from linkedin_mcp_server.scraping.link_metadata import Reference


class PostComments:
    """Own the one workflow whose subject is a single post's comment thread.

    One navigation, one section (``post``). The permalink is validated before
    the capture is built, so a value that does not name a post never reaches
    the browser.
    """

    def __init__(self, capture: SectionCapture):
        self._capture = capture

    async def get_post_comments(
        self,
        post_url: str,
        max_scrolls: int | None = None,
    ) -> dict[str, Any]:
        """Read a post and paginate its comments and collapsed replies.

        Args:
            post_url: A LinkedIn post permalink or bare post URN; see
                ``identifiers.normalize_post_url``.
            max_scrolls: Pagination budget, in rounds (default 5).

        Returns:
            {url, sections: {post: text}} plus optional
            ``references["post"]`` (the author, commenters and linked posts)
            and ``section_errors``.

        Raises:
            InvalidReferenceError: when *post_url* is not a post permalink.
        """
        url = normalize_post_url(post_url)
        extracted = await self._capture.capture(
            url,
            section_name="post",
            plan=CapturePlan(CaptureMode.COMMENT_THREAD, max_scrolls),
        )

        sections: dict[str, str] = {}
        references: dict[str, list[Reference]] = {}
        section_errors: dict[str, dict[str, Any]] = {}
        if extracted.text and extracted.text != RATE_LIMITED_SECTION_TEXT:
            sections["post"] = extracted.text
            if extracted.references:
                references["post"] = extracted.references
        elif extracted.text == RATE_LIMITED_SECTION_TEXT:
            section_errors["post"] = rate_limited_section_error()
        elif extracted.error:
            section_errors["post"] = extracted.error

        result: dict[str, Any] = {"url": url, "sections": sections}
        if references:
            result["references"] = references
        if section_errors:
            result["section_errors"] = section_errors
        return result
