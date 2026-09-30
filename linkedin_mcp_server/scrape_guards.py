"""Annotate scrape results that came back empty with no explanation.

A scraping tool that ends with no section text and no ``section_errors`` entry
says nothing about why. "LinkedIn did not hydrate", "the session is degraded"
and "another process holds the profile" all look exactly like "there is nothing
here", and an agent that concludes the latter stops looking. This module turns
the silent case into a visible one and leaves every result that already
explains itself, or is empty by contract, untouched.

The rule
--------
A result is annotated when all of these hold:

* the tool is tagged ``scraping`` or ``search`` and returns a ``sections`` dict;
* no section holds non-blank text (a missing key, ``""`` and whitespace all
  count as no text), and no other payload such as ``job_ids`` stands in for it;
* ``section_errors`` does not already explain it (a rate limit, an overlay that
  never mounted and an extraction failure all set one), and
* the tool is not in ``EMPTY_IS_AN_ANSWER``.

Why a rendered "nothing" is not caught: LinkedIn draws an empty inbox, a search
with no matches or an empty saved-jobs list as text ("No results found"), so a
tool that reads them returns that text and ``sections`` is not empty. Empty
``sections`` therefore means the page yielded no characters at all after the
chrome was stripped, which a healthy page never does. The one exception is a
tool that skips reading when it already knows the answer, listed below.

Partial results (some sections filled, others absent) are not annotated: that
shape is normal for a profile whose optional sections do not exist.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import mcp.types as mt
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult

from linkedin_mcp_server.error_diagnostics import build_issue_diagnostics

logger = logging.getLogger(__name__)

#: Tags that mark a tool as reading page text into ``sections``.
SCRAPING_TAGS: frozenset[str] = frozenset({"scraping", "search"})

#: Tools whose contract says an empty ``sections`` is a valid answer.
#: ``get_pending_invitations`` returns before reading the page when the
#: received counter is zero, and documents empty ``sections`` as "none pending".
EMPTY_IS_AN_ANSWER: frozenset[str] = frozenset({"get_pending_invitations"})

_ALIAS_PREFIX = "linkedin_"

EMPTY_SCRAPE_ERROR_TYPE = "EmptyScrapeSection"


def _has_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    # Anything that is not a string is a structured payload, not blank text.
    return value is not None


def is_unexplained_empty(result: dict[str, Any]) -> bool:
    """Whether *result* has no section text and nothing that explains it."""
    sections = result.get("sections")
    if not isinstance(sections, dict):
        return False
    if any(_has_text(value) for value in sections.values()):
        return False
    if result.get("section_errors"):
        return False
    # ``job_ids`` and similar payloads are content of their own.
    if result.get("job_ids"):
        return False
    return True


def annotate_empty_scrape_result(
    result: dict[str, Any], *, tool_name: str
) -> dict[str, Any]:
    """Annotate *result* in place when it is an unexplained empty scrape.

    Adds ``section_errors`` (one ``EmptyScrapeSection`` entry per blank section,
    or one under ``"sections"`` when none are named), ``empty_scrape: true`` and
    a ``warnings`` entry. Building the diagnostics retains the debug trace
    directory, so the failure can be inspected after the fact.
    """
    if not is_unexplained_empty(result):
        return result

    names = sorted(result["sections"]) or ["sections"]
    target_url = result.get("url") if isinstance(result.get("url"), str) else None
    message = (
        f"{tool_name} returned no page text (empty: {', '.join(names)}) and no "
        "error. LinkedIn may not have hydrated, the session may be degraded, or "
        "another process may be holding the browser profile. This is not "
        "evidence that there is nothing to find; retry before concluding that."
    )
    logger.warning("%s url=%s", message, target_url)

    diagnostics: dict[str, Any] = {}
    try:
        diagnostics = build_issue_diagnostics(
            RuntimeWarning(message),
            context=tool_name,
            target_url=target_url,
            section_name=",".join(names),
        )
    except Exception:
        logger.debug("Could not build empty-scrape diagnostics", exc_info=True)

    result["section_errors"] = {
        name: {
            **diagnostics,
            "error_type": EMPTY_SCRAPE_ERROR_TYPE,
            "error_message": message,
        }
        for name in names
    }
    result["empty_scrape"] = True
    warnings = result.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
        result["warnings"] = warnings
    warnings.append(message)
    return result


class EmptyScrapeMiddleware(Middleware):
    """Annotate unexplained empty results of scraping tools.

    Register it only on a process that runs the tools. A proxy forwards the
    owner's result, which already carries the annotation.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        result = await call_next(context)
        try:
            return await self._annotate(context, result)
        except Exception:
            # Diagnostics must never turn a returned result into a failure.
            logger.debug("Empty-scrape check failed", exc_info=True)
            return result

    async def _annotate(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        result: ToolResult,
    ) -> ToolResult:
        structured = result.structured_content
        if result.is_error or not isinstance(structured, dict):
            return result
        if not is_unexplained_empty(structured):
            return result

        name = context.message.name
        base = name.removeprefix(_ALIAS_PREFIX)
        if name in EMPTY_IS_AN_ANSWER or base in EMPTY_IS_AN_ANSWER:
            return result
        fastmcp_context = context.fastmcp_context
        if fastmcp_context is None:
            return result
        tool = await fastmcp_context.fastmcp.get_tool(name)
        if tool is None or SCRAPING_TAGS.isdisjoint(tool.tags or ()):
            return result

        annotated = annotate_empty_scrape_result(dict(structured), tool_name=base)
        content = list(result.content)
        for index, block in enumerate(content):
            if isinstance(block, mt.TextContent):
                content[index] = mt.TextContent(
                    type="text", text=json.dumps(annotated, ensure_ascii=False)
                )
                break
        return ToolResult(
            content=content, structured_content=annotated, meta=result.meta
        )
