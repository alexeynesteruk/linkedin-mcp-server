"""Soft guards for scrape results that succeed with no usable content.

Empty ``sections`` (or blank critical section bodies) are a common LinkedIn UI
failure mode: the tool returns 200-style structured data while the page never
hydrated. Raise nothing - morning routines and agents still get a payload - but
attach ``section_errors``, keep traces, and write issue diagnostics so the
failure is debuggable.
"""

from __future__ import annotations

import logging
from typing import Any

from linkedin_mcp_server.debug_trace import mark_trace_for_retention
from linkedin_mcp_server.error_diagnostics import build_issue_diagnostics

logger = logging.getLogger(__name__)

# Minimum non-whitespace characters before a section body counts as content.
# LinkedIn chrome-only pages can still emit a few UI labels; real scrapes are
# much larger. Keep this low so tiny-but-real threads still pass.
_MIN_SECTION_CHARS = 20


def _section_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return len(value.strip()) < _MIN_SECTION_CHARS


def empty_critical_sections(
    result: dict[str, Any],
    *,
    required_sections: tuple[str, ...] | None = None,
) -> list[str]:
    """Return names of critical sections that are missing or effectively empty."""
    sections = result.get("sections")
    if not isinstance(sections, dict):
        return list(required_sections or ("sections",))

    if required_sections:
        missing: list[str] = []
        for name in required_sections:
            if name not in sections or _section_is_empty(sections.get(name)):
                missing.append(name)
        return missing

    if not sections:
        return ["sections"]

    non_empty = [
        name for name, value in sections.items() if not _section_is_empty(value)
    ]
    if non_empty:
        return []
    return sorted(sections.keys()) or ["sections"]


def annotate_empty_scrape_result(
    result: dict[str, Any],
    *,
    tool_name: str,
    required_sections: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Annotate *result* when critical scrape content is missing.

    Mutates and returns the same dict. Safe to call on every scrape tool return.
    """
    missing = empty_critical_sections(result, required_sections=required_sections)
    if not missing:
        return result

    # Preserve structured section_errors already set by the scraper (rate limit,
    # extraction failure with issue template, etc.). Only annotate names that
    # still lack an error entry.
    existing_errors = result.get("section_errors")
    if not isinstance(existing_errors, dict):
        existing_errors = {}
    still_missing = [name for name in missing if name not in existing_errors]
    if not still_missing:
        # Content empty but every critical section already has an error entry.
        result["empty_scrape"] = True
        return result

    target_url = result.get("url") if isinstance(result.get("url"), str) else None
    message = (
        f"Scrape returned no usable content for {tool_name} "
        f"(empty sections: {', '.join(still_missing)}). "
        "LinkedIn may not have hydrated, the session may be degraded, "
        "or another browser process may be holding the profile."
    )
    logger.warning("%s url=%s", message, target_url)

    trace_dir = mark_trace_for_retention()
    section_errors = result.setdefault("section_errors", {})
    if not isinstance(section_errors, dict):
        section_errors = {}
        result["section_errors"] = section_errors

    diagnostics: dict[str, Any] | None = None
    try:
        diagnostics = build_issue_diagnostics(
            RuntimeWarning(message),
            context=tool_name,
            target_url=target_url,
            section_name=",".join(still_missing),
        )
    except Exception:
        logger.debug("Could not build empty-scrape diagnostics", exc_info=True)

    for name in still_missing:
        entry: dict[str, Any] = {
            "error_type": "EmptyScrapeSection",
            "error_message": message,
        }
        if diagnostics is not None:
            entry.update(diagnostics)
        elif trace_dir is not None:
            entry["trace_dir"] = str(trace_dir)
        section_errors[name] = entry

    result["empty_scrape"] = True
    result.setdefault("warnings", [])
    if isinstance(result["warnings"], list) and message not in result["warnings"]:
        result["warnings"].append(message)
    return result
