"""Section config dicts controlling which LinkedIn pages are visited during scraping."""

from dataclasses import dataclass

import logging

from linkedin_mcp_server.scraping.capture import CaptureMode, CapturePlan
from linkedin_mcp_server.scraping.contracts import FilterValidationError

logger = logging.getLogger(__name__)

# Maps section name -> (url_suffix, is_overlay)
PERSON_SECTIONS: dict[str, tuple[str, bool]] = {
    "main_profile": ("/", False),
    "experience": ("/details/experience/", False),
    "education": ("/details/education/", False),
    "interests": ("/details/interests/", False),
    "honors": ("/details/honors/", False),
    "languages": ("/details/languages/", False),
    "certifications": ("/details/certifications/", False),
    "skills": ("/details/skills/", False),
    "projects": ("/details/projects/", False),
    "contact_info": ("/overlay/contact-info/", True),
    "posts": ("/recent-activity/all/", False),
}

COMPANY_SECTIONS: dict[str, tuple[str, bool]] = {
    "about": ("/about/", False),
    "posts": ("/posts/", False),
    "jobs": ("/jobs/", False),
}

# The signed-in member's own analytics dashboards ("Private to you"). Maps
# section name -> url suffix under https://www.linkedin.com/analytics. There is
# no per-username variant. /analytics/creator/ (the overview) is omitted
# because LinkedIn redirects it to the content page.
ANALYTICS_SECTIONS: dict[str, str] = {
    "content": "/creator/content/",
    "audience": "/creator/audience/",
    "top_posts": "/creator/top-posts/",
    "profile_views": "/profile-views/",
    "search_appearances": "/search-appearances/",
}

# The only dashboards whose page honours the ?timeRange= query parameter. The
# others use a fixed window (top_posts 14 days, profile_views 90 days).
ANALYTICS_TIME_RANGE_SECTIONS: frozenset[str] = frozenset({"content", "audience"})

_ANALYTICS_TIME_RANGES: dict[str, str] = {
    "7d": "past_7_days",
    "28d": "past_28_days",
    "90d": "past_90_days",
    "365d": "past_365_days",
    "past_7_days": "past_7_days",
    "past_28_days": "past_28_days",
    "past_90_days": "past_90_days",
    "past_365_days": "past_365_days",
}


@dataclass(frozen=True)
class _SectionSpec:
    name: str
    suffix: str
    plan: CapturePlan


_PERSON_SECTION_MODES = {
    "experience": CaptureMode.DETAILS,
    "education": CaptureMode.DETAILS,
    "interests": CaptureMode.DETAILS,
    "honors": CaptureMode.DETAILS,
    "languages": CaptureMode.DETAILS,
    "certifications": CaptureMode.DETAILS,
    "skills": CaptureMode.DETAILS,
    "projects": CaptureMode.DETAILS,
    "contact_info": CaptureMode.OVERLAY,
    "posts": CaptureMode.ACTIVITY,
}
_COMPANY_SECTION_MODES = {"posts": CaptureMode.ACTIVITY}


def _person_section_specs(
    sections: dict[str, tuple[str, bool]],
    max_scrolls: int | None = None,
) -> tuple[_SectionSpec, ...]:
    return tuple(
        _SectionSpec(
            name,
            suffix,
            CapturePlan(
                CaptureMode.OVERLAY
                if is_overlay
                else _PERSON_SECTION_MODES.get(name, CaptureMode.STANDARD),
                max_scrolls,
            ),
        )
        for name, (suffix, is_overlay) in sections.items()
    )


def _company_section_specs(
    sections: dict[str, tuple[str, bool]] = COMPANY_SECTIONS,
) -> tuple[_SectionSpec, ...]:
    return tuple(
        _SectionSpec(
            name,
            suffix,
            CapturePlan(
                CaptureMode.OVERLAY
                if is_overlay
                else _COMPANY_SECTION_MODES.get(name, CaptureMode.STANDARD)
            ),
        )
        for name, (suffix, is_overlay) in sections.items()
    )


def _analytics_section_specs(
    sections: dict[str, str] = ANALYTICS_SECTIONS,
    max_scrolls: int | None = None,
) -> tuple[_SectionSpec, ...]:
    return tuple(
        _SectionSpec(name, suffix, CapturePlan(CaptureMode.STANDARD, max_scrolls))
        for name, suffix in sections.items()
    )


def normalize_analytics_time_range(time_range: str | None) -> str | None:
    """Return LinkedIn's ``timeRange`` value for *time_range*.

    Accepts ``7d``/``28d``/``90d``/``365d`` or the ``past_N_days`` spelling,
    case-insensitively. ``None`` (LinkedIn's default window) passes through.

    Raises:
        FilterValidationError: for any other value, so nothing is navigated
            with a window LinkedIn would silently ignore.
    """
    if time_range is None:
        return None
    normalized = _ANALYTICS_TIME_RANGES.get(time_range.strip().lower())
    if normalized is None:
        raise FilterValidationError(
            f"Invalid time_range {time_range!r}. Valid values: 7d, 28d, 90d, "
            "365d (or past_7_days, past_28_days, past_90_days, past_365_days)."
        )
    return normalized


def parse_person_sections(
    sections: str | None,
) -> tuple[set[str], list[str]]:
    """Parse comma-separated section names into a set of requested sections.

    "main_profile" is always included. Empty/None returns {"main_profile"} only.
    Unknown section names are logged as warnings and returned.

    Returns:
        Tuple of (requested_sections, unknown_section_names).
    """
    requested: set[str] = {"main_profile"}
    unknown: list[str] = []
    if not sections:
        return requested, unknown
    for name in sections.split(","):
        name = name.strip().lower()
        if not name:
            continue
        if name in PERSON_SECTIONS:
            requested.add(name)
        else:
            unknown.append(name)
            logger.warning(
                "Unknown person section %r ignored. Valid: %s",
                name,
                ", ".join(sorted(PERSON_SECTIONS)),
            )
    return requested, unknown


def parse_company_sections(
    sections: str | None,
) -> tuple[set[str], list[str]]:
    """Parse comma-separated section names into a set of requested sections.

    "about" is always included. Empty/None returns {"about"} only.
    Unknown section names are logged as warnings and returned.

    Returns:
        Tuple of (requested_sections, unknown_section_names).
    """
    requested: set[str] = {"about"}
    unknown: list[str] = []
    if not sections:
        return requested, unknown
    for name in sections.split(","):
        name = name.strip().lower()
        if not name:
            continue
        if name in COMPANY_SECTIONS:
            requested.add(name)
        else:
            unknown.append(name)
            logger.warning(
                "Unknown company section %r ignored. Valid: %s",
                name,
                ", ".join(sorted(COMPANY_SECTIONS)),
            )
    return requested, unknown


def parse_analytics_sections(
    sections: str | None,
) -> tuple[set[str], list[str]]:
    """Parse comma-separated section names into a set of requested sections.

    Unlike the profile parsers there is no always-included baseline: the tool
    exists to read the dashboards, so empty/None selects ALL sections, and so
    does a list holding only unknown names. Unknown section names are logged
    as warnings and returned.

    Returns:
        Tuple of (requested_sections, unknown_section_names).
    """
    if not sections:
        return set(ANALYTICS_SECTIONS), []
    requested: set[str] = set()
    unknown: list[str] = []
    for name in sections.split(","):
        name = name.strip().lower()
        if not name:
            continue
        if name in ANALYTICS_SECTIONS:
            requested.add(name)
        else:
            unknown.append(name)
            logger.warning(
                "Unknown analytics section %r ignored. Valid: %s",
                name,
                ", ".join(sorted(ANALYTICS_SECTIONS)),
            )
    if not requested:
        requested = set(ANALYTICS_SECTIONS)
    return requested, unknown
