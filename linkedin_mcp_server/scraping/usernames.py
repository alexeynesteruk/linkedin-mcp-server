"""Normalize LinkedIn person vanity names from tool arguments.

Agents often pass full profile URLs (``https://www.linkedin.com/in/jane/``)
or path fragments instead of bare vanities. Scrapers build
``/in/{username}/`` paths with string format, so un-normalized input produces
broken URLs (``/in/https://.../``) and flaky scrapes.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

from linkedin_mcp_server.scraping.link_metadata import _is_linkedin_host

# LinkedIn vanity segment: letters, digits, hyphen. Reject empty, "me", and
# path-traversal style input. Length cap matches observed LinkedIn vanities.
_VANITY_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,98}[A-Za-z0-9])?$")
_RESERVED_VANITIES = frozenset({"me", "in", "pub", "www"})


def normalize_linkedin_username(value: str | None) -> str | None:
    """Return a bare LinkedIn vanity name, or None if *value* is unusable.

    Accepts:
    - bare vanity: ``williamhgates``
    - path: ``/in/williamhgates/`` or ``in/williamhgates``
    - full URL on a LinkedIn host: ``https://www.linkedin.com/in/williamhgates/``

    Rejects empty strings, reserved slugs (``me``), and values with path
    separators or query junk after stripping.
    """
    if value is None:
        return None
    raw = unquote(str(value)).strip()
    if not raw:
        return None

    candidate = raw
    if "://" in candidate or candidate.startswith("//"):
        parsed = urlparse(candidate if "://" in candidate else f"https:{candidate}")
        host = (parsed.netloc or "").lower()
        if host and not _is_linkedin_host(host):
            return None
        candidate = parsed.path or ""
    elif "?" in candidate or "#" in candidate:
        candidate = candidate.split("?", 1)[0].split("#", 1)[0]

    candidate = candidate.strip().strip("/")
    if not candidate:
        return None

    parts = [p for p in candidate.split("/") if p]
    if not parts:
        return None
    if parts[0].lower() == "in":
        if len(parts) < 2:
            return None
        vanity = parts[1]
    elif len(parts) == 1:
        vanity = parts[0]
    else:
        # Multi-segment non-/in/ path is not a profile vanity.
        return None

    vanity = vanity.strip()
    if not vanity or vanity.lower() in _RESERVED_VANITIES:
        return None
    if not _VANITY_RE.fullmatch(vanity):
        return None
    return vanity
