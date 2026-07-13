"""Shared scraping constants.

Public names are safe for tools/tests to import. Private underscore aliases
remain for backwards compatibility with existing extractor imports.
"""

# Returned as section text when LinkedIn rate-limits the page
RATE_LIMITED_MSG = (
    "[Rate limited] LinkedIn blocked this section. "
    "Try again later or request fewer sections."
)

# Back-compat alias used by extractor / older imports
_RATE_LIMITED_MSG = RATE_LIMITED_MSG
