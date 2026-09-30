"""Telling a browser that is gone from a page that merely failed.

A tool call can fail in two ways that look alike and are not. A page that did
not load, a selector that timed out or a script that threw leave the browser
usable, and the next call on the same page is expected to work. A page, context
or browser that was closed, a renderer that crashed, or a driver that exited
leave nothing to call: every later operation on the same objects fails the same
way until a new browser is launched. The server lives as long as its client, so
without a way to tell the two apart it keeps failing every call after the first.

Only the error can say which it was. Patchright's own state sees a closed page
or context and a disconnected browser (``BrowserManager.lost_reason``), but a
driver that exited changes no state here at all; the next call on it simply
raises. So both are asked: the state before a call, the error after one.
"""

from __future__ import annotations

from collections import deque

from linkedin_mcp_server.core.exceptions import BrowserLostError

#: The phrases Patchright uses for an object that is gone, with what each means.
#: Matched as lowercase substrings, so the method prefix Patchright adds
#: (``Page.goto: ...``) and the call log it appends do not matter. Read from
#: patchright 1.63.0's driver (``driver/package/lib/coreBundle.js``) and client
#: (``_impl/_errors.py``, ``_impl/_transport.py``) on 2026-09-30, not guessed:
#:
#: * ``TargetClosedError``'s default text, raised for a closed page, context or
#:   browser, and for every call pending when the driver connection dropped;
#: * the crash texts: ``Target crashed`` for an operation running when the
#:   renderer died, ``Page crashed`` for one waiting on it;
#: * ``Page has been closed.`` and ``Session already detached. Most likely the
#:   page has been closed.`` from the driver;
#: * the transport's own ``Connection closed while reading from the driver``,
#:   which is a bare ``Exception`` and so can only be recognised by its text.
#:
#: These are Patchright's words, not LinkedIn's, so the locale rule for page
#: detection does not apply: they are the same in every interface language.
#:
#: First match wins, so the more specific phrase comes first: the crash texts
#: arrive on a ``TargetClosedError`` too, and ``TargetClosedError``'s own text
#: contains ``browser has been closed``.
_LOSS_MARKERS: tuple[tuple[str, str], ...] = (
    ("target crashed", "the page crashed"),
    ("page crashed", "the page crashed"),
    ("connection closed while reading from the driver", "the browser driver exited"),
    (
        "target page, context or browser has been closed",
        "the page, context or browser was closed",
    ),
    ("page has been closed", "the page was closed"),
    ("context has been closed", "the browser context was closed"),
    ("browser has been closed", "the browser was closed"),
    ("target closed", "the page, context or browser was closed"),
)

#: How far down a chain to look. Wrappers stack a few deep at most (the tool's
#: ``ToolError``, FastMCP's masking one, a scraper exception); the bound only
#: keeps a pathological cycle of ``__context__`` links from looping.
_MAX_LINKS = 32


def _is_target_closed(error: BaseException) -> bool:
    """Whether *error* is Patchright's ``TargetClosedError`` by class.

    By name rather than by import: the class lives in ``patchright._impl``,
    which is not public API, and a text change in some release would otherwise
    leave nothing that recognises it.
    """
    return any(cls.__name__ == "TargetClosedError" for cls in type(error).__mro__)


def browser_loss_in(error: BaseException) -> str | None:
    """Say how the browser was lost if *error* reports it, else ``None``.

    Walks the whole chain, causes, contexts and exception-group members alike,
    because what reaches a caller is usually a wrapper: the tool's ``ToolError``,
    FastMCP's masking ``ToolError`` around a raw driver error, or a
    ``BrowserLostError`` a capture path raised.
    """
    pending: deque[BaseException] = deque([error])
    seen: set[int] = set()
    while pending and len(seen) < _MAX_LINKS:
        current = pending.popleft()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, BrowserLostError):
            return current.reason
        text = str(current).lower()
        for marker, reason in _LOSS_MARKERS:
            if marker in text:
                return reason
        if _is_target_closed(current):
            return "the page, context or browser was closed"
        linked = [current.__cause__, current.__context__]
        members = getattr(current, "exceptions", None)
        if isinstance(members, tuple):
            linked.extend(members)
        pending.extend(link for link in linked if isinstance(link, BaseException))
    return None


def raise_if_browser_lost(error: BaseException) -> None:
    """Raise :class:`BrowserLostError` from *error* when it reports a lost browser.

    For the broad ``except Exception`` handlers that turn a failure into a
    section error or skip past it. Called first in the handler, so a browser
    that is gone propagates instead of becoming an entry that looks like an
    empty page.
    """
    if isinstance(error, BrowserLostError):
        raise error
    reason = browser_loss_in(error)
    if reason is not None:
        raise BrowserLostError(reason) from error
