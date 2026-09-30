"""Recognising a lost browser from the error a call ended with.

The texts come from the pinned patchright's own driver and client (see
``core/browser_loss.py``), and ``TargetClosedError`` is the real class, so a
rename or a rewording upstream shows up here rather than as a server that stops
recovering.
"""

from __future__ import annotations

import threading

import pytest
from fastmcp.exceptions import ToolError
from patchright._impl._errors import Error as PatchrightError
from patchright._impl._errors import TargetClosedError

from linkedin_mcp_server.core.browser_loss import (
    browser_loss_in,
    raise_if_browser_lost,
)
from linkedin_mcp_server.core.exceptions import BrowserLostError, NetworkError
from linkedin_mcp_server.error_handler import raise_tool_error


def _chained(outer: BaseException, cause: BaseException) -> BaseException:
    outer.__cause__ = cause
    return outer


class TestBrowserLossIn:
    def test_a_closed_target_is_a_loss(self):
        assert browser_loss_in(TargetClosedError()) == (
            "the page, context or browser was closed"
        )

    def test_the_class_is_enough_when_the_text_is_unfamiliar(self):
        # A rewording in some release must not switch recovery off: the class
        # still says what happened.
        assert browser_loss_in(TargetClosedError("something new")) == (
            "the page, context or browser was closed"
        )

    def test_a_crashed_renderer_is_named_as_a_crash(self):
        crashed = PatchrightError("Page.evaluate: Target crashed ")
        assert browser_loss_in(crashed) == "the page crashed"
        # The driver raises the crash as a closed target with its own text.
        assert browser_loss_in(TargetClosedError("Page crashed")) == "the page crashed"

    def test_a_driver_that_exited_is_recognised_by_its_text(self):
        # The transport raises a bare Exception, so the text is all there is.
        exited = Exception("Connection closed while reading from the driver")
        assert browser_loss_in(exited) == "the browser driver exited"

    def test_the_masked_wrapper_fastmcp_raises_is_seen_through(self):
        masked = _chained(
            ToolError("Error calling tool 'get_feed'"), TargetClosedError()
        )
        assert browser_loss_in(masked) is not None

    def test_a_loss_several_wrappers_down_is_found(self):
        inner = _chained(NetworkError("Failed"), TargetClosedError())
        outer = _chained(ToolError("Network error"), inner)
        assert browser_loss_in(outer) == "the page, context or browser was closed"

    def test_an_implicit_context_is_followed(self):
        try:
            try:
                raise TargetClosedError()
            except TargetClosedError:
                raise RuntimeError("while handling it")  # noqa: B904 - the point
        except RuntimeError as handled:
            assert browser_loss_in(handled) is not None

    def test_an_exception_group_member_is_found(self):
        group = ExceptionGroup("two failed", [ValueError("x"), TargetClosedError()])
        assert browser_loss_in(group) is not None

    def test_a_scraper_loss_keeps_its_reason(self):
        assert browser_loss_in(BrowserLostError("the page crashed")) == (
            "the page crashed"
        )

    @pytest.mark.parametrize(
        "error",
        [
            PatchrightError("Timeout 30000ms exceeded."),
            RuntimeError("Simulated failure"),
            NetworkError("net::ERR_CONNECTION_RESET"),
            ToolError("Error calling tool 'get_feed'"),
        ],
    )
    def test_an_ordinary_failure_is_not_a_loss(self, error):
        assert browser_loss_in(error) is None

    def test_a_cycle_of_contexts_ends(self):
        first, second = RuntimeError("a"), RuntimeError("b")
        first.__context__ = second
        second.__context__ = first
        answers: list[str | None] = []
        # A daemon thread with a deadline, so a walk that loops fails this test
        # instead of hanging the run, and cannot hold the interpreter open.
        walk = threading.Thread(
            target=lambda: answers.append(browser_loss_in(first)), daemon=True
        )
        walk.start()
        walk.join(timeout=5)
        assert not walk.is_alive(), "the walk never ended"
        assert answers == [None]


class TestRaiseIfBrowserLost:
    def test_a_loss_is_raised_as_a_scraper_exception_chained_to_it(self):
        closed = TargetClosedError()
        with pytest.raises(BrowserLostError) as raised:
            raise_if_browser_lost(closed)
        assert raised.value.__cause__ is closed
        assert raised.value.reason == "the page, context or browser was closed"

    def test_an_ordinary_failure_passes(self):
        raise_if_browser_lost(RuntimeError("Simulated failure"))

    def test_an_existing_loss_is_raised_as_itself(self):
        lost = BrowserLostError("the page crashed")
        with pytest.raises(BrowserLostError) as raised:
            raise_if_browser_lost(lost)
        assert raised.value is lost


class TestTheToolErrorForALoss:
    def test_it_carries_no_issue_template_and_keeps_the_chain(self):
        lost = BrowserLostError("the page crashed")
        with pytest.raises(ToolError) as raised:
            raise_tool_error(lost, "get_person_profile")

        assert str(raised.value) == str(lost)
        assert "issue" not in str(raised.value).lower()
        # The serializing middleware finds the loss by walking to it.
        assert raised.value.__cause__ is lost
