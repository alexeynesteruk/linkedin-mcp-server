"""Tests for the comment-thread expansion loop.

``page.evaluate`` is a mock here, so the click program never executes;
``tests/test_comment_thread_dom.py`` runs it against a real DOM.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from linkedin_mcp_server.scraping.comment_thread import (
    CLICK_MORE_COMMENTS_JS,
    MAIN_TEXT_LENGTH_JS,
    expand_comment_thread,
)
from linkedin_mcp_server.scraping.session import ScrapingSession


def _page(*, clicks: list[bool], lengths: list[int]) -> MagicMock:
    """A page whose click program and text length answer from scripts."""
    click_answers = iter(clicks)
    length_answers = iter(lengths)
    page = MagicMock()
    page.viewport_size = {"width": 1000, "height": 600}
    page.mouse.move = AsyncMock()
    page.mouse.wheel = AsyncMock()

    async def evaluate(script: str, *args: Any) -> Any:
        if script == CLICK_MORE_COMMENTS_JS:
            return next(click_answers, False)
        if script == MAIN_TEXT_LENGTH_JS:
            return next(length_answers, lengths[-1])
        raise AssertionError(f"unexpected script: {script[:60]}")

    page.evaluate = AsyncMock(side_effect=evaluate)
    return page


def _click_calls(page: MagicMock) -> list[Any]:
    return [
        call.args[1]
        for call in page.evaluate.await_args_list
        if call.args[0] == CLICK_MORE_COMMENTS_JS
    ]


async def _run(page: MagicMock, rounds: int | None = None) -> None:
    with patch.object(ScrapingSession, "delay", new_callable=AsyncMock):
        await expand_comment_thread(ScrapingSession(page), rounds)


async def test_a_click_is_preferred_over_scrolling_and_growth_keeps_it_going():
    page = _page(clicks=[True, True, True], lengths=[100, 200, 300])

    await _run(page, 3)

    assert len(_click_calls(page)) == 3
    page.mouse.wheel.assert_not_awaited()


async def test_without_a_button_the_wheel_scrolls_the_post():
    page = _page(clicks=[], lengths=[100, 200, 300])

    await _run(page, 3)

    assert page.mouse.wheel.await_count == 3
    page.mouse.wheel.assert_awaited_with(0, 2000)


async def test_the_mouse_is_parked_over_the_viewport_centre_first():
    page = _page(clicks=[], lengths=[1, 2])

    await _run(page, 1)

    page.mouse.move.assert_awaited_once_with(500, 300)


async def test_two_rounds_without_growth_end_the_loop():
    page = _page(clicks=[True] * 10, lengths=[100, 100, 100, 100, 100, 100])

    await _run(page, 10)

    # Round 1 sets the baseline (growth from -1), rounds 2 and 3 are stale.
    assert len(_click_calls(page)) == 3


async def test_the_budget_ends_a_thread_that_keeps_growing():
    page = _page(clicks=[True] * 50, lengths=list(range(100, 5100, 100)))

    await _run(page, 4)

    assert len(_click_calls(page)) == 4


async def test_the_default_budget_is_five_rounds():
    page = _page(clicks=[True] * 50, lengths=list(range(100, 5100, 100)))

    await _run(page, None)

    assert len(_click_calls(page)) == 5


async def test_click_marks_are_cleared_only_after_a_round_that_grew():
    page = _page(clicks=[True, True, True], lengths=[100, 100, 300])

    await _run(page, 3)

    # First round: nothing to clear. It grew from the baseline, so the second
    # asks for a reset; the second did not grow, so the third does not.
    assert _click_calls(page) == [False, True, False]


async def test_a_failing_click_program_falls_back_to_the_wheel():
    page = _page(clicks=[], lengths=[100, 200])
    original = page.evaluate.side_effect

    async def evaluate(script: str, *args: Any) -> Any:
        if script == CLICK_MORE_COMMENTS_JS:
            raise RuntimeError("context destroyed")
        return await original(script, *args)

    page.evaluate = AsyncMock(side_effect=evaluate)

    await _run(page, 2)

    assert page.mouse.wheel.await_count == 2


async def test_a_page_that_refuses_the_mouse_is_left_alone():
    page = _page(clicks=[True], lengths=[100])
    page.mouse.move = AsyncMock(side_effect=RuntimeError("closed"))

    await _run(page, 3)

    page.evaluate.assert_not_awaited()
    page.mouse.wheel.assert_not_awaited()


async def test_a_failing_wheel_stops_the_loop():
    page = _page(clicks=[], lengths=[100, 200, 300])
    page.mouse.wheel = AsyncMock(side_effect=RuntimeError("closed"))

    await _run(page, 5)

    assert page.mouse.wheel.await_count == 1
