"""Unit tests for the lazy-list scroll loop (browser behavior is in the DOM test)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from linkedin_mcp_server.scraping.comment_thread import MAIN_TEXT_LENGTH_JS
from linkedin_mcp_server.scraping.list_scroll import (
    DEFAULT_LIST_ROUNDS,
    SCROLL_LIST_JS,
    scroll_list_until_stable,
)
from linkedin_mcp_server.scraping.session import ScrapingSession


def _page(lengths: list[int], *, probe_fails: bool = False) -> MagicMock:
    answers = iter(lengths)
    page = MagicMock()
    page.viewport_size = {"width": 1000, "height": 600}
    page.mouse.move = AsyncMock()
    page.mouse.wheel = AsyncMock()

    async def evaluate(script: str, *args: Any) -> Any:
        if script == MAIN_TEXT_LENGTH_JS:
            if probe_fails:
                raise RuntimeError("context destroyed")
            return next(answers, lengths[-1])
        assert script == SCROLL_LIST_JS
        return True

    page.evaluate = AsyncMock(side_effect=evaluate)
    return page


def _scrolls(page: MagicMock) -> int:
    return sum(
        1 for call in page.evaluate.await_args_list if call.args[0] == SCROLL_LIST_JS
    )


async def _run(page: MagicMock, rounds: int | None = None) -> None:
    with patch.object(ScrapingSession, "delay", new_callable=AsyncMock):
        await scroll_list_until_stable(ScrapingSession(page), rounds)


async def test_every_round_scrolls_the_element_and_wheels_the_viewport_centre():
    page = _page([10, 20, 30])

    await _run(page, 3)

    assert _scrolls(page) == 3
    assert page.mouse.wheel.await_count == 3
    page.mouse.move.assert_awaited_once_with(500, 300)


async def test_three_rounds_without_growth_end_the_loop():
    page = _page([10, 20, 20, 20, 20, 20, 20])

    await _run(page, 25)

    # grow, then three flat probes: the scroll after the third flat probe is
    # never issued.
    assert _scrolls(page) == 4


async def test_growth_resets_the_stale_count():
    page = _page([10, 10, 10, 20, 20, 20, 20])

    await _run(page, 25)

    assert _scrolls(page) == 6


async def test_the_round_budget_bounds_a_list_that_keeps_growing():
    page = _page(list(range(1, 200)))

    await _run(page, 7)
    assert _scrolls(page) == 7

    page = _page(list(range(1, 200)))
    await _run(page)
    assert _scrolls(page) == DEFAULT_LIST_ROUNDS


async def test_a_zero_budget_does_nothing():
    page = _page([1])

    await _run(page, 0)

    page.evaluate.assert_not_awaited()
    page.mouse.move.assert_not_awaited()


async def test_a_failed_length_probe_stops_without_scrolling():
    page = _page([1], probe_fails=True)

    await _run(page, 5)

    assert _scrolls(page) == 0


async def test_a_dead_mouse_still_scrolls_the_element():
    page = _page([1, 2, 3])
    page.mouse.move = AsyncMock(side_effect=RuntimeError("closed"))

    await _run(page, 3)

    assert _scrolls(page) == 3
    page.mouse.wheel.assert_not_awaited()


async def test_a_failing_wheel_is_dropped_but_the_loop_continues():
    page = _page([1, 2, 3])
    page.mouse.wheel = AsyncMock(side_effect=RuntimeError("closed"))

    await _run(page, 3)

    assert _scrolls(page) == 3
    assert page.mouse.wheel.await_count == 1
