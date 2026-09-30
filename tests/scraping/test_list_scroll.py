"""Unit tests for the virtualized-list scroll and merge (browser side is the DOM test)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from linkedin_mcp_server.scraping.list_scroll import (
    DEFAULT_LIST_ROUNDS,
    SCROLL_LIST_STEP_JS,
    merge_root_snapshots,
    scroll_list_collecting,
    weave_lines,
)
from linkedin_mcp_server.scraping.session import ScrapingSession

HEADER = ["Skills", "All", "Tools & Technologies"]
SIDEBAR = ["Who your viewers also viewed", "Private to you", "View"]


def _items(start: int, stop: int) -> list[str]:
    return [f"Skill {i}" for i in range(start, stop)]


class TestWeaveLines:
    def test_sliding_windows_merge_into_the_whole_list_in_order(self):
        reads = [
            HEADER + _items(0, 10) + SIDEBAR,
            HEADER + _items(6, 18) + SIDEBAR,
            HEADER + _items(15, 30) + SIDEBAR,
        ]
        merged: list[str] = []
        for read in reads:
            merged = weave_lines(merged, read)

        assert merged == HEADER + _items(0, 30) + SIDEBAR

    def test_a_window_that_skipped_ahead_still_lands_before_the_sidebar(self):
        merged = weave_lines(
            HEADER + _items(0, 5) + SIDEBAR, HEADER + _items(9, 12) + SIDEBAR
        )

        assert merged == HEADER + _items(0, 5) + _items(9, 12) + SIDEBAR

    def test_lines_repeated_across_items_do_not_scramble_the_order(self):
        items: list[str] = []
        for i in range(20):
            items += [f"Skill {i}", "8 endorsements" if i % 2 else "Endorsed by 5"]
        merged: list[str] = []
        for read in (items[0:14], items[10:30], items[26:40]):
            merged = weave_lines(merged, HEADER + read + SIDEBAR)

        assert merged == HEADER + items + SIDEBAR

    def test_the_same_read_twice_changes_nothing(self):
        read = HEADER + _items(0, 8) + SIDEBAR

        assert weave_lines(read, read) == read


def test_references_keep_their_first_seen_order_once_each():
    merged = merge_root_snapshots(
        [
            {"source": "root", "text": "a", "references": [{"href": "/in/a/"}]},
            {
                "source": "root",
                "text": "b",
                "references": [{"href": "/in/b/"}, {"href": "/in/a/"}],
            },
        ]
    )

    assert [r["href"] for r in merged["references"]] == ["/in/a/", "/in/b/"]
    assert merged["source"] == "root"


def _page(*, moves: int = 0) -> MagicMock:
    """A page whose scroller moves on the first *moves* steps, then sits still."""
    page = MagicMock()
    page.viewport_size = {"width": 1000, "height": 600}
    page.mouse.move = AsyncMock()
    page.mouse.wheel = AsyncMock()
    steps = iter([True] * moves)

    async def evaluate(script: str, *args: Any) -> Any:
        assert script == SCROLL_LIST_STEP_JS
        return next(steps, False)

    page.evaluate = AsyncMock(side_effect=evaluate)
    return page


def _reads(*windows: tuple[int, int]) -> AsyncMock:
    """One root read per scroll position; the last one repeats."""
    answers = [
        {
            "source": "root",
            "text": "\n".join(HEADER + _items(*w) + SIDEBAR),
            "references": [],
        }
        for w in windows
    ]

    async def read() -> dict[str, Any]:
        return answers.pop(0) if len(answers) > 1 else answers[0]

    return AsyncMock(side_effect=read)


async def _run(page: MagicMock, read: AsyncMock, rounds: int | None = None) -> dict:
    with patch.object(ScrapingSession, "delay", new_callable=AsyncMock):
        return await scroll_list_collecting(ScrapingSession(page), read, rounds)


async def test_a_virtualized_list_is_collected_whole():
    page = _page()

    merged = await _run(page, _reads((0, 10), (6, 18), (15, 30), (25, 30)))

    assert merged["text"].split("\n") == HEADER + _items(0, 30) + SIDEBAR
    page.mouse.move.assert_awaited_once_with(500, 300)


async def test_three_reads_that_add_nothing_end_the_loop():
    page = _page()
    read = _reads((0, 10), (0, 12))

    await _run(page, read, 25)

    # The first read, one that grew, then three flat ones.
    assert read.await_count == 5


async def test_the_round_budget_bounds_a_list_that_keeps_growing():
    windows = [(i, i + 5) for i in range(0, 400, 3)]
    read = _reads(*windows)

    await _run(_page(), read, 7)
    assert read.await_count == 8

    read = _reads(*windows)
    await _run(_page(), read)
    assert read.await_count == DEFAULT_LIST_ROUNDS + 1


async def test_a_failed_scroll_keeps_what_was_read():
    page = _page()
    page.evaluate = AsyncMock(side_effect=RuntimeError("context destroyed"))

    merged = await _run(page, _reads((0, 4)), 5)

    assert merged["text"].split("\n") == HEADER + _items(0, 4) + SIDEBAR


async def test_a_dead_mouse_still_scrolls_the_element():
    page = _page()
    page.mouse.move = AsyncMock(side_effect=RuntimeError("closed"))

    await _run(page, _reads((0, 3), (0, 6), (0, 9)), 2)

    assert page.evaluate.await_count == 2
    page.mouse.wheel.assert_not_awaited()


async def test_steps_that_still_move_the_scroller_do_not_count_as_stale():
    # A plain (not virtualized) list already fully in the DOM adds no line
    # while it is travelled; only once the scroller stops do flat reads count.
    page = _page(moves=5)
    read = _reads((0, 10))

    await _run(page, read, 25)

    # Five moving steps, then three still ones, plus the first read.
    assert read.await_count == 9


async def test_the_wheel_steps_about_a_screen_and_only_once_the_scroller_is_still():
    page = _page(moves=2)

    await _run(page, _reads((0, 3)), 3)

    # Two steps moved the element, so only the third fell back to the wheel.
    page.mouse.wheel.assert_awaited_once_with(0, 480)
