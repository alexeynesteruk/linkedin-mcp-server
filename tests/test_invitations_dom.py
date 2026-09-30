"""Browser-DOM tests for the invitation-manager page programs.

``tests/scraping/test_invitations.py`` mocks ``page.evaluate``, so the note
expansion and the empty-count check never execute there. These run the real
programs against synthetic markup in headless chromium.

The note toggle behaves the way the reader relies on: one control, marked by
``data-testid``, that expands the note on the first click and collapses it
again on the next. A second click is therefore not harmless, and the cases
below count clicks per note rather than only reading the final text.

Every fixture is rendered in four label sets, like
``tests/test_action_signals_dom.py``: English, German, opaque tokens and empty
labels. The toggle is found by its test id and never by its words, so the
answer has to be the same in all four.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest
from patchright.async_api import async_playwright

from linkedin_mcp_server.scraping.invitations import (
    EXPAND_NOTES_JS,
    RECEIVED_COUNT_IS_ZERO_JS,
)

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]


@dataclass(frozen=True, slots=True)
class Labels:
    locale: str
    see_more: str
    see_less: str
    all_tab: str
    people_tab: str


LOCALES = (
    Labels("en", "…see more", "see less", "All", "People"),
    Labels("de", "…mehr anzeigen", "weniger anzeigen", "Alle", "Personen"),
    Labels("opaque", "c1d2e3", "f4a5b6", "a7b8c9", "d0e1f2"),
    Labels("empty", "", "", "", ""),
)


@pytest.fixture
async def dom_page():
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(channel="chromium", headless=True)
            page = await browser.new_page()
        except Exception as exc:  # browser binary missing
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            yield page
        finally:
            await browser.close()


async def _in_every_locale(
    page,
    build: Callable[[Labels], str],
    expected: Any,
    read: Callable[[Any, str], Awaitable[Any]],
) -> None:
    answers = {labels.locale: await read(page, build(labels)) for labels in LOCALES}
    assert answers == {labels.locale: expected for labels in LOCALES}


# One toggle per note. The handler flips the note and counts its own clicks on
# the box, so a note clicked twice reads as collapsed *and* as clicked twice.
# ``pointer-events: none`` and ``aria-hidden`` are how LinkedIn ships it.
_TOGGLE_SCRIPT = """
<script>
  function toggleNote(button, more, less) {
    const box = button.closest('[data-testid="expandable-text-box"]');
    box.dataset.clicks = String(Number(box.dataset.clicks || 0) + 1);
    const expanded = box.dataset.state === 'expanded';
    box.dataset.state = expanded ? 'collapsed' : 'expanded';
    box.querySelector('.note').textContent =
      expanded ? box.dataset.short : box.dataset.full;
    button.textContent = expanded ? more : less;
    if (box.dataset.loads && !expanded) {
      // Expanding this note mounts another card, as the lazy list does.
      const card = document.createElement('li');
      card.innerHTML = box.dataset.loads;
      document.querySelector('main ul').appendChild(card);
      delete box.dataset.loads;
    }
  }
</script>
"""


def _note(
    labels: Labels,
    name: str,
    *,
    loads: str = "",
    expanded: bool = False,
) -> str:
    aria = ' aria-expanded="true"' if expanded else ""
    state = "expanded" if expanded else "collapsed"
    text = f"{name} full note" if expanded else f"{name} short"
    loads_attr = f" data-loads='{loads}'" if loads else ""
    return f"""
<a href="https://www.linkedin.com/in/{name}/">{name}</a>
<span data-testid="expandable-text-box" id="{name}" data-state="{state}"
  data-short="{name} short" data-full="{name} full note"{loads_attr}>
  <span class="note">{text}</span>
  <button data-testid="expandable-text-button"{aria} aria-hidden="true"
    style="pointer-events: none"
    onclick="toggleNote(this, '{labels.see_more}', '{labels.see_less}')"
    >{labels.see_less if expanded else labels.see_more}</button>
</span>
"""


def _invitations(*cards: str, outside: str = "") -> str:
    items = "".join(f"<li>{card}</li>" for card in cards)
    return (
        f"<html><head>{_TOGGLE_SCRIPT}</head><body>"
        f"<main><ul>{items}</ul></main>{outside}</body></html>"
    )


async def _notes_after(page, html: str, passes: int) -> tuple[list[int], dict]:
    """Run the expansion *passes* times: its counts, then each note's state."""
    await page.set_content(html)
    counts = [await page.evaluate(EXPAND_NOTES_JS) for _ in range(passes)]
    notes = await page.evaluate(
        """() => Object.fromEntries(
            [...document.querySelectorAll('[data-testid="expandable-text-box"]')]
              .map(box => [box.id, [box.dataset.state, Number(box.dataset.clicks || 0)]])
        )"""
    )
    return counts, notes


class TestExpandNotes:
    async def test_every_collapsed_note_is_expanded_once(self, dom_page):
        await _in_every_locale(
            dom_page,
            lambda labels: _invitations(_note(labels, "ada"), _note(labels, "bob")),
            ([2], {"ada": ["expanded", 1], "bob": ["expanded", 1]}),
            lambda page, html: _notes_after(page, html, 1),
        )

    async def test_the_second_pass_does_not_collapse_the_first_passes_notes(
        self, dom_page
    ):
        # The expanded note keeps its toggle and its test id; only the words on
        # it change. Clicking it again is the collapse.
        await _in_every_locale(
            dom_page,
            lambda labels: _invitations(_note(labels, "ada"), _note(labels, "bob")),
            ([2, 0], {"ada": ["expanded", 1], "bob": ["expanded", 1]}),
            lambda page, html: _notes_after(page, html, 2),
        )

    async def test_the_second_pass_expands_only_the_cards_the_first_one_loaded(
        self, dom_page
    ):
        def build(labels: Labels) -> str:
            later = _note(labels, "cyd").replace("'", "&#39;")
            return _invitations(_note(labels, "ada", loads=later))

        await _in_every_locale(
            dom_page,
            build,
            ([1, 1], {"ada": ["expanded", 1], "cyd": ["expanded", 1]}),
            lambda page, html: _notes_after(page, html, 2),
        )

    async def test_a_note_already_expanded_is_left_open(self, dom_page):
        await _in_every_locale(
            dom_page,
            lambda labels: _invitations(
                _note(labels, "ada", expanded=True), _note(labels, "bob")
            ),
            ([1], {"ada": ["expanded", 0], "bob": ["expanded", 1]}),
            lambda page, html: _notes_after(page, html, 1),
        )

    async def test_a_toggle_outside_main_is_not_clicked(self, dom_page):
        await _in_every_locale(
            dom_page,
            lambda labels: _invitations(
                _note(labels, "ada"), outside=f"<aside>{_note(labels, 'zed')}</aside>"
            ),
            ([1], {"ada": ["expanded", 1], "zed": ["collapsed", 0]}),
            lambda page, html: _notes_after(page, html, 1),
        )


def _manager(tabs: str, *, in_main: bool = True) -> str:
    nav = f"<nav>{tabs}</nav>"
    if in_main:
        return f"<html><body><main>{nav}<ul></ul></main></body></html>"
    return f"<html><body>{nav}<main><ul></ul></main></body></html>"


def _tab(label: str, route: str, count: str, *, current: bool) -> str:
    aria = ' aria-current="true"' if current else ""
    return (
        f'<a href="/mynetwork/invitation-manager/received/{route}/"{aria}>'
        f"{label} ({count})</a>"
    )


async def _zero(page, html: str) -> bool:
    await page.set_content(html)
    return await page.evaluate(RECEIVED_COUNT_IS_ZERO_JS)


class TestReceivedCountIsZero:
    async def test_a_selected_all_tab_at_zero(self, dom_page):
        await _in_every_locale(
            dom_page,
            lambda labels: _manager(
                _tab(labels.all_tab, "ALL", "0", current=True)
                + _tab(labels.people_tab, "CONNECTION", "0", current=False)
            ),
            True,
            _zero,
        )

    @pytest.mark.parametrize("count", ["3", "10", "100"])
    async def test_a_selected_all_tab_with_invitations(self, dom_page, count):
        await _in_every_locale(
            dom_page,
            lambda labels: _manager(_tab(labels.all_tab, "ALL", count, current=True)),
            False,
            _zero,
        )

    async def test_a_zero_on_a_tab_that_is_not_selected_says_nothing(self, dom_page):
        await _in_every_locale(
            dom_page,
            lambda labels: _manager(
                _tab(labels.all_tab, "ALL", "4", current=True)
                + _tab(labels.people_tab, "CONNECTION", "0", current=False)
            ),
            False,
            _zero,
        )

    async def test_an_unselected_all_tab_at_zero_says_nothing(self, dom_page):
        await _in_every_locale(
            dom_page,
            lambda labels: _manager(_tab(labels.all_tab, "ALL", "0", current=False)),
            False,
            _zero,
        )

    async def test_a_tab_outside_main_is_not_read(self, dom_page):
        await _in_every_locale(
            dom_page,
            lambda labels: _manager(
                _tab(labels.all_tab, "ALL", "0", current=True), in_main=False
            ),
            False,
            _zero,
        )

    async def test_no_count_is_not_zero(self, dom_page):
        await _in_every_locale(
            dom_page,
            lambda labels: _manager(
                f'<a href="/mynetwork/invitation-manager/received/ALL/"'
                f' aria-current="true">{labels.all_tab}</a>'
            ),
            False,
            _zero,
        )
