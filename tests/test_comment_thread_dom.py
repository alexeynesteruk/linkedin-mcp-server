"""Browser-DOM tests for the comment pagination click.

``page.evaluate`` is a mock in the unit suite, so ``CLICK_MORE_COMMENTS_JS``
never executes there. These run the real program against synthetic markup in
headless chromium. The button in every fixture is deliberately labelled in a
language other than English: the program must find it by position and by what
it lacks, never by its words.

Skipped automatically when chromium is not installed.
"""

from __future__ import annotations

import pytest
from patchright.async_api import async_playwright

from linkedin_mcp_server.scraping.comment_thread import CLICK_MORE_COMMENTS_JS

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]


@pytest.fixture
async def dom_page():
    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(
                channel="chromium", headless=True
            )
            page = await browser.new_page()
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            yield page
        finally:
            await browser.close()


_COMPOSER = '<div role="textbox" contenteditable="true">x</div>'
_COMMENT = '<article><a href="/in/bob/">Bob</a><p>Nice</p></article>'
_HEADER = '<header><a href="/in/alice/">Alice</a><button>Suivre</button></header>'


def _page(*thread: str, before: str = _HEADER, after: str = "") -> str:
    """A post: header, comment block (composer, then the thread), then a rail."""
    return (
        "<main>"
        f"{before}"
        f'<section id="comments">{_COMPOSER}{"".join(thread)}</section>'
        f"{after}"
        "</main>"
    )


_RECORD_CLICKS = """() => {
  if (window.__recording) return;
  window.__recording = true;
  document.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (b) window.__clicked = (window.__clicked || []).concat(b.id || b.innerText);
  }, true);
}"""


async def _click(page, reset: bool = False) -> tuple[bool, list[str]]:
    await page.evaluate(_RECORD_CLICKS)
    clicked = await page.evaluate(CLICK_MORE_COMMENTS_JS, reset)
    return clicked, await page.evaluate("() => window.__clicked || []")


async def test_clicks_the_button_below_the_thread_whatever_its_language(dom_page):
    await dom_page.set_content(
        _page(_COMMENT, '<button id="more">Charger plus de commentaires</button>')
    )

    assert await _click(dom_page) == (True, ["more"])


async def test_a_second_call_does_not_reclick_the_same_button_until_reset(dom_page):
    await dom_page.set_content(_page(_COMMENT, '<button id="more">Mehr laden</button>'))

    assert (await _click(dom_page))[0] is True
    assert await _click(dom_page) == (False, ["more"])
    assert await _click(dom_page, reset=True) == (True, ["more", "more"])


@pytest.mark.parametrize(
    "control",
    [
        '<button id="c" aria-label="Like">Like</button>',
        '<button id="c" aria-pressed="false">Aimer</button>',
        '<button id="c" aria-expanded="false">Plus</button>',
        '<button id="c" aria-haspopup="menu">Plus</button>',
        '<button id="c"><svg width="10" height="10"></svg></button>',
        '<button id="c">3 réponses</button>',
        '<button id="c" disabled>Publier</button>',
        '<button id="c" type="submit">Publier</button>',
        '<form><button id="c">Publier</button></form>',
        '<button id="c" data-testid="expandable-text-button">voir plus</button>',
        '<div role="menu"><button id="c">Signaler</button></div>',
        '<button id="c" style="display:none">Caché</button>',
    ],
)
async def test_a_control_that_is_not_pagination_is_never_clicked(dom_page, control):
    await dom_page.set_content(_page(_COMMENT, control))

    assert await _click(dom_page) == (False, [])


async def test_a_button_above_the_composer_is_out_of_reach(dom_page):
    # The header's "Suivre" carries no aria attribute and has no digit, so only
    # its position keeps it from being clicked.
    await dom_page.set_content(_page(_COMMENT))

    assert await _click(dom_page) == (False, [])


async def test_a_control_above_the_composer_inside_the_block_is_out_of_reach(
    dom_page,
):
    # A sort toggle sits in the same block as the thread, above the composer.
    # It is in scope and has no aria attribute, so only its position excludes it.
    await dom_page.set_content(
        "<main>"
        '<section><button id="sort">Trier</button>'
        f"{_COMPOSER}{_COMMENT}</section>"
        "</main>"
    )

    assert await _click(dom_page) == (False, [])


async def test_a_module_after_the_comment_block_is_out_of_reach(dom_page):
    await dom_page.set_content(
        _page(
            _COMMENT,
            after='<aside><a href="/in/carol/">Carol</a><button>Suivre</button></aside>',
        )
    )

    assert await _click(dom_page) == (False, [])


async def test_without_a_composer_nothing_is_clicked(dom_page):
    await dom_page.set_content(
        f"<main>{_HEADER}{_COMMENT}<button>Charger plus</button></main>"
    )

    assert await _click(dom_page) == (False, [])


async def test_without_a_commenter_nothing_is_clicked(dom_page):
    await dom_page.set_content(_page('<button id="more">Charger plus</button>'))

    assert await _click(dom_page) == (False, [])
