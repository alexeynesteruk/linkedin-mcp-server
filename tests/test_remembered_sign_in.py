"""Remembered-session interstitial and the Docker/CDP auth message."""

import asyncio
import unittest
from unittest.mock import patch

from patchright.async_api import Error as PlaywrightError

from linkedin_mcp_server.bootstrap import RuntimePolicy
from linkedin_mcp_server.core.auth import (
    is_remembered_sign_in,
    wait_for_remembered_sign_in,
)
from linkedin_mcp_server.dependencies import _docker_host_login_applies


def _config(cdp_url: str | None) -> object:
    browser = type("Browser", (), {"cdp_url": cdp_url})()
    return type("Config", (), {"browser": browser})()

_LOGIN = "https://www.linkedin.com/login/?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2F"
_UAS = "https://www.linkedin.com/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2F"
_BODY = (
    "We’re signing you in If you remain on this page, you’ll be signed in. "
    "Alexey Nesteruk k*****@gmail.com Cancel sign in"
)


class _Page:
    def __init__(self, frames: list[tuple[str, object]]) -> None:
        self._frames = frames
        self._index = 0

    @property
    def url(self) -> str:
        url, _body = self._frames[min(self._index, len(self._frames) - 1)]
        return url

    async def evaluate(self, _script: str) -> object:
        _url, body = self._frames[min(self._index, len(self._frames) - 1)]
        self._index += 1
        if isinstance(body, BaseException):
            raise body
        return body


class RememberedSignInTest(unittest.TestCase):
    def test_typographic_apostrophe_on_login_routes(self) -> None:
        self.assertTrue(is_remembered_sign_in(_LOGIN, _BODY))
        self.assertTrue(is_remembered_sign_in(_UAS, _BODY))

    def test_straight_apostrophe_matches_too(self) -> None:
        straight = _BODY.replace("’", "'")
        self.assertTrue(is_remembered_sign_in(_LOGIN, straight))

    def test_other_auth_routes_are_not_the_interstitial(self) -> None:
        for url in (
            "https://www.linkedin.com/checkpoint/challenge/",
            "https://www.linkedin.com/challenge/",
            "https://www.linkedin.com/authwall",
            "https://www.linkedin.com/login/login-restriction/",
            "https://www.linkedin.com/feed/",
            "https://example.com/login/",
        ):
            self.assertFalse(is_remembered_sign_in(url, _BODY), url)

    def test_password_login_copy_does_not_match(self) -> None:
        self.assertFalse(
            is_remembered_sign_in(
                _LOGIN, "Sign in Join now Email Password Forgot password?"
            )
        )

    def test_wait_follows_the_redirect(self) -> None:
        page = _Page(
            [
                (_LOGIN, _BODY),
                ("https://www.linkedin.com/feed/foryou/", "Home"),
            ]
        )

        async def _no_sleep(_seconds: float) -> None:
            return None

        with patch(
            "linkedin_mcp_server.core.auth.asyncio.sleep",
            side_effect=_no_sleep,
        ):
            finished = asyncio.run(wait_for_remembered_sign_in(page, timeout=15_000))
        self.assertTrue(finished)

    def test_wait_does_not_linger_on_a_real_login_form(self) -> None:
        page = _Page([(_LOGIN, "Email Password Sign in")])
        finished = asyncio.run(wait_for_remembered_sign_in(page, timeout=15_000))
        self.assertFalse(finished)
        self.assertEqual(page._index, 1)

    def test_unreadable_document_before_a_match_does_not_wait(self) -> None:
        page = _Page([(_LOGIN, PlaywrightError("Execution context was destroyed"))])
        finished = asyncio.run(wait_for_remembered_sign_in(page, timeout=15_000))
        self.assertFalse(finished)

    def test_docker_without_cdp_uses_the_host_login_sentence(self) -> None:
        with (
            patch(
                "linkedin_mcp_server.dependencies.get_runtime_policy",
                return_value=RuntimePolicy.DOCKER,
            ),
            patch(
                "linkedin_mcp_server.dependencies.get_config",
                return_value=_config(None),
            ),
        ):
            self.assertTrue(_docker_host_login_applies())

    def test_docker_with_cdp_keeps_the_attached_chrome_message(self) -> None:
        with (
            patch(
                "linkedin_mcp_server.dependencies.get_runtime_policy",
                return_value=RuntimePolicy.DOCKER,
            ),
            patch(
                "linkedin_mcp_server.dependencies.get_config",
                return_value=_config("http://127.0.0.1:9222"),
            ),
        ):
            self.assertFalse(_docker_host_login_applies())

    def test_managed_runtime_does_not_use_the_docker_sentence(self) -> None:
        with (
            patch(
                "linkedin_mcp_server.dependencies.get_runtime_policy",
                return_value=RuntimePolicy.MANAGED,
            ),
            patch(
                "linkedin_mcp_server.dependencies.get_config",
                side_effect=AssertionError("config should not be read"),
            ),
        ):
            self.assertFalse(_docker_host_login_applies())


if __name__ == "__main__":
    unittest.main()
