"""Tests for recognizing and safely reporting proxy failures."""

import asyncio
from collections.abc import Callable
import re
from typing import Any
from urllib.parse import quote, quote_plus, unquote, unquote_plus

import pytest
from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.config.schema import AppConfig
from linkedin_mcp_server.core.exceptions import NetworkError, ProxyConnectionError
from linkedin_mcp_server.core.proxy_errors import (
    as_proxy_error,
    goto_reporting_proxy_errors,
    is_proxy_error,
    proxy_hint,
    raise_if_proxy_error,
    redact_proxy_credentials,
)


@pytest.fixture
def proxy_config(monkeypatch):
    config = AppConfig()
    config.browser.proxy_server = "http://gate.example:7000"
    config.browser.proxy_username = "user"
    config.browser.proxy_password = "s3cr3t p@ss"
    monkeypatch.setattr("linkedin_mcp_server.config.get_config", lambda: config)
    return config


class TestClassification:
    @pytest.mark.parametrize(
        "message",
        [
            "net::ERR_PROXY_CONNECTION_FAILED at https://www.linkedin.com/feed/",
            "net::ERR_TUNNEL_CONNECTION_FAILED",
            "Page.goto: net::ERR_PROXY_AUTH_REQUESTED",
            "net::ERR_SOCKS_CONNECTION_FAILED",
            "net::ERR_NO_SUPPORTED_PROXIES",
        ],
    )
    def test_proxy_failures_are_recognized(self, message):
        assert is_proxy_error(Exception(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            "net::ERR_TOO_MANY_REDIRECTS",
            "net::ERR_ABORTED",
            "Timeout 30000ms exceeded",
            "Executable doesn't exist at /path/to/chrome",
        ],
    )
    def test_unrelated_failures_are_not_claimed(self, message):
        assert is_proxy_error(Exception(message)) is False

    def test_an_existing_proxy_error_passes_through(self):
        original = ProxyConnectionError("already converted")
        assert is_proxy_error(original) is True
        assert as_proxy_error(original) is original


class TestConversion:
    def test_message_names_the_proxy(self, proxy_config):
        converted = as_proxy_error(Exception("net::ERR_PROXY_CONNECTION_FAILED"))
        assert "http://gate.example:7000" in str(converted)

    def test_it_stays_a_network_error(self):
        # Subclassing NetworkError keeps a proxy outage out of the auth paths
        # that would retire the stored profile.
        assert issubclass(ProxyConnectionError, NetworkError)

    def test_the_raw_cause_is_dropped(self, proxy_config):
        # The top-level handlers log the whole cause chain, which would put the
        # unredacted driver message back into the log.
        original = Exception("net::ERR_PROXY_CONNECTION_FAILED user:s3cr3t p@ss@gate")
        try:
            raise_if_proxy_error(original)
        except ProxyConnectionError as converted:
            assert converted.__cause__ is None
            assert "s3cr3t" not in str(converted)
        else:  # pragma: no cover - the call above must raise
            pytest.fail("expected a ProxyConnectionError")

    def test_non_proxy_errors_are_left_alone(self):
        raise_if_proxy_error(Exception("net::ERR_TOO_MANY_REDIRECTS"))  # no raise


class TestRedaction:
    def test_plain_password_is_masked(self, proxy_config):
        assert "s3cr3t" not in redact_proxy_credentials("saw s3cr3t p@ss in the URL")

    def test_percent_encoded_password_is_masked(self, proxy_config):
        # A password inside a URL appears percent-encoded.
        redacted = redact_proxy_credentials("http://user:s3cr3t%20p%40ss@gate:7000")
        assert "s3cr3t" not in redacted and "%40ss" not in redacted

    def test_userinfo_only_password_is_masked(self, monkeypatch):
        """The secret is in PROXY_SERVER and not in the username or password fields.

        Those fields are what redaction used to read, so a password that
        appears only as userinfo used to survive into the reported text.
        """
        user = "acct" + "zone9"
        secret = "s3" + "cr3t"
        server = f"http://{user}:{secret}@gate.example:7000"
        config = AppConfig()
        config.browser.proxy_server = server
        monkeypatch.setattr("linkedin_mcp_server.config.get_config", lambda: config)

        redacted = redact_proxy_credentials(f"failed via {server}")

        assert config.browser.proxy_username is None
        assert config.browser.proxy_password is None
        assert secret not in redacted
        assert user not in redacted
        assert "gate.example" in redacted

    def test_percent_encoded_userinfo_is_masked(self, monkeypatch):
        """urlsplit does not see a userinfo whose "@" is percent-encoded.

        The password is ``p@ss``, so the encoded form ``p%40ss`` does not
        contain the literal password. Masking only the decoded fields would
        leave the encoded password in the URL.
        """
        password = "p@" + "ss"
        encoded_password = quote(password, safe="")
        userinfo = quote(f"acct:{password}", safe="")
        server = f"http://{userinfo}%40gate.example:7000"
        config = AppConfig()
        config.browser.proxy_server = server
        monkeypatch.setattr("linkedin_mcp_server.config.get_config", lambda: config)

        redacted = redact_proxy_credentials(f"proxy {server} refused ({password})")

        assert password not in redacted
        assert encoded_password not in redacted
        assert "gate.example" in redacted

    def test_percent_encoded_password_inside_userinfo_is_masked(self, monkeypatch):
        password = "p@" + "ss"
        encoded_password = quote(password, safe="")
        server = f"http://acct:{encoded_password}@gate.example:7000"
        config = AppConfig()
        config.browser.proxy_server = server
        monkeypatch.setattr("linkedin_mcp_server.config.get_config", lambda: config)

        redacted = redact_proxy_credentials(f"saw {server} and {password}")

        assert password not in redacted
        assert encoded_password not in redacted
        assert "gate.example" in redacted

    def test_a_client_error_does_not_repeat_userinfo(self, monkeypatch):
        """The proxy address is part of the error. The userinfo in it is not."""
        user = "acct" + "zone9"
        secret = "s3" + "cr3t"
        config = AppConfig()
        config.browser.proxy_server = f"http://{user}:{secret}@gate.example:7000"
        monkeypatch.setattr("linkedin_mcp_server.config.get_config", lambda: config)

        converted = as_proxy_error(Exception("net::ERR_PROXY_CONNECTION_FAILED"))
        hint = proxy_hint()

        assert secret not in str(converted)
        assert user not in str(converted)
        assert "gate.example" in str(converted)
        assert secret not in hint
        assert user not in hint
        assert "gate.example" in hint

    def test_other_encodings_of_the_password_cannot_be_recovered(self, monkeypatch):
        """Lowercase hex and quote_plus are not ``quote`` of the decoded secret.

        A driver can write either hex case, and a space as ``+``. Masking only
        the stored userinfo and ``quote`` of the password leaves a form that
        ``unquote`` turns back into the password. The host stays.
        """
        password = "p:ss" + " word"
        quoted = quote(password, safe="")
        quoted_lower = re.sub(r"%[0-9A-F]{2}", lambda m: m.group(0).lower(), quoted)
        plus = quote_plus(password, safe="")
        plus_lower = re.sub(r"%[0-9A-F]{2}", lambda m: m.group(0).lower(), plus)
        assert quoted_lower != quoted
        assert plus not in (quoted, password)
        assert plus_lower not in (quoted, quoted_lower, plus)

        config = AppConfig()
        config.browser.proxy_server = f"http://acct:{quoted}@gate.example:7000"
        monkeypatch.setattr("linkedin_mcp_server.config.get_config", lambda: config)

        redacted = redact_proxy_credentials(
            f"driver quoted {quoted_lower}, plus {plus}, and {plus_lower} "
            "via gate.example"
        )

        assert password not in redacted
        assert quoted_lower not in redacted
        assert plus not in redacted
        assert plus_lower not in redacted
        assert password not in unquote(redacted)
        assert password not in unquote_plus(redacted)
        assert "gate.example" in redacted

    def test_a_password_fragment_is_not_a_secret(self, monkeypatch):
        """Only the credential and its encodings are masked, not a piece of them."""
        password = "p:ss" + " word"
        config = AppConfig()
        config.browser.proxy_server = "http://gate.example:7000"
        config.browser.proxy_password = password
        monkeypatch.setattr("linkedin_mcp_server.config.get_config", lambda: config)

        message = "reviewed the word list at gate.example"
        assert redact_proxy_credentials(message) == message

    def test_without_a_configured_password_nothing_changes(self, monkeypatch):
        monkeypatch.setattr(
            "linkedin_mcp_server.config.get_config", lambda: AppConfig()
        )
        assert redact_proxy_credentials("untouched") == "untouched"

    def test_reporting_survives_an_unreadable_config(self, monkeypatch):
        def explode():
            raise RuntimeError("config unavailable")

        monkeypatch.setattr("linkedin_mcp_server.config.get_config", explode)
        converted = as_proxy_error(Exception("net::ERR_PROXY_CONNECTION_FAILED"))
        assert "the configured proxy" in str(converted)


class TestProxyHint:
    """Auth failures name the proxy, because a wrong password looks like one.

    Verified against a local authenticating relay: bad credentials produce no
    proxy error code at all. Chromium keeps retrying the 407 challenge until
    the navigation times out, which is indistinguishable from a slow page, so
    the failure lands in the auth path however well the classifier works.
    """

    def test_hint_names_the_proxy_when_configured(self, proxy_config):
        assert "http://gate.example:7000" in proxy_hint()

    def test_hint_never_carries_the_password(self, proxy_config):
        assert "s3cr3t" not in proxy_hint()

    def test_no_hint_without_a_proxy(self, monkeypatch):
        monkeypatch.setattr(
            "linkedin_mcp_server.config.get_config", lambda: AppConfig()
        )
        assert proxy_hint() == ""


class TestAmbiguousAuthMarker:
    """Rejected proxy credentials, which Chromium reports without naming a proxy."""

    def test_counted_when_a_proxy_is_configured(self, proxy_config):
        assert is_proxy_error(Exception("net::ERR_INVALID_AUTH_CREDENTIALS")) is True

    def test_ignored_without_a_proxy(self, monkeypatch):
        # The same code covers a site's own HTTP auth, so claiming it with no
        # proxy configured would misreport an ordinary failure.
        monkeypatch.setattr(
            "linkedin_mcp_server.config.get_config", lambda: AppConfig()
        )
        assert is_proxy_error(Exception("net::ERR_INVALID_AUTH_CREDENTIALS")) is False


class TestGotoWrapper:
    """The navigations that run before any auth check are covered too."""

    async def test_proxy_failure_is_converted(self, proxy_config):
        page = _FakePage(Exception("net::ERR_PROXY_CONNECTION_FAILED"))
        with pytest.raises(ProxyConnectionError):
            await goto_reporting_proxy_errors(page, "https://www.linkedin.com/login")

    async def test_other_failures_propagate_unchanged(self, proxy_config):
        page = _FakePage(RuntimeError("net::ERR_ABORTED"))
        with pytest.raises(RuntimeError):
            await goto_reporting_proxy_errors(page, "https://www.linkedin.com/login")

    async def test_success_returns_the_response(self, proxy_config):
        page = _FakePage(None)
        assert await goto_reporting_proxy_errors(page, "https://example.com") == "ok"


class _FakePage:
    def __init__(self, error):
        self._error = error

    async def goto(self, url, **kwargs):
        if self._error:
            raise self._error
        return "ok"


class _Request:
    def __init__(self, *, navigation=True, frame=None):
        self.frame = frame
        self._navigation = navigation

    def is_navigation_request(self):
        return self._navigation


class _ClockPage:
    def __init__(self, events, *, finish_after, main_frame=None, error=None):
        self._events = events
        self._finish_after = finish_after
        self.main_frame = main_frame
        self._error = error
        self._listeners: list[tuple[str, Callable[..., Any]]] = []

    def on(self, event, handler):
        self._listeners.append((event, handler))

    def remove_listener(self, event, handler):
        self._listeners = [
            (found, callback)
            for found, callback in self._listeners
            if not (found == event and callback == handler)
        ]

    async def goto(self, url, **kwargs):
        timeout = kwargs.get("timeout", 30_000)

        async def body():
            if self._error is not None:
                raise self._error
            elapsed = 0.0
            for delay, request in self._events:
                await asyncio.sleep(delay - elapsed)
                elapsed = delay
                for event, handler in list(self._listeners):
                    if event == "request":
                        handler(request)
            await asyncio.sleep(self._finish_after)
            return "ok"

        if not timeout:
            return await body()
        return await asyncio.wait_for(body(), timeout / 1000)


class TestNavigationBudget:
    async def test_a_request_sent_after_the_old_clock_still_completes(self):
        page = _ClockPage([(0.25, _Request())], finish_after=0.02)
        assert (
            await goto_reporting_proxy_errors(
                page, "https://www.linkedin.com/feed/", timeout=100
            )
            == "ok"
        )

    async def test_the_budget_still_bounds_the_answer(self):
        page = _ClockPage([(0.01, _Request())], finish_after=0.25)
        with pytest.raises(PlaywrightTimeoutError, match="after the request was sent"):
            await goto_reporting_proxy_errors(
                page, "https://www.linkedin.com/feed/", timeout=100
            )

    async def test_a_request_that_is_never_sent_fails(self, monkeypatch):
        monkeypatch.setattr(
            "linkedin_mcp_server.core.proxy_errors.STARTUP_BUDGET_MS", 50
        )
        page = _ClockPage([(2.0, _Request())], finish_after=0.01)
        with pytest.raises(PlaywrightTimeoutError, match="before the request was sent"):
            await goto_reporting_proxy_errors(
                page, "https://www.linkedin.com/feed/", timeout=100
            )

    async def test_an_overall_timeout_bounds_the_wait_for_the_request(self):
        page = _ClockPage([(2.0, _Request())], finish_after=0.01)
        with pytest.raises(PlaywrightTimeoutError, match="before the request was sent"):
            await asyncio.wait_for(
                goto_reporting_proxy_errors(
                    page,
                    "https://www.linkedin.com/feed/",
                    timeout=100,
                    overall_timeout=100,
                ),
                1.0,
            )

    async def test_an_overall_timeout_bounds_the_answer_after_a_late_request(self):
        page = _ClockPage([(0.15, _Request())], finish_after=0.1)
        with pytest.raises(PlaywrightTimeoutError, match="after the request was sent"):
            await goto_reporting_proxy_errors(
                page,
                "https://www.linkedin.com/feed/",
                timeout=200,
                overall_timeout=200,
            )

    async def test_a_subresource_does_not_start_the_budget(self):
        page = _ClockPage(
            [(0.01, _Request(navigation=False)), (0.2, _Request())],
            finish_after=0.02,
        )
        assert (
            await goto_reporting_proxy_errors(
                page, "https://www.linkedin.com/feed/", timeout=150
            )
            == "ok"
        )

    async def test_an_iframe_navigation_does_not_start_the_budget(self):
        main = object()
        page = _ClockPage(
            [(0.01, _Request(frame=object())), (0.30, _Request(frame=main))],
            finish_after=0.02,
            main_frame=main,
        )
        assert (
            await goto_reporting_proxy_errors(
                page, "https://www.linkedin.com/feed/", timeout=200
            )
            == "ok"
        )


class TestUsernameIsTreatedAsSecret:
    """Provider usernames carry account, zone and session identity."""

    USERNAME = "brd-customer-acct1-zone-resi"

    def _config(self, monkeypatch):
        config = AppConfig()
        config.browser.proxy_server = "http://gate.example:7000"
        config.browser.proxy_username = self.USERNAME
        monkeypatch.setattr("linkedin_mcp_server.config.get_config", lambda: config)
        return config

    def test_username_is_masked(self, monkeypatch):
        self._config(monkeypatch)
        redacted = redact_proxy_credentials(f"http://{self.USERNAME}:pw@gate:7000")
        assert self.USERNAME not in redacted

    def test_username_absent_from_the_config_repr(self, monkeypatch):
        # cli_main logs the whole config at DEBUG level.
        config = self._config(monkeypatch)
        assert self.USERNAME not in repr(config)

    def test_hint_still_names_the_server(self, proxy_config):
        # The server holds no secret and is what you need to diagnose.
        assert "gate.example" in proxy_hint()
