"""Tests for the section contracts every scraping workflow returns."""

from typing import Any

import pytest

from linkedin_mcp_server.scraping import contracts
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    SEND_INTERRUPTED_WARNING,
    ExtractedSection,
    FilterValidationError,
    message_action_result,
    normalize_message,
    rate_limited_section_error,
    refuse_an_invalid_message,
    refuse_an_invalid_thread_message,
)

CONTROL_REASON = (
    "Message must not contain control characters other than line breaks "
    "(LF or CRLF). Tabs, a lone CR, DEL and every other C0 character are "
    "refused."
)


class TestRateLimitedSection:
    def test_the_sentinel_text_is_what_reaches_the_client(self):
        # Pinned as a literal on purpose. Every other assertion in the suite
        # compares a result against this same constant, so it moves with any
        # edit and none of them can see the message a client would read.
        assert RATE_LIMITED_SECTION_TEXT == (
            "[Rate limited] LinkedIn blocked this section. "
            "Try again later or request fewer sections."
        )

    def test_the_reported_error_repeats_the_sentinel_verbatim(self):
        # The tools compare a section's text against the sentinel and then
        # report this error, so the two drifting apart would describe a
        # section the caller never saw.
        assert rate_limited_section_error() == {
            "error_type": "rate_limit",
            "error_message": RATE_LIMITED_SECTION_TEXT,
        }


class TestExtractedSection:
    def test_a_section_without_an_error_carries_none(self):
        section = ExtractedSection(text="Bill Gates", references=[])

        assert section.error is None

    def test_an_error_is_kept_beside_the_text(self):
        section = ExtractedSection(
            text="", references=[], error=rate_limited_section_error()
        )

        assert section.text == ""
        assert section.error == rate_limited_section_error()


class TestFilterValidationError:
    def test_it_is_still_a_value_error(self):
        # Direct extractor callers catch ValueError; the tool wrappers catch
        # this subclass to surface the message past mask_error_details.
        assert issubclass(FilterValidationError, ValueError)


class TestMessageActionResult:
    def test_the_retry_contract_is_explicit_on_every_result(self):
        assert message_action_result(
            "https://www.linkedin.com/messaging/compose/",
            "sent",
            "Message submitted.",
            recipient_selected=True,
            sent=True,
            retry_safe=False,
        ) == {
            "url": "https://www.linkedin.com/messaging/compose/",
            "status": "sent",
            "message": "Message submitted.",
            "recipient_selected": True,
            "sent": True,
            "retry_safe": False,
        }

    def test_the_interruption_warning_names_duplicate_delivery(self):
        assert SEND_INTERRUPTED_WARNING == (
            "Message submission was interrupted while in flight. The send outcome "
            "is unknown; check the conversation before retrying, as a retry may "
            "deliver the message twice."
        )


class TestRefuseAnInvalidMessage:
    @pytest.mark.parametrize(
        "message",
        [
            f"First{chr(codepoint)}Second"
            for codepoint in (*range(32), 127)
            if codepoint != 10
        ],
        ids=[
            f"U+{codepoint:04X}" for codepoint in (*range(32), 127) if codepoint != 10
        ],
    )
    def test_every_c0_or_del_character_but_lf_is_refused(self, message: str):
        assert refuse_an_invalid_message("alice", message) == message_action_result(
            "https://www.linkedin.com/in/alice/",
            "invalid_message",
            CONTROL_REASON,
        )

    def test_the_control_refusal_names_what_is_allowed(self):
        # Pinned as a literal: the caller reads this to correct its input.
        assert refuse_an_invalid_message("alice", "a\tb")["message"] == (
            "Message must not contain control characters other than line "
            "breaks (LF or CRLF). Tabs, a lone CR, DEL and every other C0 "
            "character are refused."
        )

    @pytest.mark.parametrize(
        "message",
        ["Hi Ada,\n\nThanks!\nBest,\nBob", "Hi Ada,\r\n\r\nThanks!", "a\n"],
        ids=["lf", "crlf", "trailing-lf"],
    )
    def test_line_breaks_are_accepted(self, message: str):
        assert refuse_an_invalid_message("alice", message) is None

    @pytest.mark.parametrize(
        "message",
        ["a\r", "a\rb", "a\n\rb", "a\r\rb"],
        ids=["trailing", "inner", "after-lf", "double"],
    )
    def test_a_carriage_return_outside_crlf_is_refused(self, message: str):
        assert refuse_an_invalid_message("alice", message)["message"] == (
            CONTROL_REASON
        )

    @pytest.mark.parametrize("message", ["\n", "\r\n\r\n", " \n \n "])
    def test_line_breaks_alone_are_blank(self, message: str):
        assert refuse_an_invalid_message("alice", message)["message"] == (
            "Message must contain non-whitespace characters."
        )

    def test_whitespace_is_refused_before_normal_message_text(self):
        assert refuse_an_invalid_message("alice", "   ") == message_action_result(
            "https://www.linkedin.com/in/alice/",
            "invalid_message",
            "Message must contain non-whitespace characters.",
        )

    def test_safe_single_line_text_is_accepted(self):
        assert refuse_an_invalid_message("alice", "Hello, Alice!") is None

    def test_the_refusal_calls_the_owner_constructor_directly(self, monkeypatch):
        calls: list[tuple[str, str, str]] = []
        sentinel: dict[str, Any] = {"owner": "contracts"}

        def constructor(url: str, status: str, message: str) -> dict[str, Any]:
            calls.append((url, status, message))
            return sentinel

        monkeypatch.setattr(contracts, "message_action_result", constructor)

        assert refuse_an_invalid_message("alice", "") is sentinel
        assert calls == [
            (
                "https://www.linkedin.com/in/alice/",
                "invalid_message",
                "Message must contain non-whitespace characters.",
            )
        ]


class TestRefuseAnInvalidThreadMessage:
    THREAD_ID = "2-abc=="

    @pytest.mark.parametrize(
        ("message", "reason"),
        [
            ("  ", "Message must contain non-whitespace characters."),
            ("a\tb", CONTROL_REASON),
            ("a\rb", CONTROL_REASON),
        ],
        ids=["blank", "tab", "lone-cr"],
    )
    def test_it_refuses_like_a_profile_send_but_names_the_thread(self, message, reason):
        assert refuse_an_invalid_thread_message(self.THREAD_ID, message) == {
            "url": "https://www.linkedin.com/messaging/thread/2-abc==/",
            "status": "invalid_message",
            "message": reason,
            "recipient_selected": False,
            "sent": False,
            "retry_safe": True,
        }
        assert refuse_an_invalid_message("alice", message)["message"] == reason

    @pytest.mark.parametrize("message", ["Hello!", "Hello,\n\nBob"])
    def test_a_usable_message_is_not_refused(self, message):
        assert refuse_an_invalid_thread_message(self.THREAD_ID, message) is None


class TestNormalizeMessage:
    """The exact text a send types and then looks for (#441)."""

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("Hello!", "Hello!"),
            ("Hi Ada,\r\n\r\nThanks!\r\nBob", "Hi Ada,\n\nThanks!\nBob"),
            ("Hi Ada,\n\n\nThanks!", "Hi Ada,\n\n\nThanks!"),
            ("\n \nHi Ada,\nBob\n\n", "Hi Ada,\nBob"),
            ("Hi Ada,   \nBob ", "Hi Ada,\nBob"),
            ("Hi Ada,\n  - Monday\n  - Tuesday", "Hi Ada,\n  - Monday\n  - Tuesday"),
            ("  Hello", "  Hello"),
            ("Hi\n   \nBob", "Hi\n\nBob"),
        ],
        ids=[
            "single-line",
            "crlf",
            "paragraph-breaks-kept",
            "edge-blank-lines",
            "trailing-spaces",
            "indentation-kept",
            "leading-space-kept",
            "space-only-line-is-blank",
        ],
    )
    def test_normalization(self, message, expected):
        assert normalize_message(message) == expected

    def test_a_normalized_message_is_still_accepted_and_stable(self):
        message = normalize_message("Hi Ada, \r\n\r\nThanks!\r\n")

        assert refuse_an_invalid_message("alice", message) is None
        assert normalize_message(message) == message
