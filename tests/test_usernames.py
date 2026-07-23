"""Tests for LinkedIn vanity normalization (scraping/usernames.py)."""

import pytest

from linkedin_mcp_server.scraping.usernames import normalize_linkedin_username


class TestNormalizeLinkedinUsername:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("williamhgates", "williamhgates"),
            ("WilliamHGates", "WilliamHGates"),
            ("/in/williamhgates/", "williamhgates"),
            ("in/williamhgates", "williamhgates"),
            ("https://www.linkedin.com/in/williamhgates/", "williamhgates"),
            ("https://linkedin.com/in/williamhgates", "williamhgates"),
            ("https://www.linkedin.com/in/williamhgates/?trk=foo", "williamhgates"),
            ("//www.linkedin.com/in/jane-doe/", "jane-doe"),
            ("  stickerdaniel  ", "stickerdaniel"),
            ("a", "a"),
            ("ab", "ab"),
        ],
    )
    def test_accepts_valid_forms(self, value: str, expected: str):
        assert normalize_linkedin_username(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "   ",
            "me",
            "ME",
            "in",
            "https://example.com/in/evil/",
            "https://evil.com/in/williamhgates/",
            "/company/acme/",
            "foo/bar",
            "../etc",
            "has space",
            "has_underscore",
            "name!",
            "https://www.linkedin.com/in/",
            "https://www.linkedin.com/in/me/",
        ],
    )
    def test_rejects_invalid_forms(self, value: str | None):
        assert normalize_linkedin_username(value) is None
