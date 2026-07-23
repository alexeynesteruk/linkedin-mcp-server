"""Tests for empty-scrape soft guards."""

from __future__ import annotations

from unittest.mock import patch

from linkedin_mcp_server.scrape_guards import (
    annotate_empty_scrape_result,
    empty_critical_sections,
)


class TestEmptyCriticalSections:
    def test_missing_sections_key(self):
        assert empty_critical_sections({}) == ["sections"]

    def test_empty_sections_dict(self):
        assert empty_critical_sections({"sections": {}}) == ["sections"]

    def test_required_section_missing(self):
        result = {"sections": {"other": "x" * 50}}
        assert empty_critical_sections(result, required_sections=("inbox",)) == [
            "inbox"
        ]

    def test_required_section_whitespace_only(self):
        result = {"sections": {"inbox": "   \n  "}}
        assert empty_critical_sections(result, required_sections=("inbox",)) == [
            "inbox"
        ]

    def test_required_section_present(self):
        result = {"sections": {"inbox": "Conversation list\n" + ("x" * 40)}}
        assert empty_critical_sections(result, required_sections=("inbox",)) == []

    def test_any_section_non_empty_passes(self):
        result = {
            "sections": {
                "main_profile": "Name\nTitle\n" + ("y" * 40),
                "experience": "",
            }
        }
        assert empty_critical_sections(result) == []

    def test_all_sections_empty_lists_names(self):
        result = {"sections": {"a": "", "b": "  "}}
        assert empty_critical_sections(result) == ["a", "b"]


class TestAnnotateEmptyScrapeResult:
    def test_noop_when_content_present(self):
        result = {
            "url": "https://www.linkedin.com/messaging/",
            "sections": {"inbox": "Messaging\n" + ("z" * 40)},
        }
        out = annotate_empty_scrape_result(
            result, tool_name="get_inbox", required_sections=("inbox",)
        )
        assert out is result
        assert "empty_scrape" not in out
        assert "section_errors" not in out

    def test_annotates_and_keeps_trace(self):
        result = {
            "url": "https://www.linkedin.com/messaging/",
            "sections": {},
        }
        with (
            patch(
                "linkedin_mcp_server.scrape_guards.mark_trace_for_retention",
                return_value=None,
            ) as mark,
            patch(
                "linkedin_mcp_server.scrape_guards.build_issue_diagnostics",
                return_value={
                    "issue_template_path": "/tmp/issue.md",
                    "runtime": {"trace_dir": "/tmp/trace"},
                },
            ),
        ):
            out = annotate_empty_scrape_result(
                result, tool_name="get_inbox", required_sections=("inbox",)
            )

        mark.assert_called_once()
        assert out["empty_scrape"] is True
        assert "inbox" in out["section_errors"]
        assert out["section_errors"]["inbox"]["error_type"] == "EmptyScrapeSection"
        assert out["section_errors"]["inbox"]["issue_template_path"] == "/tmp/issue.md"
        assert any("no usable content" in w for w in out["warnings"])
