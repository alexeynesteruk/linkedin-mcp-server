"""Tests for the empty-scrape annotation."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP

from linkedin_mcp_server.scrape_guards import (
    EMPTY_IS_AN_ANSWER,
    EmptyScrapeMiddleware,
    annotate_empty_scrape_result,
    is_unexplained_empty,
)
from linkedin_mcp_server.server import create_mcp_server
from linkedin_mcp_server.server_role import ServerRole

URL = "https://www.linkedin.com/feed/"


@pytest.fixture(autouse=True)
def diagnostics():
    """Keep issue reports out of the real profile directory."""
    with patch(
        "linkedin_mcp_server.scrape_guards.build_issue_diagnostics",
        return_value={"issue_template_path": "/tmp/issue.md", "trace_dir": "/tmp/t"},
    ) as build:
        yield build


class TestIsUnexplainedEmpty:
    @pytest.mark.parametrize(
        "sections", [{}, {"feed": ""}, {"feed": "  \n\t "}, {"a": "", "b": " "}]
    )
    def test_no_text_and_no_error_is_unexplained(self, sections):
        assert is_unexplained_empty({"url": URL, "sections": sections})

    def test_any_text_is_content(self):
        assert not is_unexplained_empty(
            {"url": URL, "sections": {"a": "", "b": "Short"}}
        )

    def test_a_short_real_section_is_not_flagged(self):
        """No length floor: a two-line contact overlay is real content."""
        assert not is_unexplained_empty(
            {"url": URL, "sections": {"contact_info": "Email\nada@x.io"}}
        )

    def test_an_existing_error_explains_the_emptiness(self):
        assert not is_unexplained_empty(
            {"url": URL, "sections": {}, "section_errors": {"feed": {"e": 1}}}
        )

    def test_job_ids_are_content_of_their_own(self):
        assert not is_unexplained_empty({"url": URL, "sections": {}, "job_ids": ["1"]})

    @pytest.mark.parametrize("result", [{}, {"url": URL}, {"sections": None}])
    def test_results_without_a_sections_dict_are_not_scrapes(self, result):
        assert not is_unexplained_empty(result)


class TestAnnotate:
    def test_content_is_returned_untouched(self, diagnostics):
        result = {"url": URL, "sections": {"feed": "post text"}}
        assert annotate_empty_scrape_result(result, tool_name="get_feed") == {
            "url": URL,
            "sections": {"feed": "post text"},
        }
        diagnostics.assert_not_called()

    def test_an_empty_result_gains_error_flag_warning_and_trace(self, diagnostics):
        result = annotate_empty_scrape_result(
            {"url": URL, "sections": {}}, tool_name="get_feed"
        )

        assert result["empty_scrape"] is True
        entry = result["section_errors"]["sections"]
        assert entry["error_type"] == "EmptyScrapeSection"
        assert entry["issue_template_path"] == "/tmp/issue.md"
        assert "not evidence that there is nothing" in entry["error_message"]
        assert result["warnings"] == [entry["error_message"]]
        assert result["sections"] == {}
        kwargs = diagnostics.call_args.kwargs
        assert kwargs["context"] == "get_feed"
        assert kwargs["target_url"] == URL

    def test_blank_sections_are_named(self):
        result = annotate_empty_scrape_result(
            {"url": URL, "sections": {"inbox": " ", "a": ""}}, tool_name="get_inbox"
        )
        assert sorted(result["section_errors"]) == ["a", "inbox"]

    def test_a_diagnostics_failure_still_annotates(self, diagnostics):
        diagnostics.side_effect = OSError("disk full")
        result = annotate_empty_scrape_result(
            {"url": URL, "sections": {}}, tool_name="get_feed"
        )
        assert result["empty_scrape"] is True
        assert result["section_errors"]["sections"]["error_type"] == (
            "EmptyScrapeSection"
        )


def _server(results: dict[str, dict[str, Any]], tags: dict[str, set[str]]) -> FastMCP:
    mcp = FastMCP("test")
    mcp.add_middleware(EmptyScrapeMiddleware())
    for name, result in results.items():

        def make(value: dict[str, Any]):
            async def tool() -> dict[str, Any]:
                return dict(value)

            return tool

        mcp.tool(make(result), name=name, tags=tags[name])
    return mcp


EMPTY = {"url": URL, "sections": {}}


class TestMiddleware:
    async def test_a_scraping_tool_with_empty_sections_is_annotated(self):
        mcp = _server({"get_feed": EMPTY}, {"get_feed": {"feed", "scraping"}})

        result = await mcp.call_tool("get_feed", {})

        assert result.structured_content is not None
        assert result.structured_content["empty_scrape"] is True
        assert "sections" in result.structured_content["section_errors"]
        # The text block a client reads carries the same annotation.
        text = result.content[0].text  # type: ignore[union-attr]
        assert json.loads(text) == result.structured_content

    async def test_a_search_tool_is_covered_by_its_tag(self):
        mcp = _server({"search_x": EMPTY}, {"search_x": {"search"}})
        result = await mcp.call_tool("search_x", {})
        assert result.structured_content["empty_scrape"] is True  # type: ignore[index]

    async def test_get_pending_invitations_empty_is_an_answer(self):
        assert "get_pending_invitations" in EMPTY_IS_AN_ANSWER
        mcp = _server(
            {"get_pending_invitations": EMPTY},
            {"get_pending_invitations": {"network", "scraping"}},
        )
        result = await mcp.call_tool("get_pending_invitations", {})
        assert result.structured_content == EMPTY

    async def test_the_linkedin_alias_of_an_exempt_tool_is_exempt_too(self):
        mcp = _server(
            {"linkedin_get_pending_invitations": EMPTY},
            {"linkedin_get_pending_invitations": {"scraping"}},
        )
        result = await mcp.call_tool("linkedin_get_pending_invitations", {})
        assert result.structured_content == EMPTY

    async def test_an_alias_of_a_scraping_tool_is_annotated_under_its_own_name(
        self, diagnostics
    ):
        mcp = _server({"linkedin_get_feed": EMPTY}, {"linkedin_get_feed": {"scraping"}})
        await mcp.call_tool("linkedin_get_feed", {})
        assert diagnostics.call_args.kwargs["context"] == "get_feed"

    async def test_an_untagged_tool_is_never_annotated(self):
        mcp = _server({"linkedin_health": EMPTY}, {"linkedin_health": {"meta"}})
        result = await mcp.call_tool("linkedin_health", {})
        assert result.structured_content == EMPTY

    async def test_an_action_tool_is_never_annotated(self):
        mcp = _server({"send_message": EMPTY}, {"send_message": {"actions"}})
        result = await mcp.call_tool("send_message", {})
        assert result.structured_content == EMPTY

    async def test_a_result_that_explains_itself_is_left_alone(self):
        explained = {
            "url": URL,
            "sections": {},
            "section_errors": {"feed": {"error_type": "rate_limit"}},
        }
        mcp = _server({"get_feed": explained}, {"get_feed": {"scraping"}})
        result = await mcp.call_tool("get_feed", {})
        assert result.structured_content == explained

    async def test_a_result_without_sections_is_left_alone(self):
        sidebar = {"url": URL, "sidebar_profiles": {}}
        mcp = _server(
            {"get_sidebar_profiles": sidebar}, {"get_sidebar_profiles": {"scraping"}}
        )
        result = await mcp.call_tool("get_sidebar_profiles", {})
        assert result.structured_content == sidebar

    async def test_a_tool_that_returns_content_is_left_alone(self, diagnostics):
        ok = {"url": URL, "sections": {"feed": "posts"}}
        mcp = _server({"get_feed": ok}, {"get_feed": {"scraping"}})
        result = await mcp.call_tool("get_feed", {})
        assert result.structured_content == ok
        diagnostics.assert_not_called()

    async def test_a_failing_check_never_breaks_the_result(self, diagnostics):
        mcp = _server({"get_feed": EMPTY}, {"get_feed": {"scraping"}})
        with patch(
            "linkedin_mcp_server.scrape_guards.annotate_empty_scrape_result",
            side_effect=RuntimeError("boom"),
        ):
            result = await mcp.call_tool("get_feed", {})
        assert result.structured_content == EMPTY


class TestRegistration:
    @pytest.mark.parametrize("role", list(ServerRole))
    def test_only_roles_that_run_tools_have_it(self, role):
        mcp = _server_for(role)
        has = any(isinstance(m, EmptyScrapeMiddleware) for m in mcp.middleware)
        assert has == role.drives_browser

    async def test_the_real_get_feed_tool_is_annotated_end_to_end(self):
        """Through the registered tool, not a stand-in: empty feed text."""
        from linkedin_mcp_server.scraping.contracts import ExtractedSection

        extractor = MagicMock()
        extractor.extract_feed = AsyncMock(
            return_value=ExtractedSection(text="", references=[])
        )
        mcp = create_mcp_server()
        with patch(
            "linkedin_mcp_server.tools.feed.get_ready_extractor",
            AsyncMock(return_value=extractor),
        ):
            result = await mcp.call_tool("get_feed", {})
        assert result.structured_content is not None
        assert result.structured_content["empty_scrape"] is True
        assert result.structured_content["sections"] == {}

    async def test_the_real_pending_invitations_tool_stays_unannotated(self):
        extractor = MagicMock()
        extractor.get_pending_invitations = AsyncMock(
            return_value={"url": URL, "sections": {}}
        )
        mcp = create_mcp_server()
        with patch(
            "linkedin_mcp_server.tools.network.get_ready_extractor",
            AsyncMock(return_value=extractor),
        ):
            result = await mcp.call_tool("get_pending_invitations", {})
        assert result.structured_content == {"url": URL, "sections": {}}


def _server_for(role: ServerRole) -> FastMCP:
    from linkedin_mcp_server.server_role import reset_process_role_for_testing

    reset_process_role_for_testing()
    if role is ServerRole.PROXY:
        return create_mcp_server(role=role, proxy_backend=MagicMock())
    return create_mcp_server(role=role)
