"""Tests for the optional file export of a job tool's result.

The export is the only place these tools touch the local disk, and the caller
is an agent composing the path. Every refusal is therefore asserted twice: the
error, and that nothing was created or altered on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from linkedin_mcp_server.result_export import (
    OutputPathError,
    apply_output_mode,
    check_output_target,
    resolve_export_path,
)


def _result() -> dict[str, Any]:
    return {
        "url": "https://www.linkedin.com/jobs/search/?keywords=python",
        "sections": {"search_results": "Job A\nJob B"},
        "job_ids": ["1", "2"],
        "total": {"count": 2, "exact": True},
        "section_errors": {"search_results": {"error_type": "x"}},
    }


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def root(home: Path) -> Path:
    return home.resolve() / ".linkedin-mcp" / "exports"


def _files(directory: Path) -> list[str]:
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*"))


class TestModes:
    def test_display_returns_the_same_object_and_writes_nothing(self, home, root):
        result = _result()

        assert apply_output_mode(result, None, "display") is result
        assert apply_output_mode(result, "a.json", "display") is result
        assert not root.exists()

    def test_file_writes_json_and_returns_a_confirmation(self, home, root):
        result = _result()

        returned = apply_output_mode(result, "jobs.json", "file")

        target = root / "jobs.json"
        assert json.loads(target.read_text(encoding="utf-8")) == result
        assert returned == {
            "saved_path": str(target),
            "url": result["url"],
            "job_ids": ["1", "2"],
            "total": {"count": 2, "exact": True},
            "section_errors": result["section_errors"],
            "section_names": ["search_results"],
        }

    def test_file_confirmation_keeps_section_errors(self, home, root):
        """A file-only call must not hide that the scrape came back empty."""
        result = {
            "url": "u",
            "sections": {},
            "section_errors": {"a": {"error_type": "e"}},
        }

        returned = apply_output_mode(result, "e.json", "file")

        assert returned["section_errors"] == result["section_errors"]

    def test_both_returns_the_full_result_plus_the_path(self, home, root):
        result = _result()

        returned = apply_output_mode(result, "jobs.json", "both")

        assert returned == {**result, "saved_path": str(root / "jobs.json")}
        assert "saved_path" not in result

    def test_a_non_json_name_gets_a_text_rendering(self, home, root):
        apply_output_mode(_result(), "jobs.md", "file")

        text = (root / "jobs.md").read_text(encoding="utf-8")
        assert "URL: https://www.linkedin.com/jobs/search/?keywords=python" in text
        assert "## search_results\nJob A\nJob B" in text
        assert "JOB_IDS: 1, 2" in text
        assert "section_errors" not in text

    def test_non_ascii_content_round_trips_as_utf8(self, home, root):
        result = {"url": "u", "sections": {"s": "Zurich ü 中"}}

        apply_output_mode(result, "u.json", "file")

        assert json.loads((root / "u.json").read_bytes().decode("utf-8")) == result

    @pytest.mark.parametrize("mode", ["file", "both"])
    def test_a_writing_mode_requires_a_path(self, home, root, mode):
        with pytest.raises(OutputPathError, match="output_path is required"):
            apply_output_mode(_result(), None, mode)
        with pytest.raises(OutputPathError, match="output_path is required"):
            apply_output_mode(_result(), "", mode)

    def test_an_unknown_mode_is_refused(self, home, root):
        with pytest.raises(OutputPathError, match="output_mode must be one of"):
            apply_output_mode(_result(), "a.json", "append")
        assert not root.exists()


class TestPathSafety:
    def test_a_relative_path_resolves_under_the_export_directory(self, home, root):
        assert resolve_export_path("a/b/c.json") == root / "a" / "b" / "c.json"

    def test_parent_directories_are_created_private(self, home, root):
        apply_output_mode(_result(), "deep/er/jobs.json", "file")

        assert (root / "deep" / "er" / "jobs.json").is_file()
        assert (root / "deep" / "er").stat().st_mode & 0o777 == 0o700
        assert (root / "deep" / "er" / "jobs.json").stat().st_mode & 0o777 == 0o600

    def test_an_absolute_path_inside_the_root_is_accepted(self, home, root):
        assert resolve_export_path(str(root / "x.json")) == root / "x.json"

    def test_tilde_is_expanded_before_the_containment_check(self, home, root):
        assert resolve_export_path("~/.linkedin-mcp/exports/x.json") == root / "x.json"

    def test_tilde_outside_the_root_is_refused_not_taken_literally(self, home, root):
        with pytest.raises(OutputPathError, match="inside the LinkedIn MCP export"):
            resolve_export_path("~/notes.json")
        assert not (root / "~").exists()

    @pytest.mark.parametrize(
        "path", ["../outside.json", "a/../../outside.json", "/etc/passwd"]
    )
    def test_a_path_that_leaves_the_root_is_refused(self, home, root, path):
        with pytest.raises(OutputPathError, match="inside the LinkedIn MCP export"):
            apply_output_mode(_result(), path, "file")
        assert not (home / "outside.json").exists()
        assert not root.exists()

    @pytest.mark.parametrize("path", ["", "   ", "a\0b.json"])
    def test_an_empty_or_nul_path_is_refused(self, home, root, path):
        with pytest.raises(OutputPathError):
            resolve_export_path(path)

    @pytest.mark.parametrize("path", ["sub/", "sub" + os.sep])
    def test_a_trailing_separator_names_a_directory(self, home, root, path):
        with pytest.raises(OutputPathError, match="not a directory"):
            resolve_export_path(path)

    @pytest.mark.parametrize("path", [".", "sub/.."])
    def test_the_export_directory_itself_is_refused(self, home, root, path):
        with pytest.raises(OutputPathError, match="not the export directory"):
            resolve_export_path(path)

    def test_an_existing_directory_is_refused(self, home, root):
        (root / "d").mkdir(parents=True)

        with pytest.raises(OutputPathError, match="is a directory"):
            apply_output_mode(_result(), "d", "file")

        assert _files(root) == ["d"]

    def test_an_existing_file_is_never_overwritten(self, home, root):
        root.mkdir(parents=True)
        (root / "keep.json").write_text("precious")

        for mode in ("file", "both"):
            with pytest.raises(OutputPathError, match="never overwritten"):
                apply_output_mode(_result(), "keep.json", mode)

        assert (root / "keep.json").read_text() == "precious"
        assert _files(root) == ["keep.json"]

    def test_a_symlink_final_component_is_refused_even_when_dangling(self, home, root):
        root.mkdir(parents=True)
        outside = home / "outside.json"
        (root / "link.json").symlink_to(outside)

        with pytest.raises(OutputPathError):
            apply_output_mode(_result(), "link.json", "file")

        assert not outside.exists()

    def test_a_symlinked_directory_escaping_the_root_is_refused(self, home, root):
        root.mkdir(parents=True)
        outside = home / "outside"
        outside.mkdir()
        (root / "escape").symlink_to(outside, target_is_directory=True)

        with pytest.raises(OutputPathError, match="inside the LinkedIn MCP export"):
            apply_output_mode(_result(), "escape/job.json", "file")

        assert _files(outside) == []

    def test_a_symlinked_export_root_is_refused(self, home):
        linkedin = home / ".linkedin-mcp"
        outside = home / "outside"
        linkedin.mkdir()
        outside.mkdir()
        (linkedin / "exports").symlink_to(outside, target_is_directory=True)

        with pytest.raises(OutputPathError, match="must not be a symlink"):
            apply_output_mode(_result(), "job.json", "file")

        assert _files(outside) == []

    def test_a_file_where_a_parent_directory_is_needed_is_refused(self, home, root):
        root.mkdir(parents=True)
        (root / "blocker").write_text("x")

        with pytest.raises(OutputPathError, match="not a directory"):
            resolve_export_path("blocker/job.json")

    def test_the_check_touches_nothing(self, home, root):
        check_output_target("a/b.json", "file")

        assert not root.exists()


class TestAtomicWrite:
    def test_no_temporary_file_is_left_behind(self, home, root):
        apply_output_mode(_result(), "jobs.json", "file")

        assert _files(root) == ["jobs.json"]

    def test_a_failed_write_leaves_neither_target_nor_temp(
        self, home, root, monkeypatch
    ):
        monkeypatch.setattr(
            json, "dumps", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        with pytest.raises(RuntimeError):
            apply_output_mode(_result(), "jobs.json", "file")

        assert not (root / "jobs.json").exists()
        assert [f for f in _files(root) if f.endswith(".tmp")] == []

    def test_a_name_taken_after_the_check_is_still_not_overwritten(
        self, home, root, monkeypatch
    ):
        """The race between the pre-check and the write must not clobber."""
        import linkedin_mcp_server.result_export as export

        real = export.resolve_export_path

        def resolve_then_create(output_path: str) -> Path:
            path = real(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("raced in")
            return path

        monkeypatch.setattr(export, "resolve_export_path", resolve_then_create)
        root.mkdir(parents=True)

        with pytest.raises(OutputPathError, match="already exists"):
            export.apply_output_mode(_result(), "jobs.json", "file")

        assert (root / "jobs.json").read_text() == "raced in"
        assert _files(root) == ["jobs.json"]

    def test_a_filesystem_without_hard_links_falls_back_to_a_checked_replace(
        self, home, root, monkeypatch
    ):
        def no_link(*_args: Any, **_kwargs: Any) -> None:
            raise PermissionError("links unsupported")

        monkeypatch.setattr(os, "link", no_link)

        apply_output_mode(_result(), "jobs.json", "file")

        assert json.loads((root / "jobs.json").read_text())["job_ids"] == ["1", "2"]
        assert _files(root) == ["jobs.json"]
