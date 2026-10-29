"""Optionally write a job tool's result to a file under the export directory.

The tools return their result to the caller as before; ``output_path`` and
``output_mode`` add a copy on disk. Writing is confined to
``~/.linkedin-mcp/exports`` because the caller of a tool is an agent, and a path
it composes must never be able to reach an arbitrary file: a relative path
resolves under that directory, an absolute one must already be inside it, and
``~`` is expanded before that comparison rather than after.

A file is never replaced. A name that already exists is refused, so a repeated
call cannot silently destroy an earlier export, and the check is made twice:
before the browser is started, so a bad path costs no scrape, and again at
write time, where it is enforced by a hard link that fails if the name exists.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from linkedin_mcp_server.common_utils import secure_mkdir
from linkedin_mcp_server.core.exceptions import InvalidReferenceError
from linkedin_mcp_server.scrape_guards import annotate_empty_scrape_result

OutputMode = Literal["display", "file", "both"]
_OUTPUT_MODES = ("display", "file", "both")
_FILE_MODE = 0o600
# Keys a file-only confirmation carries over. Anything that says the result is
# incomplete or empty must survive, or "file" would hide a failed scrape.
_CONFIRMATION_KEYS = (
    "url",
    "job_ids",
    "total",
    "promoted_job_ids",
    "section_errors",
    "empty_scrape",
    "warnings",
)


class OutputPathError(InvalidReferenceError):
    """The requested export path is unusable; the message says how to fix it."""


def export_root() -> Path:
    """The export directory, refusing one that is a symlink."""
    declared = Path.home().resolve() / ".linkedin-mcp" / "exports"
    resolved = declared.resolve()
    if resolved != declared:
        raise OutputPathError("LinkedIn MCP export directory must not be a symlink")
    return resolved


def resolve_export_path(output_path: str) -> Path:
    """Return the file *output_path* names, or raise :class:`OutputPathError`.

    Touches nothing on disk. Refuses an empty path, a NUL byte, a trailing
    separator (a directory), the export directory itself, a path that leaves the
    export directory once ``~`` is expanded and symlinks are resolved, a final
    component that is a symlink, an existing name of any kind, and a parent that
    exists but is not a directory.
    """
    if not output_path or not output_path.strip():
        raise OutputPathError("output_path must not be empty")
    if "\0" in output_path:
        raise OutputPathError("output_path must not contain a NUL byte")
    if output_path.endswith(("/", os.sep)):
        raise OutputPathError("output_path must name a file, not a directory")

    root = export_root()
    requested = Path(output_path).expanduser()
    candidate = requested if requested.is_absolute() else root / requested
    path = candidate.resolve()
    if not path.is_relative_to(root):
        raise OutputPathError(
            f"output_path must be inside the LinkedIn MCP export directory ({root})"
        )
    if path == root:
        raise OutputPathError("output_path must name a file, not the export directory")
    if candidate.is_symlink():
        raise OutputPathError("output_path must not be a symlink")
    if os.path.lexists(path):
        if path.is_dir():
            raise OutputPathError(f"output_path {path} is a directory; name a file")
        raise OutputPathError(
            f"output_path {path} already exists; existing files are never "
            "overwritten, choose another name"
        )
    for parent in path.parents:
        if parent == root:
            break
        if os.path.lexists(parent) and not parent.is_dir():
            raise OutputPathError(f"{parent} exists and is not a directory")
    return path


def check_output_target(output_path: str | None, output_mode: str) -> None:
    """Refuse a bad export request before any browser work; write nothing."""
    if output_mode not in _OUTPUT_MODES:
        raise OutputPathError(
            f"output_mode must be one of {', '.join(_OUTPUT_MODES)}; "
            f"got {output_mode!r}"
        )
    if output_mode == "display":
        return
    if not output_path:
        raise OutputPathError("output_path is required when output_mode is not display")
    resolve_export_path(output_path)


def _render_result_text(result: dict[str, Any]) -> str:
    """Readable plain text for a non-JSON export: url, sections and job ids."""
    parts: list[str] = []
    url = result.get("url")
    if url:
        parts.append(f"URL: {url}")
    for name, text in (result.get("sections") or {}).items():
        parts.append(f"\n## {name}\n{text}")
    job_ids = result.get("job_ids")
    if job_ids:
        parts.append("\nJOB_IDS: " + ", ".join(job_ids))
    return "\n".join(parts) + "\n"


def _write_new_file(path: Path, content: str) -> None:
    """Create *path* atomically with *content*; raise if the name is taken.

    The content is written to a temporary file beside the target, then published
    with ``os.link``, which is the rename that refuses to replace. The temporary
    name is removed either way, so a failure leaves nothing behind. A filesystem
    without hard links uses exclusive creation instead; publication there is
    not atomic, but still never replaces an existing export.
    """
    secure_mkdir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(tmp, _FILE_MODE)
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise OutputPathError(
                f"output_path {path} already exists; existing files are never "
                "overwritten, choose another name"
            ) from None
        except OSError:
            try:
                target_fd = os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE
                )
            except FileExistsError:
                raise OutputPathError(
                    f"output_path {path} already exists; existing files are never "
                    "overwritten, choose another name"
                ) from None
            identity = os.fstat(target_fd)
            try:
                with os.fdopen(target_fd, "w", encoding="utf-8") as handle:
                    handle.write(content)
                    handle.flush()
            except BaseException:
                # Close before unlinking so Windows can remove the partial
                # file; leave a replacement owned by another writer alone.
                try:
                    current = path.stat()
                    if (current.st_dev, current.st_ino) == (
                        identity.st_dev,
                        identity.st_ino,
                    ):
                        path.unlink()
                except OSError:
                    pass
                raise
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def apply_output_mode(
    result: dict[str, Any],
    output_path: str | None,
    output_mode: str,
    *,
    tool_name: str = "job export",
) -> dict[str, Any]:
    """Optionally write *result* to disk and shape what the caller gets back.

    - ``display``: return *result* itself and write nothing; ``output_path`` is
      ignored.
    - ``file``: write, and return a compact confirmation (``saved_path``,
      ``section_names`` and whichever of url, job_ids, total, promoted_job_ids
      and section_errors the result had, including empty-scrape warnings).
    - ``both``: write, and return *result* with ``saved_path`` added.

    A ``.json`` name receives the whole result; any other name receives a text
    rendering of url, sections and job ids only.
    """
    check_output_target(output_path, output_mode)
    if output_mode == "display":
        return result
    # File mode removes `sections` before middleware sees the return value.
    # Diagnose before exporting, or an empty scrape looks like a successful save.
    result = annotate_empty_scrape_result(dict(result), tool_name=tool_name)
    assert output_path is not None
    path = resolve_export_path(output_path)
    if path.suffix == ".json":
        content = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        content = _render_result_text(result)
    _write_new_file(path, content)

    if output_mode == "both":
        return {**result, "saved_path": str(path)}
    confirmation: dict[str, Any] = {"saved_path": str(path)}
    for key in _CONFIRMATION_KEYS:
        if key in result:
            confirmation[key] = result[key]
    confirmation["section_names"] = sorted((result.get("sections") or {}).keys())
    return confirmation
