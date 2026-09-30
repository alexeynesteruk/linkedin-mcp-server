"""Best-effort trace capture with on-error retention."""

from __future__ import annotations

import itertools
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Literal

from linkedin_mcp_server.common_utils import secure_mkdir, slugify_fragment
from linkedin_mcp_server.exceptions import ProfileRootRefusedError
from linkedin_mcp_server.session_state import (
    _owned,
    auth_root_dir,
    get_source_profile_dir,
)

logger = logging.getLogger(__name__)

TraceMode = Literal["off", "on_error", "always"]

_TRACE_COUNTER = itertools.count(1)
_TRACE_DIR: Path | None = None
_TRACE_KEEP = False
_EXPLICIT_TRACE_DIR = False

#: Run directories are made by ``tempfile.mkdtemp(prefix="run-")``: the prefix
#: and eight characters from its ``[a-z0-9_]`` alphabet. Pruning matches exactly
#: that shape, so a file or directory someone else put in the trace root is
#: never a candidate.
_RUN_PREFIX = "run-"
_RUN_DIR_NAME = re.compile(r"^run-[a-z0-9_]{8}$")

#: Retention for ``trace-runs``: the newest ``_TRACE_RUNS_KEEP`` runs survive,
#: and of those only the ones younger than ``_TRACE_RUNS_MAX_AGE_DAYS``.
_TRACE_RUNS_KEEP = 50
_TRACE_RUNS_MAX_AGE_DAYS = 14.0


def _trace_mode() -> TraceMode:
    raw = os.getenv("LINKEDIN_TRACE_MODE", "").strip().lower()
    if raw in {"off", "false", "0", "no"}:
        return "off"
    if raw in {"always", "keep", "persist"}:
        return "always"
    return "on_error"


def _trace_root() -> Path:
    source_profile = _safe_source_profile_dir()
    root = auth_root_dir(source_profile) / "trace-runs"
    secure_mkdir(root)
    return root


def trace_enabled() -> bool:
    return (
        bool(os.getenv("LINKEDIN_DEBUG_TRACE_DIR", "").strip())
        or _trace_mode() != "off"
    )


def get_trace_dir() -> Path | None:
    global _TRACE_DIR, _EXPLICIT_TRACE_DIR

    explicit = os.getenv("LINKEDIN_DEBUG_TRACE_DIR", "").strip()
    if explicit:
        _EXPLICIT_TRACE_DIR = True
        if _TRACE_DIR is None:
            _TRACE_DIR = Path(explicit).expanduser().resolve()
        return _TRACE_DIR

    if _trace_mode() == "off":
        return None

    if _TRACE_DIR is None:
        _TRACE_DIR = Path(
            tempfile.mkdtemp(
                prefix=_RUN_PREFIX,
                dir=_trace_root(),
            )
        ).resolve()
        _prune_trace_runs_best_effort(_TRACE_DIR)
    return _TRACE_DIR


def _prune_trace_runs_best_effort(current: Path) -> None:
    try:
        prune_trace_runs(current.parent, keep_dir=current)
    except Exception:
        logger.debug("Trace-run pruning skipped", exc_info=True)


def prune_trace_runs(
    trace_root: Path,
    *,
    keep_dir: Path | None = None,
    keep: int = _TRACE_RUNS_KEEP,
    max_age_days: float = _TRACE_RUNS_MAX_AGE_DAYS,
    now: float | None = None,
) -> list[Path]:
    """Delete old ``run-*`` directories from *trace_root*; return what went.

    Bounded so retained traces (``on_error`` keeps one per failing run) cannot
    accumulate forever. A run survives only if it is among the newest *keep* and
    younger than *max_age_days*; *keep_dir*, this process's own run, always
    survives.

    Only deletes what this module made: a real (not symlinked) directory named
    exactly like ``tempfile.mkdtemp(prefix="run-")`` output, directly inside
    ``<auth root>/trace-runs``. The auth root has to pass the same ownership
    check as every other destructive operation (``session_state._owned``); a
    root this server cannot prove it owns is left untouched. Symlinks are
    neither followed nor removed.
    """
    try:
        source_profile = _owned(_safe_source_profile_dir())
    except ProfileRootRefusedError:
        logger.debug("Not pruning trace runs: profile root is not owned")
        return []
    expected_root = auth_root_dir(source_profile) / "trace-runs"
    if trace_root.is_symlink() or trace_root.resolve() != expected_root.resolve():
        return []

    current = time.time() if now is None else now
    runs: list[tuple[float, Path]] = []
    with os.scandir(trace_root) as entries:
        for entry in entries:
            if not _RUN_DIR_NAME.match(entry.name):
                continue
            try:
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    continue
                runs.append((entry.stat(follow_symlinks=False).st_mtime, Path(entry)))
            except OSError:
                continue

    runs.sort(key=lambda run: run[0], reverse=True)
    cutoff = current - max_age_days * 86400
    removed: list[Path] = []
    for rank, (mtime, path) in enumerate(runs):
        if keep_dir is not None and path.resolve() == keep_dir.resolve():
            continue
        if rank < keep and mtime >= cutoff:
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            logger.debug("Could not prune trace run %s", path, exc_info=True)
            continue
        removed.append(path)
    return removed


def mark_trace_for_retention() -> Path | None:
    global _TRACE_KEEP
    trace_dir = get_trace_dir()
    if trace_dir is not None:
        secure_mkdir(trace_dir)
        _TRACE_KEEP = True
    return trace_dir


def should_keep_traces() -> bool:
    return _EXPLICIT_TRACE_DIR or _TRACE_KEEP or _trace_mode() == "always"


def cleanup_trace_dir() -> None:
    global _TRACE_DIR, _TRACE_KEEP, _EXPLICIT_TRACE_DIR

    trace_dir = _TRACE_DIR
    if trace_dir is None or should_keep_traces():
        return
    try:
        shutil.rmtree(trace_dir)
    except OSError:
        return
    _TRACE_DIR = None
    _TRACE_KEEP = False
    _EXPLICIT_TRACE_DIR = False


def reset_trace_state_for_testing() -> None:
    global _TRACE_COUNTER, _TRACE_DIR, _TRACE_KEEP, _EXPLICIT_TRACE_DIR
    _TRACE_COUNTER = itertools.count(1)
    _TRACE_DIR = None
    _TRACE_KEEP = False
    _EXPLICIT_TRACE_DIR = False


def _slugify_step(step: str) -> str:
    return slugify_fragment(step)


def _safe_source_profile_dir() -> Path:
    try:
        return get_source_profile_dir()
    except Exception:
        return Path("~/.linkedin-mcp/profile").expanduser()


async def record_page_trace(
    page: Any, step: str, *, extra: dict[str, Any] | None = None
) -> None:
    """Persist a screenshot and basic page state when trace capture is enabled."""
    trace_dir = get_trace_dir()
    if trace_dir is None:
        return

    secure_mkdir(trace_dir)
    screenshot_dir = trace_dir / "screens"
    secure_mkdir(screenshot_dir)
    step_id = next(_TRACE_COUNTER)
    slug = _slugify_step(step) or "step"

    try:
        title = await page.title()
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        title = f"<error: {exc}>"

    try:
        body_text = await page.evaluate("() => document.body?.innerText || ''")
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        body_text = f"<error: {exc}>"

    if not isinstance(body_text, str):
        body_text = ""

    try:
        remember_me = (await page.locator("#rememberme-div").count()) > 0
    except Exception:  # pragma: no cover - best effort diagnostics
        remember_me = False

    try:
        cookies = await page.context.cookies()
    except Exception:  # pragma: no cover - best effort diagnostics
        cookies = []

    linkedin_cookie_names = sorted(
        {
            cookie["name"]
            for cookie in cookies
            if "linkedin.com" in cookie.get("domain", "")
        }
    )

    screenshot_path = screenshot_dir / f"{step_id:03d}-{slug}.png"
    screenshot: str | None = None
    try:
        await page.screenshot(path=str(screenshot_path), full_page=True)
        screenshot = str(screenshot_path)
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        screenshot = f"<error: {exc}>"

    payload = {
        "step_id": step_id,
        "step": step,
        "url": getattr(page, "url", ""),
        "title": title,
        "remember_me": remember_me,
        "body_length": len(body_text),
        "body_marker": " ".join(body_text.split())[:200],
        "linkedin_cookie_names": linkedin_cookie_names,
        "screenshot": screenshot,
        "extra": extra or {},
    }

    trace_jsonl = trace_dir / "trace.jsonl"
    try:
        fd = os.open(str(trace_jsonl), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    except FileExistsError:
        pass
    with trace_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
