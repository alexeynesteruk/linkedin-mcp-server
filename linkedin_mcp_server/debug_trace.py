"""Best-effort trace capture with on-error retention."""

from __future__ import annotations

import itertools
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Literal

from linkedin_mcp_server.common_utils import secure_mkdir, slugify_fragment
from linkedin_mcp_server.session_state import auth_root_dir, get_source_profile_dir

TraceMode = Literal["off", "on_error", "always"]

logger = logging.getLogger(__name__)

_TRACE_COUNTER = itertools.count(1)
_TRACE_DIR: Path | None = None
_TRACE_KEEP = False
_EXPLICIT_TRACE_DIR = False

# Default GC: drop empty/near-empty run dirs older than 7 days; hard-cap at 200 dirs.
_DEFAULT_TRACE_MAX_AGE_DAYS = 7.0
_DEFAULT_TRACE_MAX_RUNS = 200
_MIN_USEFUL_BYTES = 64


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
                prefix="run-",
                dir=_trace_root(),
            )
        ).resolve()
    return _TRACE_DIR


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


def _run_dir_byte_size(path: Path) -> int:
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _is_empty_or_useless_run(path: Path) -> bool:
    """True when a run dir has no meaningful artifacts (empty logs only)."""
    if not path.is_dir():
        return False
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    if not children:
        return True
    return _run_dir_byte_size(path) < _MIN_USEFUL_BYTES


def garbage_collect_trace_runs(
    *,
    max_age_days: float | None = None,
    max_runs: int | None = None,
    now: float | None = None,
) -> dict[str, int]:
    """Delete stale empty trace-run dirs and enforce a simple retention cap.

    Env overrides:
    - ``LINKEDIN_TRACE_MAX_AGE_DAYS`` (default 7; 0 disables age-based GC)
    - ``LINKEDIN_TRACE_MAX_RUNS`` (default 200; 0 disables count-based GC)

    Never deletes the active process ``_TRACE_DIR`` or an explicit
    ``LINKEDIN_DEBUG_TRACE_DIR`` target. Best-effort: IO errors are ignored.
    """
    if max_age_days is None:
        raw_age = os.getenv("LINKEDIN_TRACE_MAX_AGE_DAYS", "").strip()
        if raw_age:
            try:
                max_age_days = float(raw_age)
            except ValueError:
                max_age_days = _DEFAULT_TRACE_MAX_AGE_DAYS
        else:
            max_age_days = _DEFAULT_TRACE_MAX_AGE_DAYS

    if max_runs is None:
        raw_runs = os.getenv("LINKEDIN_TRACE_MAX_RUNS", "").strip()
        if raw_runs:
            try:
                max_runs = int(raw_runs)
            except ValueError:
                max_runs = _DEFAULT_TRACE_MAX_RUNS
        else:
            max_runs = _DEFAULT_TRACE_MAX_RUNS

    stats = {"scanned": 0, "deleted_empty": 0, "deleted_over_cap": 0}
    try:
        root = _trace_root()
    except Exception:
        return stats
    if not root.is_dir():
        return stats

    protected: set[Path] = set()
    if _TRACE_DIR is not None:
        try:
            protected.add(_TRACE_DIR.resolve())
        except OSError:
            protected.add(_TRACE_DIR)
    explicit = os.getenv("LINKEDIN_DEBUG_TRACE_DIR", "").strip()
    if explicit:
        try:
            protected.add(Path(explicit).expanduser().resolve())
        except OSError:
            pass

    now_ts = now if now is not None else time.time()
    age_cutoff = (
        now_ts - (max_age_days * 86400.0) if max_age_days and max_age_days > 0 else None
    )

    try:
        run_dirs = sorted(
            (p for p in root.iterdir() if p.is_dir() and p.name.startswith("run-")),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        )
    except OSError:
        return stats

    stats["scanned"] = len(run_dirs)
    remaining: list[Path] = []
    for run_dir in run_dirs:
        try:
            resolved = run_dir.resolve()
        except OSError:
            remaining.append(run_dir)
            continue
        if resolved in protected:
            remaining.append(run_dir)
            continue
        try:
            mtime = run_dir.stat().st_mtime
        except OSError:
            remaining.append(run_dir)
            continue
        if (
            age_cutoff is not None
            and mtime < age_cutoff
            and _is_empty_or_useless_run(run_dir)
        ):
            try:
                shutil.rmtree(run_dir)
                stats["deleted_empty"] += 1
                continue
            except OSError:
                remaining.append(run_dir)
                continue
        remaining.append(run_dir)

    if max_runs and max_runs > 0 and len(remaining) > max_runs:
        # Drop oldest empty/useless dirs first, then oldest anything if still over.
        overflow = len(remaining) - max_runs
        candidates = sorted(
            remaining,
            key=lambda p: (
                0 if _is_empty_or_useless_run(p) else 1,
                p.stat().st_mtime if p.exists() else 0.0,
            ),
        )
        for run_dir in candidates:
            if overflow <= 0:
                break
            try:
                resolved = run_dir.resolve()
            except OSError:
                continue
            if resolved in protected:
                continue
            if not _is_empty_or_useless_run(run_dir):
                # Prefer empty dirs; stop once only contentful runs remain under pressure
                # unless still wildly over (2x cap).
                if len(remaining) - stats["deleted_over_cap"] <= max_runs * 2:
                    continue
            try:
                shutil.rmtree(run_dir)
                stats["deleted_over_cap"] += 1
                overflow -= 1
                if run_dir in remaining:
                    remaining.remove(run_dir)
            except OSError:
                continue

    if stats["deleted_empty"] or stats["deleted_over_cap"]:
        logger.info(
            "Trace-run GC: scanned=%d deleted_empty=%d deleted_over_cap=%d",
            stats["scanned"],
            stats["deleted_empty"],
            stats["deleted_over_cap"],
        )
    return stats


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
