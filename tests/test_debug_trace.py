import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from linkedin_mcp_server.debug_trace import (
    _safe_source_profile_dir,
    cleanup_trace_dir,
    get_trace_dir,
    mark_trace_for_retention,
    record_page_trace,
    reset_trace_state_for_testing,
)


def setup_function():
    reset_trace_state_for_testing()


def teardown_function():
    reset_trace_state_for_testing()


def test_get_trace_dir_creates_ephemeral_dir_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))

    trace_dir = get_trace_dir()

    assert trace_dir is not None
    assert trace_dir.exists()
    assert "trace-runs" in str(trace_dir)


def test_cleanup_trace_dir_removes_ephemeral_dir_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    trace_dir = get_trace_dir()
    assert trace_dir is not None

    cleanup_trace_dir()

    assert not trace_dir.exists()


def test_mark_trace_for_retention_keeps_trace_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    trace_dir = mark_trace_for_retention()
    assert trace_dir is not None

    cleanup_trace_dir()

    assert trace_dir.exists()


def test_explicit_trace_dir_is_preserved(monkeypatch, tmp_path):
    trace_dir = tmp_path / "explicit-trace"
    monkeypatch.setenv("LINKEDIN_DEBUG_TRACE_DIR", str(trace_dir))

    resolved = get_trace_dir()
    assert resolved == trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)

    cleanup_trace_dir()

    assert trace_dir.exists()


def test_trace_mode_off_disables_trace_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("LINKEDIN_TRACE_MODE", "off")

    assert get_trace_dir() is None


@pytest.mark.asyncio
async def test_reset_trace_state_resets_step_counter(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))

    page = MagicMock()
    page.url = "https://www.linkedin.com/feed/"
    page.title = AsyncMock(return_value="LinkedIn")
    page.evaluate = AsyncMock(return_value="Feed")
    locator = MagicMock()
    locator.count = AsyncMock(return_value=0)
    page.locator = MagicMock(return_value=locator)
    page.context.cookies = AsyncMock(return_value=[])
    page.screenshot = AsyncMock()

    await record_page_trace(page, "first")
    trace_dir = get_trace_dir()
    assert trace_dir is not None
    first_payload = json.loads((trace_dir / "trace.jsonl").read_text().splitlines()[0])
    assert first_payload["step_id"] == 1

    reset_trace_state_for_testing()
    monkeypatch.setenv("USER_DATA_DIR", str((tmp_path / "second") / "profile"))

    await record_page_trace(page, "first-again")
    second_trace_dir = get_trace_dir()
    assert second_trace_dir is not None
    second_payload = json.loads(
        (second_trace_dir / "trace.jsonl").read_text().splitlines()[0]
    )
    assert second_payload["step_id"] == 1


def test_safe_source_profile_dir_ignores_generic_env_fallback(monkeypatch):
    monkeypatch.setenv("USER_DATA_DIR", "/tmp/unrelated-user-data")
    monkeypatch.setattr(
        "linkedin_mcp_server.debug_trace.get_source_profile_dir",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert _safe_source_profile_dir() == Path("~/.linkedin-mcp/profile").expanduser()


# --- pruning of old run directories -----------------------------------------

import os  # noqa: E402
import time  # noqa: E402

from linkedin_mcp_server.debug_trace import prune_trace_runs  # noqa: E402
from linkedin_mcp_server.profile_claim import ensure_profile_claim  # noqa: E402

_DAY = 86400.0


def _point_trace_at(monkeypatch, profile: Path) -> None:
    """The conftest pins this module's view of the profile root; move it."""
    monkeypatch.setattr(
        "linkedin_mcp_server.debug_trace.get_source_profile_dir", lambda: profile
    )


def _claimed_root(monkeypatch, tmp_path) -> Path:
    """An owned auth root with an existing trace-runs directory."""
    profile = tmp_path / "profile"
    monkeypatch.setenv("USER_DATA_DIR", str(profile))
    _point_trace_at(monkeypatch, profile)
    ensure_profile_claim(profile)
    root = tmp_path / "trace-runs"
    root.mkdir()
    return root


def _run(root: Path, name: str, age_days: float) -> Path:
    path = root / name
    path.mkdir()
    (path / "trace.jsonl").write_text("{}\n")
    stamp = time.time() - age_days * _DAY
    os.utime(path, (stamp, stamp))
    return path


def test_prune_removes_runs_older_than_the_age_limit(monkeypatch, tmp_path):
    root = _claimed_root(monkeypatch, tmp_path)
    old = _run(root, "run-abcd1234", age_days=30)
    fresh = _run(root, "run-wxyz_789", age_days=1)

    removed = prune_trace_runs(root)

    assert removed == [old]
    assert not old.exists()
    assert fresh.exists()


def test_prune_keeps_only_the_newest_runs_by_count(monkeypatch, tmp_path):
    root = _claimed_root(monkeypatch, tmp_path)
    runs = [_run(root, f"run-aaaaaa{i:02d}", age_days=i / 10) for i in range(6)]

    prune_trace_runs(root, keep=3)

    assert [p.exists() for p in runs] == [True, True, True, False, False, False]


def test_prune_never_removes_the_current_run(monkeypatch, tmp_path):
    root = _claimed_root(monkeypatch, tmp_path)
    current = _run(root, "run-current1", age_days=90)

    assert prune_trace_runs(root, keep_dir=current) == []
    assert current.exists()


def test_prune_only_touches_directories_named_like_its_own(monkeypatch, tmp_path):
    root = _claimed_root(monkeypatch, tmp_path)
    lookalikes = [
        _run(root, "run-abcd12345", age_days=90),  # nine characters
        _run(root, "run-ABCD1234", age_days=90),  # outside mkdtemp's alphabet
        _run(root, "keep-abcd1234", age_days=90),
        _run(root, "run-abcd123", age_days=90),
    ]
    stray = root / "run-file0001"
    stray.write_text("not a directory")
    os.utime(stray, (0, 0))

    assert prune_trace_runs(root) == []
    assert all(p.exists() for p in [*lookalikes, stray])


def test_prune_does_not_follow_or_remove_a_symlinked_run(monkeypatch, tmp_path):
    root = _claimed_root(monkeypatch, tmp_path)
    target = tmp_path / "precious"
    target.mkdir()
    (target / "keep.txt").write_text("data")
    link = root / "run-linked01"
    link.symlink_to(target, target_is_directory=True)

    assert prune_trace_runs(root, max_age_days=0) == []
    assert link.is_symlink()
    assert (target / "keep.txt").read_text() == "data"


def test_prune_refuses_a_symlinked_trace_root(monkeypatch, tmp_path):
    root = _claimed_root(monkeypatch, tmp_path)
    root.rmdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    old = _run(elsewhere, "run-abcd1234", age_days=90)
    root.symlink_to(elsewhere, target_is_directory=True)

    assert prune_trace_runs(root) == []
    assert old.exists()


def test_prune_ignores_a_directory_that_is_not_the_trace_root(monkeypatch, tmp_path):
    _claimed_root(monkeypatch, tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    old = _run(other, "run-abcd1234", age_days=90)

    assert prune_trace_runs(other) == []
    assert old.exists()


def test_prune_leaves_an_unowned_profile_root_alone(monkeypatch, tmp_path):
    """No claim marker on this root: the ownership guard refuses, so nothing is
    deleted. (The autouse fixture claims only its own tmp profile.)"""
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "somebody-elses-file").write_text("x")
    monkeypatch.setenv("USER_DATA_DIR", str(foreign / "profile"))
    _point_trace_at(monkeypatch, foreign / "profile")
    root = foreign / "trace-runs"
    root.mkdir()
    old = _run(root, "run-abcd1234", age_days=90)

    assert prune_trace_runs(root) == []
    assert old.exists()


def test_a_new_process_run_prunes_the_stale_ones(monkeypatch, tmp_path):
    root = _claimed_root(monkeypatch, tmp_path)
    old = _run(root, "run-abcd1234", age_days=90)

    trace_dir = get_trace_dir()

    assert trace_dir is not None and trace_dir.exists()
    assert not old.exists()


def test_a_prune_failure_never_breaks_trace_setup(monkeypatch, tmp_path):
    _claimed_root(monkeypatch, tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("linkedin_mcp_server.debug_trace.prune_trace_runs", boom)

    assert get_trace_dir() is not None
