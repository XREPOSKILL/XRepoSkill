import json

import pytest

from xreposkill.pool import (_load_pool_file, find_trajectory, format_steps_excerpt,
                             load_pool, load_steps_window, load_submitted_patch)
from tests.conftest import traj, write_jsonl


@pytest.fixture(autouse=True)
def _clear_cache():
    _load_pool_file.cache_clear()
    yield
    _load_pool_file.cache_clear()


def test_format_steps_excerpt_truncates():
    out = format_steps_excerpt([{"kind": "thought", "content": "x" * 2000, "raw": {}}],
                               max_chars=1000, max_step_chars=100)
    assert "…" in out and len(out) < 200
    steps = [{"kind": "action", "content": "a" * 500, "raw": {}} for _ in range(10)]
    assert "...truncated..." in format_steps_excerpt(steps, max_chars=1200)


def test_find_and_window(tmp_path):
    pool = tmp_path / "pool"
    write_jsonl(pool / "S.jsonl", [traj("x__x-1", "m", "pass", "S::x__x-1",
                                        ["cat a.py", "grep foo b.py", "pytest"])])
    assert find_trajectory(pool, "S::x__x-1")["model"] == "m"
    assert find_trajectory(pool, "S::nope") is None
    assert find_trajectory(pool, "T::x__x-1") is None      # falls back to a scan
    txt = load_steps_window(pool, "S::x__x-1", start=1)
    assert txt.startswith("[1 action]") and "[2 action]" in txt
    assert load_steps_window(pool, "S::nope", start=0) == ""
    assert len(load_pool(pool)) == 1


def test_load_submitted_patch(tmp_path):
    raw = tmp_path / "raw.traj.json"
    raw.write_text(json.dumps({"info": {"submission": "diff --git a/x b/x\n+hello\n"},
                               "messages": [], "instance_id": "x__x-1"}))
    row = {**traj("x__x-1", "m", "pass", "S::x__x-1", []), "source_path": str(raw)}
    pool = tmp_path / "pool"
    write_jsonl(pool / "S.jsonl", [row])
    assert "hello" in load_submitted_patch(pool, "S::x__x-1")
    assert load_submitted_patch(pool, "nope::x") == ""
