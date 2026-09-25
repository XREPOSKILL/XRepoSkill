"""Divergence localization for fail/pass pairs.

Both trajectories are abstracted to event sequences (events.py), aligned on
(action_type, anchor) keys, and the shared prefix is extended across
matching blocks.  A gap of unmatched events between two matching blocks is
tolerated only if it holds at most GAP_TOLERANCE events on each side and
none of them is an edit, test run, script run, install, or submission.
The divergence point of each side is its first event after the shared
prefix.
"""
from __future__ import annotations
import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from xreposkill.events import Event, extract_events
from xreposkill.pool import find_trajectory

GAP_TOLERANCE = 3          # unmatched events allowed inside the shared prefix
MAX_PREFIX_RENDER = 30     # compact events rendered into the fork summary
MAX_CONTINUATION = 25
SIGNIFICANT = {"edit", "run_test", "run_python", "install", "submit"}


def anchor_key(ev: Event) -> tuple[str, str]:
    t = ev.target.strip().lower()
    if "/" in t:
        t = t.rsplit("/", 1)[-1]
    t = re.sub(r"['\"]", "", t)[:24]
    return (ev.action_type, t)


def _compact(ev: Event) -> str:
    return f"{ev.action_type}:{ev.target}" if ev.target else ev.action_type


class Fork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pair_id: str
    instance_id: str
    repo: str
    fail_traj_id: str
    pass_traj_id: str
    n_shared: int
    shared_prefix: list[str]
    fail_div_step: int | None
    pass_div_step: int | None
    fail_continuation: list[str]
    pass_continuation: list[str]


def shared_prefix_end(fail_keys: list, pass_keys: list) -> tuple[int, int, int]:
    """``(fail_end, pass_end, n_matched)`` where the shared prefix ends."""
    sm = SequenceMatcher(a=fail_keys, b=pass_keys, autojunk=False)
    fail_end = pass_end = n_matched = 0
    for blk in sm.get_matching_blocks():
        if blk.size == 0:
            break
        if (blk.a - fail_end > GAP_TOLERANCE
                or blk.b - pass_end > GAP_TOLERANCE):
            break
        gap = fail_keys[fail_end:blk.a] + pass_keys[pass_end:blk.b]
        if any(k[0] in SIGNIFICANT for k in gap):
            break
        fail_end = blk.a + blk.size
        pass_end = blk.b + blk.size
        n_matched += blk.size
    return fail_end, pass_end, n_matched


def localize_fork(*, pair: dict, fail_steps: list[dict],
                  pass_steps: list[dict]) -> Fork:
    fail_ev = extract_events(fail_steps)
    pass_ev = extract_events(pass_steps)
    f_end, p_end, n_matched = shared_prefix_end(
        [anchor_key(e) for e in fail_ev], [anchor_key(e) for e in pass_ev])
    prefix = [_compact(e) for e in fail_ev[:f_end]]
    if len(prefix) > MAX_PREFIX_RENDER:
        half = MAX_PREFIX_RENDER // 2
        prefix = prefix[:half] + ["..."] + prefix[-half:]
    return Fork(
        pair_id=pair["pair_id"], instance_id=pair["instance_id"],
        repo=pair.get("repo", ""),
        fail_traj_id=pair["fail_traj_id"], pass_traj_id=pair["pass_traj_id"],
        n_shared=n_matched, shared_prefix=prefix,
        fail_div_step=fail_ev[f_end].step_idx if f_end < len(fail_ev) else None,
        pass_div_step=pass_ev[p_end].step_idx if p_end < len(pass_ev) else None,
        fail_continuation=[_compact(e) for e in fail_ev[f_end:f_end + MAX_CONTINUATION]],
        pass_continuation=[_compact(e) for e in pass_ev[p_end:p_end + MAX_CONTINUATION]],
    )


def build_forks(*, pairs_path: Path, pool_dir: Path, out_path: Path
                ) -> tuple[int, int]:
    n_ok = n_skip = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for line in Path(pairs_path).read_text().splitlines():
            if not line.strip():
                continue
            pair = json.loads(line)
            fail = find_trajectory(pool_dir, pair["fail_traj_id"])
            pas = find_trajectory(pool_dir, pair["pass_traj_id"])
            if fail is None or pas is None:
                n_skip += 1
                continue
            fork = localize_fork(pair=pair, fail_steps=fail.get("steps", []),
                                 pass_steps=pas.get("steps", []))
            fout.write(fork.model_dump_json() + "\n")
            n_ok += 1
    return n_ok, n_skip


def load_forks(forks_path: Path) -> dict[str, dict]:
    forks: dict[str, dict] = {}
    for line in Path(forks_path).read_text().splitlines():
        if line.strip():
            f = json.loads(line)
            forks[f["pair_id"]] = f
    return forks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Locate the divergence point of each pair.")
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--pool-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    n_ok, n_skip = build_forks(pairs_path=a.pairs, pool_dir=a.pool_dir,
                               out_path=a.out)
    print(f"[fork] forks={n_ok} skipped(missing trajectory)={n_skip}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
