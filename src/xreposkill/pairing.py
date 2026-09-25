"""Pair a failed with a successful trajectory on the same issue.

Backbone LLMs are ranked by their resolution rate on the pool.  On each
issue with both outcomes, every failed trajectory is combined with every
successful trajectory of a backbone with a higher rate, the combinations
are ranked by the gap in resolution rate, and the top K become pairs.
A check before pairing asserts that no issue of the pool appears in the
evaluation benchmark id list.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from xreposkill.pool import load_pool

TIER_NAMES = ("weak", "mid", "strong")


def preflight_no_leakage(*, pool_instance_ids: set[str],
                         eval_set_file: Path) -> None:
    eval_ids = {l.strip() for l in Path(eval_set_file).read_text().splitlines()
                if l.strip()}
    overlap = pool_instance_ids & eval_ids
    assert not overlap, f"leakage: {sorted(overlap)[:5]}..."


def resolve_rates(pool: list[dict]) -> dict[str, float]:
    """Per-backbone resolution rate over pass/fail outcomes."""
    n_pass: dict[str, int] = defaultdict(int)
    n_tot: dict[str, int] = defaultdict(int)
    for t in pool:
        if t["outcome"] not in ("pass", "fail"):
            continue
        n_tot[t["model"]] += 1
        n_pass[t["model"]] += t["outcome"] == "pass"
    return {m: n_pass[m] / n_tot[m] for m in n_tot}


def ability_tiers(rates: dict[str, float]) -> dict[str, str]:
    """Split backbones into weak/mid/strong terciles by resolution rate."""
    ranked = sorted(rates, key=rates.get)
    n = len(ranked)
    return {m: TIER_NAMES[min(i * len(TIER_NAMES) // n, 2)]
            for i, m in enumerate(ranked)}


def frontier_instances(pool: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """Issues with both outcomes: ``{iid: {"pass": [...], "fail": [...]}}``."""
    sides: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"pass": [], "fail": []})
    for t in pool:
        if t["outcome"] in ("pass", "fail"):
            sides[t["instance_id"]][t["outcome"]].append(t)
    return {iid: s for iid, s in sides.items() if s["pass"] and s["fail"]}


class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    pair_id: str
    instance_id: str
    repo: str
    fail_traj_id: str
    pass_traj_id: str
    fail_meta: dict
    pass_meta: dict
    fail_tier: str
    pass_tier: str
    capability_gap: float          # rate(pass backbone) - rate(fail backbone)


def select_pairs(pool: list[dict], *, K: int = 4) -> tuple[list[Pair], dict]:
    rates = resolve_rates(pool)
    tiers = ability_tiers(rates)
    frontier = frontier_instances(pool)
    pairs: list[Pair] = []
    n_no_positive_gap = 0
    for iid in sorted(frontier):
        sides = frontier[iid]
        scored = []
        for ft in sides["fail"]:
            for pt in sides["pass"]:
                gap = rates[pt["model"]] - rates[ft["model"]]
                scored.append((gap, ft, pt))
        scored.sort(key=lambda x: (-x[0], x[1]["traj_id"], x[2]["traj_id"]))
        chosen = [s for s in scored if s[0] > 0][:K]
        if not chosen:
            n_no_positive_gap += 1
            continue
        for gap, ft, pt in chosen:
            pairs.append(Pair(
                pair_id=f"{iid}::{ft['traj_id']}::{pt['traj_id']}",
                instance_id=iid, repo=ft["repo"],
                fail_traj_id=ft["traj_id"], pass_traj_id=pt["traj_id"],
                fail_meta={"agent": ft["agent"], "model": ft["model"]},
                pass_meta={"agent": pt["agent"], "model": pt["model"]},
                fail_tier=tiers[ft["model"]], pass_tier=tiers[pt["model"]],
                capability_gap=round(gap, 4)))
    stats = {
        "n_instances_pool": len({t["instance_id"] for t in pool}),
        "n_frontier": len(frontier),
        "n_no_positive_gap": n_no_positive_gap,
        "n_pairs": len(pairs),
        "rates": {m: round(r, 4) for m, r in
                  sorted(rates.items(), key=lambda x: x[1])},
        "tiers": tiers,
    }
    return pairs, stats


def load_pairs(pairs_path: Path) -> dict[str, dict]:
    pairs: dict[str, dict] = {}
    for line in Path(pairs_path).read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            pairs[p["pair_id"]] = p
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Select fail/pass pairs.")
    ap.add_argument("--pool-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stats-out", type=Path, default=None)
    ap.add_argument("--K", type=int, default=4, help="pairs per issue")
    ap.add_argument("--eval-ids", type=Path, action="append", default=[],
                    help="benchmark id list(s) that must not overlap the pool")
    a = ap.parse_args(argv)
    pool = load_pool(a.pool_dir)
    for f in a.eval_ids:
        preflight_no_leakage(pool_instance_ids={t["instance_id"] for t in pool},
                             eval_set_file=f)
    pairs, stats = select_pairs(pool, K=a.K)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for p in pairs:
            f.write(p.model_dump_json() + "\n")
    if a.stats_out:
        a.stats_out.write_text(json.dumps(stats, indent=2))
    print(f"[pair] frontier issues={stats['n_frontier']} pairs={stats['n_pairs']} "
          f"no-positive-gap={stats['n_no_positive_gap']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
