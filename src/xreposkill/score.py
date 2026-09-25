"""Stage 2c: resolution gain and z score of every merged rule.

For a rule with predicate b of repository R: the held-out set is every
pool trajectory of the mixed-outcome issues of R, minus the two sides of
every pair the rule was merged from.  Each trajectory is labelled b / not b
by the predicate executor (no LLM call), and the gain is the
Mantel-Haenszel risk difference across issues, with Sato's variance.
The z score is gain / standard error.  Scoring never removes a rule: a
rule without a usable predicate or without a usable issue stratum gets a
"fallback" record and is ordered by support later.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from xreposkill.events import Event, extract_events
from xreposkill.pairing import frontier_instances, load_pairs
from xreposkill.pool import load_pool
from xreposkill.predicate import Node, eval_node, parse_predicate, PredicateError
from xreposkill.schemas import Rule

Z95 = 1.959964


@dataclass(frozen=True)
class LabeledTraj:
    traj_id: str
    instance_id: str
    model: str
    outcome: str               # "pass" | "fail"
    label: bool


@dataclass(frozen=True)
class Stratum:
    instance_id: str
    a: int      # pass & b
    n1: int     # b total
    c: int      # pass & not b
    n0: int     # not b total


@dataclass
class LiftResult:
    lift: float | None          # MH risk difference; None if no usable strata
    se: float | None
    ci_low: float | None
    ci_high: float | None
    n_strata_used: int
    n_strata_empty: int         # strata with an empty b or not-b cell
    n_labeled: int
    occurrence: float | None    # fraction of labelled trajectories with b


def build_holdout(pool: list[dict], *, instances: set[str],
                  exclude_traj_ids: set[str] = frozenset()) -> list[dict]:
    return [t for t in pool
            if t["instance_id"] in instances
            and t["outcome"] in ("pass", "fail")
            and t["traj_id"] not in exclude_traj_ids]


def label_holdout(node: Node, holdout: list[dict], *,
                  events_cache: dict[str, list[Event]] | None = None
                  ) -> list[LabeledTraj]:
    out = []
    for t in holdout:
        if events_cache is not None:
            events = events_cache.get(t["traj_id"])
            if events is None:
                events = events_cache[t["traj_id"]] = extract_events(t.get("steps", []))
        else:
            events = extract_events(t.get("steps", []))
        out.append(LabeledTraj(traj_id=t["traj_id"], instance_id=t["instance_id"],
                               model=t["model"], outcome=t["outcome"],
                               label=eval_node(node, events)))
    return out


def strata_from_labels(labeled: list[LabeledTraj]) -> tuple[list[Stratum], int]:
    by_inst: dict[str, list[LabeledTraj]] = defaultdict(list)
    for l in labeled:
        by_inst[l.instance_id].append(l)
    strata, n_empty = [], 0
    for iid, rows in sorted(by_inst.items()):
        n1 = sum(r.label for r in rows)
        n0 = len(rows) - n1
        if n1 == 0 or n0 == 0:
            n_empty += 1
            continue
        a = sum(r.label and r.outcome == "pass" for r in rows)
        c = sum((not r.label) and r.outcome == "pass" for r in rows)
        strata.append(Stratum(iid, a, n1, c, n0))
    return strata, n_empty


def mh_risk_difference(strata: list[Stratum]) -> tuple[float, float] | None:
    """Mantel-Haenszel risk difference with Sato's (1989) variance: ``(rd, se)``."""
    W = sum(s.n1 * s.n0 / (s.n1 + s.n0) for s in strata)
    if W <= 0:
        return None
    rd = sum((s.a * s.n0 - s.c * s.n1) / (s.n1 + s.n0) for s in strata) / W
    P = sum((s.n1 ** 2 * s.c - s.n0 ** 2 * s.a
             + s.n1 * s.n0 * (s.n0 - s.n1) / 2) / (s.n1 + s.n0) ** 2
            for s in strata)
    Q = sum((s.a * (s.n0 - s.c) + s.c * (s.n1 - s.a)) / (2 * (s.n1 + s.n0))
            for s in strata)
    var = (rd * P + Q) / W ** 2
    return rd, math.sqrt(max(var, 0.0))


def stratified_lift(labeled: list[LabeledTraj]) -> LiftResult:
    strata, n_empty = strata_from_labels(labeled)
    occ = (sum(l.label for l in labeled) / len(labeled)) if labeled else None
    res = mh_risk_difference(strata) if strata else None
    if res is None:
        return LiftResult(None, None, None, None, 0, n_empty, len(labeled), occ)
    rd, se = res
    return LiftResult(lift=rd, se=se, ci_low=rd - Z95 * se, ci_high=rd + Z95 * se,
                      n_strata_used=len(strata), n_strata_empty=n_empty,
                      n_labeled=len(labeled), occurrence=occ)


def score_rule(rule: Rule, *, pool: list[dict], repo_frontier: set[str],
               pairs: dict[str, dict], events_cache: dict[str, list[Event]]
               ) -> dict:
    """Attach gain and z to one rule; never rejects."""
    if not rule.predicate:
        return {"status": "fallback", "reason": "no_predicate"}
    try:
        node = parse_predicate(rule.predicate)
    except PredicateError as e:
        return {"status": "fallback", "reason": f"predicate_error: {e}"}
    src = [pairs[pid] for pid in rule.source_pair_ids if pid in pairs]
    exclude = {p["fail_traj_id"] for p in src} | {p["pass_traj_id"] for p in src}
    holdout = build_holdout(pool, instances=repo_frontier,
                            exclude_traj_ids=exclude)
    r = stratified_lift(label_holdout(node, holdout, events_cache=events_cache))
    if r.lift is None:
        return {"status": "fallback", "reason": "no_usable_strata"}
    if not r.se or r.se <= 0:
        return {"status": "fallback", "reason": "zero_se"}
    rec = {k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in asdict(r).items()}
    return {"status": "scored", "z": round(r.lift / r.se, 4), **rec}


def score_tree(*, rules_root: Path, out: Path, pool_dir: Path,
               pairs_path: Path, log_path: Path | None = None) -> dict[str, int]:
    """Score every ``<rules_root>/<repo_slug>/rules.jsonl``; write the same layout to ``out``."""
    pool = load_pool(pool_dir)
    frontier = set(frontier_instances(pool))
    inst_repo = {t["instance_id"]: t["repo"] for t in pool}
    frontier_by_repo: dict[str, set[str]] = defaultdict(set)
    for iid in frontier:
        frontier_by_repo[inst_repo[iid]].add(iid)
    pairs = load_pairs(pairs_path) if Path(pairs_path).exists() else {}
    events_cache: dict[str, list[Event]] = {}
    stats = {"scored": 0, "fallback": 0}
    log_f = Path(log_path).open("w") if log_path else None
    for rf in sorted(Path(rules_root).glob("*/rules.jsonl")):
        slug = rf.parent.name
        repo = slug.replace("__", "/", 1)
        scored: list[Rule] = []
        for line in rf.read_text().splitlines():
            if not line.strip():
                continue
            rule = Rule.model_validate_json(line)
            rule.validation = score_rule(
                rule, pool=pool, repo_frontier=frontier_by_repo.get(repo, set()),
                pairs=pairs, events_cache=events_cache)
            stats[rule.validation["status"]] += 1
            scored.append(rule)
            if log_f:
                log_f.write(json.dumps({"rule_id": rule.rule_id,
                                        "predicate": rule.predicate,
                                        **rule.validation}) + "\n")
        dest = Path(out) / slug / "rules.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w") as f:
            for r in scored:
                f.write(r.model_dump_json() + "\n")
    if log_f:
        log_f.close()
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2c: gain and z score.")
    ap.add_argument("--rules-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pool-dir", required=True, type=Path)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--log", type=Path, default=None)
    a = ap.parse_args(argv)
    stats = score_tree(rules_root=a.rules_root, out=a.out, pool_dir=a.pool_dir,
                       pairs_path=a.pairs, log_path=a.log)
    print(f"[score] {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
