"""Stage 2e: cross-repository generalization into transferable rules.

Two scored rules of different repositories prescribe the same behavior
when their predicates agree on the trajectory pool.  Every predicate is
evaluated on every pool trajectory, so each rule becomes a binary vector;
agreement is the phi correlation of two vectors.  Rules whose predicate is
true on at least ``min_fires`` trajectories and false on at least
``min_fires`` are "powered" and are clustered by complete-linkage
agglomerative clustering over distance 1 - phi; the cutoff is the local
minimum of the phi histogram just left of its rightmost mode.  Every other
rule (no predicate, unparseable, or under-powered) is assigned by one
batched multiple-choice LLM query to one of the clusters that contain a
text-embedding neighbour of it, or to none.

A cluster spanning at least two repositories becomes a transferable rule:
its members are rewritten by the skill-learning LLM into one rule with a
trigger and an action, member gains are pooled with inverse-variance
weights, and Cochran's Q marks heterogeneous evidence (HET).  Clusters of
a single repository stay as repository-specific residual rules.

Outputs under ``--out-dir``: components.jsonl, assignments.jsonl,
strategies.jsonl, repo_residual.jsonl, summary.json.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from xreposkill import llm
from xreposkill.config import load_config
from xreposkill.pool import load_pool, pool_events
from xreposkill.predicate import eval_node, parse_predicate, validate_predicate

ASSIGN_SYSTEM = """You classify process rules (distilled from software-\
engineering agent trajectories) into strategy groups. For each numbered ITEM \
you get one rule and a list of candidate GROUPS (each shown by example member \
rules). Pick the group whose members prescribe the SAME underlying behavior/\
strategy as the rule, ignoring repository-specific anchors (file paths, API \
names) and wording. "Same" means: an agent following this rule would take the \
same kind of action at the same kind of moment as one following the group's \
rules. If no group matches, answer -1. When in doubt, answer -1.
Reply with ONLY: {"assignments": [{"i": <item number>, "group": <group \
number or -1>}, ...]} covering every item."""

MERGE_SYSTEM = """You deduplicate rule candidates distilled from agent \
trajectories. The numbered candidates below come from DIFFERENT repositories \
but were judged to prescribe the same underlying strategy. Merge them into \
ONE transferable rule.

Rules for merging:
- The merged rule must be repository-agnostic: state the TRIGGER (what \
situation in the issue or in the agent's own trajectory activates it) and the \
ACTION, without repository-specific file paths or API names in the main text.
- Repository-specific anchors (paths, commands, function names) go into \
"anchor_examples": at most one short example per repository, verbatim from a \
candidate. Do NOT invent anchors.
- If some candidates give different advice, merge the majority theme and \
list the dissenting candidate numbers in "outliers".

Reply with ONLY this JSON object:
{"title": "<imperative, <=80 chars>",
 "trigger": "<one sentence: when this applies>",
 "action": "<1-3 sentences: what to do>",
 "anchor_examples": ["<repo>: <anchor>", ...],
 "outliers": [<candidate numbers>]}"""


# ---------------------------------------------------------------- loading

def load_scored_rules(rules_root: Path) -> list[dict]:
    """Flatten ``<rules_root>/<repo_slug>/rules.jsonl`` into rule dicts."""
    rules: list[dict] = []
    for f in sorted(Path(rules_root).glob("*/rules.jsonl")):
        repo = f.parent.name
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            expr = r.get("predicate") or None
            if expr and validate_predicate(expr) is not None:
                expr = None
            v = r.get("validation") or {}
            scored = v.get("status") == "scored"
            rules.append({
                "idx": len(rules), "rule_id": r["rule_id"], "repo": repo,
                "title": r.get("title", ""), "body": r.get("body", ""),
                "category": r.get("category", ""),
                "support": int(r.get("support") or 0),
                "z": v.get("z") if scored else None,
                "lift": v.get("lift") if scored else None,
                "predicate": expr,
            })
    return rules


def label_matrix(rules: list[dict], pool: list[dict]) -> dict[int, np.ndarray]:
    """``idx -> bool vector over the pool`` for rules with a predicate."""
    ev = pool_events(pool)
    out: dict[int, np.ndarray] = {}
    for r in rules:
        if not r["predicate"]:
            continue
        node = parse_predicate(r["predicate"])
        out[r["idx"]] = np.fromiter((eval_node(node, e) for e in ev),
                                    dtype=bool, count=len(ev))
    return out


# ---------------------------------------------------------------- blocking

def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """Sentence embeddings, unit-normalised. Tests replace this function."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name).encode(
        texts, normalize_embeddings=True, show_progress_bar=False)


def candidate_pairs(rules: list[dict], k: int, *, embedding_model: str
                    ) -> list[tuple[int, int, float]]:
    """kNN text neighbours; decides only WHICH pairs the assignment step offers."""
    from sklearn.neighbors import NearestNeighbors
    if len(rules) < 2:
        return []
    X = embed_texts([r["title"] + ". " + r["body"] for r in rules],
                    embedding_model)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(rules)),
                          metric="cosine").fit(X)
    dist, ind = nn.kneighbors(X)
    pairs: dict[tuple[int, int], float] = {}
    for i, (drow, irow) in enumerate(zip(dist, ind)):
        for d, j in zip(drow, irow):
            if i == j:
                continue
            key = (min(i, int(j)), max(i, int(j)))
            pairs[key] = min(float(d), pairs.get(key, 1.0))
    return [(a, b, d) for (a, b), d in sorted(pairs.items())]


# ---------------------------------------------------------------- phi cutoff

def phi_valley(phis: list[float]) -> float | None:
    """Cutoff at the valley just left of the RIGHTMOST mode of the phi histogram.

    The distribution has a spike near 1 (same behavior) and a broader mass
    at lower values (behaviors that successful trajectories bundle
    together); the cutoff sits between them.  ``None`` when the histogram
    has a single mode.
    """
    hist, edges = np.histogram(phis, bins=40, range=(-1.0, 1.0))
    smooth = np.convolve(hist, np.ones(3) / 3, mode="same")
    n = len(smooth)
    modes = [k for k in range(1, n - 1)
             if smooth[k] >= smooth[k - 1] and smooth[k] > smooth[k + 1]]
    if len(modes) < 2:
        return None
    prev, r = modes[-2], modes[-1]
    valley = prev + int(np.argmin(smooth[prev:r + 1]))
    return float((edges[valley] + edges[valley + 1]) / 2)


# ---------------------------------------------------------------- assignment

def build_assignment_jobs(rules: list[dict], unresolved: set[int],
                          neighbors: dict[int, list[tuple[int, float]]],
                          root_of: dict[int, int], max_groups: int = 6,
                          ) -> tuple[list[dict], int]:
    """One job per unresolved rule: the clusters of its neighbours as choices."""
    jobs, auto_none = [], 0
    for i in sorted(unresolved):
        seen: dict[int, tuple[float, list[int]]] = {}
        for j, d in sorted(neighbors.get(i, []), key=lambda x: x[1]):
            if j in unresolved:
                continue
            root = root_of[j]
            if root not in seen:
                seen[root] = (d, [])
            seen[root][1].append(j)
        if not seen:
            auto_none += 1
            continue
        groups = sorted(seen.items(), key=lambda kv: kv[1][0])[:max_groups]
        jobs.append({"idx": i, "groups": [(root, members[:3])
                                          for root, (_, members) in groups]})
    return jobs, auto_none


def assign_unresolved(rules: list[dict], jobs: list[dict], *, model: str,
                      workers: int, batch: int = 8) -> dict[int, int]:
    """rule idx -> chosen cluster root (rules answered none are omitted)."""
    batches = [jobs[i:i + batch] for i in range(0, len(jobs), batch)]

    def run(bi: int, chunk: list[dict]) -> tuple[int, dict[int, int]]:
        lines = []
        for n, job in enumerate(chunk):
            r = rules[job["idx"]]
            lines.append(f"ITEM {n}:\nRULE: {r['title']}: {r['body']}")
            for g, (_, members) in enumerate(job["groups"]):
                lines.append(f"  GROUP {g}: "
                             + "; ".join(rules[m]["title"] for m in members))
        reply = llm.chat(model=model, system=ASSIGN_SYSTEM, user="\n".join(lines),
                         max_tokens=6000, temperature=0.0, timeout=240,
                         reasoning_effort="low")
        picks: dict[int, int] = {}
        for v in llm.parse_json_obj(reply.text).get("assignments", []):
            if not (isinstance(v, dict) and isinstance(v.get("i"), int)
                    and 0 <= v["i"] < len(chunk)):
                continue
            job = chunk[v["i"]]
            g = v.get("group")
            if isinstance(g, int) and 0 <= g < len(job["groups"]):
                picks[job["idx"]] = job["groups"][g][0]
        return bi, picks

    out: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(run, i, c) for i, c in enumerate(batches)]):
            _, picks = fut.result()
            out.update(picks)
    print(f"[generalize/assign] rules={len(jobs)} assigned={len(out)} "
          f"none_or_missing={len(jobs) - len(out)}", flush=True)
    return out


# ---------------------------------------------------------------- pooling

def pool_and_q(members: list[dict], *, alpha: float = 0.10) -> dict:
    """Inverse-variance pooled gain and Cochran's Q over members with gain and z."""
    from scipy.stats import chi2
    eff = []
    for m in members:
        z, lift = m.get("z"), m.get("lift")
        if isinstance(z, (int, float)) and isinstance(lift, (int, float)) \
                and abs(z) > 1e-9:
            se = abs(lift / z)
            if se > 1e-9:
                eff.append((lift, se))
    if len(eff) < 2:
        return {"n_scored": len(eff),
                "pooled_lift": eff[0][0] if eff else None,
                "pooled_z": (eff[0][0] / eff[0][1]) if eff else None,
                "q": None, "q_p": None, "homogeneous": None}
    w = [1 / se ** 2 for _, se in eff]
    pooled = sum(wi * l for wi, (l, _) in zip(w, eff)) / sum(w)
    se_pooled = math.sqrt(1 / sum(w))
    q = sum(wi * (l - pooled) ** 2 for wi, (l, _) in zip(w, eff))
    q_p = float(chi2.sf(q, len(eff) - 1))
    return {"n_scored": len(eff), "pooled_lift": round(pooled, 4),
            "pooled_z": round(pooled / se_pooled, 3),
            "q": round(q, 3), "q_p": round(q_p, 4), "homogeneous": q_p >= alpha}


def majority_predicate(members: list[dict]) -> dict | None:
    exprs = Counter(m["predicate"] for m in members if m.get("predicate"))
    if not exprs:
        return None
    top, n = exprs.most_common(1)[0]
    return {"predicate": top, "agreement": f"{n}/{len(members)}"}


def merge_component(cid: int, members: list[dict], *, model: str,
                    alpha: float = 0.10) -> dict:
    lines = [f"[{i}] ({m['repo']}, support {m['support']}"
             + (f", z {m['z']:+.2f}" if isinstance(m.get("z"), (int, float)) else "")
             + f") {m['title']}: {m['body']}"
             for i, m in enumerate(members)]
    user = "\n\n".join(lines)
    merged = llm.parse_json_obj(llm.chat(
        model=model, system=MERGE_SYSTEM, user=user, max_tokens=8000,
        temperature=0.0, timeout=240, reasoning_effort="low").text)
    if not (merged.get("title") and merged.get("action")):   # one retry
        merged = llm.parse_json_obj(llm.chat(
            model=model, system=MERGE_SYSTEM, user=user, max_tokens=8000,
            temperature=0.0, timeout=240, reasoning_effort="low").text)
    ok = bool(merged.get("title") and merged.get("action"))
    return {
        "strategy_id": f"strategy/{cid}", "merge_ok": ok,
        "title": merged.get("title") or members[0]["title"],
        "trigger": merged.get("trigger", ""),
        "action": merged.get("action", ""),
        "anchor_examples": merged.get("anchor_examples", []),
        "outliers": merged.get("outliers", []),
        "repos": sorted({m["repo"] for m in members}),
        "n_members": len(members),
        "support_total": sum(m["support"] for m in members),
        **pool_and_q(members, alpha=alpha),
        "predicate_majority": majority_predicate(members),
        "member_rule_ids": [m["rule_id"] for m in members],
    }


# ---------------------------------------------------------------- driver

def run_generalize(*, rules_root: Path, pool_dir: Path, out_dir: Path,
                   model: str, knn: int = 15, min_fires: int = 20,
                   alpha: float = 0.10, workers: int = 12,
                   embedding_model: str = "all-MiniLM-L6-v2",
                   phi_cutoff: float | None = None,
                   dry_run: bool = False) -> dict:
    from sklearn.cluster import AgglomerativeClustering
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = load_scored_rules(rules_root)
    pool = load_pool(pool_dir)
    labels = label_matrix(rules, pool)
    fires = {i: int(v.sum()) for i, v in labels.items()}
    n_pool = len(pool)
    print(f"[generalize] rules={len(rules)} with_predicate={len(labels)} "
          f"pool={n_pool}", flush=True)

    pairs = candidate_pairs(rules, knn, embedding_model=embedding_model)
    neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, j, d in pairs:
        neighbors[i].append((j, d))
        neighbors[j].append((i, d))

    # symmetric power floor: a predicate true on almost none or almost all
    # trajectories carries no discriminative signal either way
    powered = sorted(i for i in labels
                     if min_fires <= fires[i] <= n_pool - min_fires)
    unresolved = {r["idx"] for r in rules} - set(powered)
    if len(powered) >= 2:
        L = np.stack([labels[i] for i in powered]).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(L)
        phis = C[np.triu_indices(len(powered), 1)]
        phis = phis[np.isfinite(phis)]
    else:
        C = np.zeros((len(powered), len(powered)))
        phis = np.array([])
    cutoff = phi_cutoff if phi_cutoff is not None else (
        phi_valley(list(phis)) if len(phis) else None)
    print(f"[generalize/phi] powered_rules={len(powered)} phi_pairs={len(phis)} "
          f"unresolved_rules={len(unresolved)} cutoff={cutoff}", flush=True)
    if len(phis):
        hist, edges = np.histogram(phis, bins=20, range=(-1, 1))
        print("[generalize/phi] hist " + " ".join(
            f"{edges[k]:+.1f}:{hist[k]}" for k in range(len(hist))), flush=True)
    if cutoff is None:
        print("[generalize] ABORT: the phi histogram has no second mode; pass "
              "--phi-cutoff to set the cutoff explicitly.", flush=True)
        return {"aborted": "no_phi_cutoff", "n_rules": len(rules),
                "n_powered": len(powered)}

    if len(powered) >= 2:
        D = np.clip(1.0 - np.nan_to_num(C, nan=-1.0), 0.0, 2.0)
        np.fill_diagonal(D, 0.0)
        lab = AgglomerativeClustering(
            n_clusters=None, distance_threshold=float(1.0 - cutoff),
            metric="precomputed", linkage="complete").fit_predict(D)
        root_of = {idx: int(c) for idx, c in zip(powered, lab)}
    else:
        root_of = {idx: k for k, idx in enumerate(powered)}
    jobs, auto_none = build_assignment_jobs(rules, unresolved, neighbors, root_of)
    print(f"[generalize/assign] unresolved={len(unresolved)} auto_none={auto_none} "
          f"jobs={len(jobs)} (~{-(-len(jobs) // 8)} LLM calls)", flush=True)
    if dry_run:
        return {"dry_run": True, "phi_cutoff": cutoff, "n_rules": len(rules),
                "n_powered": len(powered), "n_unresolved": len(unresolved),
                "n_jobs": len(jobs)}

    assignments = assign_unresolved(rules, jobs, model=model, workers=workers) \
        if jobs else {}
    comp_of: dict[int, int] = dict(root_of)
    comp_of.update(assignments)
    next_id = max(root_of.values(), default=-1) + 1
    for r in rules:
        if r["idx"] not in comp_of:
            comp_of[r["idx"]] = next_id
            next_id += 1
    with (out_dir / "assignments.jsonl").open("w") as fh:
        for job in jobs:
            i = job["idx"]
            fh.write(json.dumps({
                "rule_id": rules[i]["rule_id"],
                "assigned_root": assignments.get(i),
                "choices": [[rules[m]["rule_id"] for m in ms]
                            for _, ms in job["groups"]]}, ensure_ascii=False) + "\n")

    comps: dict[int, list[dict]] = defaultdict(list)
    for r in rules:
        comps[comp_of[r["idx"]]].append(r)
    multi = {c: ms for c, ms in comps.items()
             if len({m["repo"] for m in ms}) >= 2}
    residual = [m for c, ms in comps.items() if c not in multi for m in ms]
    pos = {idx: k for k, idx in enumerate(powered)}

    def cohesion(ms: list[dict]) -> float | None:
        ks = [pos[m["idx"]] for m in ms if m["idx"] in pos]
        if len(ks) < 2:
            return None
        return round(float(C[np.ix_(ks, ks)].min()), 3)

    with (out_dir / "components.jsonl").open("w") as fh:
        for c, ms in sorted(comps.items()):
            fh.write(json.dumps({
                "component": c, "multi_repo": c in multi,
                "repos": sorted({m["repo"] for m in ms}), "min_phi": cohesion(ms),
                "members": [{k: m[k] for k in ("rule_id", "repo", "title",
                                                 "support", "z")} for m in ms]},
                ensure_ascii=False) + "\n")
    print(f"[generalize/graph] components={len(comps)} multi_repo={len(multi)} "
          f"rules_in_multi={sum(len(ms) for ms in multi.values())} "
          f"residual={len(residual)}", flush=True)

    strategies, failed = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(merge_component, c, ms, model=model, alpha=alpha)
                for c, ms in multi.items()]
        for fut in as_completed(futs):
            s = fut.result()
            strategies.append(s)
            failed += 0 if s["merge_ok"] else 1
            print(f"[generalize/merge] {s['strategy_id']:14s} ok={s['merge_ok']} "
                  f"repos={len(s['repos'])} support={s['support_total']} "
                  f"z={s['pooled_z']} homogeneous={s['homogeneous']} :: "
                  f"{s['title'][:55]}", flush=True)
    strategies.sort(key=lambda s: -(s["pooled_z"] if s["pooled_z"] is not None
                                    else -99))
    with (out_dir / "strategies.jsonl").open("w") as fh:
        for s in strategies:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    with (out_dir / "repo_residual.jsonl").open("w") as fh:
        for m in residual:
            fh.write(json.dumps({k: v for k, v in m.items() if k != "idx"},
                                ensure_ascii=False) + "\n")
    summary = {
        "knn": knn, "min_fires": min_fires, "phi_cutoff": cutoff,
        "n_rules": len(rules), "n_candidate_pairs": len(pairs),
        "n_phi_pairs": int(len(phis)), "n_powered": len(powered),
        "n_unresolved": len(unresolved), "n_auto_none": auto_none,
        "n_assigned": len(assignments), "n_components": len(comps),
        "n_strategy_components": len(multi),
        "n_rules_merged": sum(len(ms) for ms in multi.values()),
        "n_residual_rules": len(residual), "merge_failed": failed,
        "n_homogeneous": sum(1 for s in strategies if s["homogeneous"]),
        "n_heterogeneous": sum(1 for s in strategies if s["homogeneous"] is False),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[generalize] {json.dumps(summary)}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2e: cross-repository generalization.")
    ap.add_argument("--rules-root", required=True, type=Path,
                    help="scored rules tree (Stage 2c output)")
    ap.add_argument("--pool-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--knn", type=int, default=None)
    ap.add_argument("--min-fires", type=int, default=None)
    ap.add_argument("--phi-cutoff", type=float, default=None,
                    help="explicit phi cutoff (default: histogram valley)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="stop after the phi histogram and the job count")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    g = cfg["generalize"]
    summary = run_generalize(
        rules_root=a.rules_root, pool_dir=a.pool_dir, out_dir=a.out_dir,
        model=llm.resolve_model(a.model or cfg["models"]["skill_learning"]),
        knn=a.knn or g["knn"], min_fires=a.min_fires or g["min_fires"],
        alpha=g["heterogeneity_alpha"], workers=a.workers or g["workers"],
        embedding_model=g["embedding_model"], phi_cutoff=a.phi_cutoff,
        dry_run=a.dry_run)
    return 1 if summary.get("aborted") else 0


if __name__ == "__main__":
    raise SystemExit(main())
