"""Stage 2a: route candidate rules to their repository and merge them.

Every candidate is bucketed by the repository of its source pair (read
from the pair id, never from the LLM's own labelling).  Within a bucket,
candidates of the same category are handed to the skill-learning LLM in
batches, each batch is merged into 3 to 10 rules, and rules with the same
title from different batches are merged again.  The number of candidates
merged into a rule is its support, and the merged rule takes the predicate
that occurs most often among its candidates.
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from xreposkill import llm
from xreposkill.config import load_config
from xreposkill.cost import CostLogger
from xreposkill.schemas import Rule, repo_slug_from_pair_id, safe_name

MERGE_PROMPT = (Path(__file__).parent / "prompts" / "merge_rules.txt").read_text()


def route_candidates_by_repo(candidate_files: list[Path], out_dir: Path) -> int:
    """Write ``<out_dir>/<repo_slug>.jsonl`` per repository; return the count."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict]] = defaultdict(list)
    for cf in candidate_files:
        for line in Path(cf).read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                groups[repo_slug_from_pair_id(c["pair_id"])].append(c)
    n = 0
    for slug, items in sorted(groups.items()):
        with (out_dir / f"{safe_name(slug)}.jsonl").open("w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
                n += 1
    return n


def group_by_category(candidates: list[dict]) -> dict[str, list[dict]]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        if c.get("category"):
            by_cat[c["category"]].append(c)
    return dict(by_cat)


def representative_predicate(cands: list[dict]) -> str | None:
    """Most common predicate among the candidates; ties keep the first seen."""
    counts: Counter = Counter()
    order: list[str] = []
    for c in cands:
        p = c.get("predicate")
        if isinstance(p, str) and p.strip():
            if p not in counts:
                order.append(p)
            counts[p] += 1
    if not counts:
        return None
    return max(order, key=lambda k: counts[k])


def _batch_merge(*, batch: list[dict], category: str, bucket_id: str,
                 model: str, cost_logger: CostLogger) -> list[Rule]:
    user_msg = "\n\n".join(f"[{i}] {c['rule']}" for i, c in enumerate(batch))
    reply = llm.chat(model=model, system=MERGE_PROMPT, user=user_msg,
                     max_tokens=24000, temperature=0.1, timeout=180)
    cost_logger.log(stage="merge", sub="merge_rules", model=model,
                    in_tokens=reply.in_tokens, out_tokens=reply.out_tokens,
                    cost_usd=reply.cost, bucket_id=bucket_id,
                    category=category, batch_size=len(batch))
    if reply.finish_reason == "length":
        # truncated output: split the batch instead of dropping it
        if len(batch) == 1:
            print(f"[merge] WARN {bucket_id}/{category}: single-item batch "
                  "still truncated, dropped", flush=True)
            return []
        mid = len(batch) // 2
        return (_batch_merge(batch=batch[:mid], category=category,
                             bucket_id=bucket_id, model=model,
                             cost_logger=cost_logger)
                + _batch_merge(batch=batch[mid:], category=category,
                               bucket_id=bucket_id, model=model,
                               cost_logger=cost_logger))
    obj = llm.parse_json_obj(reply.text)
    if not obj:
        print(f"[merge] WARN {bucket_id}/{category}: unparseable reply "
              f"(finish={reply.finish_reason!r}), batch of {len(batch)} dropped",
              flush=True)
        return []
    rules: list[Rule] = []
    for j, r in enumerate(obj.get("rules", [])):
        if not isinstance(r, dict):
            continue
        indices = [i for i in r.get("support_indices", [])
                   if isinstance(i, int) and 0 <= i < len(batch)]
        if not indices:
            continue
        try:
            rules.append(Rule(
                rule_id=f"{bucket_id}:{category}:{j}",
                title=str(r.get("title", "")).strip(),
                body=str(r.get("body", "")).strip(),
                category=category, support=len(indices),
                source_pair_ids=[batch[i]["pair_id"] for i in indices],
                predicate=representative_predicate([batch[i] for i in indices])))
        except Exception:
            continue
    return rules


def merge_rules(*, candidates: list[dict], category: str, bucket_id: str,
                model: str, cost_logger: CostLogger, batch_size: int
                ) -> list[Rule]:
    """Merge one category of a bucket; same-title rules across batches fold."""
    if not candidates:
        return []
    by_title: dict[str, Rule] = {}
    for i in range(0, len(candidates), batch_size):
        chunk = candidates[i:i + batch_size]
        for r in _batch_merge(batch=chunk, category=category,
                              bucket_id=bucket_id, model=model,
                              cost_logger=cost_logger):
            key = r.title.lower()
            if key in by_title:
                prev = by_title[key]
                by_title[key] = Rule(
                    rule_id=prev.rule_id, title=prev.title,
                    body=prev.body if len(prev.body) >= len(r.body) else r.body,
                    category=category, support=prev.support + r.support,
                    source_pair_ids=prev.source_pair_ids + r.source_pair_ids,
                    predicate=prev.predicate or r.predicate)
            else:
                by_title[key] = r
    final = sorted(by_title.values(), key=lambda r: r.support, reverse=True)
    return [Rule(rule_id=f"{bucket_id}:{r.category}:{j}", title=r.title,
                 body=r.body, category=r.category, support=r.support,
                 source_pair_ids=r.source_pair_ids, predicate=r.predicate)
            for j, r in enumerate(final)]


def merge_buckets(*, buckets_dir: Path, out_root: Path, model: str,
                  cost_logger: CostLogger, batch_size: int = 32,
                  resume: bool = True, merge_fn=merge_rules,
                  ) -> tuple[int, int, int]:
    """For each ``<buckets_dir>/<repo>.jsonl`` write ``<out_root>/<repo>/rules.jsonl``.

    Returns ``(n_done, n_skipped, n_errored)``.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    err_path = out_root.parent / "merge_errors.jsonl"
    n_done = n_skip = n_err = 0
    for bucket_file in sorted(Path(buckets_dir).glob("*.jsonl")):
        bucket_id = bucket_file.stem
        rules_path = out_root / bucket_id / "rules.jsonl"
        if resume and rules_path.exists() and rules_path.stat().st_size > 0:
            n_skip += 1
            continue
        try:
            cands = [json.loads(l) for l in bucket_file.read_text().splitlines()
                     if l.strip()]
            if not cands:
                continue
            all_rules: list[Rule] = []
            for cat, items in group_by_category(cands).items():
                all_rules.extend(merge_fn(
                    candidates=items, category=cat, bucket_id=bucket_id,
                    model=model, cost_logger=cost_logger, batch_size=batch_size))
            rules_path.parent.mkdir(parents=True, exist_ok=True)
            with rules_path.open("w") as f:
                for r in all_rules:
                    f.write(r.model_dump_json() + "\n")
            n_done += 1
            print(f"[merge] {bucket_id}: candidates={len(cands)} rules={len(all_rules)}",
                  flush=True)
        except Exception as e:
            n_err += 1
            with err_path.open("a") as f:
                f.write(json.dumps({"bucket": bucket_id, "error": repr(e),
                                    "traceback": traceback.format_exc()}) + "\n")
            print(f"[merge] ERROR on {bucket_id}: {e!r}", file=sys.stderr, flush=True)
    print(f"[merge] done={n_done} skipped={n_skip} errored={n_err}", flush=True)
    return n_done, n_skip, n_err


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2a: within-repository merge.")
    ap.add_argument("--candidates-dir", required=True, type=Path,
                    help="Stage 1 output dir with <analyst_type>.jsonl files")
    ap.add_argument("--buckets-dir", required=True, type=Path,
                    help="where per-repository candidate files are written")
    ap.add_argument("--out-root", required=True, type=Path,
                    help="writes <out-root>/<repo>/rules.jsonl")
    ap.add_argument("--cost-log", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    n = route_candidates_by_repo(sorted(a.candidates_dir.glob("*.jsonl")),
                                 a.buckets_dir)
    print(f"[merge] routed {n} candidates into "
          f"{len(list(a.buckets_dir.glob('*.jsonl')))} repository buckets", flush=True)
    merge_buckets(
        buckets_dir=a.buckets_dir, out_root=a.out_root,
        model=llm.resolve_model(a.model or cfg["models"]["skill_learning"]),
        cost_logger=CostLogger(a.cost_log or a.out_root.parent / "cost_log.jsonl"),
        batch_size=cfg["merge"]["batch_size"], resume=a.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
