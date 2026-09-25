"""Stage 2d: n-gram overlap audit between rule text and source patches.

A rule that copies solution content out of the successful side's patch
shares long token n-grams with that patch.  Every rule is compared with
the patch submitted by the successful trajectory of each pair it was merged
from; a rule sharing ``n`` or more consecutive tokens is flagged.  The audit
reports, it does not drop.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

from xreposkill.pairing import load_pairs
from xreposkill.pool import load_submitted_patch

_STOP = {"the", "a", "an", "to", "of", "in", "for", "and", "or", "is",
         "be", "before", "after", "with", "not", "this", "that", "it"}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*|\S", text.lower())
            if t not in _STOP]


def ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def overlap_hits(rule_text: str, patch_text: str, *, n: int) -> list[str]:
    shared = ngrams(tokens(rule_text), n) & ngrams(tokens(patch_text), n)
    return sorted(" ".join(g) for g in shared)


def audit_tree(*, rules_root: Path, pairs_path: Path, pool_dir: Path,
               n: int = 6, out: Path | None = None) -> tuple[int, int]:
    pairs = load_pairs(pairs_path)
    rows = []
    n_rules = n_flagged = 0
    for rf in sorted(Path(rules_root).glob("*/rules.jsonl")):
        for line in rf.read_text().splitlines():
            if not line.strip():
                continue
            rule = json.loads(line)
            n_rules += 1
            hits: list[str] = []
            for pid in rule.get("source_pair_ids", []):
                pair = pairs.get(pid)
                if pair is None:
                    continue
                patch = load_submitted_patch(pool_dir, pair["pass_traj_id"])
                if patch:
                    hits += overlap_hits(rule.get("title", "") + "\n"
                                         + rule.get("body", ""), patch, n=n)
            rows.append({"rule_id": rule["rule_id"], "file": str(rf),
                         "flagged": bool(hits), "n_hits": len(set(hits)),
                         "hits": sorted(set(hits))[:10]})
            if hits:
                n_flagged += 1
                print(f"[audit] FLAG {rule['rule_id']}: {sorted(set(hits))[:3]}",
                      flush=True)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[audit] rules={n_rules} flagged={n_flagged} (>= {n}-token overlap)",
          flush=True)
    return n_rules, n_flagged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2d: n-gram audit.")
    ap.add_argument("--rules-root", required=True, type=Path)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--pool-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fail-on-flag", action="store_true")
    a = ap.parse_args(argv)
    _, flagged = audit_tree(rules_root=a.rules_root, pairs_path=a.pairs,
                            pool_dir=a.pool_dir, n=a.n, out=a.out)
    return 1 if (a.fail_on_flag and flagged) else 0


if __name__ == "__main__":
    raise SystemExit(main())
