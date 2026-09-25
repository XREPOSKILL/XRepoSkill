#!/usr/bin/env python3
"""Print pass@1 of the baseline and skills arms from the evaluator output.

Usage: summarize_eval.py <eval_dir> <instance_ids.txt>
Reads <eval_dir>/{baseline,skills}/eval_results.json written by the
SWE-bench Pro evaluator ({instance_id: {"resolved": bool, ...}}).
"""
from __future__ import annotations
import json
import sys
from math import comb
from pathlib import Path


def resolved(d: Path) -> set[str]:
    f = d / "eval_results.json"
    if not f.exists():
        return set()
    return {i for i, r in json.loads(f.read_text()).items() if r.get("resolved")}


def main() -> int:
    eval_dir, ids_file = Path(sys.argv[1]), Path(sys.argv[2])
    ids = [l.strip() for l in ids_file.read_text().splitlines() if l.strip()]
    base, sk = resolved(eval_dir / "baseline"), resolved(eval_dir / "skills")
    n = len(ids)
    b = len(sk - base)
    c = len(base - sk)
    p = 2 * sum(comb(b + c, k) for k in range(0, min(b, c) + 1)) / 2 ** (b + c) \
        if b + c else 1.0
    print(f"baseline pass@1: {len(base)}/{n} ({100 * len(base) / n:.1f}%)")
    print(f"skills   pass@1: {len(sk)}/{n} ({100 * len(sk) / n:.1f}%)")
    print(f"skills-only wins: {b}, baseline-only wins: {c}, "
          f"McNemar exact p={min(p, 1.0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
