"""Write the skill as SKILL.md files.

Layout (read back by retrieval.load_rules):

    <out>/repo_conventions/universal__strategies/SKILL.md   transferable rules
    <out>/repo_conventions/<repo_slug>/SKILL.md              residual rules

Each rule is one bullet ``- **title** (tag) — body``; the tag holds the
support, the z score, and HET when the pooled evidence is heterogeneous.
Bullets are in strategies.jsonl order (z descending), so the rule id
``universal__strategies/skill/<i>`` maps to ``strategies[i]``.
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

STRATEGIES_BUCKET = "universal__strategies"


def one_line(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def evidence_tag(*, support: int, z: float | None, het: bool = False) -> str:
    tag = f"support {support}"
    if isinstance(z, (int, float)):
        tag += f", z {z:+.2f}"
    if het:
        tag += ", HET"
    return tag


def strategy_bullet(s: dict) -> str:
    tag = evidence_tag(support=s["support_total"], z=s.get("pooled_z"),
                       het=s.get("homogeneous") is False)
    parts = [f"Trigger: {one_line(s['trigger'])}",
             f"Action: {one_line(s['action'])}"]
    anchors = [one_line(a) for a in (s.get("anchor_examples") or [])[:2]]
    if anchors:
        parts.append(f"Anchors: {' | '.join(anchors)}")
    return f"- **{one_line(s['title'])}** ({tag}) — {' '.join(parts)}"


def residual_bullet(r: dict) -> str:
    tag = evidence_tag(support=r["support"], z=r.get("z"))
    return f"- **{one_line(r['title'])}** ({tag}) — {one_line(r['body'])}"


def write_bucket(root: Path, bucket: str, desc: str, bullets: list[str]) -> None:
    d = Path(root) / "repo_conventions" / bucket
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        "---\n"
        f"name: {bucket}\n"
        "axis: repo_convention\n"
        f"n_rules: {len(bullets)}\n"
        f"description: {desc}\n"
        "---\n\n" + "\n".join(bullets) + "\n")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def pack_skills(*, strategies: list[dict], residual: list[dict], out_root: Path,
                include_residual: bool = True) -> dict[str, int]:
    out_root = Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    write_bucket(out_root, STRATEGIES_BUCKET,
                 "Transferable rules pooled across repositories by behavioral "
                 "equivalence; z is the inverse-variance pooled estimate and "
                 "HET marks heterogeneous evidence across repositories.",
                 [strategy_bullet(s) for s in strategies])
    n_buckets = 0
    if include_residual:
        by_repo: dict[str, list[dict]] = defaultdict(list)
        for r in residual:
            by_repo[r["repo"]].append(r)
        for repo, rs in sorted(by_repo.items()):
            rs.sort(key=lambda r: -(r["z"] if isinstance(r.get("z"), (int, float))
                                    else -99))
            write_bucket(out_root, repo,
                         f"Repository-specific residual rules for {repo}, ordered "
                         "by z score.", [residual_bullet(r) for r in rs])
            n_buckets += 1
    (out_root / "packed_ids.json").write_text(json.dumps(
        {f"{STRATEGIES_BUCKET}/skill/{i}": s.get("strategy_id", str(i))
         for i, s in enumerate(strategies)}, indent=1))
    return {"strategies": len(strategies),
            "residual": len(residual) if include_residual else 0,
            "repo_buckets": n_buckets}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pack the skill into SKILL.md files.")
    ap.add_argument("--generalize-dir", required=True, type=Path,
                    help="Stage 2e output dir (strategies.jsonl, repo_residual.jsonl)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--strategies-only", action="store_true",
                    help="omit the repository-specific residual rules")
    a = ap.parse_args(argv)
    strategies = read_jsonl(a.generalize_dir / "strategies.jsonl")
    residual_path = a.generalize_dir / "repo_residual.jsonl"
    residual = read_jsonl(residual_path) if residual_path.exists() else []
    counts = pack_skills(strategies=strategies, residual=residual, out_root=a.out,
                         include_residual=not a.strategies_only)
    print(f"[pack] {counts} -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
