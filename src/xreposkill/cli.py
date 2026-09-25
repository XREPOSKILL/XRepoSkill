"""``python -m xreposkill <stage> [options]``: one entry point for every stage."""
from __future__ import annotations
import importlib
import sys

STAGES: dict[str, tuple[str, str, str]] = {
    # name: (module, function, one-line help)
    "download":     ("xreposkill.download", "main", "fetch trajectories and pass/fail labels"),
    "ingest":       ("xreposkill.ingest", "main", "build the trajectory pool"),
    "pair":         ("xreposkill.pairing", "main", "pair failed with successful trajectories"),
    "fork":         ("xreposkill.fork", "main", "locate the divergence point of each pair"),
    "discover":     ("xreposkill.discover", "main", "Stage 1: candidate rules"),
    "merge":        ("xreposkill.merge", "main", "Stage 2a: within-repository merge"),
    "judge":        ("xreposkill.judge", "main", "Stage 2b: coarse LLM screen"),
    "score":        ("xreposkill.score", "main", "Stage 2c: resolution gain and z score"),
    "audit":        ("xreposkill.audit", "main", "Stage 2d: n-gram audit"),
    "generalize":   ("xreposkill.generalize", "main", "Stage 2e: cross-repository generalization"),
    "pack":         ("xreposkill.pack", "main", "write SKILL.md files"),
    "retrieve":     ("xreposkill.retrieval", "main", "Stage 3a: two-call rule selection"),
    "rollout":      ("xreposkill.rollout", "main", "Stage 3b: run mini-SWE-agent"),
    "deepswe-prep": ("xreposkill.rollout", "deepswe_prep_main", "Stage 3a for DeepSWE tasks"),
}


def usage() -> str:
    width = max(len(s) for s in STAGES)
    lines = ["usage: python -m xreposkill <stage> [options]", "", "stages:"]
    lines += [f"  {s:<{width}}  {h}" for s, (_, _, h) in STAGES.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    stage, rest = argv[0], argv[1:]
    if stage not in STAGES:
        print(f"unknown stage {stage!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module, fn, _ = STAGES[stage]
    return int(getattr(importlib.import_module(module), fn)(rest) or 0)
