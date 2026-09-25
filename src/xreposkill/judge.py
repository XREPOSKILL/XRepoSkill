"""Stage 2b: coarse LLM screen of merged rules.

The judge scores four binary dimensions; a rule is kept when every
dimension in ``judge.required_dims`` (default: is_discipline and
not_repo_conflicting) is 1.  An unparseable verdict is retried once and
then fails OPEN, because the resolution gain of Stage 2c is the main
filter and a rule must not vanish on a transport error.
"""
from __future__ import annotations
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

from xreposkill import llm
from xreposkill.config import load_config
from xreposkill.cost import CostLogger
from xreposkill.schemas import Rule

JUDGE_PROMPT = (Path(__file__).parent / "prompts" / "judge.txt").read_text()
ALL_DIMS = ("is_discipline", "not_redundant_with_prior",
            "behavior_actionable", "not_repo_conflicting")


class Judge(Protocol):
    def evaluate(self, *, section_text: str, bucket: str
                 ) -> tuple[bool, dict[str, Any]]: ...


class LLMJudge:
    def __init__(self, *, model: str, cost_logger: CostLogger,
                 dims: tuple[str, ...] = ALL_DIMS):
        self.model = model
        self.cost = cost_logger
        self.dims = tuple(dims)

    def evaluate(self, *, section_text: str, bucket: str
                 ) -> tuple[bool, dict[str, Any]]:
        user = json.dumps({"bucket": bucket, "rule": section_text}, indent=2)
        d: dict | None = None
        text = ""
        for _attempt in (1, 2):
            reply = llm.chat(model=self.model, system=JUDGE_PROMPT, user=user,
                             max_tokens=24000, temperature=0.0, timeout=180)
            text = reply.text
            self.cost.log(stage="judge", sub="judge", model=self.model,
                          in_tokens=reply.in_tokens, out_tokens=reply.out_tokens,
                          cost_usd=reply.cost, bucket=bucket)
            d = llm.parse_json_obj(text) or None
            if d:
                break
        if d is None:
            return True, {"error": "judge_unavailable_fail_open",
                          "_raw_response": text}
        keep = all(d.get(k) == 1 for k in self.dims)
        d["_raw_response"] = text
        return keep, d


def load_keep_map(judge_log_path: Path, dims: tuple[str, ...] | None = None
                  ) -> dict[str, bool]:
    """Prior verdicts from a judge log; recomputed against ``dims`` when given."""
    out: dict[str, bool] = {}
    p = Path(judge_log_path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = rec.get("rule_id")
        if not rid:
            continue
        dec = rec.get("decision")
        if dims is not None and isinstance(dec, dict) and not dec.get("error"):
            out[rid] = all(dec.get(k) == 1 for k in dims)
        else:
            out[rid] = bool(rec.get("keep"))
    return out


def judge_rules(*, rules_root: Path, out: Path, judge: Judge,
                judge_log_path: Path, workers: int = 1,
                dims: tuple[str, ...] | None = None) -> tuple[int, int]:
    """Judge every ``<rules_root>/<bucket>/rules.jsonl``; mirror kept rules to ``out``."""
    rules_root, out = Path(rules_root), Path(out)
    judge_log_path = Path(judge_log_path)
    judge_log_path.parent.mkdir(parents=True, exist_ok=True)
    keep_map = load_keep_map(judge_log_path, dims)
    rules_files = sorted(rules_root.glob("*/rules.jsonl"))
    work: list[tuple[str, Rule]] = []
    for rf in rules_files:
        bucket = rf.parent.name
        for line in rf.read_text().splitlines():
            if line.strip():
                r = Rule.model_validate_json(line)
                if r.rule_id not in keep_map:
                    work.append((bucket, r))

    log_lock = threading.Lock()
    log_f = judge_log_path.open("a")

    def one(item: tuple[str, Rule]) -> tuple[str, bool]:
        bucket, r = item
        section = r.title + "\n" + r.body
        keep, dec = judge.evaluate(section_text=section, bucket=bucket)
        with log_lock:
            log_f.write(json.dumps({"bucket": bucket, "rule_id": r.rule_id,
                                    "section": section, "keep": keep,
                                    "decision": dec}, ensure_ascii=False) + "\n")
            log_f.flush()
        return r.rule_id, keep

    try:
        if workers <= 1:
            for it in work:
                rid, keep = one(it)
                keep_map[rid] = keep
        elif work:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for fut in as_completed([ex.submit(one, it) for it in work]):
                    rid, keep = fut.result()
                    keep_map[rid] = keep
    finally:
        log_f.close()

    n_keep = n_drop = 0
    for rf in rules_files:
        dest = out / rf.parent.name / "rules.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        kept = []
        for line in rf.read_text().splitlines():
            if not line.strip():
                continue
            r = Rule.model_validate_json(line)
            if keep_map.get(r.rule_id, False):
                kept.append(r)
                n_keep += 1
            else:
                n_drop += 1
        with dest.open("w") as f:
            for r in kept:
                f.write(r.model_dump_json() + "\n")
    return n_keep, n_drop


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2b: coarse LLM screen.")
    ap.add_argument("--rules-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--judge-log", type=Path, default=None)
    ap.add_argument("--cost-log", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    dims = tuple(cfg["judge"]["required_dims"]) or ALL_DIMS
    judge = LLMJudge(
        model=llm.resolve_model(a.model or cfg["models"]["skill_learning"]),
        cost_logger=CostLogger(a.cost_log or a.out.parent / "cost_log.jsonl"),
        dims=dims)
    n_keep, n_drop = judge_rules(
        rules_root=a.rules_root, out=a.out, judge=judge,
        judge_log_path=a.judge_log or a.out.parent / "judge_log.jsonl",
        workers=a.workers or cfg["judge"]["workers"], dims=dims)
    print(f"[judge] kept={n_keep} dropped={n_drop} dims={list(dims)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
