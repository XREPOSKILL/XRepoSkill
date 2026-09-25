"""Stage 1: candidate rule discovery from divergence-localized pairs.

Each pair is read by three prompts, one per reference point:
  skipped_step          failed suffix + the successful patch: the earliest
                        step the failed trajectory skipped
  discipline_observer   successful suffix only: up to three steps taken
                        before its first edit
  process_delta         both suffixes at the divergence point: the earliest
                        step only the successful trajectory took
Each prompt returns candidate rules as JSON; a guard rejects a candidate
whose paragraph is too long, whose action rule states bug facts, whose
predicate does not parse, or whose predicate is true on fewer than 10% or
more than 90% of the pool.  A rejected reply is re-prompted once with the
reason and dropped if it fails again.
"""
from __future__ import annotations
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from xreposkill import llm
from xreposkill.config import load_config
from xreposkill.cost import CostLogger, StageSummaryLogger, TranscriptLogger
from xreposkill.fork import load_forks
from xreposkill.pool import (load_pool, load_steps_window, load_submitted_patch,
                             pool_events)
from xreposkill.predicate import occurrence_rate, validate_predicate
from xreposkill.schemas import CATEGORIES, RULE_KINDS, Candidate

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    text = (PROMPTS_DIR / f"analyst_{name}.txt").read_text()
    return (text.replace("{shared_rules}",
                         (PROMPTS_DIR / "_shared_rules.txt").read_text().strip())
                .replace("{shared_predicate}",
                         (PROMPTS_DIR / "_shared_predicate.txt").read_text().strip()))


# ------------------------------------------------------------------ guards

_FILEPATH_LINE = re.compile(
    r"\b[\w./_-]+\.(?:py|js|ts|go|rs|c|cpp|h)(?::\d+|\s+line\s+\d+)\b")
_RARE_SYMBOL = re.compile(r"\b__\w+__\b|\b[A-Z][a-z]+(?:[A-Z][a-z]+){2,}\b")
_ANCHOR = re.compile(r"`[^`]+`|\b[\w-]+/[\w./-]+|\b\w+\.\w{1,5}\b(?!\.)")


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""


def anti_fact_hit(text: str, *, category: str) -> bool:
    """An action rule must not carry single-bug facts (file:line, rare symbols)."""
    if category == "search_recipe":
        return False
    if _FILEPATH_LINE.search(text):
        return True
    return len(_RARE_SYMBOL.findall(text)) >= 3


def fact_anchored(text: str) -> bool:
    """A fact rule must name at least one path, symbol, or command."""
    return bool(_ANCHOR.search(text))


def check_candidate(obj: dict, *, max_chars: int,
                    occurrence_checker: Callable[[str], float] | None = None,
                    occurrence_bounds: tuple[float, float] = (0.10, 0.90),
                    ) -> GuardResult:
    kind = obj.get("kind")
    if kind not in RULE_KINDS:
        return GuardResult(False, "bad_kind")
    category = obj.get("category")
    if category not in CATEGORIES:
        return GuardResult(False, "bad_category")
    rule = obj.get("rule")
    if not isinstance(rule, str) or not rule.strip():
        return GuardResult(False, "empty_rule")
    if len(rule) > max_chars:
        return GuardResult(False, "too_long")
    pred = obj.get("predicate")
    if kind == "action":
        if anti_fact_hit(rule, category=category):
            return GuardResult(False, "anti_fact")
        if not isinstance(pred, str) or not pred.strip():
            return GuardResult(False, "predicate_missing")
        err = validate_predicate(pred)
        if err:
            return GuardResult(False, f"predicate_parse: {err}")
        if occurrence_checker is not None:
            lo, hi = occurrence_bounds
            rate = occurrence_checker(pred)
            if not lo <= rate <= hi:
                return GuardResult(False, f"predicate_occurrence: {rate:.2f} "
                                          f"outside [{lo:.2f}, {hi:.2f}]")
        return GuardResult(True)
    if not fact_anchored(rule):
        return GuardResult(False, "fact_unanchored")
    if isinstance(pred, str) and pred.strip():
        if validate_predicate(pred):
            obj["predicate"] = None     # facts may go without a predicate
    else:
        obj["predicate"] = None
    return GuardResult(True)


def build_occurrence_checker(pool_dir: Path) -> Callable[[str], float]:
    """Pool-wide occurrence rate of a predicate, memoized and thread-safe."""
    events = pool_events(load_pool(pool_dir))
    memo: dict[str, float] = {}
    lock = threading.Lock()

    def rate(expr: str) -> float:
        with lock:
            if expr not in memo:
                memo[expr] = occurrence_rate(expr, events)
            return memo[expr]
    return rate


# ------------------------------------------------------------------ prompts

@dataclass(frozen=True)
class AnalystSpec:
    label: str            # analyst_type in Candidate
    sub: str              # short tag in logs
    sides: tuple[str, ...]
    window_chars: int
    with_patch: bool


ANALYSTS: tuple[AnalystSpec, ...] = (
    AnalystSpec("skipped_step", "A-", ("fail",), 5000, True),
    AnalystSpec("discipline_observer", "A+", ("pass",), 5000, False),
    AnalystSpec("process_delta", "A_diff", ("fail", "pass"), 3500, False),
)


def fork_context(fork: dict | None, pair: dict, pool_dir: Path, *,
                 sides: tuple[str, ...], window_chars: int) -> dict[str, Any]:
    """Shared prefix, divergence excerpts, and continuations for a prompt."""
    fork = fork or {}
    ctx: dict[str, Any] = {
        "shared_prefix_events": fork.get("shared_prefix", []),
        "n_shared_events": fork.get("n_shared", 0),
    }
    for side in sides:
        tid = pair[f"{side}_traj_id"]
        div = fork.get(f"{side}_div_step")
        ctx[f"{side}_traj_id"] = tid
        ctx[f"{side}_continuation_events"] = fork.get(f"{side}_continuation", [])
        ctx[f"{side}_divergence_step"] = div
        ctx[f"{side}_divergence_excerpt"] = load_steps_window(
            pool_dir, tid, start=div or 0, max_chars=window_chars)
    return ctx


def build_user_message(spec: AnalystSpec, pair: dict, fork: dict | None,
                       pool_dir: Path) -> str:
    msg: dict[str, Any] = {"pair_id": pair["pair_id"],
                           "instance_id": pair["instance_id"],
                           "repo": pair.get("repo", "")}
    for side in spec.sides:
        msg[f"{side}_traj_id"] = pair[f"{side}_traj_id"]
    if spec.with_patch:
        msg["successful_patch"] = load_submitted_patch(
            pool_dir, pair["pass_traj_id"])[:4000]
    msg["fork"] = fork_context(fork, pair, pool_dir, sides=spec.sides,
                               window_chars=spec.window_chars)
    return json.dumps(msg, indent=2)


def run_analyst(spec: AnalystSpec, *, pair: dict, pool_dir: Path,
                forks: dict[str, dict], model: str, cost_logger: CostLogger,
                transcript_logger: TranscriptLogger | None = None,
                retry_on_guard_reject: int = 1, max_chars: int = 1000,
                occurrence_checker: Callable[[str], float] | None = None,
                occurrence_bounds: tuple[float, float] = (0.10, 0.90),
                ) -> list[dict[str, Any]]:
    system = load_prompt(spec.label)
    user_msg = build_user_message(spec, pair, forks.get(pair["pair_id"]),
                                  pool_dir)
    for attempt in range(retry_on_guard_reject + 1):
        reply = llm.chat(model=model, system=system, user=user_msg,
                         max_tokens=24000, temperature=0.2, timeout=180)
        cost_logger.log(stage="discover", sub=spec.sub, model=model,
                        in_tokens=reply.in_tokens, out_tokens=reply.out_tokens,
                        cost_usd=reply.cost, pair_id=pair["pair_id"],
                        attempt=attempt)
        parsed = llm.parse_json(reply.text)
        if parsed is None:
            if transcript_logger:
                transcript_logger.log(
                    stage="discover", sub=spec.sub, pair_id=pair["pair_id"],
                    attempt=attempt, prompt_user=user_msg, response=reply.text,
                    parse_ok=False, guard_ok=False, guard_reason="json_parse")
            continue
        objs = parsed if isinstance(parsed, list) else [parsed]
        accepted: list[dict] = []
        first_reason = ""
        for obj in objs:
            if not isinstance(obj, dict):
                first_reason = first_reason or "not_an_object"
                continue
            g = check_candidate(obj, max_chars=max_chars,
                                occurrence_checker=occurrence_checker,
                                occurrence_bounds=occurrence_bounds)
            if g.ok:
                accepted.append({
                    "pair_id": pair["pair_id"], "analyst_type": spec.label,
                    "kind": obj["kind"], "category": obj["category"],
                    "rule": obj["rule"].strip(),
                    "evidence": obj.get("evidence") or {},
                    "predicate": obj.get("predicate")})
            elif not first_reason:
                first_reason = g.reason
        if transcript_logger:
            transcript_logger.log(
                stage="discover", sub=spec.sub, pair_id=pair["pair_id"],
                attempt=attempt, prompt_user=user_msg, response=reply.text,
                parse_ok=True, guard_ok=bool(accepted),
                guard_reason="" if accepted else (first_reason or "empty"),
                extracted=objs[0] if objs and isinstance(objs[0], dict) else None)
        if accepted:
            return accepted
        user_msg += (f"\n\n[GUARD REJECTED: {first_reason or 'empty'}] Fix the "
                     "offending part (fact-shaped content in an action rule, "
                     "an unanchored fact rule, or an unparseable / power-less "
                     "predicate) and re-emit the full JSON.")
    return []


# ------------------------------------------------------------------ driver

def run_discovery(*, pairs_path: Path, forks_path: Path | None, pool_dir: Path,
                  out_dir: Path, failed_path: Path, model: str, workers: int,
                  cost_logger: CostLogger, transcripts_dir: Path | None = None,
                  retry_on_guard_reject: int = 1, max_chars: int = 1000,
                  occurrence_bounds: tuple[float, float] = (0.10, 0.90),
                  occurrence_check: bool = True,
                  analyst_fn: Callable[..., list[dict]] = run_analyst,
                  ) -> dict[str, tuple[int, int]]:
    """Run every (prompt x pair) task through one shared worker pool.

    Returns ``{analyst_type: (n_pairs_with_candidates, n_failures)}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_path = Path(failed_path)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in Path(pairs_path).read_text().splitlines()
             if l.strip()]
    forks = load_forks(forks_path) if forks_path else {}
    checker = build_occurrence_checker(pool_dir) if occurrence_check else None
    transcripts = {s.label: TranscriptLogger(transcripts_dir / f"discover_{s.label}.jsonl")
                   for s in ANALYSTS} if transcripts_dir else {}
    counts = {s.label: [0, 0] for s in ANALYSTS}
    out_locks = {s.label: threading.Lock() for s in ANALYSTS}
    count_lock = threading.Lock()
    fail_lock = threading.Lock()

    def work(spec: AnalystSpec, pair: dict) -> None:
        try:
            cands = analyst_fn(
                spec, pair=pair, pool_dir=pool_dir, forks=forks, model=model,
                cost_logger=cost_logger,
                transcript_logger=transcripts.get(spec.label),
                retry_on_guard_reject=retry_on_guard_reject,
                max_chars=max_chars, occurrence_checker=checker,
                occurrence_bounds=occurrence_bounds)
        except Exception as e:
            with fail_lock, failed_path.open("a") as f:
                f.write(json.dumps({"pair_id": pair["pair_id"],
                                    "analyst": spec.label,
                                    "error": repr(e)}) + "\n")
            with count_lock:
                counts[spec.label][1] += 1
            return
        validated = []
        for c in cands:
            try:
                Candidate.model_validate(c)
                validated.append(c)
            except Exception as e:
                with fail_lock, failed_path.open("a") as f:
                    f.write(json.dumps({"pair_id": pair["pair_id"],
                                        "analyst": spec.label,
                                        "schema_error": repr(e),
                                        "raw": c}) + "\n")
                with count_lock:
                    counts[spec.label][1] += 1
        if validated:
            with out_locks[spec.label], \
                    (out_dir / f"{spec.label}.jsonl").open("a") as f:
                for c in validated:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            with count_lock:
                counts[spec.label][0] += 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, spec, p) for p in pairs for spec in ANALYSTS]
        for fut in as_completed(futs):
            fut.result()
    wall = time.time() - t0
    if transcripts_dir:
        summary = StageSummaryLogger(Path(transcripts_dir) / "stage_summary.jsonl")
        for spec in ANALYSTS:
            n_ok, n_fail = counts[spec.label]
            summary.log(stage="discover", sub=spec.sub, pairs_in=len(pairs),
                        candidates_out=n_ok,
                        outcomes=transcripts[spec.label].summary(
                            stage="discover", sub=spec.sub),
                        wall_seconds=wall, n_fail=n_fail)
    return {label: tuple(c) for label, c in counts.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1: candidate rule discovery.")
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--forks", type=Path, default=None)
    ap.add_argument("--pool-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="writes <out-dir>/<analyst_type>.jsonl")
    ap.add_argument("--failed", type=Path, default=None,
                    help="default <out-dir>/../failed_discover.jsonl")
    ap.add_argument("--cost-log", type=Path, default=None,
                    help="default <out-dir>/../cost_log.jsonl")
    ap.add_argument("--transcripts-dir", type=Path, default=None,
                    help="default <out-dir>/../transcripts")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model", default=None, help="override models.skill_learning")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--occurrence-check", action=argparse.BooleanOptionalAction,
                    default=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    run_root = a.out_dir.parent
    model = llm.resolve_model(a.model or cfg["models"]["skill_learning"])
    counts = run_discovery(
        pairs_path=a.pairs, forks_path=a.forks, pool_dir=a.pool_dir,
        out_dir=a.out_dir, failed_path=a.failed or run_root / "failed_discover.jsonl",
        model=model, workers=a.workers or cfg["discover"]["workers"],
        cost_logger=CostLogger(a.cost_log or run_root / "cost_log.jsonl"),
        transcripts_dir=a.transcripts_dir or run_root / "transcripts",
        retry_on_guard_reject=cfg["discover"]["retry_on_guard_reject"],
        max_chars=cfg["discover"]["rule_max_chars"],
        occurrence_bounds=tuple(cfg["discover"]["occurrence_bounds"]),
        occurrence_check=a.occurrence_check)
    for label, (n_ok, n_fail) in counts.items():
        print(f"[discover/{label}] pairs_with_candidates={n_ok} failures={n_fail}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
