"""Append-only JSONL loggers for LLM cost, prompt transcripts, and stage summaries."""
from __future__ import annotations
import collections
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CostEvent:
    ts: float
    stage: str
    sub: str
    model: str
    in_tokens: int
    out_tokens: int
    cost_usd: float
    extra: dict[str, Any]


class CostLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, *, stage: str, sub: str, model: str, in_tokens: int,
            out_tokens: int, cost_usd: float, **extra: Any) -> None:
        ev = CostEvent(ts=time.time(), stage=stage, sub=sub, model=model,
                       in_tokens=in_tokens, out_tokens=out_tokens,
                       cost_usd=cost_usd, extra=extra)
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(asdict(ev)) + "\n")


class TranscriptLogger:
    """Per-call record of prompt, reply, and guard verdict, with counters."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str, str], int] = collections.Counter()

    def log(self, *, stage: str, sub: str, pair_id: str, attempt: int,
            prompt_user: str, response: str, parse_ok: bool, guard_ok: bool,
            guard_reason: str = "", extracted: dict | None = None) -> None:
        if not parse_ok:
            outcome = "parse_fail"
        elif guard_ok:
            outcome = "accepted"
        else:
            outcome = f"guard_fail:{guard_reason or 'unknown'}"
        rec = {
            "ts": time.time(), "stage": stage, "sub": sub,
            "pair_id": pair_id, "attempt": attempt, "outcome": outcome,
            "prompt_user_chars": len(prompt_user),
            "prompt_user_preview": prompt_user[:300],
            "response": response, "parse_ok": parse_ok,
            "guard_ok": guard_ok, "guard_reason": guard_reason,
            "kind": (extracted or {}).get("kind"),
            "category": (extracted or {}).get("category"),
        }
        with self._lock:
            self._counts[(stage, sub, outcome)] += 1
            with self.path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def summary(self, *, stage: str, sub: str) -> dict[str, int]:
        with self._lock:
            return {outcome: n for (s, ss, outcome), n in self._counts.items()
                    if s == stage and ss == sub}


class StageSummaryLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, *, stage: str, sub: str, pairs_in: int, candidates_out: int,
            outcomes: dict[str, int], wall_seconds: float, **extra: Any) -> None:
        rec = {"ts": time.time(), "stage": stage, "sub": sub,
               "pairs_in": pairs_in, "candidates_out": candidates_out,
               "pair_yield": (candidates_out / pairs_in) if pairs_in else 0.0,
               "outcomes": outcomes, "wall_seconds": round(wall_seconds, 1),
               **extra}
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
