"""Normalise mini-SWE-agent 2.x trajectories into the trajectory pool.

Input layout:  ``<trajs_root>/<submission>/trajs/<iid>/<iid>.traj.json`` and
``<evals_root>/<submission>/per_instance_details.json``.
Output:        ``<pool_dir>/<submission>.jsonl`` (one NormalizedTraj per line).

Two message encodings occur in the raw files: the chat format
(``{"role": ..., "content": ..., "tool_calls": [...]}``) and the OpenAI
Responses format used by gpt-5-* submissions (assistant turns are response
objects with an ``output`` list; tool results are ``function_call_output``
items).  Both become the same step stream, and Responses-format assistant
turns are rewritten to the chat shape so every consumer reads
``raw["tool_calls"]`` uniformly.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from xreposkill.schemas import NormalizedTraj, TrajStep, repo_from_instance_id


def _extract_model(raw: dict, fallback: str) -> str:
    model_val = ((raw.get("info") or {}).get("config") or {}).get("model")
    if model_val is None:
        return fallback
    if isinstance(model_val, dict):
        return str(model_val.get("model_name") or fallback)
    return str(model_val)


def _is_responses_turn(m: dict) -> bool:
    return m.get("role") is None and isinstance(m.get("output"), list)


def _responses_turn_to_step(m: dict) -> TrajStep:
    texts: list[str] = []
    tool_calls: list[dict] = []
    for item in m["output"]:
        t = item.get("type")
        if t == "message":
            for part in item.get("content") or []:
                if part.get("text"):
                    texts.append(str(part["text"]))
        elif t == "function_call":
            tool_calls.append({
                "id": item.get("call_id", ""), "type": "function",
                "function": {"name": item.get("name", ""),
                             "arguments": item.get("arguments", "")}})
    content = "\n".join(texts)
    kind = "action" if tool_calls else "thought"
    norm_raw = {"role": "assistant", "content": content,
                "tool_calls": tool_calls or None,
                "response_id": m.get("id", "")}
    return TrajStep(kind=kind, content=content or "None", raw=norm_raw)


def _responses_output_to_step(m: dict) -> TrajStep:
    extra = m.get("extra") or {}
    content = str(m.get("output") or extra.get("raw_output") or "")
    norm_raw = {"role": "tool", "content": content, "extra": extra,
                "tool_call_id": m.get("call_id", "")}
    return TrajStep(kind="observation", content=content, raw=norm_raw)


def adapt_trajectory(path: Path, *, submission_id: str,
                     outcome: str) -> NormalizedTraj:
    raw = json.loads(Path(path).read_text())
    instance_id = raw["instance_id"]
    messages = raw.get("messages", [])
    steps: list[TrajStep] = []
    for i, m in enumerate(messages):
        role = m.get("role", "")
        if role == "assistant":
            next_role = (messages[i + 1].get("role", "")
                         if i + 1 < len(messages) else "")
            kind = "action" if next_role == "tool" else "thought"
            steps.append(TrajStep(kind=kind, content=str(m["content"]), raw=m))
        elif role in ("user", "tool"):
            steps.append(TrajStep(kind="observation", content=str(m["content"]),
                                  raw=m))
        elif _is_responses_turn(m):
            steps.append(_responses_turn_to_step(m))
        elif m.get("type") == "function_call_output":
            steps.append(_responses_output_to_step(m))
        # system and exit messages are dropped
    return NormalizedTraj(
        instance_id=instance_id,
        repo=repo_from_instance_id(instance_id),
        agent="mini-swe-agent",
        model=_extract_model(raw, fallback=submission_id.split("_", 2)[-1]),
        outcome=outcome, steps=steps, submission_id=submission_id,
        source_path=str(path))


def load_outcomes(evals_root: Path, submission: str) -> dict[str, str]:
    f = Path(evals_root) / submission / "per_instance_details.json"
    if not f.exists():
        raise FileNotFoundError(f"missing {f}")
    raw = json.loads(f.read_text())
    return {iid: ("pass" if v.get("resolved") else "fail")
            for iid, v in raw.items()}


def ingest_submission(submission: str, trajs_root: Path, evals_root: Path,
                      out_path: Path, *, files: list[Path] | None = None,
                      strict: bool = True) -> int:
    """Write ``out_path`` with one normalised trajectory per raw file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes = load_outcomes(evals_root, submission)
    if files is None:
        files = sorted((Path(trajs_root) / submission / "trajs")
                       .glob("*/*.traj.json"))
    n = 0
    with out_path.open("w") as fout:
        for tp in files:
            iid = json.loads(Path(tp).read_text())["instance_id"]
            if iid not in outcomes:
                if strict:
                    raise KeyError(f"no outcome for {iid} in {submission}")
                continue
            nt = adapt_trajectory(tp, submission_id=submission,
                                  outcome=outcomes[iid])
            fout.write(nt.model_dump_json() + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the trajectory pool.")
    ap.add_argument("--submissions", required=True, type=Path)
    ap.add_argument("--trajs-root", required=True, type=Path)
    ap.add_argument("--evals-root", required=True, type=Path)
    ap.add_argument("--pool-dir", required=True, type=Path)
    ap.add_argument("--strict", action=argparse.BooleanOptionalAction,
                    default=True)
    a = ap.parse_args(argv)
    from xreposkill.download import read_list
    total = 0
    for sub in read_list(a.submissions):
        n = ingest_submission(sub, a.trajs_root, a.evals_root,
                              a.pool_dir / f"{sub}.jsonl", strict=a.strict)
        print(f"[ingest] {sub}: {n} trajectories", flush=True)
        total += n
    print(f"[ingest] pool={a.pool_dir} trajectories={total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
