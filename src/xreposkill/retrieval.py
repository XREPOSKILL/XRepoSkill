"""Stage 3a: issue-adaptive rule selection with two calls to the backbone LLM.

  call 1 (analyze): issue text + the titles of all rules -> a summary of the
          bug and the kinds of guidance that would help
  call 2 (select):  issue text + that summary + every rule with its id,
          evidence tag, title, and body in file (z) order -> at most top_n
          rule ids with a one-line reason each

The result is written to ``<out_dir>/<instance_id>.json``.  The rollout
later injects ONLY the selected rules' title and body, never the summary,
the reasons, or the tags.
"""
from __future__ import annotations
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from xreposkill import llm
from xreposkill.config import load_config

ANALYZE_SYSTEM = (
    "You are the analysis step of a skill-retrieval flow for software bug "
    "fixing. Given a bug report and an index of rule titles distilled from "
    "past fixes, characterise the bug and what guidance would help.\n"
    "Reply with a single JSON object:\n"
    '{"bug_summary": "<2-3 sentences>", '
    '"needed_guidance": ["<short description>", ...]}'
)

SELECT_SYSTEM = (
    "You are the selection step of a skill-retrieval flow for software bug "
    "fixing. Given a bug report, an analysis of it, and a list of rules "
    "(full text), select the rules most likely to help an engineer fix "
    "this bug.\n"
    "Reply with a single JSON object: {\"selected\": [{\"rule_id\": "
    "\"<id verbatim>\", \"reason\": \"<one line>\"}, ...]} — at most "
    "{top_n} entries, most relevant first. If nothing helps, use []."
)

_REPO_BULLET = re.compile(
    r"^- \*\*(?P<title>.+?)\*\* \((?P<tag>[^)]*)\) — (?P<body>.+)$", re.MULTILINE)


@dataclass
class RuleEntry:
    rule_id: str      # "<bucket>/skill/<idx>"
    bucket: str
    title: str
    support: int
    body: str
    tag: str = ""     # evidence tag, e.g. "support 576, z +15.59, HET"


def parse_skill_md(path: Path, bucket: str) -> list[RuleEntry]:
    entries: list[RuleEntry] = []
    for i, m in enumerate(_REPO_BULLET.finditer(Path(path).read_text())):
        sup = re.search(r"support (\d+)", m.group("tag"))
        entries.append(RuleEntry(
            rule_id=f"{bucket}/skill/{i}", bucket=bucket,
            title=m.group("title").strip(),
            support=int(sup.group(1)) if sup else 0,
            body=m.group("body").strip(), tag=m.group("tag").strip()))
    return entries


def load_rules(skills_root: Path) -> list[RuleEntry]:
    """Every rule of a packed skill, bucket by bucket, in file order."""
    root = Path(skills_root) / "repo_conventions"
    entries: list[RuleEntry] = []
    if root.exists():
        for bucket_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            skill_md = bucket_dir / "SKILL.md"
            if skill_md.exists():
                entries.extend(parse_skill_md(skill_md, bucket_dir.name))
    return entries


def render_index(rules: list[RuleEntry]) -> str:
    """One line per rule with its id and title, grouped by bucket."""
    lines: list[str] = []
    bucket = None
    for r in rules:
        if r.bucket != bucket:
            bucket = r.bucket
            lines.append(f"### {bucket}")
        lines.append(f"- [{r.rule_id}] {r.title}")
    return "\n".join(lines)


def render_rules(rules: list[RuleEntry]) -> str:
    """Title and body of the given rules: what the agent receives."""
    return "\n\n".join(f"[{r.rule_id}] {r.title}\n{r.body}" for r in rules)


def default_llm(model: str) -> Callable[[str, str], tuple[str, dict]]:
    def call(system: str, user: str) -> tuple[str, dict]:
        reply = llm.chat(model=model, system=system, user=user,
                         max_tokens=8000, temperature=0.0, timeout=180,
                         reasoning_effort="low", drop_params=True)
        return reply.text, {"in_tokens": reply.in_tokens,
                            "out_tokens": reply.out_tokens}
    return call


class RetrievalFlow:
    """Two-call rule selection; ``llm_fn(system, user) -> (text, usage)``."""

    def __init__(self, *, skills_root: Path, model: str, top_n: int = 3,
                 issue_max_chars: int = 6000,
                 llm_fn: Callable[[str, str], tuple[str, dict]] | None = None):
        self.rules: list[RuleEntry] = load_rules(Path(skills_root))
        self.by_id = {r.rule_id: r for r in self.rules}
        self.model = model
        self.top_n = top_n
        self.issue_max_chars = issue_max_chars
        self.llm_fn = llm_fn or default_llm(model)

    def corpus_block(self) -> str:
        return "\n\n".join(
            f"[{r.rule_id}]{f' ({r.tag})' if r.tag else ''} {r.title}\n{r.body}"
            for r in self.rules)

    def run_instance(self, *, instance_id: str, issue_text: str) -> dict:
        issue = issue_text[:self.issue_max_chars]
        t0 = time.time()
        usage: list[dict] = []
        error = ""
        analysis: dict = {}
        selected: list[dict] = []
        if self.rules:
            try:
                txt, u = self.llm_fn(ANALYZE_SYSTEM,
                                     f"BUG REPORT:\n{issue}\n\nRULE INDEX:\n"
                                     + render_index(self.rules))
                usage.append(u)
                analysis = llm.parse_json_obj(txt)
                txt, u = self.llm_fn(
                    SELECT_SYSTEM.replace("{top_n}", str(self.top_n)),
                    f"BUG REPORT:\n{issue}\n\nANALYSIS:\n"
                    f"{json.dumps(analysis, ensure_ascii=False)}\n\n"
                    f"RULES:\n{self.corpus_block()}")
                usage.append(u)
                for item in llm.parse_json_obj(txt).get("selected", []):
                    rid = item.get("rule_id") if isinstance(item, dict) else None
                    if rid in self.by_id and rid not in [s["rule_id"] for s in selected]:
                        selected.append({"rule_id": rid,
                                         "reason": str(item.get("reason", ""))})
                selected = selected[:self.top_n]
            except Exception as e:  # noqa: BLE001 record, never crash the batch
                error = f"{type(e).__name__}: {e}"
        return {"instance_id": instance_id,
                "bug_summary": analysis.get("bug_summary", ""),
                "needed_guidance": analysis.get("needed_guidance", []),
                "selected": selected, "model": self.model, "usage": usage,
                "seconds": round(time.time() - t0, 2), "error": error}


def selected_rule_ids(retrieval_json: Path) -> list[str]:
    d = json.loads(Path(retrieval_json).read_text())
    return [s["rule_id"] for s in d.get("selected", [])]


def build_skill_block(*, skills_root: Path, instance_id: str,
                      retrieval_dir: Path) -> str:
    """Title and body of the rules selected for ``instance_id``, nothing else."""
    rj = Path(retrieval_dir) / f"{instance_id}.json"
    if not rj.exists():
        raise FileNotFoundError(
            f"retrieval output missing for {instance_id}: {rj}; run "
            "`python -m xreposkill retrieve` first")
    by_id = {r.rule_id: r for r in load_rules(Path(skills_root))}
    rules = [by_id[rid] for rid in selected_rule_ids(rj) if rid in by_id]
    return render_rules(rules)


def run_flow_batch(*, flow: RetrievalFlow, instances: list[dict], out_dir: Path,
                   workers: int = 4, resume: bool = True) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(inst: dict) -> dict:
        iid = inst["instance_id"]
        dest = out_dir / f"{iid}.json"
        if resume and dest.exists():
            return json.loads(dest.read_text())
        res = flow.run_instance(instance_id=iid,
                                issue_text=inst.get("problem_statement", ""))
        dest.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return res

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, inst) for inst in instances]):
            res = fut.result()
            results.append(res)
            print(f"[retrieve] {res['instance_id'][:60]:60s} "
                  f"selected={len(res['selected'])} {res['seconds']:.0f}s"
                  + (f" ERROR={res['error']}" if res["error"] else ""), flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    from xreposkill.instances import (DEFAULT_DATASET, DEFAULT_DOCKERHUB_USER,
                                      read_instance_id_file, resolve_instances)
    ap = argparse.ArgumentParser(description="Stage 3a: two-call rule selection.")
    ap.add_argument("--skills-root", required=True, type=Path)
    ap.add_argument("--instance-ids", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--instances-jsonl", type=Path, default=None,
                    help="read instance dicts from this JSONL instead of Hugging Face")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--dockerhub-user", default=DEFAULT_DOCKERHUB_USER)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model", default=None, help="override models.backbone")
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    instances = resolve_instances(read_instance_id_file(a.instance_ids),
                                  instances_jsonl=a.instances_jsonl,
                                  dataset=a.dataset, dockerhub_user=a.dockerhub_user)
    flow = RetrievalFlow(skills_root=a.skills_root,
                         model=llm.resolve_model(a.model or cfg["models"]["backbone"]),
                         top_n=a.top_n or cfg["retrieval"]["top_n"],
                         issue_max_chars=cfg["retrieval"]["issue_max_chars"])
    print(f"[retrieve] rules={len(flow.rules)} instances={len(instances)} "
          f"model={flow.model}", flush=True)
    results = run_flow_batch(flow=flow, instances=instances, out_dir=a.out_dir,
                             workers=a.workers or cfg["retrieval"]["workers"],
                             resume=not a.no_resume)
    print(f"[retrieve] done={len(results)} "
          f"errors={sum(1 for r in results if r['error'])} "
          f"empty_selection={sum(1 for r in results if not r['selected'])}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
