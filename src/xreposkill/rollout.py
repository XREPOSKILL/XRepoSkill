"""Stage 3b: run mini-SWE-agent with the selected rules in its first message.

Per instance the selected rules (title and body) are rendered into the
``{{ skill_block }}`` variable of the mini-SWE-agent config
(configs/mini_swe_agent/swebench_pro_with_skills.yaml), which places them
above the issue description with the framing that they are guidance on how
to investigate and must be verified against the code.  The baseline
condition passes an empty block, so the agent runs without added text.

Anti-leak guards for SWE-bench Pro images: the container's git history is
rebuilt as one baseline commit (the images ship the fix commit), and the
config runs the container with ``--network none``.

Outputs under ``--out-dir``: ``<iid>/{<iid>.patch, <iid>.traj.json,
skill_block.txt}``, ``results.jsonl``, ``preds.json`` (for the SWE-bench Pro
evaluator), ``instances.jsonl``.
"""
from __future__ import annotations
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import yaml

from xreposkill.config import load_config
from xreposkill.instances import (DEFAULT_DATASET, DEFAULT_DOCKERHUB_USER,
                                  read_instance_id_file, resolve_instances,
                                  write_instances_jsonl)
from xreposkill.retrieval import RetrievalFlow, build_skill_block

CONDITIONS = ("baseline", "skills")
DEFAULT_MINI_CONFIG = (Path(__file__).resolve().parents[2] / "configs"
                       / "mini_swe_agent" / "swebench_pro_with_skills.yaml")

# The sweap images ship /app with full git history, including the instance's
# own fix commit.  Rebuild .git as a single baseline commit over exactly the
# tracked files so `git show <fix>` cannot reveal the patch while
# `git diff`-based submission keeps working.  Runs in the container's
# writable layer, so the image on disk is untouched.
GIT_HISTORY_GUARD = (
    "set -e; cd /app; "
    "git config --global --add safe.directory /app >/dev/null 2>&1 || true; "
    "git ls-files -z > /tmp/.tracked_files; "
    "rm -rf .git; git init -q; "
    "git config user.email eval@local; git config user.name eval; "
    "xargs -0 -r git add -f -- < /tmp/.tracked_files; "
    "git commit -qm baseline; rm -f /tmp/.tracked_files; "
    'test "$(git rev-list --all | wc -l)" = 1'
)

SKILL_FRAME = """<relevant_skills>
The following skills were retrieved from a distilled corpus of past
SWE-bench fixes because they may apply to the task below. Treat them as
guidance for HOW to investigate and acquire facts at runtime, not as
answers. Verify each step against the actual code before acting.

{block}
</relevant_skills>
"""


def apply_git_history_guard(env) -> None:
    res = env.execute({"command": GIT_HISTORY_GUARD}, timeout=600)
    if res.get("returncode") != 0:
        raise RuntimeError(f"git_history_guard failed (rc={res.get('returncode')}): "
                           f"{str(res.get('output', ''))[:300]}")


def skill_block_for(condition: str, *, skills_root: Path | None,
                    instance_id: str, retrieval_dir: Path | None) -> str:
    if condition == "baseline":
        return ""
    if condition == "skills":
        if skills_root is None or retrieval_dir is None:
            raise ValueError("condition 'skills' needs --skills-root and --retrieval-dir")
        return build_skill_block(skills_root=skills_root, instance_id=instance_id,
                                 retrieval_dir=retrieval_dir)
    raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")


def derive_mini_config(base: dict, *, model: str | None = None,
                       reasoning_effort: str | None = None,
                       container_timeout: str | None = None,
                       max_format_errors: int | None = None) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg.setdefault("model", {})
    if model:
        cfg["model"]["model_name"] = model
    if reasoning_effort:
        cfg["model"].setdefault("model_kwargs", {})
        cfg["model"]["model_kwargs"]["reasoning_effort"] = reasoning_effort
        cfg["model"]["model_kwargs"]["drop_params"] = True
    if container_timeout:
        cfg.setdefault("environment", {})["container_timeout"] = container_timeout
    if max_format_errors is not None:
        cfg.setdefault("agent", {})["max_consecutive_format_errors"] = max_format_errors
    return cfg


def process_instance(*, instance: dict, out_dir: Path, skill_block: str,
                     mini_config: dict, agent_factory: Callable | None = None,
                     env_factory: Callable | None = None,
                     model_factory: Callable | None = None) -> dict:
    """Run one instance; the factories are test hooks, production uses mini-SWE-agent."""
    instance_id = instance["instance_id"]
    instance_dir = Path(out_dir) / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / "skill_block.txt").write_text(skill_block)

    if model_factory is None:
        from minisweagent.models import get_model
        model_factory = lambda: get_model(config=mini_config.get("model", {}))
    if env_factory is None:
        from minisweagent.run.benchmarks.swebench import get_sb_environment

        def env_factory():
            env = get_sb_environment(mini_config, instance)
            apply_git_history_guard(env)
            return env
    if agent_factory is None:
        from minisweagent.agents.default import DefaultAgent

        def agent_factory(model, env):
            return DefaultAgent(model, env, **mini_config.get("agent", {}))

    model = model_factory()
    env = env_factory()
    agent = agent_factory(model, env)
    traj_path = instance_dir / f"{instance_id}.traj.json"
    agent.config.output_path = traj_path

    exit_status, submission, error = "", "", None
    try:
        info = agent.run(task=instance["problem_statement"], skill_block=skill_block)
        exit_status = info.get("exit_status", "")
        submission = info.get("submission", "") or ""
    except Exception as e:  # surface to results.jsonl, do not crash the batch
        exit_status, error = type(e).__name__, str(e)
    finally:
        try:
            agent.save(traj_path, {"info": {"exit_status": exit_status,
                                            "submission": submission}})
        except Exception:
            pass
    patch_path = instance_dir / f"{instance_id}.patch"
    patch_path.write_text(submission)
    return {"instance_id": instance_id, "exit_status": exit_status,
            "patch_path": str(patch_path), "traj_path": str(traj_path),
            "skill_block_chars": len(skill_block),
            "cost": getattr(agent, "cost", 0.0),
            "n_calls": getattr(agent, "n_calls", 0), "error": error}


def _read(p) -> str:
    try:
        return Path(p).read_text() if p else ""
    except Exception:
        return ""


def run_batch(*, instances: list[dict], out_dir: Path, condition: str,
              mini_config: dict, skills_root: Path | None = None,
              retrieval_dir: Path | None = None, workers: int = 4,
              process_fn: Callable = process_instance) -> list[dict]:
    """Run every instance in ``workers`` threads; write results.jsonl and preds.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = mini_config.get("model", {}).get("model_name", "?")
    results_f = (out_dir / "results.jsonl").open("a")
    lock = threading.Lock()

    def one(inst: dict) -> dict:
        iid = inst["instance_id"]
        try:
            block = skill_block_for(condition, skills_root=skills_root,
                                    instance_id=iid, retrieval_dir=retrieval_dir)
            res = process_fn(instance=inst, out_dir=out_dir, skill_block=block,
                             mini_config=mini_config)
        except Exception as e:
            res = {"instance_id": iid, "exit_status": f"driver_crash:{type(e).__name__}",
                   "patch_path": "", "traj_path": "", "skill_block_chars": 0,
                   "cost": 0.0, "n_calls": 0, "error": str(e)[:300]}
        with lock:
            results_f.write(json.dumps(res) + "\n")
            results_f.flush()
        return res

    start = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, fut in enumerate(as_completed([ex.submit(one, i) for i in instances]), 1):
            res = fut.result()
            results.append(res)
            print(f"[rollout {k}/{len(instances)}] {res['instance_id'][:60]:60s} "
                  f"exit={res['exit_status']:<14} patch={len(_read(res.get('patch_path'))):>6} "
                  f"calls={res.get('n_calls', 0):>3} cost=${res.get('cost', 0):.2f} "
                  f"elapsed={time.time() - start:.0f}s", flush=True)
    results_f.close()
    preds = {r["instance_id"]: {"model_name_or_path": model_name,
                                "instance_id": r["instance_id"],
                                "model_patch": _read(r.get("patch_path"))}
             for r in results}
    (out_dir / "preds.json").write_text(json.dumps(preds, indent=2))
    n_sub = sum(1 for r in results if r.get("exit_status") == "Submitted")
    n_patch = sum(1 for r in results if _read(r.get("patch_path")))
    print(f"[rollout] total={len(results)} submitted={n_sub} with_patch={n_patch} "
          f"cost=${sum(r.get('cost', 0.0) for r in results):.2f} "
          f"wall={time.time() - start:.0f}s -> {out_dir / 'preds.json'}", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3b: skill-guided rollout.")
    ap.add_argument("--instance-ids", required=True, type=Path)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--skills-root", type=Path, default=None)
    ap.add_argument("--retrieval-dir", type=Path, default=None,
                    help="Stage 3a output dir (required for condition skills)")
    ap.add_argument("--instances-jsonl", type=Path, default=None)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--dockerhub-user", default=DEFAULT_DOCKERHUB_USER)
    ap.add_argument("--mini-config", type=Path, default=DEFAULT_MINI_CONFIG)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model", default=None, help="backbone LLM (litellm name)")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--container-timeout", default=None, help="e.g. 6h")
    ap.add_argument("--max-format-errors", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.condition == "skills" and (a.skills_root is None or a.retrieval_dir is None):
        print("FATAL: condition skills needs --skills-root and --retrieval-dir",
              file=sys.stderr)
        return 2
    a.out_dir.mkdir(parents=True, exist_ok=True)
    instances = resolve_instances(read_instance_id_file(a.instance_ids),
                                  instances_jsonl=a.instances_jsonl,
                                  dataset=a.dataset, dockerhub_user=a.dockerhub_user)
    write_instances_jsonl(instances, a.out_dir / "instances.jsonl")
    mini_cfg = derive_mini_config(
        yaml.safe_load(a.mini_config.read_text()),
        model=a.model or cfg["models"]["backbone"],
        reasoning_effort=a.reasoning_effort, container_timeout=a.container_timeout,
        max_format_errors=a.max_format_errors)
    (a.out_dir / "mini_config.yaml").write_text(
        yaml.safe_dump(mini_cfg, sort_keys=False, allow_unicode=True))
    print(f"[rollout] model={mini_cfg['model'].get('model_name')} "
          f"condition={a.condition} instances={len(instances)}", flush=True)
    run_batch(instances=instances, out_dir=a.out_dir, condition=a.condition,
              mini_config=mini_cfg, skills_root=a.skills_root,
              retrieval_dir=a.retrieval_dir,
              workers=a.workers or cfg["rollout"]["workers"])
    return 0


# ------------------------------------------------------------ DeepSWE prep

def prepare_deepswe_task(*, task_dir: Path, skills_root: Path, model: str,
                         out_dir: Path, top_n: int = 3,
                         flow: RetrievalFlow | None = None) -> tuple[dict, Path]:
    """Run Stage 3a on a DeepSWE task and write the Pier prompt template.

    Writes ``<out_dir>/<task_id>.json`` (retrieval record) and
    ``<out_dir>/<task_id>.prompt.j2``, which Pier's mini-SWE-agent adapter
    consumes via ``--ak prompt_template_path=<file>``.
    """
    task_id = Path(task_dir).name
    issue = (Path(task_dir) / "instruction.md").read_text()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flow = flow or RetrievalFlow(skills_root=skills_root, model=model, top_n=top_n)
    res = flow.run_instance(instance_id=task_id, issue_text=issue)
    (out_dir / f"{task_id}.json").write_text(json.dumps(res, indent=1))
    block = "" if res.get("error") else build_skill_block(
        skills_root=skills_root, instance_id=task_id, retrieval_dir=out_dir)
    tmpl = out_dir / f"{task_id}.prompt.j2"
    tmpl.write_text((SKILL_FRAME.format(block=block) + "\n{{ instruction }}\n")
                    if block.strip() else "{{ instruction }}\n")
    return res, tmpl


def deepswe_prep_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3a for DeepSWE tasks (Pier).")
    ap.add_argument("--task-dir", required=True, type=Path, nargs="+")
    ap.add_argument("--skills-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--top-n", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    from xreposkill import llm
    model = llm.resolve_model(a.model or cfg["models"]["backbone"])
    flow = RetrievalFlow(skills_root=a.skills_root, model=model,
                         top_n=a.top_n or cfg["retrieval"]["top_n"])
    rc = 0
    for td in a.task_dir:
        res, tmpl = prepare_deepswe_task(task_dir=td, skills_root=a.skills_root,
                                         model=model, out_dir=a.out_dir, flow=flow)
        status = f"ERROR={res['error']}" if res.get("error") else \
            f"selected={len(res.get('selected', []))}"
        print(f"[deepswe-prep] {td.name}: {status} -> {tmpl}", flush=True)
        rc = rc or (1 if res.get("error") else 0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
