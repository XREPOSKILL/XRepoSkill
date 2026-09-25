"""End-to-end smoke test with every LLM call mocked."""
import json
import shutil

import numpy as np
import pytest

from xreposkill import generalize, llm
from xreposkill.audit import audit_tree
from xreposkill.cost import CostLogger
from xreposkill.discover import run_discovery
from xreposkill.fork import build_forks
from xreposkill.generalize import run_generalize
from xreposkill.ingest import ingest_submission
from xreposkill.judge import judge_rules
from xreposkill.merge import merge_buckets, route_candidates_by_repo
from xreposkill.pack import pack_skills, read_jsonl
from xreposkill.pairing import select_pairs
from xreposkill.pool import load_pool
from xreposkill.retrieval import RetrievalFlow, build_skill_block, run_flow_batch
from xreposkill.rollout import process_instance, run_batch
from xreposkill.score import score_tree

pytestmark = pytest.mark.slow


def _fake_chat(*, model, system, user, **kw):
    """Route every mocked LLM call by the prompt it receives."""
    if "Acquisition-Discipline Analyst" in system:
        text = json.dumps({"kind": "action", "category": "replay",
                           "rule": "Run the failing test before editing any file.",
                           "predicate": "before(run_test, first_edit)",
                           "evidence": {"fail_traj_id": "sub-A::x__x-1",
                                        "skipped_step_index": 1,
                                        "gold_patch_hunk_ref": "x.py:1"}})
    elif "Acquisition-Discipline Observer" in system:
        text = json.dumps([{"kind": "fact", "category": "lookup",
                            "rule": "x: tests live under tests/test_x.py; read them first.",
                            "predicate": None,
                            "evidence": {"pass_traj_id": "sub-B::x__x-1"}}])
    elif "Process-Delta Analyst" in system:
        text = json.dumps({"kind": "action", "category": "verification",
                           "rule": "Re-run the test suite after each edit.",
                           "predicate": "after(run_test, first_edit)",
                           "evidence": {"fail_traj_id": "sub-A::x__x-1",
                                        "pass_traj_id": "sub-B::x__x-1",
                                        "skipped_step_index": 1,
                                        "cited_pass_step_index": 2}})
    elif "merging candidate rules" in system:
        n = user.count("\n[") + 1
        text = json.dumps({"rules": [{"title": "Reproduce before editing",
                                      "body": "Run the failing test first.",
                                      "support_indices": list(range(n))}]})
    elif "Rule-Quality Judge" in system:
        text = json.dumps({"is_discipline": 1, "not_redundant_with_prior": 0,
                           "behavior_actionable": 1, "not_repo_conflicting": 1})
    elif "classify process rules" in system:
        text = json.dumps({"assignments": []})
    elif "deduplicate rule candidates" in system:
        text = json.dumps({"title": "Reproduce before editing", "trigger": "t",
                           "action": "a", "anchor_examples": [], "outliers": []})
    elif "analysis step" in system:
        text = json.dumps({"bug_summary": "s", "needed_guidance": []})
    elif "selection step" in system:
        text = json.dumps({"selected": [{"rule_id": "universal__strategies/skill/0",
                                         "reason": "r"}]})
    else:
        raise AssertionError(f"unexpected prompt: {system[:60]}")
    return llm.Reply(text=text, in_tokens=1, out_tokens=1, cost=0.0, finish_reason="stop")


def test_full_pipeline_smoke(fixtures, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "chat", _fake_chat)
    monkeypatch.setattr(generalize, "embed_texts",
                        lambda texts, m: np.ones((len(texts), 2)) / np.sqrt(2))
    smoke = tmp_path / "smoke"
    shutil.copytree(fixtures / "smoke", smoke)
    run = tmp_path / "run"
    pool_dir = run / "traj_pool"

    # ingest: fixture pair on one instance, plus synthetic issues of a second
    # repository so the cross-repository stage has two repositories to merge
    for sub in ("sub-A", "sub-B"):
        assert ingest_submission(sub, smoke / "trajs_root" / "bash-only",
                                 smoke / "eval_root", pool_dir / f"{sub}.jsonl") == 1
    from tests.conftest import traj, write_jsonl
    # per issue: two failed and two successful trajectories of four backbones,
    # so that after the pair sides are excluded every issue keeps one
    # trajectory of each outcome for the held-out scoring
    TEST_FIRST = ["pytest t.py", "sed -i s/a/b/ f.py", "pytest t.py"]
    EDIT_ONLY = ["sed -i s/a/b/ f.py"]
    rows = []
    for repo in ("y", "z"):
        for k in range(6):
            for m, outcome, cmds in (("weak", "fail", EDIT_ONLY), ("mid2", "fail", EDIT_ONLY),
                                     ("mid1", "pass", TEST_FIRST), ("strong", "pass", TEST_FIRST)):
                rows.append(traj(f"{repo}__{repo}-{k}", m, outcome,
                                 f"sub-{m}::{repo}__{repo}-{k}", cmds, repo=f"{repo}/{repo}"))
        # one failed test-first trajectory so the separation is imperfect
        # (a perfectly separated stratum has zero Sato variance)
        rows.append(traj(f"{repo}__{repo}-0", "zextra", "fail", f"sub-zextra::{repo}__{repo}-0",
                         TEST_FIRST, repo=f"{repo}/{repo}"))
    for m in ("weak", "mid2", "mid1", "strong", "zextra"):
        write_jsonl(pool_dir / f"sub-{m}.jsonl", [r for r in rows if r["model"] == m])
    pool = load_pool(pool_dir)
    assert len(pool) == 52

    # pair + fork (the fixture pair shares one backbone, so it has no
    # positive resolution-rate gap and yields no pair)
    pairs, stats = select_pairs(pool, K=1)
    assert stats["n_frontier"] == 13 and stats["n_no_positive_gap"] == 1
    assert len(pairs) == 12
    pairs_path = run / "pairs" / "pairs.jsonl"
    write_jsonl(pairs_path, [p.model_dump() for p in pairs])
    n_ok, _ = build_forks(pairs_path=pairs_path, pool_dir=pool_dir,
                          out_path=run / "pairs" / "forks.jsonl")
    assert n_ok == len(pairs)

    # Stage 1
    counts = run_discovery(pairs_path=pairs_path, forks_path=run / "pairs" / "forks.jsonl",
                           pool_dir=pool_dir, out_dir=run / "candidates",
                           failed_path=run / "failed.jsonl", model="m", workers=2,
                           cost_logger=CostLogger(run / "cost.jsonl"),
                           occurrence_bounds=(0.0, 1.0))
    assert all(c[1] == 0 for c in counts.values())
    assert len(list((run / "candidates").glob("*.jsonl"))) == 3

    # Stage 2
    n = route_candidates_by_repo(sorted((run / "candidates").glob("*.jsonl")), run / "buckets")
    assert n == 3 * len(pairs)
    merge_buckets(buckets_dir=run / "buckets", out_root=run / "rules_merged", model="m",
                  cost_logger=CostLogger(run / "cost.jsonl"), resume=False)
    assert sorted(p.name for p in (run / "rules_merged").iterdir()) == ["y__y", "z__z"]

    from xreposkill.judge import LLMJudge
    judge = LLMJudge(model="m", cost_logger=CostLogger(run / "cost.jsonl"),
                     dims=("is_discipline", "not_repo_conflicting"))
    n_keep, n_drop = judge_rules(rules_root=run / "rules_merged", out=run / "rules_judged",
                                 judge=judge, judge_log_path=run / "judge.jsonl", workers=2)
    assert n_keep > 0 and n_drop == 0

    stats = score_tree(rules_root=run / "rules_judged", out=run / "rules_scored",
                       pool_dir=pool_dir, pairs_path=pairs_path)
    assert stats["scored"] >= 1
    y_rules = read_jsonl(run / "rules_scored" / "y__y" / "rules.jsonl")
    scored = [r for r in y_rules if r["validation"]["status"] == "scored"]
    assert scored and scored[0]["validation"]["z"] > 0
    audit_tree(rules_root=run / "rules_scored", pairs_path=pairs_path, pool_dir=pool_dir,
               out=run / "audit.jsonl")

    summary = run_generalize(rules_root=run / "rules_scored", pool_dir=pool_dir,
                             out_dir=run / "generalize", model="m", knn=5, min_fires=1,
                             workers=2, phi_cutoff=0.5)
    assert summary["n_strategy_components"] >= 1
    strategies = read_jsonl(run / "generalize" / "strategies.jsonl")
    residual_path = run / "generalize" / "repo_residual.jsonl"
    residual = read_jsonl(residual_path) if residual_path.exists() else []
    counts = pack_skills(strategies=strategies, residual=residual, out_root=run / "skills")
    assert counts["strategies"] == len(strategies)

    # Stage 3
    flow = RetrievalFlow(skills_root=run / "skills", model="m")
    insts = [{"instance_id": "new__repo-1", "problem_statement": "a fresh bug"}]
    run_flow_batch(flow=flow, instances=insts, out_dir=run / "retrieval", workers=1)
    block = build_skill_block(skills_root=run / "skills", instance_id="new__repo-1",
                              retrieval_dir=run / "retrieval")
    assert block.startswith("[universal__strategies/skill/0] Reproduce before editing")

    class Agent:
        def __init__(self, *a, **k):
            self.config = type("C", (), {"output_path": None})()
            self.cost, self.n_calls = 0.0, 1

        def run(self, task, **kw):
            assert kw["skill_block"] == block
            return {"exit_status": "Submitted", "submission": "diff"}

        def save(self, *a):
            pass

    def proc(*, instance, out_dir, skill_block, mini_config):
        return process_instance(instance=instance, out_dir=out_dir, skill_block=skill_block,
                                mini_config=mini_config, agent_factory=lambda m, e: Agent(),
                                env_factory=lambda: None, model_factory=lambda: None)
    results = run_batch(instances=insts, out_dir=run / "rollout", condition="skills",
                        mini_config={"model": {"model_name": "m"}},
                        skills_root=run / "skills", retrieval_dir=run / "retrieval",
                        workers=1, process_fn=proc)
    assert results[0]["exit_status"] == "Submitted"
    assert json.loads((run / "rollout" / "preds.json").read_text())["new__repo-1"]["model_patch"] == "diff"
