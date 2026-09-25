import json

import numpy as np

from xreposkill import generalize, llm
from xreposkill.generalize import (build_assignment_jobs, load_scored_rules,
                                   majority_predicate, phi_valley, pool_and_q,
                                   run_generalize)
from tests.conftest import read_jsonl, traj, write_jsonl


def test_phi_valley_finds_cutoff_left_of_rightmost_mode():
    rng = np.random.default_rng(0)
    phis = list(rng.normal(0.0, 0.15, 400)) + list(rng.normal(0.95, 0.02, 60))
    cut = phi_valley(phis)
    assert cut is not None and 0.4 < cut < 0.95
    assert phi_valley(list(rng.normal(0.0, 0.1, 200))) is None


def test_pool_and_q():
    same = [{"z": 4.0, "lift": 0.2}, {"z": 3.0, "lift": 0.21}]
    r = pool_and_q(same)
    assert r["n_scored"] == 2 and r["homogeneous"] is True
    assert abs(r["pooled_lift"] - 0.2) < 0.02 and r["pooled_z"] > 4
    diff = [{"z": 8.0, "lift": 0.4}, {"z": 8.0, "lift": -0.4}]
    assert pool_and_q(diff)["homogeneous"] is False
    one = pool_and_q([{"z": 2.0, "lift": 0.1}, {"z": None, "lift": None}])
    assert one["n_scored"] == 1 and one["pooled_z"] == 2.0 and one["homogeneous"] is None


def test_majority_predicate_and_assignment_jobs():
    assert majority_predicate([{"predicate": "a"}, {"predicate": "a"}, {"predicate": "b"}]) \
        == {"predicate": "a", "agreement": "2/3"}
    assert majority_predicate([{"predicate": None}]) is None
    rules = [{"idx": i, "title": f"t{i}"} for i in range(4)]
    neighbors = {3: [(0, 0.1), (1, 0.2), (2, 0.3)], 2: []}
    jobs, auto_none = build_assignment_jobs(rules, {2, 3}, neighbors,
                                            {0: 7, 1: 7, 2: 9})
    assert auto_none == 1 and len(jobs) == 1
    assert jobs[0]["idx"] == 3 and jobs[0]["groups"] == [(7, [0, 1])]


def _scored_rule(rid, repo, title, predicate, z=3.0, lift=0.3):
    return {"rule_id": rid, "title": title, "body": f"{title} body.",
            "category": "verification", "support": 5, "source_pair_ids": [],
            "predicate": predicate,
            "validation": ({"status": "scored", "z": z, "lift": lift} if z is not None
                           else {"status": "fallback", "reason": "no_predicate"})}


def _synthetic(tmp_path):
    """Pool of 40 trajectories; two behaviours with independent 50% occurrence."""
    rng = np.random.default_rng(1)
    rows = []
    for k in range(40):
        cmds = []
        if rng.random() < 0.5:
            cmds.append("pytest tests/test_a.py")
        if rng.random() < 0.5:
            cmds.append("grep -rn 'def ' src/")
        cmds.append("sed -i s/a/b/ f.py")
        rows.append(traj(f"i{k}", "m", "pass" if k % 2 else "fail", f"s::i{k}", cmds))
    pool = tmp_path / "pool"
    write_jsonl(pool / "s.jsonl", rows)
    root = tmp_path / "rules"
    write_jsonl(root / "django__django" / "rules.jsonl", [
        _scored_rule("d:v:0", "django__django", "Reproduce before editing",
                     "before(run_test, first_edit)"),
        _scored_rule("d:v:1", "django__django", "Grep definitions first",
                     'before(search("def "), first_edit)'),
        _scored_rule("d:v:2", "django__django", "Read the django docs", None, z=None)])
    write_jsonl(root / "sympy__sympy" / "rules.jsonl", [
        _scored_rule("s:v:0", "sympy__sympy", "Run the failing test before edits",
                     "before(run_test, first_edit)", z=2.5, lift=0.28),
        _scored_rule("s:v:1", "sympy__sympy", "Inventory definitions with grep",
                     'before(search("def "), first_edit)', z=2.0, lift=0.2)])
    return pool, root


def test_load_scored_rules_drops_unparseable_predicates(tmp_path):
    _, root = _synthetic(tmp_path)
    write_jsonl(root / "x__y" / "rules.jsonl",
                [_scored_rule("x:v:0", "x__y", "Bad", "frob(")])
    rules = load_scored_rules(root)
    assert len(rules) == 6
    bad = [r for r in rules if r["rule_id"] == "x:v:0"][0]
    assert bad["predicate"] is None and bad["repo"] == "x__y"
    fallback = [r for r in rules if r["rule_id"] == "d:v:2"][0]
    assert fallback["z"] is None


def test_run_generalize_end_to_end(monkeypatch, tmp_path):
    pool, root = _synthetic(tmp_path)
    # embeddings: rules 0/3 alike, 1/4 alike, 2 near 0
    vecs = {"Reproduce before editing": [1, 0], "Run the failing test before edits": [1, 0.1],
            "Grep definitions first": [0, 1], "Inventory definitions with grep": [0.1, 1],
            "Read the django docs": [0.9, 0.2]}

    def embed(texts, model_name):
        X = np.array([vecs[t.split(". ")[0]] for t in texts], dtype=float)
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    monkeypatch.setattr(generalize, "embed_texts", embed)
    calls = []

    def chat(*, model, system, user, **kw):
        calls.append(system[:20])
        if "classify" in system:
            return llm.Reply(text=json.dumps({"assignments": [{"i": 0, "group": -1}]}),
                             in_tokens=1, out_tokens=1, cost=0.0, finish_reason="stop")
        return llm.Reply(text=json.dumps({
            "title": "Reproduce before editing", "trigger": "starting a fix",
            "action": "run the failing test", "anchor_examples": ["django: runtests.py"],
            "outliers": []}), in_tokens=1, out_tokens=1, cost=0.0, finish_reason="stop")
    monkeypatch.setattr(llm, "chat", chat)
    summary = run_generalize(rules_root=root, pool_dir=pool, out_dir=tmp_path / "g",
                             model="m", knn=3, min_fires=5, workers=2,
                             phi_cutoff=0.8)
    assert summary["n_powered"] == 4 and summary["n_unresolved"] == 1
    assert summary["n_strategy_components"] == 2 and summary["n_residual_rules"] == 1
    strategies = read_jsonl(tmp_path / "g" / "strategies.jsonl")
    assert len(strategies) == 2 and all(len(s["repos"]) == 2 for s in strategies)
    assert strategies[0]["pooled_z"] >= strategies[1]["pooled_z"]
    assert strategies[0]["predicate_majority"]["agreement"] == "2/2"
    residual = read_jsonl(tmp_path / "g" / "repo_residual.jsonl")
    assert residual[0]["rule_id"] == "d:v:2"
    assert len(read_jsonl(tmp_path / "g" / "assignments.jsonl")) == 1
    assert (tmp_path / "g" / "summary.json").exists()


def test_run_generalize_dry_run_and_abort(monkeypatch, tmp_path):
    pool, root = _synthetic(tmp_path)
    monkeypatch.setattr(generalize, "embed_texts",
                        lambda texts, m: np.eye(len(texts)))
    s = run_generalize(rules_root=root, pool_dir=pool, out_dir=tmp_path / "g",
                       model="m", knn=2, min_fires=5, phi_cutoff=0.8, dry_run=True)
    assert s["dry_run"] and s["n_jobs"] == 1
    s = run_generalize(rules_root=root, pool_dir=pool, out_dir=tmp_path / "g2",
                       model="m", knn=2, min_fires=5)
    assert s.get("aborted") == "no_phi_cutoff"      # 6 phi values: no bimodal histogram
