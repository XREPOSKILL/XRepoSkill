import json
from unittest.mock import patch

from xreposkill import llm
from xreposkill.audit import audit_tree, overlap_hits
from xreposkill.cost import CostLogger
from xreposkill.judge import LLMJudge, judge_rules, load_keep_map
from xreposkill.merge import (group_by_category, merge_buckets, merge_rules,
                              representative_predicate, route_candidates_by_repo)
from xreposkill.schemas import Rule
from tests.conftest import read_jsonl, traj, write_jsonl


def _cand(pair_id, kind="action", category="lookup", rule="body", predicate=None):
    return {"pair_id": pair_id, "analyst_type": "discipline_observer", "kind": kind,
            "category": category, "rule": rule, "evidence": {"pass_traj_id": "t"},
            "predicate": predicate}


def _reply(text, finish="stop"):
    return llm.Reply(text=text, in_tokens=10, out_tokens=10, cost=0.0,
                     finish_reason=finish)


def test_route_by_repo_and_group(tmp_path):
    src = write_jsonl(tmp_path / "c.jsonl", [
        _cand("django__django-1::f::p"), _cand("django__django-2::f::p", kind="fact"),
        _cand("sphinx-doc__sphinx-8721::f::p", category="verification")])
    n = route_candidates_by_repo([src], tmp_path / "buckets")
    assert n == 3
    names = sorted(p.stem for p in (tmp_path / "buckets").glob("*.jsonl"))
    assert names == ["django__django", "sphinx-doc__sphinx"]
    assert len(read_jsonl(tmp_path / "buckets" / "django__django.jsonl")) == 2
    by_cat = group_by_category([_cand("a"), _cand("b"), _cand("c", category="replay"),
                                {"pair_id": "d", "category": ""}])
    assert {k: len(v) for k, v in by_cat.items()} == {"lookup": 2, "replay": 1}


def test_representative_predicate():
    assert representative_predicate([{"predicate": "a"}, {"predicate": "b"},
                                     {"predicate": "a"}, {"predicate": None}]) == "a"
    assert representative_predicate([{"predicate": None}]) is None


def test_merge_rules_batches_and_folds_titles(monkeypatch, tmp_path):
    fake = json.dumps({"rules": [
        {"title": "Inventory peers", "body": "Run grep.", "support_indices": [0, 1]},
        {"title": "Hallucinated", "body": "x", "support_indices": [99]}]})
    calls = {"n": 0}

    def chat(**kw):
        calls["n"] += 1
        return _reply(fake)
    monkeypatch.setattr(llm, "chat", chat)
    cands = [_cand(f"p{i}", predicate="exists(read)" if i < 2 else None)
             for i in range(70)]
    rules = merge_rules(candidates=cands, category="lookup", bucket_id="b", model="m",
                        cost_logger=CostLogger(tmp_path / "c.jsonl"), batch_size=32)
    assert calls["n"] == 3
    assert len(rules) == 1 and rules[0].support == 6     # 2 per batch x 3 batches
    assert rules[0].rule_id == "b:lookup:0" and rules[0].predicate == "exists(read)"
    assert rules[0].source_pair_ids[:2] == ["p0", "p1"]


def test_merge_rules_splits_truncated_batch_and_drops_bad_json(monkeypatch, tmp_path):
    calls = {"n": 0}

    def chat(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _reply('{"rules": [{"tit', "length")
        return _reply(json.dumps({"rules": [{"title": "R", "body": "b",
                                             "support_indices": [0]}]}))
    monkeypatch.setattr(llm, "chat", chat)
    rules = merge_rules(candidates=[_cand(f"p{i}") for i in range(4)], category="lookup",
                        bucket_id="b", model="m",
                        cost_logger=CostLogger(tmp_path / "c.jsonl"), batch_size=32)
    assert calls["n"] == 3 and len(rules) == 1 and rules[0].support == 2
    monkeypatch.setattr(llm, "chat", lambda **kw: _reply("not json"))
    assert merge_rules(candidates=[_cand("p0")], category="lookup", bucket_id="b",
                       model="m", cost_logger=CostLogger(tmp_path / "c.jsonl"),
                       batch_size=32) == []


def test_merge_buckets_writes_tree_and_resumes(tmp_path):
    buckets = tmp_path / "buckets"
    write_jsonl(buckets / "django__django.jsonl", [_cand("p1"), _cand("p2", category="replay")])
    out = tmp_path / "rules"

    def fake_merge(*, candidates, category, bucket_id, **kw):
        return [Rule(rule_id=f"{bucket_id}:{category}:0", title=f"T {category}",
                     body="b", category=category, support=len(candidates),
                     source_pair_ids=[c["pair_id"] for c in candidates])]
    n = merge_buckets(buckets_dir=buckets, out_root=out, model="m",
                      cost_logger=CostLogger(tmp_path / "c.jsonl"), merge_fn=fake_merge,
                      resume=False)
    assert n == (1, 0, 0)
    rules = read_jsonl(out / "django__django" / "rules.jsonl")
    assert sorted(r["category"] for r in rules) == ["lookup", "replay"]
    with patch("xreposkill.merge.merge_rules") as m:
        assert merge_buckets(buckets_dir=buckets, out_root=out, model="m",
                             cost_logger=CostLogger(tmp_path / "c.jsonl"))[1] == 1
        m.assert_not_called()


def _verdict(**over):
    d = {"is_discipline": 1, "not_redundant_with_prior": 1,
         "behavior_actionable": 1, "not_repo_conflicting": 1, "rationale": "ok"}
    d.update(over)
    return json.dumps(d)


def test_llm_judge_dims_and_fail_open(monkeypatch, tmp_path):
    cl = CostLogger(tmp_path / "c.jsonl")
    monkeypatch.setattr(llm, "chat", lambda **kw: _reply(_verdict(not_redundant_with_prior=0)))
    keep, dec = LLMJudge(model="m", cost_logger=cl).evaluate(section_text="t", bucket="b")
    assert keep is False
    keep, _ = LLMJudge(model="m", cost_logger=cl,
                       dims=("is_discipline", "not_repo_conflicting")).evaluate(
        section_text="t", bucket="b")
    assert keep is True
    calls = {"n": 0}

    def bad(**kw):
        calls["n"] += 1
        return _reply("garbage")
    monkeypatch.setattr(llm, "chat", bad)
    keep, dec = LLMJudge(model="m", cost_logger=cl).evaluate(section_text="t", bucket="b")
    assert calls["n"] == 2 and keep is True and dec["error"] == "judge_unavailable_fail_open"


def test_judge_rules_parallel_and_resume(tmp_path):
    root = tmp_path / "rules"
    rules = [Rule(rule_id=f"b:lookup:{i}", title=f"T{i}", body="body", category="lookup",
                  support=1, source_pair_ids=[f"p{i}"]) for i in range(10)]
    write_jsonl(root / "b" / "rules.jsonl", [r.model_dump() for r in rules])
    log = tmp_path / "judge.jsonl"
    log.write_text("".join(json.dumps({
        "rule_id": f"b:lookup:{i}", "keep": i in (0, 2, 4),
        "decision": {"is_discipline": 1 if i in (0, 2, 4) else 0}}) + "\n"
        for i in range(5)))
    seen = []

    class J:
        def evaluate(self, *, section_text, bucket):
            seen.append(section_text)
            idx = int(section_text.split("\n", 1)[0][1:])
            return idx in (6, 8), {"is_discipline": 1 if idx in (6, 8) else 0}
    n_keep, n_drop = judge_rules(rules_root=root, out=tmp_path / "out", judge=J(),
                                 judge_log_path=log, workers=4)
    assert len(seen) == 5 and (n_keep, n_drop) == (5, 5)
    kept = sorted(r["rule_id"] for r in read_jsonl(tmp_path / "out" / "b" / "rules.jsonl"))
    assert kept == [f"b:lookup:{i}" for i in (0, 2, 4, 6, 8)]
    assert len(log.read_text().splitlines()) == 10
    assert load_keep_map(log, ("is_discipline",))["b:lookup:1"] is False
    assert load_keep_map(tmp_path / "missing.jsonl") == {}


def test_audit_flags_copied_patch_text(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"info": {"submission":
        "def compute_separability_matrix(left, right): return cleft"}, "messages": [],
        "instance_id": "x__x-1"}))
    pool = tmp_path / "pool"
    write_jsonl(pool / "S2.jsonl", [{**traj("x__x-1", "m", "pass", "S2::x__x-1", []),
                                     "source_path": str(raw)}])
    pairs = write_jsonl(tmp_path / "pairs.jsonl", [
        {"pair_id": "pp", "instance_id": "x__x-1", "fail_traj_id": "S::x__x-1",
         "pass_traj_id": "S2::x__x-1"}])
    copied = Rule(rule_id="b:lookup:0", title="Copy",
                  body="def compute_separability_matrix(left, right): return cleft",
                  category="lookup", support=1, source_pair_ids=["pp"])
    clean = Rule(rule_id="b:lookup:1", title="Clean", body="Run the failing test first.",
                 category="lookup", support=1, source_pair_ids=["pp"])
    write_jsonl(tmp_path / "rules" / "b" / "rules.jsonl",
                [copied.model_dump(), clean.model_dump()])
    n, flagged = audit_tree(rules_root=tmp_path / "rules", pairs_path=pairs,
                            pool_dir=pool, n=4, out=tmp_path / "audit.jsonl")
    assert (n, flagged) == (2, 1)
    rows = {r["rule_id"]: r for r in read_jsonl(tmp_path / "audit.jsonl")}
    assert rows["b:lookup:0"]["flagged"] and not rows["b:lookup:1"]["flagged"]
    assert overlap_hits("a b c d", "x y z", n=2) == []
