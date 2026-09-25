import json

import pytest

from xreposkill import discover, llm
from xreposkill.cost import CostLogger
from xreposkill.discover import (ANALYSTS, anti_fact_hit, build_occurrence_checker,
                                 build_user_message, check_candidate, fact_anchored,
                                 load_prompt, run_analyst, run_discovery)
from tests.conftest import read_jsonl, traj, write_jsonl

PAIR = {"pair_id": "x__x-1::S::x__x-1::S2::x__x-1", "instance_id": "x__x-1",
        "repo": "x/x", "fail_traj_id": "S::x__x-1", "pass_traj_id": "S2::x__x-1"}
SPECS = {s.label: s for s in ANALYSTS}


def _good(analyst="skipped_step", **over):
    obj = {"kind": "action", "category": "search_recipe",
           "rule": "Before editing operator overloads run `grep -rn 'def __<op>__' <module>/`.",
           "predicate": "before(run_test, first_edit)",
           "evidence": {"fail_traj_id": "S::x__x-1", "skipped_step_index": 1,
                        "gold_patch_hunk_ref": "x.py:1"}}
    if analyst == "discipline_observer":
        obj["evidence"] = {"pass_traj_id": "S2::x__x-1"}
    if analyst == "process_delta":
        obj["evidence"] = {"fail_traj_id": "S::x__x-1", "pass_traj_id": "S2::x__x-1",
                           "skipped_step_index": 1, "cited_pass_step_index": 2}
    obj.update(over)
    return obj


def fake_chat(*responses):
    """Replacement for llm.chat returning canned texts in order."""
    queue = list(responses)
    seen = []

    def chat(*, model, system, user, **kw):
        seen.append(user)
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        return llm.Reply(text=text, in_tokens=10, out_tokens=10, cost=0.0,
                         finish_reason="stop")
    chat.seen = seen
    return chat


def test_prompts_load_and_share_sections():
    for spec in ANALYSTS:
        p = load_prompt(spec.label)
        assert "{shared_rules}" not in p and "{shared_predicate}" not in p
        assert "FACT RULES" in p and "first_edit" in p


def test_guard_helpers():
    assert anti_fact_hit("The bug is in separable.py line 244.", category="lookup")
    assert not anti_fact_hit("Run `grep -rn 'def __<op>__' src/`.", category="search_recipe")
    assert fact_anchored("django: run tests with tests/runtests.py")
    assert fact_anchored("use `runtests` wrapper")
    assert not fact_anchored("reproduce the failure before editing")


def test_check_candidate_reasons():
    ok = _good()
    assert check_candidate(ok, max_chars=1000).ok
    assert check_candidate({**ok, "kind": "issue_type"}, max_chars=1000).reason == "bad_kind"
    assert check_candidate({**ok, "category": "x"}, max_chars=1000).reason == "bad_category"
    assert check_candidate({**ok, "rule": ""}, max_chars=1000).reason == "empty_rule"
    assert check_candidate({**ok, "rule": "x" * 20}, max_chars=10).reason == "too_long"
    assert check_candidate({**ok, "category": "lookup", "rule": "The fix is at foo.py:244"},
                           max_chars=1000).reason == "anti_fact"
    assert check_candidate({**ok, "predicate": None}, max_chars=1000).reason == "predicate_missing"
    g = check_candidate({**ok, "predicate": "frob("}, max_chars=1000)
    assert g.reason.startswith("predicate_parse")
    g = check_candidate(ok, max_chars=1000, occurrence_checker=lambda e: 0.95)
    assert g.reason.startswith("predicate_occurrence")
    assert check_candidate(ok, max_chars=1000, occurrence_checker=lambda e: 0.4).ok
    fact = {**ok, "kind": "fact", "rule": "django: run tests via tests/runtests.py",
            "predicate": "frob("}
    assert check_candidate(fact, max_chars=1000).ok and fact["predicate"] is None
    assert check_candidate({**ok, "kind": "fact", "rule": "reproduce first"},
                           max_chars=1000).reason == "fact_unanchored"


def _pool(tmp_path):
    pool = tmp_path / "pool"
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"info": {"submission": "diff --git a/x b/x\n+REAL_GOLD\n"},
                               "messages": [], "instance_id": "x__x-1"}))
    fail = {**traj("x__x-1", "m", "fail", "S::x__x-1", ["sed -i s/a/b/ f.py"]),
            "source_path": str(raw)}
    fail["steps"].insert(0, {"kind": "thought", "content": "REAL_FAIL_STEP", "raw": {}})
    pas = {**traj("x__x-1", "m2", "pass", "S2::x__x-1",
                  ["pytest t.py", "sed -i s/a/b/ f.py"]), "source_path": str(raw)}
    write_jsonl(pool / "S.jsonl", [fail])
    write_jsonl(pool / "S2.jsonl", [pas])
    return pool


def test_user_message_carries_patch_and_fork_window(tmp_path):
    pool = _pool(tmp_path)
    fork = {"pair_id": PAIR["pair_id"], "shared_prefix": [], "n_shared": 0,
            "fail_div_step": 1, "pass_div_step": 0, "fail_continuation": ["edit:f.py"],
            "pass_continuation": ["run_test:t.py"]}
    msg = json.loads(build_user_message(SPECS["skipped_step"], PAIR, fork, pool))
    assert "REAL_GOLD" in msg["successful_patch"]
    assert msg["fork"]["fail_divergence_excerpt"].startswith("[1 action]")
    assert "pass_divergence_excerpt" not in msg["fork"]
    msg = json.loads(build_user_message(SPECS["process_delta"], PAIR, None, pool))
    assert "REAL_FAIL_STEP" in msg["fork"]["fail_divergence_excerpt"]
    assert "successful_patch" not in msg


def test_run_analyst_accepts_and_retries(monkeypatch, tmp_path):
    pool = _pool(tmp_path)
    bad = {**_good(), "category": "lookup", "rule": "The fix is at foo.py:244"}
    chat = fake_chat(json.dumps(bad), json.dumps(_good()))
    monkeypatch.setattr(llm, "chat", chat)
    out = run_analyst(SPECS["skipped_step"], pair=PAIR, pool_dir=pool, forks={},
                      model="m", cost_logger=CostLogger(tmp_path / "c.jsonl"))
    assert len(out) == 1 and out[0]["analyst_type"] == "skipped_step"
    assert out[0]["category"] == "search_recipe"
    assert "GUARD REJECTED: anti_fact" in chat.seen[1]


def test_run_analyst_drops_after_retries_and_parse_failures(monkeypatch, tmp_path):
    pool = _pool(tmp_path)
    monkeypatch.setattr(llm, "chat", fake_chat("not json at all"))
    assert run_analyst(SPECS["process_delta"], pair=PAIR, pool_dir=pool, forks={},
                       model="m", cost_logger=CostLogger(tmp_path / "c.jsonl")) == []
    bad = {**_good(), "predicate": None}
    monkeypatch.setattr(llm, "chat", fake_chat(json.dumps(bad)))
    assert run_analyst(SPECS["skipped_step"], pair=PAIR, pool_dir=pool, forks={},
                       model="m", cost_logger=CostLogger(tmp_path / "c.jsonl"),
                       retry_on_guard_reject=1) == []


def test_observer_keeps_partial_batch_and_single_object(monkeypatch, tmp_path):
    pool = _pool(tmp_path)
    mixed = [_good("discipline_observer"),
             {**_good("discipline_observer"), "category": "lookup",
              "rule": "See foo.py:42"}]
    monkeypatch.setattr(llm, "chat", fake_chat(json.dumps(mixed)))
    out = run_analyst(SPECS["discipline_observer"], pair=PAIR, pool_dir=pool, forks={},
                      model="m", cost_logger=CostLogger(tmp_path / "c.jsonl"))
    assert len(out) == 1
    monkeypatch.setattr(llm, "chat", fake_chat(json.dumps(_good("discipline_observer"))))
    out = run_analyst(SPECS["discipline_observer"], pair=PAIR, pool_dir=pool, forks={},
                      model="m", cost_logger=CostLogger(tmp_path / "c.jsonl"))
    assert len(out) == 1


def test_occurrence_checker(tmp_path):
    rate = build_occurrence_checker(_pool(tmp_path))
    assert rate("before(run_test, first_edit)") == 0.5
    assert rate("exists(edit)") == 1.0


def test_run_discovery_interleaves_and_isolates_failures(tmp_path):
    pool = _pool(tmp_path)
    pairs = write_jsonl(tmp_path / "pairs.jsonl", [PAIR, {**PAIR, "pair_id": "p2"}])

    def fn(spec, *, pair, **kw):
        if spec.label == "process_delta":
            raise RuntimeError("boom")
        c = _good(spec.label)
        if spec.label == "discipline_observer":
            del c["evidence"]        # schema violation
        return [{"pair_id": pair["pair_id"], "analyst_type": spec.label, **c}]

    counts = run_discovery(
        pairs_path=pairs, forks_path=None, pool_dir=pool, out_dir=tmp_path / "cands",
        failed_path=tmp_path / "failed.jsonl", model="m", workers=4,
        cost_logger=CostLogger(tmp_path / "c.jsonl"), transcripts_dir=tmp_path / "tr",
        occurrence_check=False, analyst_fn=fn)
    assert counts == {"skipped_step": (2, 0), "discipline_observer": (0, 2),
                      "process_delta": (0, 2)}
    assert len(read_jsonl(tmp_path / "cands" / "skipped_step.jsonl")) == 2
    failed = read_jsonl(tmp_path / "failed.jsonl")
    assert sum("schema_error" in f for f in failed) == 2
    assert sum("boom" in f.get("error", "") for f in failed) == 2
    assert (tmp_path / "tr" / "stage_summary.jsonl").exists()
