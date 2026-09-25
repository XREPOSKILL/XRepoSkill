import json

import pytest
from pydantic import ValidationError

from xreposkill import llm
from xreposkill.cost import CostLogger, StageSummaryLogger, TranscriptLogger
from xreposkill.schemas import (Candidate, Evidence, Rule, repo_from_instance_id,
                                repo_slug_from_pair_id)


def test_parse_json_variants():
    assert llm.parse_json('{"a": 1}') == {"a": 1}
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('Sure: {"a": [1, 2]} done') == {"a": [1, 2]}
    assert llm.parse_json("[1, 2]") == [1, 2]
    assert llm.parse_json("no json here") is None
    assert llm.parse_json_obj("[1]") == {}
    assert llm.parse_json("") is None


def test_call_with_retry():
    calls = {"n": 0}

    class Transient(Exception):
        pass

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Transient()
        return "ok"
    assert llm.call_with_retry(fn, retry_on=(Transient,), wait_seconds=0.0) == "ok"
    assert calls["n"] == 3
    with pytest.raises(ValueError):
        llm.call_with_retry(lambda: (_ for _ in ()).throw(ValueError("x")),
                            retry_on=(Transient,), wait_seconds=0.0)


def test_chat_retries_transient_provider_error(monkeypatch):
    import litellm
    from unittest.mock import MagicMock
    calls = {"n": 0}

    def fake_completion(**kw):
        calls["n"] += 1
        if calls["n"] < 2:
            raise litellm.exceptions.InternalServerError(
                message="empty stream", model="x", llm_provider="openai")
        m = MagicMock()
        m.choices = [MagicMock(message=MagicMock(content='{"ok":1}'), finish_reason="stop")]
        m.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        m.get = lambda k, default=0: default
        return m
    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    r = llm.chat(model="m", system="s", user="u")
    assert r.text == '{"ok":1}' and calls["n"] == 2 and r.in_tokens == 10


def test_resolve_model(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODEL", "openai/x")
    assert llm.resolve_model("deepseek/y") == "openai/x"
    monkeypatch.delenv("PIPELINE_MODEL")
    assert llm.resolve_model("deepseek/y") == "deepseek/y"


def test_loggers(tmp_path):
    cl = CostLogger(tmp_path / "cost.jsonl")
    cl.log(stage="discover", sub="A-", model="m", in_tokens=1, out_tokens=2, cost_usd=0.5, x=1)
    rec = json.loads((tmp_path / "cost.jsonl").read_text())
    assert rec["cost_usd"] == 0.5 and rec["extra"] == {"x": 1}
    tl = TranscriptLogger(tmp_path / "t.jsonl")
    tl.log(stage="discover", sub="A-", pair_id="p1", attempt=0, prompt_user="h",
           response="bad", parse_ok=False, guard_ok=False, guard_reason="json_parse")
    tl.log(stage="discover", sub="A-", pair_id="p2", attempt=0, prompt_user="h",
           response="{}", parse_ok=True, guard_ok=False, guard_reason="anti_fact",
           extracted={"kind": "action", "category": "lookup"})
    tl.log(stage="discover", sub="A-", pair_id="p3", attempt=0, prompt_user="h",
           response="{}", parse_ok=True, guard_ok=True)
    assert tl.summary(stage="discover", sub="A-") == {
        "parse_fail": 1, "guard_fail:anti_fact": 1, "accepted": 1}
    sl = StageSummaryLogger(tmp_path / "s.jsonl")
    sl.log(stage="discover", sub="A-", pairs_in=100, candidates_out=20, outcomes={},
           wall_seconds=42.5)
    assert json.loads((tmp_path / "s.jsonl").read_text())["pair_yield"] == 0.2


def test_repo_helpers():
    assert repo_from_instance_id("astropy__astropy-12907") == "astropy/astropy"
    assert repo_slug_from_pair_id("django__django-1::f::p") == "django__django"


def test_candidate_evidence_matrix():
    base = dict(pair_id="p1", kind="action", category="lookup", rule="r",
                predicate="exists(read)")
    with pytest.raises(ValidationError):
        Candidate(analyst_type="skipped_step", evidence={}, **base)
    Candidate(analyst_type="discipline_observer", evidence={"pass_traj_id": "t"}, **base)
    with pytest.raises(ValidationError):
        Candidate(analyst_type="process_delta",
                  evidence={"fail_traj_id": "f", "pass_traj_id": "p",
                            "skipped_step_index": 2}, **base)
    with pytest.raises(ValidationError):
        Candidate(analyst_type="process_delta", kind="issue_type",
                  **{k: v for k, v in base.items() if k != "kind"},
                  evidence={"fail_traj_id": "f", "pass_traj_id": "p",
                            "skipped_step_index": 2, "cited_pass_step_index": 3})


def test_evidence_coerces_multi_step_citation():
    assert Evidence(pass_traj_id="t", cited_pass_step_index="4,6,10").cited_pass_step_index == 4
    assert Evidence(fail_traj_id="t", skipped_step_index="5-6").skipped_step_index == 5


def test_rule_validation():
    Rule(rule_id="b:lookup:0", title="t", body="b", category="lookup", support=1,
         source_pair_ids=[])
    with pytest.raises(ValidationError):
        Rule(rule_id="b:x:0", title="t", body="b", category="nope", support=1,
             source_pair_ids=[])
    with pytest.raises(ValidationError):
        Rule(rule_id="b:x:0", title="t", body="b", category="lookup", support=-1,
             source_pair_ids=[])
