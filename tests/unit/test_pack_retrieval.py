import json

import pytest

from xreposkill.pack import STRATEGIES_BUCKET, pack_skills, strategy_bullet
from xreposkill.retrieval import (RetrievalFlow, build_skill_block, load_rules,
                                  render_index, render_rules, run_flow_batch,
                                  selected_rule_ids)

def _strategy(i, z=5.0, het=False):
    return {"strategy_id": f"strategy/{i}", "title": f"Strategy {i}",
            "trigger": "when X", "action": f"do Y{i}", "anchor_examples": ["django: a", "sympy: b", "c"],
            "support_total": 10 + i, "pooled_z": z, "homogeneous": not het}


@pytest.fixture
def skills(tmp_path):
    residual = [{"rule_id": "d:v:0", "repo": "django__django", "title": "Use runtests",
                 "body": "Run tests/runtests.py.", "support": 3, "z": 1.5},
                {"rule_id": "d:v:1", "repo": "django__django", "title": "High z",
                 "body": "B.", "support": 1, "z": 4.0},
                {"rule_id": "d:v:2", "repo": "django__django", "title": "No z",
                 "body": "C.", "support": 9, "z": None}]
    out = tmp_path / "skills"
    counts = pack_skills(strategies=[_strategy(0), _strategy(1, 2.0, het=True)],
                         residual=residual, out_root=out)
    assert counts == {"strategies": 2, "residual": 3, "repo_buckets": 1}
    return out


def test_strategy_bullet_format():
    b = strategy_bullet(_strategy(0, het=True))
    assert b.startswith("- **Strategy 0** (support 10, z +5.00, HET) — Trigger: when X Action: do Y0")
    assert "Anchors: django: a | sympy: b" in b and " c" not in b.split("Anchors:")[1]


def test_pack_then_load_round_trip(skills):
    rules = load_rules(skills)
    assert [r.rule_id for r in rules] == [
        "django__django/skill/0", "django__django/skill/1", "django__django/skill/2",
        f"{STRATEGIES_BUCKET}/skill/0", f"{STRATEGIES_BUCKET}/skill/1"]
    dj = rules[:3]
    assert [r.title for r in dj] == ["High z", "Use runtests", "No z"]   # z order, no-z tail
    assert dj[0].tag == "support 1, z +4.00" and dj[2].tag == "support 9"
    st = rules[3:]
    assert st[1].tag.endswith("HET") and st[0].support == 10
    assert (skills / "packed_ids.json").exists()
    assert "Strategy 0" in render_index(rules) and "do Y0" not in render_index(rules)
    assert render_rules(st[:1]).startswith(f"[{STRATEGIES_BUCKET}/skill/0] Strategy 0\n")


def test_strategies_only_pack(tmp_path):
    counts = pack_skills(strategies=[_strategy(0)], residual=[{"repo": "r", "title": "t",
                         "body": "b", "support": 1, "z": 1.0}], out_root=tmp_path / "s",
                         include_residual=False)
    assert counts["repo_buckets"] == 0
    assert [r.bucket for r in load_rules(tmp_path / "s")] == [STRATEGIES_BUCKET]


def two_step_llm(selected_id=f"{STRATEGIES_BUCKET}/skill/0"):
    calls = []

    def fake(system, user):
        calls.append((system, user))
        if "analysis step" in system:
            assert "RULE INDEX" in user and "do Y0" not in user
            return json.dumps({"bug_summary": "a bug", "needed_guidance": ["g"]}), {}
        assert "ANALYSIS" in user and "RULES" in user and "(support 10, z +5.00)" in user
        return json.dumps({"selected": [{"rule_id": selected_id, "reason": "r"},
                                        {"rule_id": "nope/skill/9", "reason": "x"},
                                        {"rule_id": selected_id, "reason": "dup"}]}), {}
    fake.calls = calls
    return fake


def test_flow_two_calls_and_output_shape(skills):
    fn = two_step_llm()
    flow = RetrievalFlow(skills_root=skills, model="m", llm_fn=fn)
    res = flow.run_instance(instance_id="i-1", issue_text="the bug")
    assert len(fn.calls) == 2 and res["bug_summary"] == "a bug"
    assert res["selected"] == [{"rule_id": f"{STRATEGIES_BUCKET}/skill/0", "reason": "r"}]
    assert res["error"] == ""


def test_flow_batch_resume_and_block(skills, tmp_path):
    flow = RetrievalFlow(skills_root=skills, model="m", llm_fn=two_step_llm())
    out = tmp_path / "retrieval"
    insts = [{"instance_id": "i-1", "problem_statement": "p"}]
    run_flow_batch(flow=flow, instances=insts, out_dir=out, workers=1)
    assert selected_rule_ids(out / "i-1.json") == [f"{STRATEGIES_BUCKET}/skill/0"]
    flow2 = RetrievalFlow(skills_root=skills, model="m", llm_fn=two_step_llm())
    run_flow_batch(flow=flow2, instances=insts, out_dir=out, workers=1)
    assert flow2.llm_fn.calls == []
    block = build_skill_block(skills_root=skills, instance_id="i-1", retrieval_dir=out)
    assert block == f"[{STRATEGIES_BUCKET}/skill/0] Strategy 0\nTrigger: when X Action: do Y0 Anchors: django: a | sympy: b"
    assert "a bug" not in block and "support 10" not in block
    with pytest.raises(FileNotFoundError):
        build_skill_block(skills_root=skills, instance_id="x", retrieval_dir=out)


def test_flow_records_error(skills):
    def broken(system, user):
        raise RuntimeError("api down")
    res = RetrievalFlow(skills_root=skills, model="m", llm_fn=broken).run_instance(
        instance_id="i", issue_text="p")
    assert res["selected"] == [] and "api down" in res["error"]
