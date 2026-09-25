import json

import pytest

from xreposkill.pairing import (ability_tiers, frontier_instances, load_pairs,
                                preflight_no_leakage, resolve_rates, select_pairs)


def T(iid, model, outcome):
    return {"instance_id": iid, "repo": "org/name", "agent": "mini",
            "model": model, "outcome": outcome, "traj_id": f"{model}::{iid}"}


def _pool():
    outcomes = {"weak": ["pass", "fail", "fail", "fail"],
                "mid": ["pass", "pass", "fail", "fail"],
                "strong": ["pass", "pass", "pass", "fail"]}
    return [T(f"i{k}", m, o) for m, outs in outcomes.items() for k, o in enumerate(outs)]


def test_resolve_rates_and_tiers():
    rates = resolve_rates(_pool())
    assert rates["weak"] == 0.25 and rates["strong"] == 0.75
    assert ability_tiers(rates) == {"weak": "weak", "mid": "mid", "strong": "strong"}


def test_frontier_excludes_all_pass_and_all_fail():
    assert set(frontier_instances(_pool())) == {"i1", "i2"}


def test_select_pairs_positive_gap_only_and_K():
    pairs, stats = select_pairs(_pool(), K=4)
    assert stats["n_frontier"] == 2 and pairs
    for p in pairs:
        assert p.capability_gap > 0
    i1 = [p for p in pairs if p.instance_id == "i1"]
    assert i1[0].pass_tier == "strong" and i1[0].fail_tier == "weak"
    pairs, _ = select_pairs(_pool(), K=1)
    per_inst = {}
    for p in pairs:
        per_inst[p.instance_id] = per_inst.get(p.instance_id, 0) + 1
    assert all(v == 1 for v in per_inst.values())


def test_preflight_no_leakage(tmp_path):
    f = tmp_path / "eval.txt"
    f.write_text("astropy__astropy-12907\n")
    with pytest.raises(AssertionError):
        preflight_no_leakage(pool_instance_ids={"astropy__astropy-12907"}, eval_set_file=f)
    preflight_no_leakage(pool_instance_ids={"foo__bar-1"}, eval_set_file=f)


def test_load_pairs(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text(json.dumps({"pair_id": "a", "x": 1}) + "\n\n")
    assert load_pairs(p) == {"a": {"pair_id": "a", "x": 1}}
