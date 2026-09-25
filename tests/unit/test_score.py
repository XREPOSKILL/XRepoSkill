import math

from xreposkill.predicate import parse_predicate
from xreposkill.schemas import Rule
from xreposkill.score import (LabeledTraj, Stratum, build_holdout, label_holdout,
                              mh_risk_difference, score_rule, score_tree,
                              stratified_lift)
from tests.conftest import read_jsonl, traj, write_jsonl


def L(iid, model, outcome, label, tid=""):
    return LabeledTraj(traj_id=tid or f"{iid}:{model}", instance_id=iid,
                       model=model, outcome=outcome, label=label)


def test_build_holdout_excludes_extraction_sides():
    pool = [{"instance_id": "i1", "outcome": "pass", "traj_id": "a"},
            {"instance_id": "i1", "outcome": "fail", "traj_id": "b"},
            {"instance_id": "i1", "outcome": "unknown", "traj_id": "c"},
            {"instance_id": "i2", "outcome": "pass", "traj_id": "d"}]
    h = build_holdout(pool, instances={"i1"}, exclude_traj_ids={"b"})
    assert [t["traj_id"] for t in h] == ["a"]


def test_single_stratum_reduces_to_plain_rd():
    rd, se = mh_risk_difference([Stratum("i", a=30, n1=40, c=10, n0=40)])
    assert math.isclose(rd, 0.5)
    p1, p0 = 30 / 40, 10 / 40
    se_binom = math.sqrt(p1 * (1 - p1) / 40 + p0 * (1 - p0) / 40)
    assert abs(se - se_binom) / se_binom < 0.05


def test_stratification_defuses_simpson():
    labeled = [L("hard", "m", "pass", True, "h1"), L("hard", "m", "fail", True, "h2")]
    labeled += [L("hard", "m", "fail", False, f"h{i}") for i in range(3, 11)]
    labeled += [L("easy", "m", "pass", True, f"e{i}") for i in range(8)]
    labeled += [L("easy", "m", "pass", False, "e8"), L("easy", "m", "fail", False, "e9")]
    r = stratified_lift(labeled)
    assert r.lift is not None and r.lift > 0.3
    assert r.n_strata_used == 2 and r.ci_low < r.lift < r.ci_high


def test_empty_cell_strata_are_counted_not_used():
    r = stratified_lift([L("i1", "m", "pass", True), L("i1", "m", "fail", True)])
    assert r.lift is None and r.n_strata_empty == 1 and r.occurrence == 1.0


def test_label_holdout_uses_events_cache():
    t = traj("i1", "m", "pass", "s::i1", ["pytest t.py", "sed -i s/a/b/ f.py"])
    cache = {}
    out = label_holdout(parse_predicate("before(run_test, first_edit)"), [t],
                        events_cache=cache)
    assert out[0].label is True and "s::i1" in cache


def _mk_pool(tmp_path):
    """4 mixed-outcome issues; passers run tests before editing."""
    rows = []
    for k in range(4):
        iid = f"inst{k}"
        for m in ("weak", "mid", "strong"):
            rows.append(traj(iid, m, "pass", f"p_{m}::{iid}",
                             ["pytest tests/test_a.py", "sed -i s/a/b/ f.py"]))
            rows.append(traj(iid, m, "fail", f"f_{m}::{iid}", ["sed -i s/a/b/ f.py"]))
    rows.append(traj("inst0", "mid", "fail", "f_mid_b::inst0",
                     ["pytest tests/test_a.py", "sed -i s/a/b/ f.py"]))
    pool_dir = tmp_path / "pool"
    write_jsonl(pool_dir / "all.jsonl", rows)
    return pool_dir


def _rule(rid, predicate, pairs=()):
    return Rule(rule_id=rid, title="Reproduce first",
                body="Run the failing test before editing.", category="verification",
                support=3, source_pair_ids=list(pairs), predicate=predicate)


def test_score_rule_and_tree(tmp_path):
    pool_dir = _mk_pool(tmp_path)
    root = tmp_path / "rules"
    write_jsonl(root / "o__n" / "rules.jsonl", [
        _rule("o__n:verification:0", "before(run_test, first_edit)").model_dump(),
        _rule("o__n:lookup:1", None).model_dump(),
        _rule("o__n:lookup:2", "frob(").model_dump(),
    ])
    pairs = write_jsonl(tmp_path / "pairs.jsonl", [])
    stats = score_tree(rules_root=root, out=tmp_path / "scored", pool_dir=pool_dir,
                       pairs_path=pairs, log_path=tmp_path / "log.jsonl")
    assert stats == {"scored": 1, "fallback": 2}
    out = {r["rule_id"]: r for r in read_jsonl(tmp_path / "scored" / "o__n" / "rules.jsonl")}
    v = out["o__n:verification:0"]["validation"]
    assert v["status"] == "scored" and v["lift"] > 0 and v["z"] > 0
    assert v["n_strata_used"] == 4
    assert out["o__n:lookup:1"]["validation"] == {"status": "fallback", "reason": "no_predicate"}
    assert out["o__n:lookup:2"]["validation"]["reason"].startswith("predicate_error")
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 3


def test_score_rule_excludes_extraction_sides(tmp_path):
    pool_dir = _mk_pool(tmp_path)
    from xreposkill.pool import load_pool
    pool = load_pool(pool_dir)
    pairs = {"pp": {"pair_id": "pp", "instance_id": "inst0",
                    "fail_traj_id": "f_weak::inst0", "pass_traj_id": "p_strong::inst0"}}
    frontier = {f"inst{k}" for k in range(4)}
    with_pair = score_rule(_rule("r", "before(run_test, first_edit)", ["pp"]),
                           pool=pool, repo_frontier=frontier, pairs=pairs, events_cache={})
    without = score_rule(_rule("r", "before(run_test, first_edit)"),
                         pool=pool, repo_frontier=frontier, pairs=pairs, events_cache={})
    assert with_pair["n_labeled"] == without["n_labeled"] - 2
