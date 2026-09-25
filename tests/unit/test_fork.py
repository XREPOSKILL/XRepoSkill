from xreposkill.fork import Fork, build_forks, localize_fork, shared_prefix_end
from tests.conftest import action_step, traj, write_jsonl

PAIR = {"pair_id": "p1", "instance_id": "i1", "repo": "org/name",
        "fail_traj_id": "f1", "pass_traj_id": "s1"}


def test_identical_prefix_then_divergence():
    shared = [action_step("grep -rn 'separability' astropy/"),
              action_step("cat astropy/modeling/separable.py")]
    fail_steps = shared + [action_step("sed -i 's/a/b/' astropy/modeling/separable.py")]
    pass_steps = shared + [action_step("python -m pytest astropy/tests/test_separable.py"),
                           action_step("sed -i 's/a/b/' astropy/modeling/separable.py")]
    fork = localize_fork(pair=PAIR, fail_steps=fail_steps, pass_steps=pass_steps)
    assert fork.n_shared == 2
    assert fork.fail_continuation[0].startswith("edit:")
    assert fork.pass_continuation[0].startswith("run_test:")
    assert fork.fail_div_step == 2 and fork.pass_div_step == 2


def test_alignment_tolerates_small_gaps():
    shared = [action_step("grep -rn 'foo' src/"), action_step("cat src/foo.py")]
    noise = [action_step("ls src/"), action_step("git status")]
    fork = localize_fork(pair=PAIR, fail_steps=[shared[0]] + noise + [shared[1]],
                         pass_steps=shared)
    assert fork.n_shared == 2


def test_large_gap_or_significant_gap_ends_prefix():
    fail_keys = [("search", "foo")] + [("read", f"x{i}") for i in range(5)] + [("edit", "a.py")]
    pass_keys = [("search", "foo"), ("edit", "a.py")]
    assert shared_prefix_end(fail_keys, pass_keys) == (1, 1, 1)
    fail_keys = [("search", "foo"), ("run_test", "t"), ("read", "a.py")]
    pass_keys = [("search", "foo"), ("read", "a.py")]
    assert shared_prefix_end(fail_keys, pass_keys) == (1, 1, 1)


def test_no_events_side_and_anchor_normalisation():
    fork = localize_fork(pair=PAIR, fail_steps=[], pass_steps=[action_step("cat a.py")])
    assert fork.n_shared == 0 and fork.fail_div_step is None and fork.pass_div_step == 0
    fork = localize_fork(pair=PAIR, fail_steps=[action_step("cat /testbed/astropy/x.py")],
                         pass_steps=[action_step("cat astropy/x.py")])
    assert fork.n_shared == 1
    Fork.model_validate_json(fork.model_dump_json())


def test_build_forks_skips_missing_trajectories(tmp_path):
    pool = tmp_path / "pool"
    write_jsonl(pool / "s.jsonl", [traj("i1", "m", "fail", "s::i1", ["cat a.py"]),
                                   traj("i1", "m2", "pass", "s2::i1", ["cat a.py"])])
    pairs = write_jsonl(tmp_path / "pairs.jsonl", [
        {**PAIR, "fail_traj_id": "s::i1", "pass_traj_id": "s2::i1"},
        {**PAIR, "pair_id": "p2", "fail_traj_id": "nope::i1", "pass_traj_id": "s2::i1"}])
    assert build_forks(pairs_path=pairs, pool_dir=pool, out_path=tmp_path / "forks.jsonl") == (1, 1)
