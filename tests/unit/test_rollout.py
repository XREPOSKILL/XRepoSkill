import json
from pathlib import Path

import pytest

from xreposkill.instances import (build_pro_problem_statement, load_instances_jsonl,
                                  read_instance_id_file, resolve_instances)
from xreposkill.pack import STRATEGIES_BUCKET, pack_skills
from xreposkill.rollout import (SKILL_FRAME, derive_mini_config, prepare_deepswe_task,
                                process_instance, run_batch, skill_block_for)


class StubAgent:
    def __init__(self, model, env, **cfg):
        self.config = type("Cfg", (), {"output_path": None})()
        self.cost, self.n_calls, self.captured = 0.42, 7, {}

    def run(self, task, **kwargs):
        self.captured = {"task": task, **kwargs}
        return {"exit_status": "Submitted", "submission": "<PATCH>"}

    def save(self, path, *extra):
        Path(path).write_text(json.dumps({"info": extra[0] if extra else {}}))


@pytest.fixture
def factories():
    agents = []

    def agent_factory(model, env):
        a = StubAgent(model, env)
        agents.append(a)
        return a
    return dict(agent_factory=agent_factory, env_factory=lambda: "env",
                model_factory=lambda: "model"), agents


def test_process_instance_writes_artifacts(tmp_path, factories):
    hooks, agents = factories
    res = process_instance(instance={"instance_id": "d__d-9", "problem_statement": "boom"},
                           out_dir=tmp_path, skill_block="SKILL_X",
                           mini_config={"agent": {}, "model": {}}, **hooks)
    assert res["exit_status"] == "Submitted" and res["cost"] == 0.42
    assert (tmp_path / "d__d-9" / "d__d-9.patch").read_text() == "<PATCH>"
    assert (tmp_path / "d__d-9" / "skill_block.txt").read_text() == "SKILL_X"
    assert (tmp_path / "d__d-9" / "d__d-9.traj.json").exists()
    assert agents[0].captured == {"task": "boom", "skill_block": "SKILL_X"}


def test_process_instance_records_failure(tmp_path):
    class Boom(StubAgent):
        def run(self, task, **kw):
            raise RuntimeError("model exploded")
    res = process_instance(instance={"instance_id": "x__y-1", "problem_statement": "p"},
                           out_dir=tmp_path, skill_block="", mini_config={},
                           agent_factory=lambda m, e: Boom(m, e),
                           env_factory=lambda: None, model_factory=lambda: None)
    assert res["exit_status"] == "RuntimeError" and "exploded" in res["error"]
    assert (tmp_path / "x__y-1" / "x__y-1.patch").read_text() == ""


def _skills(tmp_path):
    out = tmp_path / "skills"
    pack_skills(strategies=[{"strategy_id": "s/0", "title": "Reproduce", "trigger": "t",
                             "action": "a", "anchor_examples": [], "support_total": 3,
                             "pooled_z": 2.0, "homogeneous": True}],
                residual=[], out_root=out)
    return out


def test_skill_block_for_conditions(tmp_path):
    skills = _skills(tmp_path)
    rd = tmp_path / "retrieval"
    rd.mkdir()
    (rd / "i-1.json").write_text(json.dumps({"selected": [
        {"rule_id": f"{STRATEGIES_BUCKET}/skill/0", "reason": "x"}]}))
    assert skill_block_for("baseline", skills_root=None, instance_id="i-1",
                           retrieval_dir=None) == ""
    blk = skill_block_for("skills", skills_root=skills, instance_id="i-1", retrieval_dir=rd)
    assert blk.startswith(f"[{STRATEGIES_BUCKET}/skill/0] Reproduce")
    with pytest.raises(ValueError):
        skill_block_for("skills", skills_root=skills, instance_id="i-1", retrieval_dir=None)
    with pytest.raises(ValueError):
        skill_block_for("nope", skills_root=None, instance_id="i", retrieval_dir=None)


def test_run_batch_writes_results_and_preds(tmp_path, factories):
    hooks, _ = factories
    skills = _skills(tmp_path)
    rd = tmp_path / "retrieval"
    rd.mkdir()
    (rd / "a-1.json").write_text(json.dumps({"selected": [
        {"rule_id": f"{STRATEGIES_BUCKET}/skill/0", "reason": "x"}]}))
    insts = [{"instance_id": "a-1", "problem_statement": "p1"},
             {"instance_id": "b-2", "problem_statement": "p2"}]   # b-2: no retrieval file

    def proc(*, instance, out_dir, skill_block, mini_config):
        return process_instance(instance=instance, out_dir=out_dir,
                                skill_block=skill_block, mini_config=mini_config, **hooks)
    results = run_batch(instances=insts, out_dir=tmp_path / "out", condition="skills",
                        mini_config={"model": {"model_name": "m"}}, skills_root=skills,
                        retrieval_dir=rd, workers=2, process_fn=proc)
    by_id = {r["instance_id"]: r for r in results}
    assert by_id["a-1"]["exit_status"] == "Submitted" and by_id["a-1"]["skill_block_chars"] > 0
    assert by_id["b-2"]["exit_status"].startswith("driver_crash")
    preds = json.loads((tmp_path / "out" / "preds.json").read_text())
    assert preds["a-1"]["model_patch"] == "<PATCH>" and preds["a-1"]["model_name_or_path"] == "m"
    assert preds["b-2"]["model_patch"] == ""
    assert len((tmp_path / "out" / "results.jsonl").read_text().splitlines()) == 2


def test_derive_mini_config():
    base = {"model": {"model_name": "x"}, "environment": {}, "agent": {"step_limit": 1}}
    cfg = derive_mini_config(base, model="openai/gpt-5.6-luna", reasoning_effort="high",
                             container_timeout="6h", max_format_errors=10)
    assert cfg["model"]["model_name"] == "openai/gpt-5.6-luna"
    assert cfg["model"]["model_kwargs"] == {"reasoning_effort": "high", "drop_params": True}
    assert cfg["environment"]["container_timeout"] == "6h"
    assert cfg["agent"] == {"step_limit": 1, "max_consecutive_format_errors": 10}
    assert base["model"] == {"model_name": "x"}


def test_instances_helpers(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("# comment\na-1\n\nb-2\n")
    assert read_instance_id_file(ids) == ["a-1", "b-2"]
    row = {"problem_statement": "P", "requirements": "R", "interface": "I"}
    assert build_pro_problem_statement(row) == "P\n\nRequirements:\nR\n\nNew interfaces introduced:\nI"
    assert build_pro_problem_statement({"problem_statement": "P"}) == "P"
    jl = tmp_path / "inst.jsonl"
    jl.write_text(json.dumps({"instance_id": "a-1", "problem_statement": "p"}) + "\n")
    assert list(load_instances_jsonl(jl)) == ["a-1"]
    assert [i["instance_id"] for i in resolve_instances(["a-1", "zz"], instances_jsonl=jl)] == ["a-1"]


def test_prepare_deepswe_task(tmp_path):
    skills = _skills(tmp_path)
    task = tmp_path / "tasks" / "task-7"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("fix the cookie store")

    class Flow:
        def run_instance(self, *, instance_id, issue_text):
            assert issue_text == "fix the cookie store"
            return {"instance_id": instance_id, "selected": [
                {"rule_id": f"{STRATEGIES_BUCKET}/skill/0", "reason": "r"}], "error": ""}
    res, tmpl = prepare_deepswe_task(task_dir=task, skills_root=skills, model="m",
                                     out_dir=tmp_path / "prep", flow=Flow())
    text = tmpl.read_text()
    assert text.startswith("<relevant_skills>") and text.rstrip().endswith("{{ instruction }}")
    assert "Reproduce" in text and (tmp_path / "prep" / "task-7.json").exists()

    class Empty:
        def run_instance(self, **kw):
            return {"selected": [], "error": ""}
    _, tmpl = prepare_deepswe_task(task_dir=task, skills_root=skills, model="m",
                                   out_dir=tmp_path / "prep2", flow=Empty())
    assert tmpl.read_text() == "{{ instruction }}\n"
