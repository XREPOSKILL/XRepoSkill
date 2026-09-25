import pytest

from xreposkill.ingest import adapt_trajectory, ingest_submission, load_outcomes
from xreposkill.schemas import NormalizedTraj

SUB = "20260217_mini-v2.0.0_claude-4-5-sonnet-high"


def test_adapter_reads_chat_format(fixtures):
    nt = adapt_trajectory(fixtures / "mini_v2_sample.traj.json", submission_id=SUB,
                          outcome="pass")
    assert nt.instance_id == "django__django-13670"
    assert nt.repo == "django/django" and nt.outcome == "pass"
    kinds = {s.kind for s in nt.steps}
    assert "action" in kinds and "observation" in kinds
    assert all(s.raw for s in nt.steps)
    assert nt.model and nt.model != SUB.split("_", 2)[-1]
    assert nt.traj_id == f"{SUB}::django__django-13670"


def test_adapter_reads_responses_format(fixtures):
    nt = adapt_trajectory(fixtures / "mini_v2_responses_sample.traj.json",
                          submission_id="20260217_mini-v2.0.0_gpt-5-mini", outcome="pass")
    actions = [s for s in nt.steps if s.kind == "action"]
    obs = [s for s in nt.steps if s.kind == "observation"]
    assert actions and obs
    assert "command" in actions[0].raw["tool_calls"][0]["function"]["arguments"]
    assert any(s.raw.get("role") == "tool" and s.content for s in obs)


def test_ingest_submission_writes_jsonl(fixtures, tmp_path):
    out = tmp_path / "pool.jsonl"
    n = ingest_submission(SUB, tmp_path / "nonexistent", fixtures / "eval_root", out,
                          files=[fixtures / "mini_v2_sample.traj.json"])
    assert n == 1
    NormalizedTraj.model_validate_json(out.read_text().splitlines()[0])


def test_strict_mode_raises_when_outcome_missing(fixtures, tmp_path):
    outcomes = load_outcomes(fixtures / "eval_root", SUB)
    assert outcomes["django__django-13670"] == "pass"
    stray = tmp_path / "stray.traj.json"
    stray.write_text('{"instance_id": "nonexistent__inst-1", "messages": [], "info": {}}')
    with pytest.raises(KeyError):
        ingest_submission(SUB, tmp_path, fixtures / "eval_root", tmp_path / "o.jsonl",
                          files=[stray])
    assert ingest_submission(SUB, tmp_path, fixtures / "eval_root", tmp_path / "o.jsonl",
                             files=[stray], strict=False) == 0
    with pytest.raises(FileNotFoundError):
        load_outcomes(tmp_path, "nope")
