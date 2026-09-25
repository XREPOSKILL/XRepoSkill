import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


def action_step(cmd: str) -> dict:
    return {"kind": "action", "content": "x",
            "raw": {"tool_calls": [{"function": {
                "name": "bash", "arguments": json.dumps({"command": cmd})}}]}}


def traj(iid: str, model: str, outcome: str, tid: str, cmds: list[str],
         repo: str = "o/n") -> dict:
    return {"instance_id": iid, "repo": repo, "agent": "mini", "model": model,
            "outcome": outcome, "steps": [action_step(c) for c in cmds],
            "submission_id": tid.split("::")[0], "source_path": "", "traj_id": tid}


def write_jsonl(p: Path, rows) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
