"""Read the trajectory pool: ``<pool_dir>/<submission>.jsonl`` of NormalizedTraj rows."""
from __future__ import annotations
import json
import re
from functools import lru_cache
from pathlib import Path

from xreposkill.events import Event, extract_events


def load_pool(pool_dir: Path) -> list[dict]:
    trajs: list[dict] = []
    for f in sorted(Path(pool_dir).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                trajs.append(json.loads(line))
    return trajs


def pool_events(pool: list[dict]) -> list[list[Event]]:
    """Event sequence of every trajectory, in pool order."""
    return [extract_events(t.get("steps", [])) for t in pool]


@lru_cache(maxsize=32)
def _load_pool_file(jsonl_path: str) -> tuple[dict[str, dict], ...]:
    by_id: dict[str, dict] = {}
    for line in Path(jsonl_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_id[r["traj_id"]] = r
    return (by_id,)


def find_trajectory(pool_dir: Path, traj_id: str) -> dict | None:
    """Look a trajectory up by ``traj_id`` (``<submission>::<instance>``)."""
    sub = traj_id.split("::", 1)[0]
    pool_file = Path(pool_dir) / f"{sub}.jsonl"
    if not pool_file.exists():
        for f in Path(pool_dir).glob("*.jsonl"):
            (idx,) = _load_pool_file(str(f))
            if traj_id in idx:
                return idx[traj_id]
        return None
    (idx,) = _load_pool_file(str(pool_file))
    return idx.get(traj_id)


def format_steps_excerpt(steps: list[dict], *, max_chars: int = 6000,
                         max_step_chars: int = 800) -> str:
    """Numbered, length-capped rendering of steps for a prompt."""
    out: list[str] = []
    used = 0
    for i, s in enumerate(steps):
        content = (s.get("content") or "").strip()
        if len(content) > max_step_chars:
            content = content[:max_step_chars] + "…"
        line = f"[{i} {s.get('kind', '?')}] {content}"
        if used + len(line) + 1 > max_chars:
            out.append("...truncated...")
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def load_steps_window(pool_dir: Path, traj_id: str, *, start: int,
                      max_chars: int = 4000) -> str:
    """Excerpt of ``steps[start:]`` with absolute step numbers kept."""
    tr = find_trajectory(Path(pool_dir), traj_id)
    if tr is None:
        return ""
    txt = format_steps_excerpt(tr.get("steps", [])[start:], max_chars=max_chars)
    if start:
        txt = re.sub(r"^\[(\d+) ",
                     lambda m: f"[{int(m.group(1)) + start} ",
                     txt, flags=re.MULTILINE)
    return txt


def load_submitted_patch(pool_dir: Path, traj_id: str) -> str:
    """The patch the trajectory submitted (``info.submission`` of the raw file)."""
    tr = find_trajectory(Path(pool_dir), traj_id)
    if tr is None or not tr.get("source_path"):
        return ""
    try:
        raw = json.loads(Path(tr["source_path"]).read_text())
        return str((raw.get("info") or {}).get("submission") or "")
    except Exception:
        return ""
