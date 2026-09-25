"""Pipeline configuration (configs/pipeline.yaml) with defaults."""
from __future__ import annotations
import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "models": {
        "skill_learning": "deepseek/deepseek-v4-pro",
        "backbone": "deepseek/deepseek-v4-flash",
    },
    "pair": {"K": 4},
    "discover": {"workers": 16, "retry_on_guard_reject": 1,
                 "rule_max_chars": 1000, "occurrence_bounds": [0.10, 0.90]},
    "merge": {"batch_size": 32},
    "judge": {"required_dims": ["is_discipline", "not_repo_conflicting"],
              "workers": 16},
    "audit": {"ngram": 6},
    "generalize": {"knn": 15, "min_fires": 20, "heterogeneity_alpha": 0.10,
                   "workers": 12, "embedding_model": "all-MiniLM-L6-v2"},
    "retrieval": {"top_n": 3, "workers": 16, "issue_max_chars": 6000},
    "rollout": {"workers": 16, "dataset": "ScaleAI/SWE-bench_Pro",
                "dockerhub_user": "jefzda"},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | str | None) -> dict:
    if path is None:
        return copy.deepcopy(DEFAULTS)
    p = Path(path)
    if not p.exists():
        return copy.deepcopy(DEFAULTS)
    return _merge(DEFAULTS, yaml.safe_load(p.read_text()) or {})
