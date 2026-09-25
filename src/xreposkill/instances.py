"""Evaluation instances: SWE-bench Pro rows or a local JSONL of instance dicts."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

DEFAULT_DATASET = "ScaleAI/SWE-bench_Pro"
DEFAULT_DOCKERHUB_USER = "jefzda"


def read_instance_id_file(p: Path) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def build_pro_problem_statement(row: dict) -> str:
    """SWE-bench Pro convention: description + requirements + new interfaces."""
    base = (row.get("problem_statement") or "").strip()
    reqs = (row.get("requirements") or "").strip()
    iface = (row.get("interface") or "").strip()
    if reqs or iface:
        return (f"{base}\n\nRequirements:\n{reqs}\n\n"
                f"New interfaces introduced:\n{iface}").strip()
    return base


def fetch_pro_instances(*, instance_ids: Iterable[str],
                        dataset: str = DEFAULT_DATASET,
                        dockerhub_user: str = DEFAULT_DOCKERHUB_USER,
                        ) -> tuple[list[dict], list[str]]:
    """Rows from Hugging Face as repair-shaped dicts; ``(instances, skipped_ids)``."""
    from datasets import load_dataset
    ds = load_dataset(dataset, split="test")
    by_id = {d["instance_id"]: d for d in ds}
    out, skipped = [], []
    for iid in instance_ids:
        d = by_id.get(iid)
        tag = ((d or {}).get("dockerhub_tag") or "").strip()
        if d is None or not tag:
            skipped.append(iid)
            continue
        out.append({"instance_id": iid,
                    "problem_statement": build_pro_problem_statement(d),
                    "repo": d.get("repo", ""),
                    "image_name": f"docker.io/{dockerhub_user}/sweap-images:{tag}"})
    return out, skipped


def load_instances_jsonl(p: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for line in Path(p).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            by_id[d["instance_id"]] = d
    return by_id


def write_instances_jsonl(instances: list[dict], p: Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open("w") as f:
        for d in instances:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def resolve_instances(ids: list[str], *, instances_jsonl: Path | None,
                      dataset: str = DEFAULT_DATASET,
                      dockerhub_user: str = DEFAULT_DOCKERHUB_USER) -> list[dict]:
    """Instances for ``ids`` from a local JSONL when given, else from Hugging Face."""
    if instances_jsonl is not None:
        by_id = load_instances_jsonl(instances_jsonl)
        missing = [i for i in ids if i not in by_id]
        for iid in missing:
            print(f"[instances] WARN not in {instances_jsonl}: {iid}", flush=True)
        return [by_id[i] for i in ids if i in by_id]
    instances, skipped = fetch_pro_instances(
        instance_ids=ids, dataset=dataset, dockerhub_user=dockerhub_user)
    for iid in skipped:
        print(f"[instances] WARN not in dataset or no image tag: {iid}", flush=True)
    return instances
