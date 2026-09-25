"""Fetch the trajectory pool sources.

Trajectories: the public SWE-bench submissions bucket (unsigned S3 access),
``s3://swe-bench-submissions/bash-only/<submission>/trajs/<iid>/<iid>.traj.json``.
Pass/fail labels: ``per_instance_details.json`` of the same submission in the
SWE-bench experiments repository on GitHub
(``evaluation/verified/<submission>/per_instance_details.json``).

Usage:
    python -m xreposkill download --submissions data/verified_submissions.txt \
        --trajs-root data/trajs --evals-root data/evals [--workers 8]
"""
from __future__ import annotations
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

S3_BUCKET = "swe-bench-submissions"
S3_SPLIT = "bash-only"
GITHUB_RAW = ("https://raw.githubusercontent.com/SWE-bench/experiments/main/"
              "evaluation/verified/{submission}/per_instance_details.json")


def read_list(p: Path) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def _s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_keys(s3, prefix: str) -> list[tuple[str, int]]:
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            out.append((obj["Key"], obj["Size"]))
    return out


def download_trajectories(submission: str, trajs_root: Path, *,
                          workers: int = 8, s3=None) -> int:
    """Download every ``*.traj.json`` of one submission; skip existing files."""
    s3 = s3 or _s3()
    prefix = f"{S3_SPLIT}/{submission}/trajs"
    keys = [k for k, _ in list_keys(s3, prefix) if k.endswith(".traj.json")]
    dest_root = Path(trajs_root) / submission / "trajs"
    todo = []
    for key in keys:
        local = dest_root / os.path.relpath(key, prefix)
        if not local.exists():
            todo.append((key, local))

    def one(key: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(S3_BUCKET, key, str(local))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed(ex.submit(one, k, l) for k, l in todo):
            fut.result()
    print(f"[download] {submission}: {len(keys)} trajectories "
          f"({len(todo)} new)", flush=True)
    return len(keys)


def download_outcomes(submission: str, evals_root: Path) -> Path:
    """Fetch ``per_instance_details.json`` for one submission from GitHub."""
    import requests
    dest = Path(evals_root) / submission / "per_instance_details.json"
    if dest.exists():
        return dest
    url = GITHUB_RAW.format(submission=submission)
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"{url} -> HTTP {r.status_code}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    print(f"[download] {submission}: outcomes -> {dest}", flush=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submissions", required=True, type=Path,
                    help="newline-delimited submission names")
    ap.add_argument("--trajs-root", required=True, type=Path)
    ap.add_argument("--evals-root", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outcomes-only", action="store_true")
    a = ap.parse_args(argv)
    subs = read_list(a.submissions)
    s3 = None if a.outcomes_only else _s3()
    for sub in subs:
        download_outcomes(sub, a.evals_root)
        if not a.outcomes_only:
            download_trajectories(sub, a.trajs_root, workers=a.workers, s3=s3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
