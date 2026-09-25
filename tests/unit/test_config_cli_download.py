import json

import pytest

from xreposkill import cli, download
from xreposkill.config import DEFAULTS, load_config


def test_config_defaults_and_override(tmp_path):
    cfg = load_config(None)
    assert cfg["models"]["skill_learning"] == "deepseek/deepseek-v4-pro"
    assert cfg["judge"]["required_dims"] == ["is_discipline", "not_repo_conflicting"]
    p = tmp_path / "p.yaml"
    p.write_text("models:\n  backbone: openai/x\ngeneralize:\n  knn: 3\n")
    cfg = load_config(p)
    assert cfg["models"]["backbone"] == "openai/x" and cfg["models"]["skill_learning"] \
        == DEFAULTS["models"]["skill_learning"]
    assert cfg["generalize"]["knn"] == 3 and cfg["generalize"]["min_fires"] == 20
    assert load_config(tmp_path / "missing.yaml") == DEFAULTS


def test_repo_config_matches_paper_defaults():
    from pathlib import Path
    cfg = load_config(Path(__file__).parents[2] / "configs" / "pipeline.yaml")
    assert cfg["pair"]["K"] == 4 and cfg["retrieval"]["top_n"] == 3
    assert cfg["discover"]["occurrence_bounds"] == [0.10, 0.90]
    assert cfg["generalize"]["min_fires"] == 20 and cfg["generalize"]["heterogeneity_alpha"] == 0.10


def test_cli_dispatch(capsys):
    assert cli.main([]) == 0
    assert "discover" in capsys.readouterr().out
    assert cli.main(["nope"]) == 2
    with pytest.raises(SystemExit):
        cli.main(["pack", "--help"])


def test_download_outcomes_uses_github_raw(monkeypatch, tmp_path):
    class R:
        status_code = 200
        content = json.dumps({"x__x-1": {"resolved": True}}).encode()
    seen = {}

    def get(url, timeout):
        seen["url"] = url
        return R()
    import requests
    monkeypatch.setattr(requests, "get", get)
    dest = download.download_outcomes("sub-A", tmp_path)
    assert dest == tmp_path / "sub-A" / "per_instance_details.json"
    assert seen["url"].endswith("evaluation/verified/sub-A/per_instance_details.json")
    assert download.download_outcomes("sub-A", tmp_path) == dest       # cached, no call
    R.status_code = 404
    with pytest.raises(RuntimeError):
        download.download_outcomes("sub-B", tmp_path)


def test_download_trajectories_skips_existing(monkeypatch, tmp_path):
    class S3:
        def __init__(self):
            self.downloaded = []

        def get_paginator(self, name):
            s3 = self

            class P:
                def paginate(self, Bucket, Prefix):
                    return [{"Contents": [
                        {"Key": f"{Prefix}/a__a-1/a__a-1.traj.json", "Size": 1},
                        {"Key": f"{Prefix}/b__b-2/b__b-2.traj.json", "Size": 1},
                        {"Key": f"{Prefix}/b__b-2/other.txt", "Size": 1}]}]
            return P()

        def download_file(self, bucket, key, local):
            self.downloaded.append(key)
            open(local, "w").write("{}")
    s3 = S3()
    (tmp_path / "sub-A" / "trajs" / "a__a-1").mkdir(parents=True)
    (tmp_path / "sub-A" / "trajs" / "a__a-1" / "a__a-1.traj.json").write_text("{}")
    n = download.download_trajectories("sub-A", tmp_path, s3=s3)
    assert n == 2 and s3.downloaded == ["bash-only/sub-A/trajs/b__b-2/b__b-2.traj.json"]
