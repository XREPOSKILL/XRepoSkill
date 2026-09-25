# XRepoSkill

XRepoSkill distils a transferable skill from a pool of mini-SWE-agent
trajectories and injects the selected rules into the agent for new issues.
The code implements the three stages of the paper and nothing else.

1. **Divergence-guided rule discovery.** A failed and a successful trajectory
   on the same issue are aligned action by action; three prompts read the
   trajectories from the divergence point onward and write candidate rules,
   each with an executable predicate.
2. **Rule verification and cross-repository generalization.** Candidate rules
   are merged within each repository, screened by an LLM judge, scored by the
   resolution gain (a Mantel-Haenszel risk difference with a z score) on
   held-out trajectories, audited for n-gram overlap with source patches, and
   clustered across repositories by the phi correlation of their predicate
   vectors. Clusters that span two or more repositories become transferable
   rules with a pooled gain and a Cochran's Q heterogeneity mark.
3. **Issue-adaptive rule selection.** Two calls to the backbone LLM select at
   most three rules per issue; the rules' title and body are placed above the
   issue description in the first message of an unchanged mini-SWE-agent.

## Layout

```
src/xreposkill/       one module per stage, one CLI entry (python -m xreposkill)
  download.py  ingest.py  pairing.py  fork.py         data and pairs
  discover.py  merge.py   judge.py    score.py        Stage 1, Stage 2a-2c
  audit.py     generalize.py  pack.py                 Stage 2d-2e, skill files
  retrieval.py rollout.py  instances.py               Stage 3
  events.py    predicate.py  pool.py  schemas.py      shared: action abstraction,
  llm.py       cost.py     config.py  cli.py          predicate DSL, pool, records
  prompts/                                            every LLM prompt
configs/pipeline.yaml                paper defaults for every stage
configs/mini_swe_agent/*.yaml        agent config with the {{ skill_block }} slot
data/verified_submissions.txt        the 12 leaderboard submissions of the pool
data/swebench_pro_main_200_ids.txt   evaluation ids (200 main, 50 ablation)
data/general_strategies_36.md        the 36 transferable strategies learned in the paper
scripts/run_pipeline.sh              download -> distil -> retrieve -> rollout -> evaluate
scripts/run_deepswe.sh               DeepSWE arms through Pier
tests/                               unit tests per module + mocked end-to-end smoke test
```

Run artefacts live under `runs/<run_id>/`:

```
traj_pool/      pairs/{pairs,forks}.jsonl   candidates/<analyst>.jsonl
buckets/        rules_merged/ rules_judged/ rules_scored/   (<repo>/rules.jsonl)
generalize/{strategies,repo_residual,components,assignments}.jsonl summary.json
skills/repo_conventions/<bucket>/SKILL.md
eval/retrieval/<iid>.json   eval/{baseline,skills}/{results.jsonl,preds.json}
```

## Setup

```bash
uv pip install --python .venv/bin/python -e "XRepoSkill[dev,rollout]"
cd XRepoSkill && ../.venv/bin/python -m pytest -q
```

API keys are read by litellm from the environment (`DEEPSEEK_API_KEY`,
`OPENAI_API_KEY` and `OPENAI_BASE_URL`, ...). Put them in `.env` at the
repository root; the scripts source it. `PIPELINE_MODEL` overrides every
configured model. The rollout needs Docker and the SWE-bench Pro images
(`docker.io/jefzda/sweap-images:<tag>`); the evaluate stage needs the
SWE-bench Pro evaluator (`PRO_EVALUATOR=/path/to/evaluate.py`).

## Running

Everything, in order, under `runs/v1/`:

```bash
scripts/run_pipeline.sh v1
```

Single stages, and evaluating a skill packed by an earlier run under another
backbone:

```bash
scripts/run_pipeline.sh v1 download ingest pair fork
scripts/run_pipeline.sh v1 discover merge judge score audit generalize pack
SKILLS_ROOT=runs/v1/skills BACKBONE=openai/gpt-5.6-luna \
    scripts/run_pipeline.sh luna retrieve rollout evaluate
```

Every stage is also a subcommand with its own `--help`:

```bash
.venv/bin/python -m xreposkill discover --help
```

Stage 3 calls and the agent use `models.backbone` (or `--model`); every
Stage 1 and Stage 2 call uses `models.skill_learning`. The main experiments
ran the agent with `--reasoning-effort high --container-timeout 6h
--max-format-errors 10`, which `scripts/run_pipeline.sh` sets by default.

## Data sources

* Trajectories: the public SWE-bench submissions bucket,
  `s3://swe-bench-submissions/bash-only/<submission>/trajs/`, unsigned access.
* Pass/fail labels: `evaluation/verified/<submission>/per_instance_details.json`
  in the SWE-bench experiments repository on GitHub.
* Evaluation issues: `ScaleAI/SWE-bench_Pro` on Hugging Face (ids in `data/`),
  and DeepSWE tasks through Pier (`scripts/run_deepswe.sh`).
