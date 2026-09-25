#!/usr/bin/env bash
# XRepoSkill end-to-end pipeline: data download -> skill distillation ->
# rule retrieval -> rollout -> evaluation on SWE-bench Pro.
#
# Usage:
#   scripts/run_pipeline.sh <run_id> [stage ...]
#   stages (default: all of them, in this order):
#     download ingest pair fork discover merge judge score audit generalize
#     pack retrieve rollout evaluate
#
# Environment (all optional):
#   PYTHON            interpreter, default .venv/bin/python of the repo root
#   DATA_DIR          default data/            (trajectories, labels, id lists)
#   RUNS_DIR          default runs/            (per-run artefacts)
#   SUBMISSIONS       default data/verified_submissions.txt
#   EVAL_IDS          default data/swebench_pro_main_200_ids.txt
#   SKILL_MODEL       override models.skill_learning of configs/pipeline.yaml
#   BACKBONE          override models.backbone (Stage 3 calls + the agent)
#   REASONING_EFFORT  default high (passed to the agent's model_kwargs)
#   CONTAINER_TIMEOUT default 6h
#   MAX_FORMAT_ERRORS default 10
#   WORKERS           default 16
#   SKILLS_ROOT       packed skill to evaluate; default <run>/skills
#   PRO_EVALUATOR     path to the SWE-bench Pro evaluate.py (stage evaluate)
#   SWEBENCH_PRO_OS_ROOT  checkout of SWE-bench_Pro-os used by the evaluator
#
# API keys are read by litellm from the environment (DEEPSEEK_API_KEY,
# OPENAI_API_KEY / OPENAI_BASE_URL, ...). Put them in .env at the repo root;
# this script sources it when present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f .env ]] && { set -a; source .env; set +a; }

RUN_ID="${1:?usage: run_pipeline.sh <run_id> [stage ...]}"; shift || true
STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(download ingest pair fork discover merge \
    judge score audit generalize pack retrieve rollout evaluate)

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(dirname "$ROOT")/.venv/bin/python"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
RUNS_DIR="${RUNS_DIR:-$ROOT/runs}"
RUN="$RUNS_DIR/$RUN_ID"
SUBMISSIONS="${SUBMISSIONS:-$DATA_DIR/verified_submissions.txt}"
EVAL_IDS="${EVAL_IDS:-$DATA_DIR/swebench_pro_main_200_ids.txt}"
CONFIG="$ROOT/configs/pipeline.yaml"
WORKERS="${WORKERS:-16}"
SKILLS_ROOT="${SKILLS_ROOT:-$RUN/skills}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
CONTAINER_TIMEOUT="${CONTAINER_TIMEOUT:-6h}"
MAX_FORMAT_ERRORS="${MAX_FORMAT_ERRORS:-10}"
POOL="$RUN/traj_pool"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_FLAG=(); [[ -n "${SKILL_MODEL:-}" ]] && MODEL_FLAG=(--model "$SKILL_MODEL")
BACKBONE_FLAG=(); [[ -n "${BACKBONE:-}" ]] && BACKBONE_FLAG=(--model "$BACKBONE")

mkdir -p "$RUN"
log() { echo "=== [$(date -u +%FT%TZ)] $*" >&2; }
xr() { "$PYTHON" -m xreposkill "$@"; }

for stage in "${STAGES[@]}"; do
  log "stage: $stage (run $RUN_ID)"
  case "$stage" in
    download)
      xr download --submissions "$SUBMISSIONS" \
          --trajs-root "$DATA_DIR/trajs" --evals-root "$DATA_DIR/evals" \
          --workers "$WORKERS" ;;
    ingest)
      xr ingest --submissions "$SUBMISSIONS" --trajs-root "$DATA_DIR/trajs" \
          --evals-root "$DATA_DIR/evals" --pool-dir "$POOL" ;;
    pair)
      xr pair --pool-dir "$POOL" --out "$RUN/pairs/pairs.jsonl" \
          --stats-out "$RUN/pairs/stats.json" --K 4 \
          --eval-ids "$EVAL_IDS" \
          --eval-ids "$DATA_DIR/swebench_pro_ablation_50_ids.txt" ;;
    fork)
      xr fork --pairs "$RUN/pairs/pairs.jsonl" --pool-dir "$POOL" \
          --out "$RUN/pairs/forks.jsonl" ;;
    discover)
      xr discover --pairs "$RUN/pairs/pairs.jsonl" --forks "$RUN/pairs/forks.jsonl" \
          --pool-dir "$POOL" --out-dir "$RUN/candidates" --config "$CONFIG" \
          --workers "$WORKERS" "${MODEL_FLAG[@]}" ;;
    merge)
      xr merge --candidates-dir "$RUN/candidates" --buckets-dir "$RUN/buckets" \
          --out-root "$RUN/rules_merged" --config "$CONFIG" "${MODEL_FLAG[@]}" ;;
    judge)
      xr judge --rules-root "$RUN/rules_merged" --out "$RUN/rules_judged" \
          --judge-log "$RUN/judge_log.jsonl" --config "$CONFIG" \
          --workers "$WORKERS" "${MODEL_FLAG[@]}" ;;
    score)
      xr score --rules-root "$RUN/rules_judged" --out "$RUN/rules_scored" \
          --pool-dir "$POOL" --pairs "$RUN/pairs/pairs.jsonl" \
          --log "$RUN/score_log.jsonl" ;;
    audit)
      xr audit --rules-root "$RUN/rules_scored" --pairs "$RUN/pairs/pairs.jsonl" \
          --pool-dir "$POOL" --out "$RUN/ngram_audit.jsonl" ;;
    generalize)
      xr generalize --rules-root "$RUN/rules_scored" --pool-dir "$POOL" \
          --out-dir "$RUN/generalize" --config "$CONFIG" --workers "$WORKERS" \
          "${MODEL_FLAG[@]}" ;;
    pack)
      xr pack --generalize-dir "$RUN/generalize" --out "$RUN/skills" ;;
    retrieve)
      xr retrieve --skills-root "$SKILLS_ROOT" --instance-ids "$EVAL_IDS" \
          --out-dir "$RUN/eval/retrieval" --config "$CONFIG" \
          --workers "$WORKERS" "${BACKBONE_FLAG[@]}" ;;
    rollout)
      for cond in baseline skills; do
        xr rollout --instance-ids "$EVAL_IDS" --condition "$cond" \
            --skills-root "$SKILLS_ROOT" --retrieval-dir "$RUN/eval/retrieval" \
            --out-dir "$RUN/eval/$cond" --config "$CONFIG" \
            --reasoning-effort "$REASONING_EFFORT" \
            --container-timeout "$CONTAINER_TIMEOUT" \
            --max-format-errors "$MAX_FORMAT_ERRORS" \
            --workers "$WORKERS" "${BACKBONE_FLAG[@]}"
      done ;;
    evaluate)
      : "${PRO_EVALUATOR:?set PRO_EVALUATOR to the SWE-bench Pro evaluate.py}"
      for cond in baseline skills; do
        "$PYTHON" "$PRO_EVALUATOR" -o "$RUN/eval/$cond" -w 4 2>&1 | tail -3
      done
      "$PYTHON" scripts/summarize_eval.py "$RUN/eval" "$EVAL_IDS" ;;
    *)
      echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
done
log "done: ${STAGES[*]}"
