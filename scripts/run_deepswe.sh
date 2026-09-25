#!/usr/bin/env bash
# DeepSWE evaluation through Pier: per-task rule selection (Stage 3a), then
# a baseline arm and a skills arm of mini-SWE-agent on every task.
#
# Usage: scripts/run_deepswe.sh <run_id>
# Environment:
#   DSWE          DeepSWE checkout with tasks/<id>/task.toml (required)
#   BACKBONE      backbone LLM, default deepseek/deepseek-v4-flash
#   SKILLS_ROOT   packed skill directory, e.g. runs/<run_id>/skills (required)
#   WORKERS       parallel tasks, default 16
#   PIER_EXTRA    extra flags for `pier run` (for example --ae SSL_VERIFY=False)
#   REASONING_EFFORT default high
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f .env ]] && { set -a; source .env; set +a; }
RUN_ID="${1:?usage: run_deepswe.sh <run_id>}"
: "${DSWE:?set DSWE to the DeepSWE checkout}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(dirname "$ROOT")/.venv/bin/python"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MODEL="${BACKBONE:-deepseek/deepseek-v4-flash}"
: "${SKILLS_ROOT:?set SKILLS_ROOT to a packed skill directory}"
WORKERS="${WORKERS:-16}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
PREP="$ROOT/runs/$RUN_ID/deepswe_prep"
mkdir -p "$PREP"
TASKS="$ROOT/runs/$RUN_ID/deepswe_task_ids.txt"
for t in $(ls "$DSWE/tasks"); do [[ -f "$DSWE/tasks/$t/task.toml" ]] && echo "$t"; done > "$TASKS"
echo "tasks: $(wc -l < "$TASKS")"

echo "=== Stage 3a: per-task rule selection ==="
prep_one() {
  tid="$1"
  [[ -s "$PREP/$tid.prompt.j2" ]] && return 0
  "$PYTHON" -m xreposkill deepswe-prep --task-dir "$DSWE/tasks/$tid" \
      --skills-root "$SKILLS_ROOT" --model "$MODEL" --out-dir "$PREP" \
      > "$PREP/$tid.prep.log" 2>&1 || echo "PREP_FAIL $tid"
}
export -f prep_one; export PYTHON PREP DSWE SKILLS_ROOT MODEL
xargs -a "$TASKS" -P8 -I{} bash -c 'prep_one {}'

run_one() {
  tid="$1"; prefix="$2"
  compgen -G "$DSWE/jobs/$prefix-$tid/*/verifier/reward.json" >/dev/null && return 0
  tflag=""
  [[ "${3:-}" == "skills" ]] && tflag="--ak prompt_template_path=$PREP/$tid.prompt.j2"
  pier run -p "$DSWE/tasks/$tid" --agent mini-swe-agent --model "$MODEL" \
      --ak reasoning_effort="$REASONING_EFFORT" ${PIER_EXTRA:-} $tflag \
      -n 1 --job-name "$prefix-$tid" --yes \
      > "$DSWE/jobs_logs_$prefix-$tid.log" 2>&1 || echo "RUN_FAIL $prefix-$tid"
}
export -f run_one; export REASONING_EFFORT PIER_EXTRA
echo "=== baseline arm ==="
xargs -a "$TASKS" -P"$WORKERS" -I{} bash -c 'run_one {} base'
echo "=== skills arm ==="
xargs -a "$TASKS" -P"$WORKERS" -I{} bash -c 'run_one {} skill skills'

echo "=== summary ==="
"$PYTHON" - "$DSWE" "$TASKS" <<'PY'
import glob, json, sys
from math import comb
from pathlib import Path
dswe, ids = Path(sys.argv[1]), Path(sys.argv[2]).read_text().split()
def reward(prefix):
    out = {}
    for t in ids:
        rs = glob.glob(str(dswe / "jobs" / f"{prefix}-{t}" / "*" / "verifier" / "reward.json"))
        out[t] = max((int(json.loads(Path(r).read_text()).get("reward") or 0) for r in rs), default=0)
    return out
base, sk = reward("base"), reward("skill")
n = len(ids)
print(f"baseline pass@1: {sum(base.values())}/{n}")
print(f"skills   pass@1: {sum(sk.values())}/{n}")
b = sum(1 for t in ids if sk[t] and not base[t]); c = sum(1 for t in ids if base[t] and not sk[t])
p = 2 * sum(comb(b + c, k) for k in range(0, min(b, c) + 1)) / 2 ** (b + c) if b + c else 1.0
print(f"skills-only wins: {b}, baseline-only wins: {c}, McNemar exact p={min(p, 1.0):.4f}")
PY
