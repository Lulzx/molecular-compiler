#!/usr/bin/env bash
# Phase 1 run queue (spec 10.3, Revision 12). Restricted outputs stay in
# artifacts/phase1/. Every fold checkpoints after each step, so rerunning this
# script after an interruption resumes where it stopped.
#
#   1. wait for a running `molc individual-state` (Track I0) to finish;
#   2. base leave_class_out folds, three at a time;
#   3. Section 12.6 convergence on the trained fold 0, then the report;
#   4. assumption variants (cl3, cl8, hill1) on the same folds, then the report.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-60}
PARALLEL=${PARALLEL:-3}
LOGS=artifacts/phase1/logs
mkdir -p "$LOGS"

while pgrep -f "molc individual-state" >/dev/null; do sleep 60; done

fold() {  # variant fold
  uv run molc phase1-fold --split leave_class_out --fold "$2" --variant "$1" \
    --steps "$STEPS" --batch 4 >"$LOGS/leave_class_out-$1-fold$2.log" 2>&1 \
    && echo "done $1 fold $2" || echo "FAILED $1 fold $2"
}
export -f fold
export STEPS LOGS

echo "base folds"
printf 'base %s\n' 0 1 2 3 4 | xargs -P "$PARALLEL" -n 2 bash -c 'fold "$0" "$1"'
uv run molc phase1-convergence --trained artifacts/phase1/leave_class_out-fold0.json \
  >"$LOGS/convergence-trained.log" 2>&1
uv run molc phase1-report >"$LOGS/report-base.log" 2>&1
echo "base report written"

echo "variants"
for v in cl3 cl8 hill1; do printf "$v %s\n" 0 1 2 3 4; done \
  | xargs -P "$PARALLEL" -n 2 bash -c 'fold "$0" "$1"'
uv run molc phase1-report >"$LOGS/report-variants.log" 2>&1
echo "variant report written"
