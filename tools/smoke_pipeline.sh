#!/usr/bin/env bash
# End-to-end check of the command-line pipeline on a tiny fake dataset with a random CLIP (CPU, ~3 minutes):
#   build_cache -> evaluate (main, few-shot TINS, ablations) -> summarize -> export_features -> experiments.
# The numbers are meaningless; the point is that every step runs and writes what the next one reads.
#
#   bash tools/smoke_pipeline.sh [work_dir]          (default: runs/smoke)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$(mkdir -p "${1:-$ROOT/runs/smoke}" && cd "${1:-$ROOT/runs/smoke}" && pwd)"
export OODLAB_SMOKE=1 MPLBACKEND=Agg PYTHONHASHSEED=0
PY="${PYTHON:-python}"
step() { echo; echo "=== $*"; }

step "1/7 feature cache (fake ImageNet + OpenOOD, random tiny CLIP)"
"$PY" "$ROOT/scripts/build_cache.py" --home "$W/home" > "$W/build_cache.log" 2>&1 || { tail -30 "$W/build_cache.log"; exit 1; }
tail -3 "$W/build_cache.log"
CACHE="$W/home/working/cache"

step "2/7 evaluate: main table (TANL + ReNeg operating points + few-shot rows), one stream order"
"$PY" "$ROOT/scripts/evaluate.py" --config "$ROOT/configs/main_vitb16.yaml" --cache "$CACHE" --out "$W/runs/main" \
    --random-clip --seeds 0 | tail -4

step "3/7 evaluate: few-shot table (TINS + TANL + nine ReNeg heads) and the ablations"
"$PY" "$ROOT/scripts/evaluate.py" --config "$ROOT/configs/fewshot_tins.yaml" --cache "$CACHE" --out "$W/runs/fewshot" \
    --random-clip --seeds 0 --datasets ninco places | tail -3
"$PY" "$ROOT/scripts/evaluate.py" --config "$ROOT/configs/ablation_tanl_base.yaml" --cache "$CACHE" --out "$W/runs/abl" \
    --random-clip --seeds 0 --datasets ninco | tail -2
"$PY" "$ROOT/scripts/evaluate.py" --config "$ROOT/configs/ablation_neglabel_base.yaml" --cache "$CACHE" \
    --out "$W/runs/abl_nl" --random-clip --seeds 0 --datasets ninco | tail -2

step "4/7 summarize"
"$PY" "$ROOT/scripts/summarize.py" "$W/runs/main/runs.csv" --ref tanl > "$W/summary.txt"
head -12 "$W/summary.txt"

step "5/7 export 8-bit features for the CPU experiments"
"$PY" "$ROOT/scripts/export_features.py" --cache "$CACHE" --out "$W/exports" --random-clip \
    --what streams four_ood textbank pool | tail -2

step "6/7 experiments: equivalence check, TANL traces, replays"
export RENEG_EXPORTS="$W/exports" RENEG_EXP_OUT="$W/experiments"
"$PY" "$ROOT/experiments/00_check_equivalence.py" --n 300 | tail -2
"$PY" "$ROOT/experiments/01_build_traces.py" --seeds 0 > "$W/traces.log" 2>&1 || { tail -20 "$W/traces.log"; exit 1; }
tail -2 "$W/traces.log"
"$PY" "$ROOT/experiments/02_replay.py" paper --seeds 0 | tail -2
"$PY" "$ROOT/experiments/02_replay.py" table --ref tanl > "$W/replay_table.txt"
head -8 "$W/replay_table.txt"
"$PY" "$ROOT/experiments/04_four_ood_fixes.py" run --stage screen-2 | tail -2
"$PY" "$ROOT/experiments/05_fewshot_and_oracle.py" run --stage oracle-2 | tail -2

step "7/7 experiments: generic replay on another base (TINS scores replaced by TANL's own, selftest)"
"$PY" "$ROOT/experiments/07_reneg_on_tins.py" selftest

echo
echo "SMOKE PIPELINE PASSED (outputs in $W)"
