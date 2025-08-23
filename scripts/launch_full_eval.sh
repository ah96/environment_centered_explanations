#!/usr/bin/env bash
set -euo pipefail

# --- knobs ---
SIZES="20x20 30x30 40x40"
DENS="0.20 0.30 0.40"
PLANNERS="a_star,dijkstra,bfs,dfs,theta_star"
EXPL="shap,lime,cose,geodesic,rand"
KMAX=8
LIME_SAMPLES=500
SHAP_PERM=100
FOCUS_TOP_M=20
NUM_ENVS=150
CONNECTIVITY=8
JOBS=16   # set <= core count (leave a little headroom)

# --- reproducibility & perf ---
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

RUN_ID="$1"  # first arg is run id (e.g., 20250822_153000_full)

ROOT="runs/$RUN_ID"
OUTDIR="$ROOT/csv"
FIGDIR="$ROOT/figs"
LOGDIR="$ROOT/logs"
JOBDIR="$ROOT/joblogs"
MANIDIR="$ROOT/manifest"
RESDIR="$ROOT/results"   # per-job stdout/stderr (from parallel --results)

mkdir -p "$OUTDIR" "$FIGDIR" "$LOGDIR" "$JOBDIR" "$MANIDIR" "$RESDIR"

# metadata (for reproducibility)
git rev-parse HEAD            > "$ROOT/meta_git_commit.txt" 2>/dev/null || true
git status --porcelain        > "$ROOT/meta_git_dirty.txt"  2>/dev/null || true
uname -a                      > "$ROOT/meta_uname.txt"      2>/dev/null || true
lscpu                         > "$ROOT/meta_lscpu.txt"      2>/dev/null || true
python -V                     > "$ROOT/meta_python_version.txt"
pip freeze                    > "$ROOT/meta_pip_freeze.txt" 2>/dev/null || true

# Build manifest WITHOUT a seed; we’ll inject the seed via line numbers.
MANIFEST="$MANIDIR/eval_cmds.txt"
: > "$MANIFEST"
for s in $SIZES; do
  for d in $DENS; do
    echo "python -m cli.run_eval --sizes $s --densities $d \
      --num-envs $NUM_ENVS \
      --planners $PLANNERS --explainers $EXPL \
      --kmax $KMAX --connectivity $CONNECTIVITY \
      --lime-samples $LIME_SAMPLES --shap-perm $SHAP_PERM \
      --focus-top-m $FOCUS_TOP_M --robustness false \
      --outdir $OUTDIR" >> "$MANIFEST"
  done
done

TOTAL=$(wc -l < "$MANIFEST")
JOBLOG="$JOBDIR/eval.tsv"

# Run with ETA; {1} (line number) becomes the seed; {2} is the command line.
# --results captures per-job stdout/stderr under $RESDIR for deep debugging.
nl -ba "$MANIFEST" | \
parallel -j "$JOBS" --lb --eta --joblog "$JOBLOG" --colsep '\t' \
         --results "$RESDIR/eval" \
         '{2} --seed {1}' 2>&1 | tee "$LOGDIR/eval.log"

echo "[DONE] Launched $TOTAL eval jobs for RUN_ID=$RUN_ID"
echo "CSV:   $OUTDIR"
echo "LOGS:  $LOGDIR (combined), $RESDIR/eval (per-job)"
echo "JOBLOG:$JOBLOG"
