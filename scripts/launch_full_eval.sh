#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Launch the full evaluation with GNU Parallel:
#  - cores-aware (-j 0 uses number of visible cores)
#  - sharded jobs per size×density for better utilization and faster feedback
#  - deterministic seeding (seed = manifest line number)
#  - neat run folder layout under runs/$RUN_ID
#
# USAGE:
#   bash scripts/launch_full_eval.sh [RUN_ID]
# EXAMPLE:
#   RUN_ID=$(date +%Y%m%d_%H%M%S)_full
#   bash scripts/launch_full_eval.sh "$RUN_ID"
#
# Optional env knobs before calling:
#   JOBS=0                 # -j 0 → number of visible cores (default)
#   REPS=6                 # shards per (size,density)
#   NUM_ENVS=150           # total envs per (size,density) (will be split across REPS)
# -----------------------------------------------------------------------------
set -euo pipefail

# --------------------------- knobs you likely tweak ---------------------------
SIZES="20x20 30x30 40x40"
DENS="0.20 0.30 0.40"
PLANNERS="a_star,dijkstra,bfs,dfs,theta_star"
EXPL="shap,lime,cose,geodesic,rand"

KMAX=8
CONNECTIVITY=8
LIME_SAMPLES=500
SHAP_PERM=100
FOCUS_TOP_M=20

# Sharding: split NUM_ENVS/REPS per job (remainder distributed to the first few shards)
: "${REPS:=6}"           # shards per (size,density)
: "${NUM_ENVS:=150}"     # total envs per (size,density)
# Concurrency (0 = auto-detect visible cores)
: "${JOBS:=0}"

# Deterministic single-thread math libs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

# --------------------------- run id & directories ----------------------------
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)_full}"

ROOT="runs/$RUN_ID"
OUTDIR="$ROOT/csv"
FIGDIR="$ROOT/figs"
LOGDIR="$ROOT/logs"
JOBDIR="$ROOT/joblogs"
MANIDIR="$ROOT/manifest"
RESDIR="$ROOT/results"

mkdir -p "$OUTDIR" "$FIGDIR" "$LOGDIR" "$JOBDIR" "$MANIDIR" "$RESDIR"

# --------------------------- capture reproducibility -------------------------
{ git rev-parse HEAD 2>/dev/null || echo "no_git"; }   > "$ROOT/meta_git_commit.txt" || true
{ git status --porcelain 2>/dev/null || true; }       > "$ROOT/meta_git_dirty.txt"  || true
{ uname -a || true; }                                 > "$ROOT/meta_uname.txt"      || true
{ lscpu || true; }                                    > "$ROOT/meta_lscpu.txt"      || true
{ python -V || true; }                                > "$ROOT/meta_python_version.txt" || true
{ pip freeze 2>/dev/null || true; }                   > "$ROOT/meta_pip_freeze.txt" || true

# --------------------------- info banner in logs -----------------------------
CORES=$(parallel --number-of-cores || echo 0)
CPUS=$(parallel --number-of-cpus  || echo 0)
echo "[INFO] RUN_ID=$RUN_ID"
echo "[INFO] Visible cores: $CORES   visible CPUs: $CPUS"
echo "[INFO] JOBS concurrency set to: ${JOBS}  (0 = use visible cores)"
echo "[INFO] Sharding: REPS=$REPS  NUM_ENVS=$NUM_ENVS  → per-shard ≈ $((NUM_ENVS/REPS)) (+ remainder)"

# --------------------------- build manifest (no seed) ------------------------
MANIFEST="$MANIDIR/eval_cmds.txt"
: > "$MANIFEST"

for s in $SIZES; do
  for d in $DENS; do
    # split NUM_ENVS into REPS shards with fair distribution
    base=$(( NUM_ENVS / REPS ))
    rem=$(( NUM_ENVS % REPS ))
    for r in $(seq 1 "$REPS"); do
      n_this=$base
      if [ "$r" -le "$rem" ]; then n_this=$(( n_this + 1 )); fi
      echo "python -m cli.run_eval \
        --sizes $s --densities $d \
        --num-envs $n_this \
        --planners $PLANNERS --explainers $EXPL \
        --kmax $KMAX --connectivity $CONNECTIVITY \
        --lime-samples $LIME_SAMPLES --shap-perm $SHAP_PERM \
        --focus-top-m $FOCUS_TOP_M --robustness false \
        --outdir $OUTDIR" >> "$MANIFEST"
    done
  done
done

TOTAL=$(wc -l < "$MANIFEST")
JOBLOG="$JOBDIR/eval.tsv"

echo "[INFO] Total jobs in manifest: $TOTAL"
echo "[INFO] Manifest: $MANIFEST"
echo "[INFO] Joblog:   $JOBLOG"
echo "[INFO] Results:  $RESDIR/eval/<host>/<seq>/{stdout,stderr}"
echo "[INFO] Combined log: $LOGDIR/eval.log"

# --------------------------- launch with ETA & per-job logs ------------------
# Seed is the manifest line number; nl gives us {1}=line#, {2}=command
# stdbuf ensures line-buffered output for smoother live tails
nl -ba "$MANIFEST" | \
stdbuf -oL -eL parallel -j "$JOBS" --lb --eta --joblog "$JOBLOG" \
         --colsep '\t' --results "$RESDIR/eval" \
         '{2} --seed {1}' 2>&1 | tee "$LOGDIR/eval.log"

echo "[DONE] Launched $TOTAL eval jobs for RUN_ID=$RUN_ID"
echo "[PATHS]"
echo "  CSV:    $OUTDIR"
echo "  FIGS:   $FIGDIR"
echo "  LOGS:   $LOGDIR  (combined)"
echo "  RESULTS:$RESDIR/eval  (per-job stdout/stderr)"
echo "  MANIFEST:$MANIFEST"
echo "  JOBLOG: $JOBLOG"
