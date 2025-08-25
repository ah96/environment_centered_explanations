#!/usr/bin/env bash
# Parallel full evaluation (uses up to 8 processes)

set -Eeuo pipefail

# ---- General config ----
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"    # number of concurrent processes
PYTHON_BIN="${PYTHON_BIN:-python}"     # override with: PYTHON_BIN=python3 ./launch_full_eval.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_ID="$(date +'%Y%m%d_%H%M%S')"
RUN_DIR="runs/${RUN_ID}"
CSV_DIR="${RUN_DIR}/csv"
FIGS_DIR="${RUN_DIR}/figs"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${CSV_DIR}" "${FIGS_DIR}" "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/full_eval_${RUN_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Full Evaluation (parallel) ==="
echo "Run: ${RUN_DIR}   Jobs: ${PARALLEL_JOBS}"
${PYTHON_BIN} -V || true

# ---- Workloads (same totals as before, split across seeds) ----
# We achieve parallelism by sharding the environment count across seeds.
EVAL_SIZES="${EVAL_SIZES:-20x20,30x30,40x40}"
EVAL_DENS="${EVAL_DENS:-0.10,0.20,0.30}"
EVAL_NUM_ENVS_TOTAL="${EVAL_NUM_ENVS_TOTAL:-100}"
EVAL_PLANNERS="${EVAL_PLANNERS:-a_star,dijkstra,bfs,dfs,theta_star}"
EVAL_EXPLAINERS="${EVAL_EXPLAINERS:-shap,lime,cose,rand,geodesic}"
EVAL_KMAX="${EVAL_KMAX:-5}"

EXACT_SIZE="${EXACT_SIZE:-20x20}"
EXACT_DENS="${EXACT_DENS:-0.20}"
EXACT_NUM_ENVS_TOTAL="${EXACT_NUM_ENVS_TOTAL:-100}"
EXACT_PLANNER="${EXACT_PLANNER:-a_star}"
EXACT_TIME_LIMIT="${EXACT_TIME_LIMIT:-60}"

ROB_SIZES="${ROB_SIZES:-30x30}"
ROB_DENS="${ROB_DENS:-0.20}"
ROB_NUM_ENVS_TOTAL="${ROB_NUM_ENVS_TOTAL:-50}"
ROB_PLANNERS="${ROB_PLANNERS:-a_star,dijkstra,bfs,dfs,theta_star}"
ROB_EXPLAINERS="${ROB_EXPLAINERS:-shap,lime,cose,rand,geodesic}"
ROB_KMAX="${ROB_KMAX:-5}"

TR_SIZES="${TR_SIZES:-20x20,30x30}"
TR_DENS="${TR_DENS:-0.10,0.20}"
TR_NUM_ENVS_TOTAL="${TR_NUM_ENVS_TOTAL:-50}"
TR_PLANNERS="${TR_PLANNERS:-a_star,dijkstra,bfs,dfs,theta_star}"
TR_EXPLAINERS="${TR_EXPLAINERS:-shap,lime,rand,geodesic}"
TR_KMAX="${TR_KMAX:-5}"

# Compute per‑shard counts (integer ceil so we don’t lose any envs)
ceil_div() { python - "$@" <<'PY'
import math,sys
n=int(sys.argv[1]); d=int(sys.argv[2]); print((n+d-1)//d)
PY
}
EVAL_PER_SHARD=$(ceil_div "${EVAL_NUM_ENVS_TOTAL}" "${PARALLEL_JOBS}")
EXACT_PER_SHARD=$(ceil_div "${EXACT_NUM_ENVS_TOTAL}" "${PARALLEL_JOBS}")
ROB_PER_SHARD=$(ceil_div "${ROB_NUM_ENVS_TOTAL}" "${PARALLEL_JOBS}")
TR_PER_SHARD=$(ceil_div "${TR_NUM_ENVS_TOTAL}" "${PARALLEL_JOBS}")

# Helper to run one eval shard with a specific seed and num_envs
run_eval_shard() {
  local seed="$1" num_envs="$2"
  ${PYTHON_BIN} -m cli.run_eval \
    --sizes "${EVAL_SIZES}" \
    --densities "${EVAL_DENS}" \
    --num-envs "${num_envs}" \
    --planners "${EVAL_PLANNERS}" \
    --explainers "${EVAL_EXPLAINERS}" \
    --kmax "${EVAL_KMAX}" \
    --seed "${seed}" \
    --outdir "${CSV_DIR}"
}

run_exact_shard() {
  local seed="$1" num_envs="$2"
  ${PYTHON_BIN} -m cli.run_exact_small \
    --size "${EXACT_SIZE}" \
    --density "${EXACT_DENS}" \
    --num-envs "${num_envs}" \
    --planner "${EXACT_PLANNER}" \
    --time-limit "${EXACT_TIME_LIMIT}" \
    --seed "${seed}" \
    --outdir "${CSV_DIR}"
}

run_robust_shard() {
  local seed="$1" num_envs="$2"
  ${PYTHON_BIN} -m cli.run_robustness \
    --sizes "${ROB_SIZES}" \
    --densities "${ROB_DENS}" \
    --num-envs "${num_envs}" \
    --planners "${ROB_PLANNERS}" \
    --explainers "${ROB_EXPLAINERS}" \
    --kmax "${ROB_KMAX}" \
    --seed "${seed}" \
    --outdir "${CSV_DIR}"
}

run_transfer_shard() {
  local seed="$1" num_envs="$2"
  ${PYTHON_BIN} -m cli.run_transfer \
    --sizes "${TR_SIZES}" \
    --densities "${TR_DENS}" \
    --num-envs "${num_envs}" \
    --planners "${TR_PLANNERS}" \
    --explainers "${TR_EXPLAINERS}" \
    --kmax "${TR_KMAX}" \
    --seed "${seed}" \
    --outdir "${CSV_DIR}"
}

# Function to fan out a task across N shards (seeds 0..N-1)
fanout() {
  local jobfun="$1" per_shard="$2" label="$3"
  echo "==> ${label}: ${PARALLEL_JOBS} shards × ${per_shard} envs each"
  if command -v parallel >/dev/null 2>&1; then
    # GNU parallel path
    seq 0 $((PARALLEL_JOBS-1)) | parallel -j "${PARALLEL_JOBS}" --halt soon,fail=1 "$jobfun" {} "${per_shard}"
  else
    # POSIX fallback: background jobs
    pids=()
    for s in $(seq 0 $((PARALLEL_JOBS-1))); do
      $jobfun "$s" "${per_shard}" &
      pids+=("$!")
    done
    # Wait and fail if any failed
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  fi
}

echo "[1/5] Main eval shards..."
if command -v parallel >/dev/null 2>&1; then
  seq 0 $((PARALLEL_JOBS-1)) | parallel -j "${PARALLEL_JOBS}" --halt soon,fail=1 \
    "${PYTHON_BIN} -m cli.run_eval \
      --sizes '${EVAL_SIZES}' \
      --densities '${EVAL_DENS}' \
      --num-envs ${EVAL_PER_SHARD} \
      --planners '${EVAL_PLANNERS}' \
      --explainers '${EVAL_EXPLAINERS}' \
      --kmax ${EVAL_KMAX} \
      --seed {} \
      --outdir '${CSV_DIR}'"
else
  pids=()
  for s in $(seq 0 $((PARALLEL_JOBS-1))); do
    ${PYTHON_BIN} -m cli.run_eval \
      --sizes "${EVAL_SIZES}" \
      --densities "${EVAL_DENS}" \
      --num-envs "${EVAL_PER_SHARD}" \
      --planners "${EVAL_PLANNERS}" \
      --explainers "${EVAL_EXPLAINERS}" \
      --kmax "${EVAL_KMAX}" \
      --seed "${s}" \
      --outdir "${CSV_DIR}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
fi

echo "[2/5] Exact minimality shards..."
if command -v parallel >/dev/null 2>&1; then
  seq 0 $((PARALLEL_JOBS-1)) | parallel -j "${PARALLEL_JOBS}" --halt soon,fail=1 \
    "${PYTHON_BIN} -m cli.run_exact_small \
      --size '${EXACT_SIZE}' \
      --density '${EXACT_DENS}' \
      --num-envs ${EXACT_PER_SHARD} \
      --planner '${EXACT_PLANNER}' \
      --time-limit ${EXACT_TIME_LIMIT} \
      --seed {} \
      --outdir '${CSV_DIR}'"
else
  pids=()
  for s in $(seq 0 $((PARALLEL_JOBS-1))); do
    ${PYTHON_BIN} -m cli.run_exact_small \
      --size "${EXACT_SIZE}" \
      --density "${EXACT_DENS}" \
      --num-envs "${EXACT_PER_SHARD}" \
      --planner "${EXACT_PLANNER}" \
      --time-limit "${EXACT_TIME_LIMIT}" \
      --seed "${s}" \
      --outdir "${CSV_DIR}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
fi

echo "[3/5] Robustness shards..."
if command -v parallel >/dev/null 2>&1; then
  seq 0 $((PARALLEL_JOBS-1)) | parallel -j "${PARALLEL_JOBS}" --halt soon,fail=1 \
    "${PYTHON_BIN} -m cli.run_robustness \
      --sizes '${ROB_SIZES}' \
      --densities '${ROB_DENS}' \
      --num-envs ${ROB_PER_SHARD} \
      --planners '${ROB_PLANNERS}' \
      --explainers '${ROB_EXPLAINERS}' \
      --kmax ${ROB_KMAX} \
      --seed {} \
      --outdir '${CSV_DIR}'"
else
  pids=()
  for s in $(seq 0 $((PARALLEL_JOBS-1))); do
    ${PYTHON_BIN} -m cli.run_robustness \
      --sizes "${ROB_SIZES}" \
      --densities "${ROB_DENS}" \
      --num-envs "${ROB_PER_SHARD}" \
      --planners "${ROB_PLANNERS}" \
      --explainers "${ROB_EXPLAINERS}" \
      --kmax "${ROB_KMAX}" \
      --seed "${s}" \
      --outdir "${CSV_DIR}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
fi

echo "[4/5] Transfer shards..."
if command -v parallel >/dev/null 2>&1; then
  seq 0 $((PARALLEL_JOBS-1)) | parallel -j "${PARALLEL_JOBS}" --halt soon,fail=1 \
    "${PYTHON_BIN} -m cli.run_transfer \
      --sizes '${TR_SIZES}' \
      --densities '${TR_DENS}' \
      --num-envs ${TR_PER_SHARD} \
      --planners '${TR_PLANNERS}' \
      --explainers '${TR_EXPLAINERS}' \
      --kmax ${TR_KMAX} \
      --seed {} \
      --outdir '${CSV_DIR}'"
else
  pids=()
  for s in $(seq 0 $((PARALLEL_JOBS-1))); do
    ${PYTHON_BIN} -m cli.run_transfer \
      --sizes "${TR_SIZES}" \
      --densities "${TR_DENS}" \
      --num-envs "${TR_PER_SHARD}" \
      --planners "${TR_PLANNERS}" \
      --explainers "${TR_EXPLAINERS}" \
      --kmax "${TR_KMAX}" \
      --seed "${s}" \
      --outdir "${CSV_DIR}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
fi

echo "[5/5] Making figures..."
${PYTHON_BIN} -m cli.make_figs \
  --eval-glob       "${CSV_DIR}/eval_*.csv" \
  --exact-glob      "${CSV_DIR}/exact_small_*.csv" \
  --transfer-glob   "${CSV_DIR}/transfer_*.csv" \
  --robustness-glob "${CSV_DIR}/robustness_*.csv" \
  --outdir          "${FIGS_DIR}"

echo "All done. Figures in ${FIGS_DIR}"
