#!/usr/bin/env bash
set -euo pipefail

# --- User-tweakable defaults (can be overridden via env or CLI flags) ---
: "${PYTHON:=python}"               # or 'python3'/'conda run -n myenv python'
: "${MODULE:=cli.plot_agreement_and_qual}"  # module path
: "${ENV_GLOB:=results/envs/E_*.npz}"
: "${PLANNER:=a_star}"
: "${CONNECTIVITY:=8}"
: "${K_TOP:=5}"
: "${MAX_ENVS:=0}"                  # 0 = no cap
: "${MAX_FAILURES:=0}"              # 0 = no cap; useful for quick preview
: "${LIME_SAMPLES:=500}"
: "${LIME_FLIP:=0.30}"
: "${SHAP_PERM:=100}"
: "${WORKERS:=$(nproc)}"            # defaults to all cores; set WORKERS=8 to cap
: "${PROGRESS_EVERY:=5}"
: "${HEARTBEAT_SECS:=60}"
: "${SKIP_QUAL:=}"                  # set to "--skip-qual" to skip qualitative plots

# --- Paths (auto-generated per run) ---
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="runs/${TS}"
OUTDIR="${RUN_ROOT}/figs"
LOG="${RUN_ROOT}/run.log"
PID_FILE="${RUN_ROOT}/run.pid"
CMD_FILE="${RUN_ROOT}/run.cmd"

mkdir -p "${OUTDIR}"

# Build the command (allow additional CLI args to override anything)
build_cmd() {
  local extra_args=("$@")
  # -u: unbuffered, so logs flush immediately
  cat <<EOF
${PYTHON} -u -m ${MODULE} \
  --env-glob "${ENV_GLOB}" \
  --outdir "${OUTDIR}" \
  --planner "${PLANNER}" --connectivity ${CONNECTIVITY} \
  --k ${K_TOP} --max-envs ${MAX_ENVS} \
  --max-failures ${MAX_FAILURES} \
  --lime-samples ${LIME_SAMPLES} --lime-flip ${LIME_FLIP} \
  --shap-perm ${SHAP_PERM} \
  --workers ${WORKERS} \
  --progress-every ${PROGRESS_EVERY} \
  --heartbeat-secs ${HEARTBEAT_SECS} \
  ${SKIP_QUAL:-} \
  ${extra_args[*]}
EOF
}

start() {
  # Pass any extra args after 'start' straight to the Python script to override defaults
  local cmd
  cmd="$(build_cmd "${@}")"

  echo "[run.sh] Output dir: ${OUTDIR}"
  echo "[run.sh] Log file  : ${LOG}"
  echo "[run.sh] PID file  : ${PID_FILE}"
  echo "${cmd}" > "${CMD_FILE}"

  # Launch under nohup, redirect stdout+stderr to log, detach to background
  nohup bash -lc "${cmd}" > "${LOG}" 2>&1 &
  echo $! > "${PID_FILE}"

  echo "[run.sh] Started with PID $(cat "${PID_FILE}")"
  echo "[run.sh] Tail progress: tail -f ${LOG}"
}

status() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if ps -p "${pid}" > /dev/null 2>&1; then
      echo "[run.sh] RUNNING (PID ${pid})"
    else
      echo "[run.sh] NOT RUNNING (stale PID file: ${PID_FILE})"
    fi
    echo "[run.sh] Log: ${LOG}"
    [[ -f "${CMD_FILE}" ]] && { echo "[run.sh] Command:"; cat "${CMD_FILE}"; }
  else
    echo "[run.sh] No active run in ${RUN_ROOT} (no PID file)."
  fi
}

tail_log() {
  if [[ -f "${LOG}" ]]; then
    echo "[run.sh] Tailing ${LOG} (Ctrl-C to stop tailing, job keeps running)"
    tail -f "${LOG}"
  else
    echo "[run.sh] No log found at ${LOG}"
  fi
}

stop() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if ps -p "${pid}" > /dev/null 2>&1; then
      echo "[run.sh] Stopping PID ${pid} ..."
      kill "${pid}" || true
      sleep 1
      if ps -p "${pid}" > /dev/null 2>&1; then
        echo "[run.sh] Still running, sending SIGKILL ..."
        kill -9 "${pid}" || true
      fi
      echo "[run.sh] Stopped."
    else
      echo "[run.sh] Process not running; removing stale ${PID_FILE}"
    fi
    rm -f "${PID_FILE}"
  else
    echo "[run.sh] No PID file; nothing to stop."
  fi
  echo "[run.sh] Last log at: ${LOG}"
}

usage() {
  cat <<USAGE
Usage: $0 <command> [extra-args...]

Commands:
  start [--any --script --flags]  Start a new run under nohup.
                                  Extra flags override defaults (e.g., --skip-qual).
  status                           Show if run (PID) is alive and print the command used.
  tail                             Tail the current run's log (live progress).
  stop                             Stop the running job (SIGTERM, then SIGKILL if needed).

Examples:
  # Quick preview: limit failures and skip qualitative PDFs
  ./run.sh start --max-failures 200 --skip-qual

  # Full run with explicit workers and k=5
  WORKERS=8 ./run.sh start --k 5

Environment overrides (before 'start'):
  PYTHON, MODULE, ENV_GLOB, PLANNER, CONNECTIVITY, K_TOP, MAX_ENVS, MAX_FAILURES,
  LIME_SAMPLES, LIME_FLIP, SHAP_PERM, WORKERS, PROGRESS_EVERY, HEARTBEAT_SECS, SKIP_QUAL

Outputs:
  ${OUTDIR}/q1_agreement_heatmap.pdf
  ${OUTDIR}/qual_case_*.pdf   (unless --skip-qual)
  ${LOG}                      (progress + heartbeat)
  ${PID_FILE}                 (running process ID)
USAGE
}

cmd="${1:-}"
shift || true

case "${cmd}" in
  start) start "$@" ;;
  status) status ;;
  tail) tail_log ;;
  stop) stop ;;
  ""|-h|--help|help) usage ;;
  *) echo "Unknown command: ${cmd}"; usage; exit 1 ;;
esac
