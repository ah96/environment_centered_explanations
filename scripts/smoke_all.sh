#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m cli.run_eval --sizes 20x20 --densities 0.30 --num-envs 6 --planners a_star \
  --explainers shap,lime,cose,geodesic,rand --kmax 5 --seed 0 --robustness false --outdir results/csv
python -m cli.run_exact_small --size 20x20 --density 0.25 --num-envs 20 --planner a_star \
  --connectivity 8 --time-limit 30 --shap-perm 40 --seed 0 --outdir results/csv
python -m cli.run_robustness --sizes 20x20 --densities 0.30 --num-envs 10 --planners a_star \
  --explainers shap,lime,cose,geodesic --kmax 5 --connectivity 8 --seed 0 \
  --lime-samples 200 --shap-perm 40 --outdir results/csv
python -m cli.run_transfer --sizes 20x20 --densities 0.30 --num-envs 30 \
  --planners a_star,dijkstra --explainers shap,lime,geodesic,rand \
  --kmax 5 --connectivity 8 --seed 0 --lime-samples 200 --shap-perm 40 --outdir results/csv
python -m cli.make_figs --eval-glob "results/csv/eval_*.csv" \
  --exact-glob "results/csv/exact_small_*.csv" --transfer-glob "results/csv/transfer_*.csv" \
  --robustness-glob "results/csv/robustness_*.csv" --outdir "results/figs" --transfer-method shap
echo "[SMOKE] Done. Figures in results/figs/"
