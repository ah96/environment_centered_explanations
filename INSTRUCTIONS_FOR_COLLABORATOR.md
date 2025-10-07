# 🧭 Environment-Centered Explanations — Collaborator Guide

Welcome aboard!  
This document will guide you through every aspect of the project — from setup to evaluation, testing, debugging, and extending the framework.

---

## 1️⃣ Project Overview

This repository implements a **planner-agnostic framework** for generating **environment-centered causal explanations** of robot navigation failures.

Instead of peeking into a planner’s internals, the system manipulates **obstacle configurations** in a grid world to identify:
- **Causal obstacles** (those that make a path infeasible)
- **Minimal modifications** (the smallest set of obstacles whose removal restores a valid plan)

We adapt three explanation methods:
- **LIME** – random perturbations and regression-based attributions  
- **SHAP** – Shapley-value reasoning for causal ranking  
- **COSE** – our novel Counterfactual Obstacle Set Explainer (greedy minimal set discovery)

These are evaluated on multiple planners: **A\***, **Dijkstra**, **BFS**, **DFS**, **Theta\***, and more.

---

## 2️⃣ Repository Structure

```
environment_centered_explanations/
├── envs/                 # Environment generation & perturbations
├── planners/             # Classical planners: A*, Dijkstra, BFS, DFS, Theta*
├── explainers/           # LIME, SHAP, COSE implementations
├── eval/                 # Evaluation metrics & ILP oracle
├── cli/                  # Command-line tools (main entry points)
├── scripts/              # Automation & monitoring utilities
├── tests/                # Unit tests & visualization scripts
├── results/              # Output CSVs, NPZs, and plots (auto-generated)
├── docs/                 # Figures and documentation
└── README.md
```

---

## 3️⃣ Setup

### 3.1. Install dependencies

```bash
git clone https://github.com/<your-username>/environment_centered_explanations.git
cd environment_centered_explanations
pip install -e .
```

If missing packages appear:
```bash
pip install numpy scipy matplotlib pandas tqdm pulp
```

---

## 4️⃣ Quick Sanity Checks

Before doing experiments:

```bash
python3 scripts/self_check.py
```

This verifies:
- Environments can be generated (`envs/`)
- Planners produce valid paths (`planners/`)
- Explainable methods (LIME, SHAP, COSE) work
- Evaluation metrics run correctly
- All CLI tools respond to `--help`

Expected output ends with:
```
[OK] All self-checks passed.
```

---

## 5️⃣ Running Evaluations

### 5.1. Core Evaluation

Runs all planners × explainers on generated environments.

```bash
python -m cli.run_eval   --sizes 30x30 --densities 0.20   --num-envs 10 --planners a_star,dijkstra   --explainers lime,shap,cose,geodesic   --outdir results/csv
```

Output: CSV files with metrics in `results/csv/`.

---

### 5.2. Robustness Evaluation

Tests explanation stability under small geometric perturbations.

```bash
python -m cli.run_robustness   --sizes 30x30 --densities 0.20   --geom-remap   --outdir results/csv
```

`--geom-remap` ensures obstacles are matched by shape, not numeric ID.

---

### 5.3. Cross-Planner Transfer Evaluation

Evaluates how well an explanation for planner A transfers to planner B.

```bash
python -m cli.run_transfer   --sizes 30x30 --densities 0.20   --planners a_star,dijkstra,bfs   --geom-remap   --outdir results/csv
```

---

### 5.4. ILP Minimality (oracle)

Computes exact minimal causal obstacle sets using Mixed Integer Linear Programming (PuLP).

```bash
python -m cli.run_exact_small --sizes 20x20 --densities 0.25
```

---

## 6️⃣ Plotting & Visualization

### 6.1. Aggregated Figures

After evaluations:

```bash
python -m cli.make_figs --outdir results/figs
```

Generates:
- `q1_success_at_k.pdf` → Faithfulness
- `q2_min_set_size_violin.pdf` → Compactness
- `q3_robustness.pdf` → Robustness
- `q4_transfer_heatmap.pdf` → Transferability

### 6.2. Agreement Plots (qualitative)
```bash
python -m cli.plot_agreement_and_qual --outdir results/figs
```

---

### 6.3. Visualizing Explanations

Generate overlays of important obstacles:
```bash
python3 tests/vis_explainers.py
```

Expected outputs:
```
tests/out/topk_overlays_seed10_k8.png
tests/out/cose_before_after_seed11.png
```

---

## 7️⃣ Planner Benchmarking

Benchmark runtime and path quality:

```bash
python3 tests/benchmark_planners.py
python3 tools/plot_planners.py tests/out/planner_benchmark.csv
```

Outputs:
- `planner_benchmark.csv`
- `planner_time_bar.png`
- `planner_length_bar.png`

---

## 8️⃣ Large-Scale Runs on HPC / Multi-Core Machine

### 8.1. Launch parallel jobs
```bash
bash scripts/launch_full_eval.sh
```

### 8.2. Monitor progress
```bash
python3 scripts/monitor_parallel.py joblog.txt 120 --watch 5
```

### 8.3. Stop all jobs
```bash
killall python3
```

---

## 9️⃣ Testing

Run all tests:
```bash
pytest -q
```

Run specific test:
```bash
pytest tests/test_explainers.py -v
```

Or visual tests (with saved images):
```bash
python3 tests/test_planners.py
```

---

## 🔍 10️⃣ Debugging and Finding Mistakes

### 10.1. Debugging locally
- Use `print()` and `pdb.set_trace()` inside suspect functions.
- Common sources of bugs:
  - Mismatched obstacle IDs after perturbations
  - Environments with no feasible path (density too high)
  - Planner connectivity mismatches (4 vs 8)
  - Typo in CLI argument names

### 10.2. Using ChatGPT or similar tools effectively

When something fails, collect:
1. The **exact traceback** (error message)
2. The **command used**
3. The **file and function** involved (e.g., `envs/generator.py:generate_environment`)

Then ask ChatGPT something like:

> “In my project `environment_centered_explanations`, running `python -m cli.run_eval` fails with a ValueError at line 140 in `lime_explainer.py`.  
> Here is the traceback: [paste here].  
> What does this mean and how can I fix it?”

You can also ask ChatGPT to:
- Explain specific functions (e.g., “Explain how `COSEExplainer.explain()` works”)
- Draw module dependencies
- Suggest code optimizations or new metrics

If inconsistencies appear, verify with:
```bash
python3 scripts/self_check.py
```

---

## 11️⃣ Adding New Components

### 11.1. New Planner
1. Create `planners/my_planner.py`
2. Implement:
   ```python
   class MyPlanner:
       def plan(self, grid, start, goal):
           ...
           return {"success": bool, "path": list[(r,c)] or None}
   ```
3. Register in `planners/__init__.py`:
   ```python
   PLANNERS["my_planner"] = MyPlanner
   ```

### 11.2. New Explainer
Follow the `LimeExplainer` pattern:
```python
class MyExplainer:
    def explain(self, env, planner):
        ...
        return {"ranking": [(id, score), ...]}
```
Then add it to `explainers/__init__.py`.

---

## 12️⃣ Common Troubleshooting

| Symptom | Likely Cause | Fix |
|----------|--------------|-----|
| Planner returns success even in blocked env | Start/goal overlap or 4-connectivity | Use `connectivity=8` |
| COSE returns empty set | Missing guide ranking | Pass `guide_ranking` |
| LIME/SHAP too slow | Too many samples/permutations | Reduce their counts |
| Robustness τ negative | ID mismatch after perturbation | Use `--geom-remap` |
| Figure scripts fail | Missing CSVs | Run `cli.run_eval` first |

---

## 13️⃣ Productivity Tips

- Always work in a **virtual environment**.  
- After each successful test:
  ```bash
  git add .
  git commit -m "Ran eval + plots, verified results"
  ```
- Keep `results/` in `.gitignore`.  
- For experiments, use smaller maps (20×20) & low density (≤0.25).  
- For new metrics → extend `eval/metrics.py`.

---

## 14️⃣ Need Help?

If stuck:
- Run `python3 scripts/self_check.py`
- Use smaller maps (`--sizes 20x20`)
- Rerun with `--debug`
- Ask Amar or consult ChatGPT with full traceback

---

**Welcome to the project!**
You now have everything you need to reproduce, debug, and extend our environment-centered explanation framework.

```
Happy explaining 🤖
```
