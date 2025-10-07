# Environment-Centered Causal Explanations of Robot Navigation Failures

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Paper](https://img.shields.io/badge/IEEE-RA--L--2025-red.svg)]()

Official implementation of our **planner-agnostic framework** for generating **environment-centered causal explanations** of robot path-planning failures.

Instead of exposing internal planner states, we perturb obstacle configurations in grid environments to identify:
1. **Causal obstacles** that make a problem infeasible, and  
2. **Minimal modifications** that restore feasibility.  

We adapt and evaluate state-of-the-art explainers:
- **LIME** – local obstacle-level importance via random perturbations  
- **SHAP** – Shapley-value-based causal importance  
- **COSE** – our novel *Counterfactual Obstacle Set Explainer* (minimal critical obstacles)  

Large-scale experiments cover six classical planners, multiple obstacle densities, and different perturbation affordances.

---

## 🚀 Features

- Planner-agnostic, environment-centered explanation pipeline  
- **LIME**, **SHAP**, and **COSE** adaptations for grid obstacles  
- Automated evaluation of:
  - Faithfulness (Success@K, AUC-S@K)  
  - Robustness under geometric perturbations  
  - Transferability across planners  
  - Compactness / minimality (vs ILP oracle)  
- Full CLI suite for reproducible experiments (`cli/`)  
- Shell & Python scripts for batch jobs, monitoring, and CI smoke tests (`scripts/`)  
- CPU-only, lightweight dependencies (NumPy, SciPy, PuLP, Matplotlib)

---

## 📂 Repository Structure

```
environment_centered_explanations/
├── envs/              # Environment generation & perturbations
├── planners/          # Classical planners: A*, Dijkstra, BFS, DFS, Theta*
├── explainers/        # LIME, SHAP, COSE implementations
├── eval/              # Metrics: faithfulness, robustness, transfer, minimality
├── cli/               # Command-line tools for experiments & plots
├── scripts/           # Automation (launch_full_eval.sh, monitor_parallel.py, self_check.py)
├── tests/             # Unit & visualization tests
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/environment_centered_explanations.git
cd environment_centered_explanations
pip install -e .
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `pulp`, `pandas`, `tqdm`.

We recommend running smoke tests first:
```bash
python3 scripts/self_check.py
```

---

## 🧑‍💻 Basic Usage

### 1️⃣ Run Core Evaluation
```bash
python -m cli.run_eval   --sizes 30x30 --densities 0.20   --num-envs 10 --planners a_star,dijkstra   --explainers lime,shap,cose,geodesic   --outdir results/csv
```

### 2️⃣ Robustness & Transfer Studies
```bash
python -m cli.run_robustness --sizes 30x30 --densities 0.20 --geom-remap
python -m cli.run_transfer   --sizes 30x30 --densities 0.20 --geom-remap
```

### 3️⃣ Aggregate and Plot
```bash
python -m cli.make_figs --outdir results/figs
```

### 4️⃣ (Optional) Monitor Parallel Jobs
```bash
bash scripts/launch_full_eval.sh &
python3 scripts/monitor_parallel.py joblog.txt 120 --watch 5
```

---

## 🧪 Self-Check & CI

```bash
python3 scripts/self_check.py
```

Checks that:
- environment generation works  
- planners produce valid paths  
- LIME, SHAP, and COSE run successfully  
- evaluation metrics are consistent  
- all CLI commands respond to `--help`

---

## 📊 Example Qualitative Result

<p align="center">
  <img src="docs/example_explanations.png" alt="Example Explanations" width="600">
</p>

- **COSE** yields minimal causal obstacle sets (compact counterfactuals)  
- **LIME** produces stable obstacle rankings under perturbations  
- **SHAP** offers highly faithful attributions  
- COSE-based repairs restore path feasibility across planners  

---

## 🧩 Scripts Overview

| Script | Description |
|---------|--------------|
| `scripts/launch_full_eval.sh` | Launches large-scale parallel evaluation (GNU Parallel). |
| `scripts/monitor_parallel.py` | Tracks progress and ETA of parallel jobs (`--watch`). |
| `scripts/self_check.py` | Runs end-to-end consistency tests. |
| `scripts/smoke_all.sh` | Minimal smoke test before experiments. |

---

## 📝 Citation

```bibtex
@article{halilovic2025environment,
  title   = {Planner-Agnostic Environment-Centered Causal Explanations of Robot Navigation Failures},
  author  = {Halilovic, Amar and ...},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2025}
}
```

---

## 📜 License

Licensed under the MIT License – see [LICENSE](LICENSE).

---

## 📫 Contact

**Amar Halilovic**  
[website / email placeholder]
