# Environment-Centered Causal Explanations of Robot Navigation Failures

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Paper](https://img.shields.io/badge/arXiv-Preprint-red.svg)]()

This repository provides the official implementation of our framework for generating **environment-centered causal explanations** of robot path planning failures.  
Instead of exposing internal planner structures, the framework perturbs obstacle configurations in grid-based environments to identify:
1. **Causal obstacles** that make a problem infeasible, and  
2. **Minimal modifications** that restore feasibility.  

We adapt and evaluate state-of-the-art model-agnostic explanation methods at the obstacle level:
- **LIME** – robust attributions under perturbations  
- **SHAP** – faithful obstacle importance rankings  
- **COSE** – a novel greedy algorithm for computing minimal critical obstacle sets (compact counterfactual explanations)  

Our large-scale experiments span six classical planners, multiple obstacle densities, and different perturbation affordances.

---

## 🚀 Features

- Planner-agnostic environment-centered explanation framework  
- Adaptations of **LIME** and **SHAP** for obstacle-level attributions  
- **COSE**: novel algorithm for compact counterfactual explanations  
- Automated evaluation pipeline with metrics for:
  - Faithfulness  
  - Robustness  
  - Compactness  
  - Generalization across planners  
- Ready-to-use scripts for generating environments, running planners, and producing explanations  
- Reproducibility-friendly: runs on CPU only, no external dependencies beyond Python scientific stack  

---

## 📂 Repository Structure

```
environment_centered_explanations/
├── cli/                   # Command-line scripts
│   ├── run_eval.py        # Run large-scale evaluations
│   ├── make_figs.py       # Generate analysis figures
│   ├── plot_agreement_and_qual.py
│   └── ...
├── explainers/            # Implementation of LIME, SHAP, and COSE
├── planners/              # Classical path planners (A*, Dijkstra, etc.)
├── envs/                  # Environment generators and perturbation tools
├── results/               # Output CSVs, NPZs, and plots (generated)
├── tests/                 # Unit tests
└── README.md
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/environment_centered_explanations.git
cd environment_centered_explanations
pip install -r requirements.txt
```

We recommend using a Python virtual environment (`venv` or `conda`).

---

## 🧑‍💻 Usage

### 1. Generate Environments
```bash
python -m cli.generate_envs     --outdir results/envs     --sizes 20 30 40     --densities 0.2 0.3
```

### 2. Run Evaluations
```bash
python -m cli.run_eval     --env-glob "results/envs/E_*.npz"     --planners a_star dijkstra bfs dfs     --explainers lime shap cose     --outdir results/csv
```

### 3. Produce Figures
```bash
python -m cli.make_figs     --eval-glob "results/csv/eval_*.csv"     --outdir results/figs
```

Figures will be saved as `.pdf` and `.png` in `results/figs/`.

---

## 📊 Example Results

- **COSE** yields the most compact explanations  
- **LIME** provides the most robust attributions under perturbations  
- **SHAP** offers the most faithful obstacle rankings  
- Explanations generalize across planners, demonstrating planner-agnosticism  

<p align="center">
  <img src="docs/example_explanations.png" alt="Example Explanations" width="600">
</p>

---

## 📝 Citation

If you use this repository in your research, please cite:

```bibtex
@article{halilovic2025environment,
  title   = {Planner-Agnostic Environment-Centered Causal Explanations of Robot Navigation Failures},
  author  = {Halilovic, Amar and ...},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2025}
}
```

---

## 📫 Contact

For questions, suggestions, or collaborations, please contact:  
**Amar Halilovic** – [your email / website link]  

---

## 📜 License

This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.
