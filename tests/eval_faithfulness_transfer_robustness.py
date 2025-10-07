import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.generator import generate_environment
from planners import AStarPlanner, DijkstraPlanner
from explainers.lime_explainer import LimeExplainer
from explainers.shap_explainer import ShapExplainer
from explainers.cose import COSEExplainer
from explainers.baselines import geodesic_line_ranking
from eval.metrics import evaluate_ranking_success_curve, kendall_tau
from eval.transfer import cross_planner_success_at_k, cross_planner_overlap
from eval.robustness import robustness_suite

def main(seed=0):
    rng = np.random.default_rng(seed)
    env = generate_environment(H=40, W=40, density=0.18, ensure_status="failure", rng=rng)
    plannerA = AStarPlanner(connectivity=8)
    plannerB = DijkstraPlanner(connectivity=8)

    lime = LimeExplainer(num_samples=150, flip_prob=0.3, random_state=seed)
    shap = ShapExplainer(permutations=30, random_state=seed)
    cose = COSEExplainer()

    # Run
    lime_out = lime.explain(env, plannerA)
    shap_out = shap.explain(env, plannerA)
    guide = geodesic_line_ranking(env)["ranking"]
    cose_out = cose.explain(env, plannerA, guide_ranking=guide)

    # Faithfulness
    for name, out in [("LIME", lime_out), ("SHAP", shap_out)]:
        r = evaluate_ranking_success_curve(env, plannerA, out["ranking"])
        print(f"{name} AUC-S@K = {r['auc_s_at_k']:.3f}")

    # Transfer (A* → Dijkstra)
    ks = [1,3,5,10]
    trans = cross_planner_success_at_k(env, lime_out["ranking"], plannerB, ks)
    overlap = cross_planner_overlap(lime_out["ranking"], shap_out["ranking"], k=5)
    print("Transfer (A*→Dijkstra):", trans)
    print("LIME vs SHAP top-5 overlap:", overlap)

    # Robustness
    perturbed = robustness_suite(env, seed=seed)
    for name, env_p in perturbed:
        lime_p = LimeExplainer(num_samples=80, flip_prob=0.3, random_state=seed).explain(env_p, plannerA)
        tau = kendall_tau(lime_out["ranking"], lime_p["ranking"])
        print(f"Robustness τ({name}) = {tau:.3f}")

if __name__ == "__main__":
    main()
