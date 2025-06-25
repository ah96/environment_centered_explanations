import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import f_oneway

# Load the results
df = pd.read_csv("contrastive_explanation_results.csv")

# Output folder
output_dir = "contrastive_plots"
os.makedirs(output_dir, exist_ok=True)

# Metrics to analyze
metrics = [
    "Faithfulness_Score",
    "Stability",
    "Explanation_Sparsity",
    "Explanation_Asymmetry",
    "Relative_Path_Diff",
    "Path_Overlap"
]

# Summary statistics per planner + perturbation mode
summary = df.groupby(["Planner", "Perturbation_Mode"])[metrics].agg(["mean", "std"]).reset_index()
summary.columns = ['_'.join(col).strip("_") for col in summary.columns.values]
summary_csv = os.path.join(output_dir, "contrastive_summary_table.csv")
summary_tex = os.path.join(output_dir, "contrastive_summary_table.tex")
summary.to_csv(summary_csv, index=False)

# LaTeX summary table
with open(summary_tex, "w") as f:
    f.write(summary.to_latex(index=False,
                              caption="Metric means and standard deviations grouped by planner and perturbation mode.",
                              label="tab:contrastive-summary",
                              column_format="ll" + "r@{\,±\,}r" * (len(metrics)),
                              multicolumn_format='c',
                              escape=False))

# ANOVA results
anova_results = []
for metric in metrics:
    groups = [group[metric].dropna().values for _, group in df.groupby("Planner")]
    if all(len(g) > 1 for g in groups):
        f_stat, p_val = f_oneway(*groups)
        anova_results.append({
            "Metric": metric,
            "F-statistic": round(f_stat, 3),
            "p-value": round(p_val, 3)
        })

anova_df = pd.DataFrame(anova_results)
anova_csv = os.path.join(output_dir, "anova_results_table.csv")
anova_tex = os.path.join(output_dir, "anova_results_table.tex")
anova_df.to_csv(anova_csv, index=False)

# LaTeX ANOVA table
with open(anova_tex, "w") as f:
    f.write(anova_df.to_latex(index=False,
                               caption="One-way ANOVA test results across planners for each explanation metric.",
                               label="tab:anova-results",
                               float_format="%.3f"))

# Combined violin plots for each metric
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        x="Planner",
        y=metric,
        hue="Perturbation_Mode",
        data=df,
        split=True,
        inner="quartile"
    )
    plt.title(f"{metric.replace('_', ' ')} by Planner and Perturbation Mode")
    plt.xlabel("Planner")
    plt.ylabel(metric.replace("_", " "))
    plt.legend(title="Perturbation Mode", loc="best")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{metric}_violin.png")
    plt.savefig(plot_path)
    plt.close()

print(f"✅ Analysis complete. Outputs saved in: {output_dir}")
