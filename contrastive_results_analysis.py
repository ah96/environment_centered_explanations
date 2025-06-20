import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load and prepare
input_csv = "contrastive_explanation_results.csv"
output_dir = "contrastive_analysis_plots"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_csv)
df.dropna(inplace=True)

df["Planner"] = df["Planner"].astype(str)
df["Perturbation_Mode"] = df["Perturbation_Mode"].astype(str)

metrics = [
    "Faithfulness_Score", "Stability", "Explanation_Sparsity",
    "Explanation_Asymmetry", "Relative_Path_Diff", "Path_Overlap"
]

def save_fig(fig, name):
    fig.savefig(os.path.join(output_dir, name), bbox_inches='tight')
    plt.close(fig)

# --- Summary statistics ---
df.describe().to_csv(os.path.join(output_dir, "summary_statistics.csv"))
df.groupby("Planner").mean(numeric_only=True).to_csv(os.path.join(output_dir, "grouped_by_planner.csv"))
df.groupby("Perturbation_Mode").mean(numeric_only=True).to_csv(os.path.join(output_dir, "grouped_by_perturbation.csv"))

# --- Distribution plots ---
for metric in metrics:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[metric], kde=True, ax=ax)
    ax.set_title(f"Distribution of {metric}")
    save_fig(fig, f"{metric}_distribution.png")

# --- Boxplots + Violin plots ---
for metric in metrics:
    # By Planner
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="Planner", y=metric, data=df, inner="box", ax=ax)
    ax.set_title(f"{metric} by Planner (Violin + Box)")
    save_fig(fig, f"{metric}_violin_by_planner.png")

    # By Perturbation Mode
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x="Perturbation_Mode", y=metric, data=df, inner="box", ax=ax)
    ax.set_title(f"{metric} by Perturbation Mode (Violin + Box)")
    save_fig(fig, f"{metric}_violin_by_mode.png")

# --- Correlation heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[metrics].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Correlation Heatmap")
save_fig(fig, "correlation_heatmap.png")

# --- Pairplots ---
sns.pairplot(df, vars=metrics, hue="Planner", plot_kws={'alpha': 0.5}).savefig(os.path.join(output_dir, "pairplot_planner.png"))
plt.close()
sns.pairplot(df, vars=metrics, hue="Perturbation_Mode", plot_kws={'alpha': 0.5}).savefig(os.path.join(output_dir, "pairplot_mode.png"))
plt.close()

# --- Rank-based comparison ---
rank_df = df.groupby("Planner")[metrics].mean(numeric_only=True).rank(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(rank_df, annot=True, cmap="YlGnBu", ax=ax)
ax.set_title("Planner Rank Across Metrics")
save_fig(fig, "planner_rank_heatmap.png")

# --- Std Dev plot ---
std_df = df.groupby("Planner")[metrics].std(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 6))
std_df.plot(kind="bar", ax=ax)
ax.set_title("Standard Deviation of Metrics by Planner (Consistency)")
save_fig(fig, "planner_stddev_bar.png")

# --- ANOVA tests ---
anova_results = []
for metric in metrics:
    for mode in df["Perturbation_Mode"].unique():
        subset = df[df["Perturbation_Mode"] == mode]
        groups = [group[metric].values for _, group in subset.groupby("Planner")]
        if len(groups) > 1:
            f_val, p_val = stats.f_oneway(*groups)
            anova_results.append({
                "Metric": metric,
                "Perturbation_Mode": mode,
                "F_statistic": f_val,
                "p_value": p_val
            })
pd.DataFrame(anova_results).to_csv(os.path.join(output_dir, "anova_results.csv"), index=False)

# --- Heatmap of means per Planner+Mode ---
pivot_means = df.pivot_table(index="Planner", columns="Perturbation_Mode", values=metrics, aggfunc="mean")
for metric in metrics:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_means[metric].unstack().to_frame().T, annot=True, cmap="Blues", ax=ax)
    ax.set_title(f"Mean {metric} by Planner and Perturbation Mode")
    save_fig(fig, f"heatmap_{metric}_by_planner_mode.png")

# --- Outlier detection (Tukey fences) ---
outliers = []
for metric in metrics:
    Q1 = df[metric].quantile(0.25)
    Q3 = df[metric].quantile(0.75)
    IQR = Q3 - Q1
    outlier_rows = df[(df[metric] < Q1 - 1.5 * IQR) | (df[metric] > Q3 + 1.5 * IQR)]
    outliers.extend(outlier_rows.index.tolist())
df.loc[sorted(set(outliers))].to_csv(os.path.join(output_dir, "detected_outliers.csv"))

# --- Top-N scoring environments per metric ---
top_n = 10
top_envs = {}
for metric in metrics:
    top_df = df.sort_values(by=metric, ascending=False).head(top_n)
    top_envs[metric] = top_df[["Pair", "Planner", "Perturbation_Mode", metric]]
    top_df.to_csv(os.path.join(output_dir, f"top10_{metric}.csv"), index=False)

print(f"[✓] Full analysis complete. All outputs saved to '{output_dir}'")
