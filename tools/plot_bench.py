import sys, os
import pandas as pd
import matplotlib.pyplot as plt

def main(p):
    df = pd.read_csv(p)
    print(df)

    agg = df.groupby("method").agg(
        mean_time_s=("wall_time_s","mean"),
        std_time_s=("wall_time_s","std"),
        mean_calls=("planner_calls","mean"),
        std_calls=("planner_calls","std"),
    ).reset_index()
    print("\nAggregate:\n", agg)

    # Plot 1: average wall time by method
    plt.figure(figsize=(7,4))
    plt.bar(agg["method"], agg["mean_time_s"], yerr=agg["std_time_s"])
    plt.title("Average wall time by method")
    plt.xlabel("Method")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out1 = os.path.join(os.path.dirname(p), "benchmark_time_bar.png")
    plt.savefig(out1, bbox_inches="tight")
    print("Saved:", out1)

    # Plot 2: average planner calls by method
    plt.figure(figsize=(7,4))
    plt.bar(agg["method"], agg["mean_calls"], yerr=agg["std_calls"])
    plt.title("Average planner calls by method")
    plt.xlabel("Method")
    plt.ylabel("Planner calls")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(p), "benchmark_calls_bar.png")
    plt.savefig(out2, bbox_inches="tight")
    print("Saved:", out2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_bench.py <path/to/benchmark_results.csv>")
        sys.exit(1)
    main(sys.argv[1])
