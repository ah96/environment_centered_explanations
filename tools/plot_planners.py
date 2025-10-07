import sys, os
import pandas as pd
import matplotlib.pyplot as plt

def main(p):
    df = pd.read_csv(p)
    # Aggregate only successful runs for path metrics
    ok = df[df["success"] == 1]

    # Time by planner
    t = df.groupby("planner")["time_s"].agg(["mean","std"]).reset_index()
    plt.figure(figsize=(7,4))
    plt.bar(t["planner"], t["mean"], yerr=t["std"])
    plt.title("Average runtime by planner")
    plt.xlabel("Planner")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=15)
    out1 = os.path.join(os.path.dirname(p), "planner_time_bar.png")
    plt.tight_layout(); plt.savefig(out1, bbox_inches="tight"); plt.close()
    print("Saved:", out1)

    # Geometric length by planner (successful only)
    g = ok.groupby("planner")["geom_length"].agg(["mean","std"]).reset_index()
    plt.figure(figsize=(7,4))
    plt.bar(g["planner"], g["mean"], yerr=g["std"])
    plt.title("Average geometric path length (successful runs)")
    plt.xlabel("Planner")
    plt.ylabel("Length")
    plt.xticks(rotation=15)
    out2 = os.path.join(os.path.dirname(p), "planner_length_bar.png")
    plt.tight_layout(); plt.savefig(out2, bbox_inches="tight"); plt.close()
    print("Saved:", out2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_planners.py <path/to/planner_benchmark.csv>")
        raise SystemExit(1)
    main(sys.argv[1])
