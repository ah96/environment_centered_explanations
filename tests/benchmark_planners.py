import os, sys, time, csv, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.generator import generate_environment
from planners import PLANNERS

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "planner_benchmark.csv")

def path_metrics(path):
    """Compute hops, geometric length, and step-type counts from a (r,c) path list."""
    if not path or len(path) < 2:
        return 0, 0.0, 0, 0
    hops = len(path) - 1
    geom = 0.0
    cardinal = 0
    diag = 0
    for (r0,c0), (r1,c1) in zip(path[:-1], path[1:]):
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        if dr == 0 and dc == 0:
            continue
        if dr == 1 and dc == 1:
            geom += math.sqrt(2.0)
            diag += 1
        elif (dr == 1 and dc == 0) or (dr == 0 and dc == 1):
            geom += 1.0
            cardinal += 1
        else:
            # Non-adjacent step (shouldn't happen), fallback to Euclidean
            geom += math.hypot(dr, dc)
    return hops, geom, cardinal, diag

def run_case(planner_name, env_kind, seed, H, W, density, connectivity=8):
    rng = np.random.default_rng(seed)
    env = generate_environment(H=H, W=W, density=density, ensure_status=env_kind, rng=rng)
    planner_cls = PLANNERS[planner_name]
    planner = planner_cls(connectivity=connectivity)

    t0 = time.perf_counter()
    out = planner.plan(env.grid, env.start, env.goal)
    t1 = time.perf_counter()

    success = bool(out["success"])
    path = out.get("path") if success else None
    hops, geom, cardinal, diag = path_metrics(path)

    return {
        "planner": planner_name,
        "connectivity": connectivity,
        "env_kind": env_kind,
        "seed": seed,
        "H": H, "W": W, "density": density,
        "success": int(success),
        "time_s": t1 - t0,
        "path_hops": hops,
        "geom_length": geom,
        "cardinal_steps": cardinal,
        "diag_steps": diag,
    }

def main():
    rows = []
    # Tweak these cases as needed
    cases = [
        #   env_kind, seed,  H,  W, density
        ("any",      0,     40, 40, 0.18),
        ("any",      1,     50, 50, 0.18),
        ("failure",  2,     40, 40, 0.20),
        ("failure",  3,     50, 50, 0.22),
    ]
    planners = ["bfs", "dfs", "dijkstra", "a_star", "theta_star"]

    for env_kind, seed, H, W, density in cases:
        for name in planners:
            rows.append(run_case(name, env_kind, seed, H, W, density, connectivity=8))

    # Write CSV
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "planner","connectivity","env_kind","seed","H","W","density",
            "success","time_s","path_hops","geom_length","cardinal_steps","diag_steps"
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {CSV_PATH}")
    # Pretty print
    print(f"{'planner':11} {'env':8} {'seed':4} {'succ':4} {'time[s]':>8} {'hops':>5} {'geom':>8}")
    for r in rows:
        print(f"{r['planner']:11} {r['env_kind']:8} {r['seed']:4d} {r['success']:4d} "
              f"{r['time_s']:8.4f} {r['path_hops']:5d} {r['geom_length']:8.2f}")

if __name__ == "__main__":
    main()
