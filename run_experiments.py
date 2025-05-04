import argparse
import os
from batch_experiment import BatchExperimentRunner

def main():
    parser = argparse.ArgumentParser(description="Run path planning experiments")
    parser.add_argument("--num_envs", type=int, default=10,
                        help="Number of environments to generate")
    parser.add_argument("--feasible", action="store_true",
                        help="Generate feasible environments (default is infeasible)")
    parser.add_argument("--grid_size", type=int, default=10,
                        help="Size of the grid")
    parser.add_argument("--num_obstacles", type=int, default=8,
                        help="Number of obstacles")
    parser.add_argument("--planner", type=str, default="A*",
                        help="Path planning algorithm to use")
    parser.add_argument("--explainer", type=str, default="LIME",
                        help="Explanation method to use")
    parser.add_argument("--output", type=str, default="results.csv",
                        help="Output CSV file for results")
    parser.add_argument("--env_dir", type=str, default=None,
                        help="Directory with existing environments (skips generation)")
    parser.add_argument("--max_attempts_per_env", type=int, default=100,
                   help="Maximum attempts to generate a single environment")
    parser.add_argument("--max_total_attempts", type=int, default=10000,
                   help="Maximum total attempts for all environments combined")
    
    args = parser.parse_args()
    
    runner = BatchExperimentRunner()
    
    # Either use existing environments or generate new ones
    if args.env_dir:
        env_paths = [os.path.join(args.env_dir, f) for f in os.listdir(args.env_dir) 
                    if f.endswith('.json')]
        print(f"Using {len(env_paths)} existing environments from {args.env_dir}")
    else:
        print(f"Generating {args.num_envs} {'feasible' if args.feasible else 'infeasible'} environments...")
        os.makedirs("environments", exist_ok=True)
        env_paths = runner.generate_environments(
            args.num_envs, 
            feasible=args.feasible,
            grid_size=args.grid_size,
            num_obstacles=args.num_obstacles,
            max_attempts_per_env=args.max_attempts_per_env,
            max_total_attempts=args.max_total_attempts,
            infeasibility_mode="block_path" if not args.feasible else None
        )
        print(f"Generated {len(env_paths)} environments")
    
    # Run experiments
    print(f"Running experiments with planner={args.planner}, explainer={args.explainer}")
    runner.run_experiment(env_paths, args.planner, args.explainer, args.output)
    print(f"Experiments complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()

# Example usage:
# Generate 20 infeasible environments and run experiments with A* and LIME
# python run_experiments.py --num_envs 20 --planner "A*" --explainer "LIME" --max_attempts_per_env 200 --max_total_attempts 20000
# python run_experiments.py --num_envs 20 --planner "A*" --explainer "LIME" --output "infeasible_results.csv"

# # Generate 20 feasible environments
# python run_experiments.py --num_envs 20 --feasible --planner "Dijkstra" --explainer "SHAP" --output "feasible_results.csv"

# # Use existing environments from a directory
# python run_experiments.py --env_dir "environments" --planner "Theta*" --explainer "Counterfactual" --output "existing_env_results.csv"