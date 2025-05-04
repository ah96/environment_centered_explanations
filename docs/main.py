from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os, json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Assume environments are pre-generated
ENV_DIR = "../environments"

@app.get("/environments")
def list_envs():
    files = sorted(f for f in os.listdir(ENV_DIR) if f.endswith(".json"))
    return files

@app.get("/environment/{idx}")
def get_env(idx: int):
    files = sorted(f for f in os.listdir(ENV_DIR) if f.endswith(".json"))
    with open(os.path.join(ENV_DIR, files[idx])) as f:
        return json.load(f)

@app.get("/explain/{method}/{idx}")
def explain(method: str, idx: int):
    # Load env and run explanation (stubbed here)
    files = sorted(f for f in os.listdir(ENV_DIR) if f.endswith(".json"))
    path = os.path.join(ENV_DIR, files[idx])
    env = json.load(open(path))
    from explanations.lime_explainer import LimeExplainer
    from explanations.shap_explainer import SHAPExplainer
    from path_planning.astar import AStarPlanner
    from gui import GridWorldEnv
    # Build environment
    e = GridWorldEnv(grid_size=env["grid_size"], num_obstacles=env["num_obstacles"])
    e.obstacle_shapes = {int(k): v for k, v in env["obstacle_shapes"].items()}
    e.obstacles = [p for shape in e.obstacle_shapes.values() for p in shape]
    e.agent_pos = env["agent_pos"]
    e.goal_pos = env["goal_pos"]
    planner = AStarPlanner()
    planner.set_environment(e.agent_pos, e.goal_pos, e.grid_size, e.obstacles)
    explainer = LimeExplainer() if method.upper() == "LIME" else SHAPExplainer()
    explainer.set_environment(e, planner)
    explanation = explainer.explain(num_samples=30)
    return list(explanation)

@app.post("/feedback")
async def receive_feedback(req: Request):
    data = await req.json()
    with open("feedback_log.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")
    return {"status": "ok"}
