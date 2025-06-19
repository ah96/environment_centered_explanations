import os
import json
import matplotlib.pyplot as plt

def draw_environment(ax, grid_size, obstacles, path_A, path_B, start, goal):
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Draw obstacles
    for shape in obstacles.values():
        for (x, y) in shape:
            ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1,
                                       color='gray', edgecolor='black'))

    # Draw factual path A
    if path_A:
        xa, ya = zip(*path_A)
        ax.plot(ya, xa, color='orange', linewidth=2, label='Path A (original)')

    # Draw contrastive path B
    if path_B:
        xb, yb = zip(*path_B)
        ax.plot(yb, xb, color='purple', linewidth=2, label='Path B (contrastive)')

    # Draw start and goal
    if start:
        ax.plot(start[1], start[0], 'bo', markersize=10, label='Start')
    if goal:
        ax.plot(goal[1], goal[0], 'g*', markersize=12, label='Goal')

    ax.legend(loc='upper right')

def visualize_and_save(pair_json_path, output_img_path):
    with open(pair_json_path, 'r') as f:
        data = json.load(f)

    original = data['original_environment']
    contrastive = data['contrastive_environment']

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_environment(
        ax=ax,
        grid_size=original['grid_size'],
        obstacles=original['obstacle_shapes'],
        path_A=original['path'],
        path_B=contrastive['path'],
        start=original['agent_pos'],
        goal=original['goal_pos']
    )

    ax.set_title("Contrastive Explanation: Path A vs B")
    plt.tight_layout()
    plt.savefig(output_img_path)
    plt.close()
    print(f"[✓] Saved image to {output_img_path}")

def visualize_all_pairs(folder="contrastive_envs"):
    subfolders = sorted([f for f in os.listdir(folder) if f.startswith("pair_")])
    for subfolder in subfolders:
        pair_path = os.path.join(folder, subfolder, "pair.json")
        if not os.path.exists(pair_path):
            print(f"[!] Skipping {subfolder}: pair.json not found")
            continue
        output_img = os.path.join(folder, subfolder, "visualization.png")
        visualize_and_save(pair_path, output_img)

if __name__ == "__main__":
    visualize_all_pairs("contrastive_envs")
