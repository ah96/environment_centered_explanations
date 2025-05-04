const canvas = document.getElementById("grid-canvas");
const ctx = canvas.getContext("2d");
const GRID_SIZE = 10;
let cellSize = canvas.width / GRID_SIZE;
let environment = null;
let heatmap = [];

async function fetchEnvironments() {
  const res = await fetch("https://your-backend.onrender.com/environments");
  const envs = await res.json();
  const select = document.getElementById("env-select");
  envs.forEach((env, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `Environment ${idx + 1}`;
    select.appendChild(opt);
  });
}

async function loadEnvironment() {
  const idx = document.getElementById("env-select").value;
  const res = await fetch(`https://your-backend.onrender.com/environment/${idx}`);
  environment = await res.json();
  drawGrid();
}

async function requestExplanation(type) {
  const res = await fetch(`https://your-backend.onrender.com/explain/${type}/${document.getElementById("env-select").value}`);
  heatmap = await res.json();
  drawGrid();
}

function drawGrid() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      ctx.strokeStyle = "#aaa";
      ctx.strokeRect(col * cellSize, row * cellSize, cellSize, cellSize);
    }
  }

  // Draw obstacles
  for (const [id, shape] of Object.entries(environment.obstacle_shapes)) {
    ctx.fillStyle = "#000";
    shape.forEach(([r, c]) => {
      ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
    });
  }

  // Draw start
  if (environment.agent_pos) {
    ctx.fillStyle = "green";
    const [r, c] = environment.agent_pos;
    ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
  }

  // Draw goal
  if (environment.goal_pos) {
    ctx.fillStyle = "red";
    const [r, c] = environment.goal_pos;
    ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
  }

  // Heatmap
  if (heatmap.length) {
    ctx.fillStyle = "rgba(255,255,0,0.6)";
    heatmap.forEach((val, i) => {
      const shape = environment.obstacle_shapes[i];
      if (val > 0.01) {
        shape.forEach(([r, c]) => {
          ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        });
      }
    });
  }
}

document.getElementById("feedback-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  const data = Object.fromEntries(form.entries());
  data.env = document.getElementById("env-select").value;
  const res = await fetch("https://your-backend.onrender.com/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  alert("Thanks for your feedback!");
  e.target.reset();
});

fetchEnvironments();
