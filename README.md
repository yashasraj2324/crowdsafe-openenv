---
title: CrowdSafeEnv
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - crowd-safety
  - simulation
  - reinforcement-learning
  - real-world
license: mit
---
# CrowdSafeEnv 🚨

**An OpenEnv crowd safety simulation for mass-gathering management.**

The agent plays the role of a real-time crowd safety coordinator at a large outdoor event. It receives sensor data (density grids, velocity fields, risk scores) and must issue commands to prevent crowd crush and stampedes — a real, life-critical decision-making task.

> Inspired by real-world tragedies: Seoul Halloween 2022, Astroworld 2021, Kumbh Mela 2013. The decisions here mirror actual crowd safety protocols.

---

## Environment Description

A 12×16 grid simulates a stadium venue. Each cell tracks:

- **Density** (people/m²): safe < 4.0, danger 4.0–6.0, crush ≥ 6.0, stampede ≥ 9.0
- **Velocity field**: crowd movement direction and speed

The agent manages:

| Resource        | Count     | Description                                 |
| --------------- | --------- | ------------------------------------------- |
| Gates           | 6         | Entry/exit flow control                     |
| Barriers        | 2         | Internal crowd channeling                   |
| Marshals        | 6         | Deployable personnel reducing local density |
| Emergency exits | 3         | High-capacity emergency egress              |
| PA broadcasts   | 5/episode | Calming announcements reducing density      |

---

## Observation Space

```python
class Observation(BaseModel):
    timestep: int                          # Current step
    density_grid: List[List[float]]        # 12×16 grid, people/m²
    velocity_field: List[List[List[float]]]# 12×16 grid, [vx, vy] m/s
    gate_states: Dict[str, bool]           # gate_id → open/closed
    marshal_positions: List[List[int]]     # [[x,y], ...] deployed marshals
    risk_scores: Dict[str, float]          # zone_id → 0.0–1.0
    active_incidents: List[str]            # current active incidents
    time_remaining: int                    # steps remaining
    pa_broadcasts_remaining: int           # PA credits left
    task_id: str                           # current task
```

## Action Space

```python
class Action(BaseModel):
    gate_operations: Dict[str, bool]       # gate_id → True=open, False=close
    marshal_deployments: List             # [[marshal_id, x, y], ...]
    pa_broadcast: Optional[str]           # PA announcement (costs 1 credit)
    barrier_changes: Dict[str, bool]      # barrier_id → True=raise, False=lower
    emergency_exit_opens: List[str]       # exit IDs to open
```

---

## Tasks

### Task 1 — Gate Routing (Easy) `task_01_gate_routing`

**Scenario:** Festival gates are opening. Arriving crowds form at the bottom of the grid. Open the correct gates before density spikes.

**Objective:** Keep all zones below 4.0 p/m² for 100 steps.

**Grader (0.0–1.0):**

- 60% — fraction of steps with no zone exceeding safe density
- 20% — max density stayed below 5.0 p/m²
- 20% — at least 3 gates opened within first 10 steps

**Expected baseline score:** ~0.55

---

### Task 2 — Surge Response (Medium) `task_02_surge_response`

**Scenario:** A speaker malfunction at step 20 triggers a crowd surge from stage-left. Dense crowd appears in rows 0–3, cols 0–5.

**Objective:** Contain the surge within 50 steps. Deploy marshals, open emergency exits.

**Grader (0.0–1.0):**

- 40% — surge contained (max density < 5.0) within 50 steps of incident
- 25% — ≥2 marshals deployed within 30 steps of surge
- 20% — emergency exit exit_L or exit_M opened
- 15% — no crush zones in final 30 steps

**Expected baseline score:** ~0.35

---

### Task 3 — Cascade Prevention (Hard) `task_03_cascade_prevention`

**Scenario:** Three simultaneous incidents at step 10–15: exit bottleneck (right side), stage surge (left), and medical panic (center). 200 steps, limited resources.

**Objective:** Prevent any zone reaching crush density (6.0 p/m²) across all three incidents.

**Grader (0.0–1.0):**

- 35% — no zone ever reaches crush density (−0.15 per zone crushed)
- 25% — early decisive action (gate + marshal ops) within first 25 steps
- 20% — ≥4 marshals deployed across zones
- 10% — PA system used ≥1 time
- 10% — no stampede and max density < 7.0

**Expected baseline score:** ~0.20

---

## Reward Function

Dense per-step reward (fires every timestep — not just terminal):

| Component         | Value                | Condition                                  |
| ----------------- | -------------------- | ------------------------------------------ |
| Density penalty   | −0.5 × crush_zones | Per step, per zone ≥ 6.0 p/m²            |
| Flow reward       | +0.10 × safe_ratio  | Fraction of cells below 4.0 p/m²          |
| Improvement bonus | +0.05                | If max density improved and no crush zones |
| Crush penalty     | −1.0                | Any zone at crush density                  |
| Stampede penalty  | −5.0                | Episode-ending event                       |
| Efficiency bonus  | +0.15                | Max density < 4.0 p/m²                    |
| Resource penalty  | −0.05               | Unnecessary marshal moves                  |

Range: `[−10.0, 1.0]` per step.

---

## Setup & Usage

### Local (Python)

```bash
git clone https://huggingface.co/spaces/your-org/crowdsafe-openenv
cd crowdsafe-openenv
pip install -r requirements.txt

# Start the API server
uvicorn app.server:app --host 0.0.0.0 --port 7860

# Run baseline inference — OpenRouter (free tier)
export HF_TOKEN="sk-or-v1-..."
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="meta-llama/llama-3.1-8b-instruct:free"
python inference.py

# Run baseline inference — Groq (fastest)
export HF_TOKEN="gsk_..."
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
python inference.py
```

### Docker

```bash
docker build -t crowdsafe-openenv .

# With OpenRouter
docker run -p 7860:7860 \
  -e API_BASE_URL="https://openrouter.ai/api/v1" \
  -e MODEL_NAME="meta-llama/llama-3.1-8b-instruct:free" \
  -e HF_TOKEN="sk-or-v1-..." \
  crowdsafe-openenv

# With Groq
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.groq.com/openai/v1" \
  -e MODEL_NAME="llama-3.1-8b-instant" \
  -e HF_TOKEN="gsk_..." \
  crowdsafe-openenv
```

### API Usage

```python
import httpx

BASE = "http://localhost:7860"

# Reset environment
obs = httpx.post(f"{BASE}/reset", json={"task_id": "task_01_gate_routing", "seed": 42}).json()

# Step
action = {
    "gate_operations": {"gate_A": True, "gate_B": True, "gate_C": True},
    "marshal_deployments": [["marshal_1", 6, 11]],
    "pa_broadcast": None,
    "barrier_changes": {},
    "emergency_exit_opens": []
}
result = httpx.post(f"{BASE}/step", json={"action": action}).json()
print(result["reward"], result["done"])

# State
state = httpx.get(f"{BASE}/state").json()
```

---

## Environment Variables

| Variable         | Description                                         |
| ---------------- | --------------------------------------------------- |
| `HF_TOKEN`     | **Required.** Your OpenRouter or Groq API key |
| `API_BASE_URL` | Provider endpoint (see below)                       |
| `MODEL_NAME`   | Model identifier (see below)                        |

### OpenRouter (default — free tier available)

```bash
export HF_TOKEN="sk-or-v1-..."                          # openrouter.ai/keys
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="meta-llama/llama-3.1-8b-instruct:free"  # free tier
# or paid: "anthropic/claude-3-haiku", "openai/gpt-4o-mini"
```

### Groq (fastest inference, free tier)

```bash
export HF_TOKEN="gsk_..."                               # console.groq.com/keys
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
# or: "llama-3.3-70b-versatile", "mixtral-8x7b-32768"
```

---

## Baseline Scores

Run with `gpt-4o-mini` (seed=42):

| Task               | Difficulty | Score           |
| ------------------ | ---------- | --------------- |
| Gate Routing       | Easy       | ~0.55           |
| Surge Response     | Medium     | ~0.35           |
| Cascade Prevention | Hard       | ~0.20           |
| **Average**  |            | **~0.37** |

---

## File Structure

```
crowdsafe-openenv/
├── inference.py          # Baseline inference script (root — required)
├── openenv.yaml          # OpenEnv spec metadata
├── Dockerfile            # Container build
├── requirements.txt      # Python dependencies
├── README.md
└── app/
    ├── env.py            # CrowdSafeEnv class (reset/step/state)
    ├── models.py         # Pydantic: Observation, Action, Reward
    ├── simulation.py     # Crowd physics engine
    ├── tasks.py          # Task definitions + graders
    ├── rewards.py        # Dense reward function
    └── server.py         # FastAPI HTTP server
```
