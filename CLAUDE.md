# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server (default port 7860)
uvicorn app.server:app --host 0.0.0.0 --port 7860 --reload

# Run baseline inference with OpenRouter (free tier)
export HF_TOKEN="your_openrouter_key"
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="meta-llama/llama-3.1-8b-instruct:free"
python inference.py

# Run baseline inference with Groq (fastest)
export HF_TOKEN="your_groq_key"
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
python inference.py

# Docker usage
docker build -t crowdsafe-openenv .
docker run -p 7860:7860 -e HF_TOKEN="your_key" -e API_BASE_URL="https://openrouter.ai/api/v1" -e MODEL_NAME="meta-llama/llama-3.1-8b-instruct:free" crowdsafe-openenv
```

### Testing
```bash
# Run unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_env.py -v
```

### API Interaction Examples
```python
import httpx

BASE = "http://localhost:7860"

# Reset environment for a specific task
obs = httpx.post(f"{BASE}/reset", json={
    "task_id": "task_01_gate_routing",
    "seed": 42
}).json()

# Take a step in the environment
action = {
    "gate_operations": {"gate_A": True, "gate_B": True},
    "marshal_deployments": [["marshal_1", 6, 11]],
    "pa_broadcast": "Please move to less crowded areas",
    "barrier_changes": {},
    "emergency_exit_opens": []
}
result = httpx.post(f"{BASE}/step", json={"action": action}).json()

# Get current state
state = httpx.get(f"{BASE}/state").json()
```

## Code Architecture

### Core Components
- **`app/env.py`**: Main `CrowdSafeEnv` class implementing the OpenEnv interface with `reset()`, `step()`, and `state()` methods
- **`app/server.py`**: FastAPI HTTP server exposing the environment via REST endpoints (`/health`, `/tasks`, `/reset`, `/step`, `/state`)
- **`app/simulation.py`**: Crowd physics engine handling density updates, velocity fields, and agent actions
- **`app/models.py`**: Pydantic models for Observation, Action, Reward, StepResult, and EnvState
- **`app/tasks.py`**: Task definitions and grader logic for the three difficulty levels
- **`app/rewards.py`**: Dense reward function computation called each timestep

### Environment Mechanics
- **Observation Space**: 12×16 density grid, velocity field, gate states, marshal positions, risk scores, incidents, time/PA remaining
- **Action Space**: Gate operations (open/close), marshal deployments (ID, x, y), PA broadcasts, barrier changes, emergency exit openings
- **Tasks**: Three progressively challenging scenarios with specific grading criteria:
  1. Gate Routing: Open gates to prevent density buildup
  2. Surge Response: Contain crowd surges using marshals and emergency exits
  3. Cascade Prevention: Manage multiple simultaneous incidents with limited resources

### Key Constants
- Grid size: 12 rows × 16 columns
- PA broadcasts: 5 per episode budget
- Density thresholds: safe < 4.0, danger 4.0-6.0, crush ≥ 6.0, stampede ≥ 9.0 people/m²
- Default max steps: 100 (varies by task)

### External Integrations
- **LLM Providers**: OpenRouter or Groq APIs accessed via `inference.py` using environment variables
- **Docker**: Containerized deployment supported via provided Dockerfile
- **CORS**: Enabled on FastAPI server for cross-origin requests

## Development Workflow
1. Modify environment logic in `app/env.py` or supporting modules
2. Test changes locally with `uvicorn app.server:app --reload`
3. Validate behavior by running inference scripts or manual API calls
4. Ensure tests pass with `pytest tests/`
5. Update task graders in `app/tasks.py` if modifying evaluation criteria