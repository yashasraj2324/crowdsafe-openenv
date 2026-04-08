# Deployment Guide

## HuggingFace Spaces

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Docker**
   - Space name: `crowdsafe-openenv`

2. Push this repo:
```bash
git init
git add .
git commit -m "Initial CrowdSafeEnv submission"
git remote add origin https://huggingface.co/spaces/YOUR_HF_USERNAME/crowdsafe-openenv
git push -u origin main
```

3. Set Space secrets (Settings → Variables and secrets):
   - `API_BASE_URL` = your LLM API endpoint
   - `MODEL_NAME`   = model identifier
   - `HF_TOKEN`     = your API key

4. The Space auto-builds from Dockerfile and serves on port 7860.

## Validation ping (automated checker)

The validator will:
```bash
# 1. GET /health → must return 200
curl https://YOUR_SPACE.hf.space/health

# 2. POST /reset → must return Observation
curl -X POST https://YOUR_SPACE.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_01_gate_routing", "seed": 42}'

# 3. POST /step → must return StepResult  
curl -X POST https://YOUR_SPACE.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"gate_operations": {"gate_A": true}, "marshal_deployments": [], "pa_broadcast": null, "barrier_changes": {}, "emergency_exit_opens": []}}'

# 4. GET /state → must return EnvState
curl https://YOUR_SPACE.hf.space/state
```

## Docker local test

```bash
docker build -t crowdsafe-openenv .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  crowdsafe-openenv

# In another terminal:
python inference.py
```

## openenv validate

```bash
pip install openenv
openenv validate --config openenv.yaml --url http://localhost:7860
```
