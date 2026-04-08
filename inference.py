#!/usr/bin/env python3
"""
CrowdSafeEnv Baseline Inference Script
=======================================

Uses the OpenAI-compatible client to run an LLM agent against all 3 tasks.
Supports OpenRouter and Groq out of the box via environment variables.

Required environment variables:
  HF_TOKEN      — Your API key (OpenRouter or Groq key)
  API_BASE_URL  — Provider endpoint (see defaults below)
  MODEL_NAME    — Model identifier (see defaults below)

Provider quick-start:
  # OpenRouter (default)
  export HF_TOKEN="sk-or-v1-..."
  export API_BASE_URL="https://openrouter.ai/api/v1"
  export MODEL_NAME="meta-llama/llama-3.1-8b-instruct:free"

  # Groq
  export HF_TOKEN="gsk_..."
  export API_BASE_URL="https://api.groq.com/openai/v1"
  export MODEL_NAME="llama-3.1-8b-instant"

Usage:
  python inference.py

Output:
  Baseline scores for each task printed to stdout.
  Results saved to baseline_results.json.

Runtime: < 20 minutes on 2vCPU / 8GB RAM.
"""
from __future__ import annotations
import os
import sys
import json
import time
import logging
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Env config — required variables
# ---------------------------------------------------------------------------

# Detect provider from API_BASE_URL if set, otherwise default to OpenRouter
_raw_base = os.environ.get("API_BASE_URL", "")

# Smart defaults per provider
if "groq.com" in _raw_base:
    _default_base = "https://api.groq.com/openai/v1"
    _default_model = "llama-3.1-8b-instant"
    _provider = "Groq"
elif "openrouter.ai" in _raw_base:
    _default_base = "https://openrouter.ai/api/v1"
    _default_model = "meta-llama/llama-3.1-8b-instruct:free"
    _provider = "OpenRouter"
else:
    # Default to OpenRouter when nothing set
    _default_base = "https://openrouter.ai/api/v1"
    _default_model = "meta-llama/llama-3.1-8b-instruct:free"
    _provider = "OpenRouter"

API_BASE_URL = os.environ.get("API_BASE_URL", _default_base)
MODEL_NAME   = os.environ.get("MODEL_NAME",   _default_model)
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set.", file=sys.stderr)
    print("  OpenRouter: export HF_TOKEN='sk-or-v1-...'", file=sys.stderr)
    print("  Groq:       export HF_TOKEN='gsk_...'", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Local env import (runs entirely in-process — no HTTP needed for baseline)
# ---------------------------------------------------------------------------
from app.env import CrowdSafeEnv
from app.models import Action
from app.tasks import TASK_METADATA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# OpenAI-compatible client (works for both OpenRouter and Groq)
# ---------------------------------------------------------------------------

_extra_headers = {}
if "openrouter.ai" in API_BASE_URL:
    # OpenRouter recommends these headers for tracking
    _extra_headers = {
        "HTTP-Referer": "https://huggingface.co/spaces/crowdsafe-openenv",
        "X-Title": "CrowdSafeEnv",
    }

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
    default_headers=_extra_headers,
)


SYSTEM_PROMPT = """You are an expert crowd safety coordinator at a mass-gathering event.
You receive real-time sensor data and must issue commands to prevent crowd crush and stampedes.

VENUE LAYOUT:
- Grid: 12 rows × 16 columns (row 0 = stage area, row 11 = entrance)
- Gates: gate_A(x=2,y=11), gate_B(x=6,y=11), gate_C(x=10,y=11), gate_D(x=14,y=11)
  Side gates: gate_E(x=0,y=6), gate_F(x=15,y=6)
- Barriers: barrier_1 (row 4, cols 3-6), barrier_2 (row 4, cols 9-13)
- Emergency exits: exit_L(x=1,y=0), exit_R(x=14,y=0), exit_M(x=8,y=0)
- Marshals: marshal_1 through marshal_6 (deploy by name to x,y position)

DENSITY THRESHOLDS:
- Safe: < 4.0 people/m²
- Danger: 4.0-6.0 people/m²
- Crush: >= 6.0 people/m²  ← CRITICAL, avoid at all costs
- Stampede: >= 9.0 people/m² → episode ends

STRATEGY GUIDELINES:
- Task 1 (Gate Routing): Open gates early to distribute crowd. Watch bottom rows.
- Task 2 (Surge Response): At step 20 a surge hits stage-left (rows 0-3, cols 0-5).
  Deploy marshals immediately, open exit_L or exit_M, lower barrier_1.
- Task 3 (Cascade): Multiple simultaneous incidents. Use PA broadcast to calm crowd.
  Deploy marshals to highest-density zones first. Open emergency exits proactively.

You must respond with ONLY a valid JSON object matching this schema:
{
  "gate_operations": {"gate_A": true/false, ...},
  "marshal_deployments": [["marshal_1", x, y], ...],
  "pa_broadcast": "optional message or null",
  "barrier_changes": {"barrier_1": true/false, ...},
  "emergency_exit_opens": ["exit_L", ...]
}
No explanation. No markdown. Just the JSON object."""


def obs_to_prompt(obs: dict, step: int, task_id: str) -> str:
    """Convert observation dict to a concise LLM prompt."""
    risk = obs.get("risk_scores", {})
    high_risk = {k: round(v, 2) for k, v in risk.items() if v > 0.5}
    incidents = obs.get("active_incidents", [])
    gate_states = obs.get("gate_states", {})
    open_gates = [g for g, s in gate_states.items() if s]

    # Find highest-density zones for the prompt
    grid = obs.get("density_grid", [])
    max_d = 0.0
    max_loc = (0, 0)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val > max_d:
                max_d = val
                max_loc = (r, c)

    return f"""STEP {step} | Task: {task_id} | Time remaining: {obs.get('time_remaining', '?')}

MAX DENSITY: {round(max_d, 2)} p/m² at cell ({max_loc[0]}, {max_loc[1]})
HIGH-RISK ZONES: {json.dumps(high_risk) if high_risk else 'none'}
ACTIVE INCIDENTS: {incidents if incidents else 'none'}
OPEN GATES: {open_gates if open_gates else 'none'}
PA BROADCASTS REMAINING: {obs.get('pa_broadcasts_remaining', 0)}
MARSHAL POSITIONS: {obs.get('marshal_positions', [])}

Issue your crowd management commands now:"""


def call_llm(prompt: str, max_retries: int = 3) -> Optional[dict]:
    """Call LLM and parse JSON action. Returns None on failure."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                return None
        except Exception as e:
            log.warning(f"LLM call failed attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1 if "groq.com" in API_BASE_URL else 2 ** attempt)
    return None


def make_action(action_dict: Optional[dict]) -> Action:
    """Convert raw dict to validated Action model. Falls back to no-op."""
    if action_dict is None:
        return Action()
    try:
        return Action(**{k: v for k, v in action_dict.items() if v is not None})
    except Exception:
        return Action()


def run_task(env: CrowdSafeEnv, task_id: str, seed: int = 42) -> dict:
    """Run one full episode of a task. Returns result dict."""
    log.info(f"Starting task: {task_id}")

    # Find max steps for this task
    max_steps = 100
    for t in TASK_METADATA:
        if t["id"] == task_id:
            max_steps = t["max_steps"]
            break

    obs = env.reset(task_id=task_id, seed=seed)
    obs_dict = obs.model_dump()

    total_reward = 0.0
    step_count = 0
    task_score = 0.0
    done = False
    info = {}

    start_time = time.time()

    for step in range(1, max_steps + 1):
        if done:
            break

        prompt = obs_to_prompt(obs_dict, step, task_id)
        action_dict = call_llm(prompt)
        action = make_action(action_dict)

        result = env.step(action)
        obs_dict = result.observation.model_dump()
        total_reward += result.reward
        done = result.done
        info = result.info
        step_count = step

        if done:
            task_score = info.get("task_score", 0.0)
            log.info(f"  Episode done at step {step}. Score: {task_score:.4f}")
            break

        # Throttle to avoid rate limits (light sleep)
        # Groq: minimal sleep; OpenRouter free tier: slightly more
        time.sleep(0.05 if "groq.com" in API_BASE_URL else 0.3)

    # If episode didn't end naturally, grade it now
    if not done:
        task_score = env.grade_episode()
        log.info(f"  Reached max steps. Score: {task_score:.4f}")

    elapsed = time.time() - start_time

    return {
        "task_id": task_id,
        "steps": step_count,
        "total_reward": round(total_reward, 4),
        "task_score": round(task_score, 4),
        "stampede": info.get("episode_stampede", False),
        "elapsed_seconds": round(elapsed, 1),
    }


def main():
    print("=" * 60)
    print("CrowdSafeEnv — Baseline Inference")
    print(f"Provider: {_provider}")
    print(f"Model:    {MODEL_NAME}")
    print(f"API:      {API_BASE_URL}")
    print("=" * 60)

    env = CrowdSafeEnv()
    results = []
    overall_start = time.time()

    tasks = [
        "task_01_gate_routing",
        "task_02_surge_response",
        "task_03_cascade_prevention",
    ]

    for task_id in tasks:
        result = run_task(env, task_id, seed=42)
        results.append(result)
        print(f"\n[{task_id}]")
        print(f"  Steps:        {result['steps']}")
        print(f"  Total reward: {result['total_reward']}")
        print(f"  Task score:   {result['task_score']} / 1.0")
        print(f"  Stampede:     {result['stampede']}")
        print(f"  Time:         {result['elapsed_seconds']}s")

        # Check time budget
        elapsed_total = time.time() - overall_start
        if elapsed_total > 1100:  # ~18 min — leave buffer for remaining tasks
            log.warning("Approaching 20-minute budget. Stopping early.")
            break

    # Summary
    print("\n" + "=" * 60)
    print("BASELINE SCORES SUMMARY")
    print("=" * 60)
    total_score = 0.0
    for r in results:
        score = r["task_score"]
        total_score += score
        status = "PASS" if 0.0 <= score <= 1.0 else "INVALID"
        print(f"  {r['task_id']:<35} {score:.4f}  [{status}]")

    avg = total_score / len(results) if results else 0.0
    print(f"\n  Average score: {avg:.4f}")
    print(f"  Total time:    {round(time.time() - overall_start, 1)}s")

    # Validate all scores in range
    all_valid = all(0.0 <= r["task_score"] <= 1.0 for r in results)
    print(f"\n  All scores in [0.0, 1.0]: {'YES ✓' if all_valid else 'NO ✗'}")

    # Save results
    output = {
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "results": results,
        "average_score": round(avg, 4),
        "all_valid": all_valid,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to baseline_results.json")

    # Exit non-zero if any score out of range
    if not all_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
