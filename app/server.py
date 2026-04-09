"""
FastAPI server exposing the OpenEnv HTTP interface.

Endpoints:
  GET  /health        → liveness check
  GET  /tasks         → list all tasks
  GET  /graders       → list all graders
  POST /reset         → start new episode
  POST /step          → advance one step
  GET  /state         → current internal state
"""
from __future__ import annotations
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.env import CrowdSafeEnv
from app.models import Action, Observation, StepResult, EnvState
from app.tasks import GRADERS

app = FastAPI(
    title="CrowdSafeEnv",
    description="OpenEnv crowd safety simulation environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton environment instance
_env = CrowdSafeEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_01_gate_routing"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action: Action


@app.get("/health")
def health():
    return {"status": "ok", "env": "crowdsafe-openenv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {"tasks": _env.get_tasks()}


@app.get("/graders")
def list_graders():
    return {
        "graders": [
            {"task_id": task_id, "has_grader": True}
            for task_id in GRADERS.keys()
        ]
    }


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    obs = _env.reset(task_id=req.task_id, seed=req.seed)
    return obs


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    try:
        result = _env.step(req.action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvState)
def state():
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def root():
    return {
        "name": "CrowdSafeEnv",
        "description": "OpenEnv crowd safety simulation",
        "endpoints": ["/health", "/tasks", "/graders", "/reset", "/step", "/state"],
        "docs": "/docs",
    }
