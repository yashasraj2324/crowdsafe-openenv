"""Typed Pydantic models for OpenEnv spec compliance."""
from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Any
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Full sensor snapshot returned by reset() and step()."""
    timestep: int = Field(..., description="Current simulation step")
    density_grid: List[List[float]] = Field(..., description="2D grid people/m²")
    velocity_field: List[List[List[float]]] = Field(..., description="2D grid [vx,vy] m/s")
    gate_states: Dict[str, bool] = Field(..., description="gate_id → open/closed")
    marshal_positions: List[List[int]] = Field(..., description="[[x,y], ...] marshal locs")
    risk_scores: Dict[str, float] = Field(..., description="zone_id → 0.0-1.0")
    active_incidents: List[str] = Field(default_factory=list)
    time_remaining: int = Field(..., description="Seconds left in episode")
    pa_broadcasts_remaining: int = Field(..., description="PA credits remaining")
    task_id: str = Field(..., description="Current task identifier")
    venue_width: int = Field(..., description="Grid columns")
    venue_height: int = Field(..., description="Grid rows")


class Action(BaseModel):
    """Commands the agent issues each timestep."""
    gate_operations: Dict[str, bool] = Field(
        default_factory=dict,
        description="gate_id → True=open, False=close"
    )
    marshal_deployments: List[List[Any]] = Field(
        default_factory=list,
        description="[[marshal_id, x, y], ...]"
    )
    pa_broadcast: Optional[str] = Field(
        default=None,
        description="PA message text; None = no broadcast"
    )
    barrier_changes: Dict[str, bool] = Field(
        default_factory=dict,
        description="barrier_id → True=raise, False=lower"
    )
    emergency_exit_opens: List[str] = Field(
        default_factory=list,
        description="List of emergency exit IDs to open"
    )


class Reward(BaseModel):
    """Detailed reward breakdown for a single step."""
    total: float = Field(..., description="Net reward for this step")
    density_penalty: float = Field(default=0.0)
    flow_reward: float = Field(default=0.0)
    crush_penalty: float = Field(default=0.0)
    stampede_penalty: float = Field(default=0.0)
    efficiency_bonus: float = Field(default=0.0)
    resource_penalty: float = Field(default=0.0)


class StepResult(BaseModel):
    """Full return value of step()."""
    observation: Observation
    reward: float
    reward_detail: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskInfo(BaseModel):
    """Metadata about a single task."""
    id: str
    name: str
    difficulty: str
    description: str
    max_steps: int


class EnvState(BaseModel):
    """Full internal state snapshot from state()."""
    task_id: str
    timestep: int
    done: bool
    total_reward: float
    episode_crush_events: int
    episode_stampede: bool
    gate_states: Dict[str, bool]
    barrier_states: Dict[str, bool]
    marshal_positions: List[List[int]]
    active_incidents: List[str]
    density_grid: List[List[float]]
    risk_scores: Dict[str, float]
