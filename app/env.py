"""
CrowdSafeEnv — main environment class implementing the OpenEnv interface.

Public API:
    reset(task_id)  → Observation
    step(action)    → StepResult
    state()         → EnvState
"""
from __future__ import annotations
from typing import Optional, Dict, Any

from app.models import (
    Observation, Action, Reward, StepResult, EnvState, TaskInfo
)
from app.simulation import CrowdSimulation
from app.rewards import compute_reward
from app.tasks import EpisodeRecord, GRADERS, TASK_METADATA


PA_BUDGET = 5  # broadcasts per episode
MAX_STEPS_DEFAULT = 100


class CrowdSafeEnv:
    """
    OpenEnv-compliant crowd safety simulation environment.

    The agent plays the role of a mass-gathering safety coordinator.
    It receives real-time sensor data and must issue crowd management
    commands to prevent crush events and stampedes.
    """

    def __init__(self):
        self._sim: Optional[CrowdSimulation] = None
        self._task_id: str = "task_01_gate_routing"
        self._timestep: int = 0
        self._max_steps: int = MAX_STEPS_DEFAULT
        self._done: bool = False
        self._total_reward: float = 0.0
        self._pa_remaining: int = PA_BUDGET
        self._prev_max_density: float = 0.0
        self._prev_marshal_positions: list = []
        self._record: Optional[EpisodeRecord] = None
        self._seed: int = 42

    # ------------------------------------------------------------------
    # OpenEnv core interface
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, seed: int = 42) -> Observation:
        """
        Start a new episode.

        Args:
            task_id: one of task_01_gate_routing, task_02_surge_response,
                     task_03_cascade_prevention. Defaults to task_01.
            seed: RNG seed for reproducibility.

        Returns:
            Initial Observation.
        """
        if task_id:
            self._task_id = task_id
        self._seed = seed

        # Look up max steps for this task
        for t in TASK_METADATA:
            if t["id"] == self._task_id:
                self._max_steps = t["max_steps"]
                break

        self._sim = CrowdSimulation(seed=seed, task_id=self._task_id)
        self._timestep = 0
        self._done = False
        self._total_reward = 0.0
        self._pa_remaining = PA_BUDGET
        self._prev_max_density = self._sim._max_density()
        self._prev_marshal_positions = []

        self._record = EpisodeRecord(task_id=self._task_id)

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Advance the environment by one timestep.

        Args:
            action: Action model with gate ops, marshal deployments, etc.

        Returns:
            StepResult with observation, reward, done flag, and info dict.
        """
        if self._sim is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._timestep += 1

        # Track PA usage
        pa_used = False
        action_dict = action.model_dump()
        if action.pa_broadcast and self._pa_remaining > 0:
            self._pa_remaining -= 1
            pa_used = True
            self._record.pa_broadcasts_used += 1
        else:
            action_dict["pa_broadcast"] = None  # revoke if out of budget

        # Track resource efficiency (marshal thrash = redundant moves)
        new_positions = [(dep[1], dep[2]) for dep in action.marshal_deployments if len(dep) >= 3]
        marshal_thrash = bool(
            new_positions and
            all(pos in [tuple(p) for p in self._prev_marshal_positions] for pos in new_positions)
        )

        # Update record
        self._record.total_steps = self._timestep
        self._record.gate_ops_count += len(action.gate_operations)
        self._record.marshal_deploys_count += len(action.marshal_deployments)
        self._record.emergency_exits_opened.extend(action.emergency_exit_opens)
        if marshal_thrash:
            self._record.resource_waste_steps += 1

        # Run simulation step
        metrics, stampede = self._sim.step(action_dict)
        metrics["stampede"] = stampede

        # Update record metrics
        crush_zones = metrics["crush_zones"]
        max_density = metrics["max_density"]
        safe_zones = metrics["safe_zones"]

        self._record.crush_zone_steps += crush_zones
        self._record.safe_zone_steps += safe_zones
        self._record.total_possible_safe += metrics["total_zones"]
        self._record.max_density_seen = max(self._record.max_density_seen, max_density)

        if stampede:
            self._record.stampede_occurred = True

        # Task-2: detect surge containment
        if self._task_id == "task_02_surge_response":
            if (self._record.surge_response_step < 0 and
                    "SURGE:stage_left" in self._sim.active_incidents and
                    max_density < 5.0):
                self._record.surge_response_step = self._timestep

        # Task-3: track crushed zones
        if self._task_id == "task_03_cascade_prevention":
            risk = self._sim.get_risk_scores()
            for zid, rs in risk.items():
                if rs >= 1.0:
                    self._record.zones_crushed_ever.add(zid)

        # Compute reward
        reward_obj = compute_reward(
            metrics=metrics,
            action=action_dict,
            prev_max_density=self._prev_max_density,
            pa_used=pa_used,
            marshal_thrash=marshal_thrash,
        )
        self._total_reward += reward_obj.total
        self._prev_max_density = max_density
        self._prev_marshal_positions = self._sim.get_marshal_positions()

        # Check terminal conditions
        self._done = (
            stampede or
            self._timestep >= self._max_steps
        )

        if self._done:
            self._record.final_active_incidents = self._sim.active_incidents[:]

        # Build info dict with grader score if done
        info: Dict[str, Any] = {
            "timestep": self._timestep,
            "max_density": round(max_density, 3),
            "crush_zones": crush_zones,
            "total_reward": round(self._total_reward, 4),
            "stampede": stampede,
            "pa_remaining": self._pa_remaining,
        }

        if self._done and self._record:
            grader_cls = GRADERS.get(self._task_id)
            if grader_cls:
                info["task_score"] = grader_cls.grade(self._record)
            info["episode_crush_events"] = self._record.crush_zone_steps
            info["episode_stampede"] = self._record.stampede_occurred

        obs = self._build_observation()

        return StepResult(
            observation=obs,
            reward=reward_obj.total,
            reward_detail=reward_obj,
            done=self._done,
            info=info,
        )

    def state(self) -> EnvState:
        """Return full internal state snapshot."""
        if self._sim is None:
            raise RuntimeError("Call reset() first.")
        return EnvState(
            task_id=self._task_id,
            timestep=self._timestep,
            done=self._done,
            total_reward=round(self._total_reward, 4),
            episode_crush_events=self._record.crush_zone_steps if self._record else 0,
            episode_stampede=self._record.stampede_occurred if self._record else False,
            gate_states=self._sim.get_gate_states(),
            barrier_states=self._sim.get_barrier_states(),
            marshal_positions=self._sim.get_marshal_positions(),
            active_incidents=self._sim.active_incidents,
            density_grid=self._sim.density,
            risk_scores=self._sim.get_risk_scores(),
        )

    def get_tasks(self):
        """Return list of available tasks."""
        return TASK_METADATA

    def grade_episode(self) -> float:
        """Run the task grader on the completed episode. Returns 0.0-1.0."""
        if not self._record:
            return 0.0
        grader_cls = GRADERS.get(self._task_id)
        if not grader_cls:
            return 0.0
        return grader_cls.grade(self._record)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Construct Observation from current simulation state."""
        sim = self._sim
        return Observation(
            timestep=self._timestep,
            density_grid=[[round(v, 3) for v in row] for row in sim.density],
            velocity_field=[
                [[round(v, 3) for v in cell] for cell in row]
                for row in sim.velocity
            ],
            gate_states=sim.get_gate_states(),
            marshal_positions=sim.get_marshal_positions(),
            risk_scores={k: round(v, 3) for k, v in sim.get_risk_scores().items()},
            active_incidents=sim.active_incidents[-10:],  # last 10 to cap size
            time_remaining=self._max_steps - self._timestep,
            pa_broadcasts_remaining=self._pa_remaining,
            task_id=self._task_id,
            venue_width=sim.cols,
            venue_height=sim.rows,
        )
