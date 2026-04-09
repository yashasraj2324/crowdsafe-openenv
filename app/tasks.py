"""
Task definitions and programmatic graders for CrowdSafeEnv.

Each grader scores an episode 0.0-1.0:
  1.0 = perfect execution
  0.0 = catastrophic failure (stampede, all zones crushed)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class EpisodeRecord:
    """Collects per-step data for end-of-episode grading."""
    task_id: str
    total_steps: int = 0
    crush_zone_steps: int = 0      # sum of crush zones per step
    safe_zone_steps: int = 0       # sum of safe zones per step
    total_possible_safe: int = 0   # rows*cols * total_steps
    stampede_occurred: bool = False
    incidents_resolved: List[str] = field(default_factory=list)
    gate_ops_count: int = 0
    marshal_deploys_count: int = 0
    emergency_exits_opened: List[str] = field(default_factory=list)
    pa_broadcasts_used: int = 0
    max_density_seen: float = 0.0
    resource_waste_steps: int = 0  # steps where marshals thrashed
    final_active_incidents: List[str] = field(default_factory=list)
    # Task-2 specific
    surge_response_step: int = -1  # step at which surge contained
    # Task-3 specific
    zones_crushed_ever: set = field(default_factory=set)


class GateRoutingGrader:
    """
    Task 1 (Easy) — Gate Routing grader.

    Scoring:
      - 60% : fraction of steps where NO zone exceeds safe density (4.0 p/m²)
      - 20% : kept max density below 5.0 p/m² throughout
      - 20% : opened at least 3 gates within first 10 steps
    Penalties:
      - -0.5 if stampede occurred (floored to 0)
    """
    TASK_ID = "task_01_gate_routing"
    MAX_STEPS = 100

    @staticmethod
    def score(record: EpisodeRecord) -> float:
        return GateRoutingGrader.grade(record)

    @staticmethod
    def grade(record: EpisodeRecord) -> float:
        if record.total_steps == 0:
            return 0.0

        # Component 1: safe steps ratio
        possible = record.total_steps * 192  # 12*16 cells
        safe_ratio = min(1.0, record.safe_zone_steps / max(1, possible))
        score = 0.60 * safe_ratio

        # Component 2: max density control
        if record.max_density_seen < 5.0:
            score += 0.20
        elif record.max_density_seen < 6.0:
            score += 0.10

        # Component 3: early gate management
        if record.gate_ops_count >= 3:
            score += 0.20
        elif record.gate_ops_count >= 1:
            score += 0.10

        # Stampede penalty
        if record.stampede_occurred:
            score -= 0.50

        return max(0.0, min(1.0, round(score, 4)))


class SurgeResponseGrader:
    """
    Task 2 (Medium) — Surge Response grader.

    Scoring:
      - 40% : surge contained (max density drops below 5.0 within 50 steps of surge)
      - 25% : at least 2 marshals deployed within 30 steps of surge
      - 20% : correct emergency exit opened (exit_L or exit_M)
      - 15% : no crush zones in final 30 steps of episode
    Penalties:
      - -0.3 per crush zone step (normalised)
      - -1.0 if stampede (floor to 0)
    """
    TASK_ID = "task_02_surge_response"
    MAX_STEPS = 150
    SURGE_STEP = 20

    @staticmethod
    def score(record: EpisodeRecord) -> float:
        return SurgeResponseGrader.grade(record)

    @staticmethod
    def grade(record: EpisodeRecord) -> float:
        if record.total_steps == 0:
            return 0.0

        score = 0.0

        # Component 1: surge contained
        if record.surge_response_step > 0:
            steps_to_contain = record.surge_response_step - SurgeResponseGrader.SURGE_STEP
            if steps_to_contain <= 30:
                score += 0.40
            elif steps_to_contain <= 50:
                score += 0.25
            elif steps_to_contain <= 80:
                score += 0.10

        # Component 2: marshal deployment speed
        if record.marshal_deploys_count >= 2:
            score += 0.25
        elif record.marshal_deploys_count >= 1:
            score += 0.12

        # Component 3: emergency exit usage
        correct_exits = {"exit_L", "exit_M"}
        if any(e in correct_exits for e in record.emergency_exits_opened):
            score += 0.20
        
        # Component 4: clean final phase
        if record.crush_zone_steps == 0:
            score += 0.15
        else:
            crush_rate = record.crush_zone_steps / max(1, record.total_steps)
            score += 0.15 * max(0, 1.0 - crush_rate * 5)

        # Stampede penalty
        if record.stampede_occurred:
            score = 0.0

        return max(0.0, min(1.0, round(score, 4)))


class CascadePreventionGrader:
    """
    Task 3 (Hard) — Cascade Prevention grader.

    Scoring:
      - 35% : no zone reaches crush density (6.0 p/m²) at any point
               (partial: penalise per unique zone crushed)
      - 25% : all 3 initial incidents have action taken within 25 steps
      - 20% : at least 4 marshals deployed across different zones
      - 10% : PA system used at least once
      - 10% : episode completes without stampede and max_density < 7.0

    Penalties:
      - -0.15 per zone ever crushed
      - -1.0 if stampede (floor to 0)
    """
    TASK_ID = "task_03_cascade_prevention"
    MAX_STEPS = 200

    @staticmethod
    def score(record: EpisodeRecord) -> float:
        return CascadePreventionGrader.grade(record)

    @staticmethod
    def grade(record: EpisodeRecord) -> float:
        if record.total_steps == 0:
            return 0.0

        score = 0.0

        # Component 1: crush prevention
        zones_crushed = len(record.zones_crushed_ever)
        if zones_crushed == 0:
            score += 0.35
        else:
            score += max(0.0, 0.35 - zones_crushed * 0.15)

        # Component 2: incident response speed
        # Proxy: gate + marshal ops within first 25 steps
        early_actions = record.gate_ops_count + record.marshal_deploys_count
        if early_actions >= 5:
            score += 0.25
        elif early_actions >= 3:
            score += 0.15
        elif early_actions >= 1:
            score += 0.05

        # Component 3: marshal coverage
        if record.marshal_deploys_count >= 4:
            score += 0.20
        elif record.marshal_deploys_count >= 2:
            score += 0.10

        # Component 4: PA usage
        if record.pa_broadcasts_used >= 1:
            score += 0.10

        # Component 5: clean finish
        if not record.stampede_occurred and record.max_density_seen < 7.0:
            score += 0.10

        # Stampede penalty
        if record.stampede_occurred:
            score = 0.0

        return max(0.0, min(1.0, round(score, 4)))


GRADERS = {
    "task_01_gate_routing": GateRoutingGrader,
    "task_02_surge_response": SurgeResponseGrader,
    "task_03_cascade_prevention": CascadePreventionGrader,
}

TASK_METADATA = [
    {
        "id": "task_01_gate_routing",
        "name": "Gate Routing",
        "difficulty": "easy",
        "grader": "app.tasks:GateRoutingGrader",
        "has_grader": True,
        "description": (
            "Manage gate opening/closing to prevent any sector exceeding safe "
            "density (4.0 p/m²) during venue fill-up. Single zone, no dynamic surge. "
            "Open the right gates at the right time before crowd density spikes."
        ),
        "max_steps": GateRoutingGrader.MAX_STEPS,
    },
    {
        "id": "task_02_surge_response",
        "name": "Surge Response",
        "difficulty": "medium",
        "grader": "app.tasks:SurgeResponseGrader",
        "has_grader": True,
        "description": (
            "A speaker malfunction causes a crowd surge from stage-left at step 20. "
            "Redirect crowd flow via barriers, deploy marshals to high-density sectors, "
            "and open emergency exits in the correct sequence under time pressure."
        ),
        "max_steps": SurgeResponseGrader.MAX_STEPS,
    },
    {
        "id": "task_03_cascade_prevention",
        "name": "Cascade Prevention",
        "difficulty": "hard",
        "grader": "app.tasks:CascadePreventionGrader",
        "has_grader": True,
        "description": (
            "Three simultaneous incidents: exit bottleneck, stage surge, and medical panic. "
            "Allocate limited marshals and PA broadcasts across all zones. "
            "Prevent any zone reaching crush density (6.0 p/m²) while managing "
            "scarce resources over 200 steps."
        ),
        "max_steps": CascadePreventionGrader.MAX_STEPS,
    },
]
