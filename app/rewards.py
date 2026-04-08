"""
Dense reward function for CrowdSafeEnv.

Fires every timestep — NOT just at episode end.
This gives the agent continuous learning signal.
"""
from __future__ import annotations
from app.models import Reward
from app.simulation import SAFE_DENSITY, CRUSH_DENSITY, MAX_DENSITY


def compute_reward(
    metrics: dict,
    action: dict,
    prev_max_density: float,
    pa_used: bool,
    marshal_thrash: bool,
) -> Reward:
    """
    Compute per-step reward.

    Args:
        metrics: dict from simulation.step() — crush_zones, max_density, etc.
        action: the raw action dict applied this step
        prev_max_density: max density from previous step (for improvement bonus)
        pa_used: whether a PA broadcast was issued this step
        marshal_thrash: whether marshals were redundantly moved

    Returns:
        Reward model with total and breakdown.
    """
    crush_zones = metrics.get("crush_zones", 0)
    max_density = metrics.get("max_density", 0.0)
    safe_zones = metrics.get("safe_zones", 0)
    total_zones = metrics.get("total_zones", 192)
    stampede = metrics.get("stampede", False)

    # --- Density penalty: -0.5 per crush zone per step ---
    density_penalty = -0.5 * crush_zones

    # --- Flow reward: +0.1 * (fraction of safe zones) ---
    safe_ratio = safe_zones / max(1, total_zones)
    flow_reward = 0.10 * safe_ratio

    # --- Improvement bonus: +0.05 if max density improved ---
    if max_density < prev_max_density and crush_zones == 0:
        flow_reward += 0.05

    # --- Crush penalty: -1.0 if any zone at crush density ---
    crush_penalty = -1.0 if crush_zones > 0 else 0.0

    # --- Stampede penalty: -5.0 terminal ---
    stampede_penalty = -5.0 if stampede else 0.0

    # --- Efficiency bonus: +0.15 if max density below safe threshold ---
    efficiency_bonus = 0.0
    if max_density < SAFE_DENSITY:
        efficiency_bonus = 0.15
    elif max_density < CRUSH_DENSITY:
        efficiency_bonus = 0.05

    # --- Resource penalty: -0.05 for unnecessary marshal moves ---
    resource_penalty = -0.05 if marshal_thrash else 0.0

    total = (
        density_penalty
        + flow_reward
        + crush_penalty
        + stampede_penalty
        + efficiency_bonus
        + resource_penalty
    )

    # Clamp to defined range
    total = max(-10.0, min(1.0, total))

    return Reward(
        total=round(total, 4),
        density_penalty=round(density_penalty, 4),
        flow_reward=round(flow_reward, 4),
        crush_penalty=round(crush_penalty, 4),
        stampede_penalty=round(stampede_penalty, 4),
        efficiency_bonus=round(efficiency_bonus, 4),
        resource_penalty=round(resource_penalty, 4),
    )
