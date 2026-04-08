"""
Crowd physics simulation engine.

Models a discrete grid where each cell has:
  - density: people per m²  (safe < 4.0, danger >= 4.0, crush >= 6.0)
  - velocity: (vx, vy) representing average crowd movement direction

Crowd flow uses a simplified Social Force Model:
  - People move toward exits/open gates
  - High-density cells repel neighbours
  - Barriers and closed gates block flow
"""
from __future__ import annotations
import random
import math
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

SAFE_DENSITY = 4.0    # p/m² — agent should keep below this
CRUSH_DENSITY = 6.0   # p/m² — immediate danger, heavy penalty
MAX_DENSITY = 9.0     # p/m² — physical maximum, triggers stampede

GRID_ROWS = 12
GRID_COLS = 16


class Gate:
    def __init__(self, gate_id: str, x: int, y: int, capacity: float = 2.0):
        self.id = gate_id
        self.x = x
        self.y = y
        self.open = False
        self.capacity = capacity  # flow rate when open (people/step/cell)


class Barrier:
    def __init__(self, barrier_id: str, cells: List[Tuple[int, int]]):
        self.id = barrier_id
        self.cells = cells  # grid cells this barrier blocks
        self.up = True       # True = barrier raised = blocking


class EmergencyExit:
    def __init__(self, exit_id: str, x: int, y: int):
        self.id = exit_id
        self.x = x
        self.y = y
        self.open = False
        self.capacity = 4.0  # large throughput when open


class Marshal:
    def __init__(self, marshal_id: str):
        self.id = marshal_id
        self.x: Optional[int] = None
        self.y: Optional[int] = None
        self.deployed = False

    def deploy(self, x: int, y: int):
        self.x = x
        self.y = y
        self.deployed = True


class CrowdSimulation:
    """Core simulation engine. Instantiated per-episode."""

    def __init__(
        self,
        rows: int = GRID_ROWS,
        cols: int = GRID_COLS,
        seed: int = 42,
        task_id: str = "task_01_gate_routing",
    ):
        self.rows = rows
        self.cols = cols
        self.rng = random.Random(seed)
        self.task_id = task_id
        self.timestep = 0
        self.done = False
        self.stampede_triggered = False

        # Density + velocity grids
        self.density: List[List[float]] = [
            [0.0] * cols for _ in range(rows)
        ]
        self.velocity: List[List[List[float]]] = [
            [[0.0, 0.0] for _ in range(cols)] for _ in range(rows)
        ]

        # Venue elements
        self.gates: Dict[str, Gate] = {}
        self.barriers: Dict[str, Barrier] = {}
        self.emergency_exits: Dict[str, EmergencyExit] = {}
        self.marshals: Dict[str, Marshal] = {}

        # Active incidents
        self.active_incidents: List[str] = []
        self.crush_events: int = 0
        self.incident_schedule: List[Tuple[int, str]] = []  # (timestep, incident)

        self._setup_venue()
        self._setup_task_scenario()

    # ------------------------------------------------------------------
    # Venue setup
    # ------------------------------------------------------------------

    def _setup_venue(self):
        """Build a generic stadium venue."""
        # Main entry gates (bottom edge)
        for i, (gid, x, cap) in enumerate([
            ("gate_A", 2, self.rows - 1),
            ("gate_B", 6, self.rows - 1),
            ("gate_C", 10, self.rows - 1),
            ("gate_D", 14, self.rows - 1),
        ]):
            self.gates[gid] = Gate(gid, x, self.rows - 1, capacity=cap or 2.0)

        # Side gates
        self.gates["gate_E"] = Gate("gate_E", 0, 6, capacity=1.5)
        self.gates["gate_F"] = Gate("gate_F", self.cols - 1, 6, capacity=1.5)

        # Internal flow barriers (horizontal at row 4)
        self.barriers["barrier_1"] = Barrier("barrier_1", [(4, c) for c in range(3, 7)])
        self.barriers["barrier_2"] = Barrier("barrier_2", [(4, c) for c in range(9, 14)])

        # Emergency exits (top edge near stage)
        self.emergency_exits["exit_L"] = EmergencyExit("exit_L", 1, 0)
        self.emergency_exits["exit_R"] = EmergencyExit("exit_R", self.cols - 2, 0)
        self.emergency_exits["exit_M"] = EmergencyExit("exit_M", 8, 0)

        # Marshals (6 available, all undeployed)
        for i in range(6):
            mid = f"marshal_{i+1}"
            self.marshals[mid] = Marshal(mid)

    def _setup_task_scenario(self):
        """Configure initial density and incident schedule per task."""
        if self.task_id == "task_01_gate_routing":
            # Light uniform crowd arriving — agent just manages gates
            for r in range(8, self.rows):
                for c in range(self.cols):
                    self.density[r][c] = self.rng.uniform(0.5, 1.5)
            # Crowd arrives in waves at gate zones
            self.incident_schedule = []  # no surprise incidents

        elif self.task_id == "task_02_surge_response":
            # Moderate base crowd
            for r in range(5, self.rows):
                for c in range(self.cols):
                    self.density[r][c] = self.rng.uniform(1.0, 2.5)
            # Surge triggers at step 20 (stage-left incident)
            self.incident_schedule = [
                (20, "SURGE:stage_left"),
                (40, "PRESSURE:center_front"),
            ]

        elif self.task_id == "task_03_cascade_prevention":
            # Heavy base crowd
            for r in range(3, self.rows):
                for c in range(self.cols):
                    self.density[r][c] = self.rng.uniform(2.0, 3.5)
            # Three simultaneous incidents
            self.incident_schedule = [
                (10, "SURGE:stage_left"),
                (10, "BOTTLENECK:exit_R"),
                (15, "PANIC:medical_zone"),
                (50, "SURGE:stage_right"),
            ]

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def step(self, action_parsed: dict) -> Tuple[dict, bool]:
        """
        Advance simulation by one timestep.
        Returns (metrics_dict, stampede_occurred).
        """
        self.timestep += 1

        # Apply agent actions
        self._apply_gate_ops(action_parsed.get("gate_operations", {}))
        self._apply_barrier_changes(action_parsed.get("barrier_changes", {}))
        self._apply_marshal_deployments(action_parsed.get("marshal_deployments", []))
        self._apply_emergency_exits(action_parsed.get("emergency_exit_opens", []))
        self._apply_pa_broadcast(action_parsed.get("pa_broadcast"))

        # Trigger scheduled incidents
        for ts, incident in self.incident_schedule:
            if self.timestep == ts:
                self._trigger_incident(incident)

        # Simulate crowd movement
        self._flow_step()

        # Add inflow from open gates
        self._process_gate_inflow()

        # Count dangerous zones
        crush_zones = self._count_crush_zones()
        if crush_zones > 0:
            self.crush_events += crush_zones

        # Check stampede condition
        stampede = self._check_stampede()
        if stampede:
            self.stampede_triggered = True
            self.active_incidents.append("STAMPEDE")

        metrics = {
            "crush_zones": crush_zones,
            "max_density": self._max_density(),
            "mean_density": self._mean_density(),
            "safe_zones": self._count_safe_zones(),
            "total_zones": self.rows * self.cols,
        }
        return metrics, stampede

    def _apply_gate_ops(self, ops: dict):
        for gid, state in ops.items():
            if gid in self.gates:
                self.gates[gid].open = bool(state)

    def _apply_barrier_changes(self, changes: dict):
        for bid, state in changes.items():
            if bid in self.barriers:
                self.barriers[bid].up = bool(state)

    def _apply_marshal_deployments(self, deployments: list):
        for dep in deployments:
            if len(dep) >= 3:
                mid, x, y = dep[0], int(dep[1]), int(dep[2])
                if mid in self.marshals:
                    x = max(0, min(self.cols - 1, x))
                    y = max(0, min(self.rows - 1, y))
                    self.marshals[mid].deploy(x, y)

    def _apply_emergency_exits(self, exit_ids: list):
        for eid in exit_ids:
            if eid in self.emergency_exits:
                self.emergency_exits[eid].open = True

    def _apply_pa_broadcast(self, message: Optional[str]):
        if message:
            # PA reduces density in random high-density zones slightly
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.density[r][c] > SAFE_DENSITY:
                        self.density[r][c] *= 0.95

    def _trigger_incident(self, incident: str):
        """Inject a crowd disturbance into the grid."""
        self.active_incidents.append(incident)
        kind = incident.split(":")[0]

        if kind == "SURGE":
            # Dense crowd appears in stage-adjacent rows
            side = incident.split(":")[1] if ":" in incident else "center"
            col_start = 0 if "left" in side else (self.cols // 2 if "right" in side else 4)
            col_end = col_start + 6
            for r in range(0, 4):
                for c in range(col_start, min(col_end, self.cols)):
                    self.density[r][c] = min(MAX_DENSITY, self.density[r][c] + self.rng.uniform(2.0, 3.5))

        elif kind == "BOTTLENECK":
            # Exit blocked — crowd piles up at that side
            for r in range(0, 5):
                for c in range(self.cols - 5, self.cols):
                    self.density[r][c] = min(MAX_DENSITY, self.density[r][c] + self.rng.uniform(1.5, 2.5))

        elif kind == "PANIC":
            # Random density spike in middle zone
            for _ in range(10):
                r = self.rng.randint(3, 8)
                c = self.rng.randint(4, 12)
                self.density[r][c] = min(MAX_DENSITY, self.density[r][c] + self.rng.uniform(1.0, 2.0))

    def _flow_step(self):
        """Simplified crowd diffusion and movement."""
        new_density = [row[:] for row in self.density]
        new_velocity = [[v[:] for v in row] for row in self.velocity]

        # Get barrier-blocked cells
        blocked = set()
        for b in self.barriers.values():
            if b.up:
                for cell in b.cells:
                    blocked.add(cell)

        for r in range(self.rows):
            for c in range(self.cols):
                d = self.density[r][c]
                if d < 0.01:
                    continue

                # Compute pressure gradient (flow toward lower density)
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if (nr, nc) not in blocked:
                            neighbors.append((nr, nc, self.density[nr][nc]))

                if not neighbors:
                    continue

                # Flow a portion of density to lower-pressure neighbours
                for nr, nc, nd in neighbors:
                    if nd < d:
                        flow = (d - nd) * 0.08
                        flow = min(flow, d * 0.15)  # cap outflow
                        new_density[r][c] -= flow
                        new_density[nr][nc] += flow

                        # Update velocity field
                        new_velocity[r][c][0] += (nc - c) * 0.1
                        new_velocity[r][c][1] += (nr - r) * 0.1

        # Marshal effect: reduce density at deployed positions
        for m in self.marshals.values():
            if m.deployed and m.x is not None:
                r, c = m.y, m.x
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    new_density[r][c] = max(0, new_density[r][c] - 0.3)
                    # Spread reduction to neighbours
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            new_density[nr][nc] = max(0, new_density[nr][nc] - 0.1)

        # Open emergency exits drain density at exit cells
        for ex in self.emergency_exits.values():
            if ex.open:
                r, c = ex.y, ex.x
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    drained = min(new_density[r][c], ex.capacity * 0.3)
                    new_density[r][c] -= drained

        # Natural decay (people leaving through normal exits)
        for r in range(self.rows):
            for c in range(self.cols):
                new_density[r][c] = max(0.0, new_density[r][c] * 0.98)

        self.density = new_density
        self.velocity = new_velocity

    def _process_gate_inflow(self):
        """Add crowd inflow through open gates (simulates arrivals)."""
        for g in self.gates.values():
            if g.open:
                r, c = g.y, g.x
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    inflow = self.rng.uniform(0.1, g.capacity * 0.3)
                    self.density[r][c] = min(
                        MAX_DENSITY,
                        self.density[r][c] + inflow
                    )

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def _count_crush_zones(self) -> int:
        return sum(
            1 for r in range(self.rows) for c in range(self.cols)
            if self.density[r][c] >= CRUSH_DENSITY
        )

    def _count_safe_zones(self) -> int:
        return sum(
            1 for r in range(self.rows) for c in range(self.cols)
            if self.density[r][c] < SAFE_DENSITY
        )

    def _max_density(self) -> float:
        return max(
            self.density[r][c]
            for r in range(self.rows) for c in range(self.cols)
        )

    def _mean_density(self) -> float:
        total = sum(
            self.density[r][c]
            for r in range(self.rows) for c in range(self.cols)
        )
        return total / (self.rows * self.cols)

    def _check_stampede(self) -> bool:
        """Stampede if any cell hits MAX_DENSITY."""
        return any(
            self.density[r][c] >= MAX_DENSITY
            for r in range(self.rows) for c in range(self.cols)
        )

    def get_risk_scores(self) -> Dict[str, float]:
        """Zone-level risk scores (0.0-1.0)."""
        zones = {}
        # Divide grid into 4 quadrants
        quad_defs = {
            "zone_NW": (0, self.rows // 2, 0, self.cols // 2),
            "zone_NE": (0, self.rows // 2, self.cols // 2, self.cols),
            "zone_SW": (self.rows // 2, self.rows, 0, self.cols // 2),
            "zone_SE": (self.rows // 2, self.rows, self.cols // 2, self.cols),
        }
        for zid, (r0, r1, c0, c1) in quad_defs.items():
            cells = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
            if not cells:
                zones[zid] = 0.0
                continue
            max_d = max(self.density[r][c] for r, c in cells)
            zones[zid] = min(1.0, max_d / CRUSH_DENSITY)
        return zones

    def get_gate_states(self) -> Dict[str, bool]:
        return {gid: g.open for gid, g in self.gates.items()}

    def get_marshal_positions(self) -> List[List[int]]:
        return [
            [m.x, m.y] for m in self.marshals.values()
            if m.deployed and m.x is not None
        ]

    def get_barrier_states(self) -> Dict[str, bool]:
        return {bid: b.up for bid, b in self.barriers.items()}
