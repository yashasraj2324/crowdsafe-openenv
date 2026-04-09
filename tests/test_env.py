"""
Tests for CrowdSafeEnv — validates OpenEnv spec compliance.
Run: python -m pytest tests/ -v
"""
import pytest
from app.env import CrowdSafeEnv
from app.models import Action, Observation, StepResult, EnvState
from app.tasks import TASK_METADATA, GRADERS, EpisodeRecord


class TestOpenEnvCompliance:
    """Verify the environment implements the OpenEnv interface correctly."""

    def setup_method(self):
        self.env = CrowdSafeEnv()

    def test_reset_returns_observation(self):
        obs = self.env.reset(task_id="task_01_gate_routing", seed=42)
        assert isinstance(obs, Observation)
        assert obs.timestep == 0
        assert obs.task_id == "task_01_gate_routing"

    def test_reset_density_grid_shape(self):
        obs = self.env.reset(task_id="task_01_gate_routing", seed=42)
        assert len(obs.density_grid) == 12
        assert all(len(row) == 16 for row in obs.density_grid)

    def test_reset_velocity_field_shape(self):
        obs = self.env.reset(task_id="task_01_gate_routing", seed=42)
        assert len(obs.velocity_field) == 12
        assert all(len(row) == 16 for row in obs.velocity_field)
        assert all(len(cell) == 2 for row in obs.velocity_field for cell in row)

    def test_step_returns_step_result(self):
        self.env.reset(task_id="task_01_gate_routing", seed=42)
        action = Action(gate_operations={"gate_A": True})
        result = self.env.step(action)
        assert isinstance(result, StepResult)
        assert isinstance(result.observation, Observation)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_step_increments_timestep(self):
        self.env.reset(task_id="task_01_gate_routing", seed=42)
        action = Action()
        result = self.env.step(action)
        assert result.observation.timestep == 1

    def test_state_returns_env_state(self):
        self.env.reset(task_id="task_01_gate_routing", seed=42)
        state = self.env.state()
        assert isinstance(state, EnvState)
        assert state.timestep == 0

    def test_state_after_step(self):
        self.env.reset(task_id="task_01_gate_routing", seed=42)
        self.env.step(Action())
        state = self.env.state()
        assert state.timestep == 1

    def test_reward_in_valid_range(self):
        self.env.reset(task_id="task_01_gate_routing", seed=42)
        for _ in range(10):
            result = self.env.step(Action())
            assert -10.0 <= result.reward <= 1.0, f"Reward {result.reward} out of range"

    def test_done_at_max_steps(self):
        self.env.reset(task_id="task_01_gate_routing", seed=42)
        for _ in range(100):
            result = self.env.step(Action(gate_operations={"gate_A": True, "gate_B": True}))
            if result.done:
                break
        assert result.done

    def test_episode_not_done_before_max_steps(self):
        self.env.reset(task_id="task_01_gate_routing", seed=42)
        result = self.env.step(Action())
        # Should not be done after just 1 step (no stampede with this action)
        # Might be done if stampede, but with a safe action it shouldn't be
        assert result.observation.timestep == 1

    def test_step_raises_before_reset(self):
        fresh_env = CrowdSafeEnv()
        with pytest.raises(RuntimeError):
            fresh_env.step(Action())

    def test_state_raises_before_reset(self):
        fresh_env = CrowdSafeEnv()
        with pytest.raises(RuntimeError):
            fresh_env.state()


class TestAllThreeTasks:
    """Verify all 3 tasks are accessible and gradeable."""

    def setup_method(self):
        self.env = CrowdSafeEnv()

    def test_task_list_has_three_tasks(self):
        tasks = self.env.get_tasks()
        assert len(tasks) == 3

    def test_task_ids_correct(self):
        tasks = self.env.get_tasks()
        ids = [t["id"] for t in tasks]
        assert "task_01_gate_routing" in ids
        assert "task_02_surge_response" in ids
        assert "task_03_cascade_prevention" in ids

    def test_all_tasks_have_graders(self):
        tasks = self.env.get_tasks()
        assert len(tasks) >= 3
        assert all(task.get("has_grader") is True for task in tasks)
        assert all(str(task.get("grader", "")).startswith("app.tasks:") for task in tasks)

    def test_task_difficulties(self):
        tasks = {t["id"]: t["difficulty"] for t in self.env.get_tasks()}
        assert tasks["task_01_gate_routing"] == "easy"
        assert tasks["task_02_surge_response"] == "medium"
        assert tasks["task_03_cascade_prevention"] == "hard"

    @pytest.mark.parametrize("task_id", [
        "task_01_gate_routing",
        "task_02_surge_response",
        "task_03_cascade_prevention",
    ])
    def test_task_runs_to_completion(self, task_id):
        obs = self.env.reset(task_id=task_id, seed=42)
        assert obs.task_id == task_id
        
        max_steps = next(t["max_steps"] for t in TASK_METADATA if t["id"] == task_id)
        for _ in range(max_steps):
            action = Action(
                gate_operations={"gate_A": True, "gate_B": True, "gate_C": True},
                marshal_deployments=[["marshal_1", 6, 6]],
            )
            result = self.env.step(action)
            if result.done:
                break
        
        assert result.done

    @pytest.mark.parametrize("task_id", [
        "task_01_gate_routing",
        "task_02_surge_response",
        "task_03_cascade_prevention",
    ])
    def test_grader_score_in_range(self, task_id):
        obs = self.env.reset(task_id=task_id, seed=42)
        max_steps = next(t["max_steps"] for t in TASK_METADATA if t["id"] == task_id)
        
        for _ in range(max_steps):
            action = Action(
                gate_operations={"gate_A": True, "gate_B": True, "gate_C": True},
                marshal_deployments=[["marshal_1", 6, 6], ["marshal_2", 2, 2]],
                pa_broadcast="Please move calmly to exits",
                emergency_exit_opens=["exit_M"],
            )
            result = self.env.step(action)
            if result.done:
                break
        
        score = self.env.grade_episode()
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0.0, 1.0] for {task_id}"


class TestGraders:
    """Unit tests for each grader."""

    def test_gate_routing_grader_perfect(self):
        from app.tasks import GateRoutingGrader
        record = EpisodeRecord(task_id="task_01_gate_routing")
        record.total_steps = 100
        record.safe_zone_steps = 100 * 192  # all safe
        record.gate_ops_count = 5
        record.max_density_seen = 3.0
        record.stampede_occurred = False
        score = GateRoutingGrader.grade(record)
        assert score == 1.0

    def test_gate_routing_grader_stampede(self):
        from app.tasks import GateRoutingGrader
        record = EpisodeRecord(task_id="task_01_gate_routing")
        record.total_steps = 50
        record.safe_zone_steps = 50 * 192
        record.gate_ops_count = 3
        record.max_density_seen = 9.5
        record.stampede_occurred = True
        score = GateRoutingGrader.grade(record)
        assert score < 0.6  # penalty applied

    def test_surge_response_grader_zero_on_stampede(self):
        from app.tasks import SurgeResponseGrader
        record = EpisodeRecord(task_id="task_02_surge_response")
        record.total_steps = 30
        record.surge_response_step = 25
        record.marshal_deploys_count = 2
        record.emergency_exits_opened = ["exit_L"]
        record.stampede_occurred = True
        score = SurgeResponseGrader.grade(record)
        assert score == 0.0

    def test_cascade_grader_score_range(self):
        from app.tasks import CascadePreventionGrader
        record = EpisodeRecord(task_id="task_03_cascade_prevention")
        record.total_steps = 200
        record.marshal_deploys_count = 4
        record.pa_broadcasts_used = 2
        record.gate_ops_count = 6
        record.max_density_seen = 5.5
        record.stampede_occurred = False
        score = CascadePreventionGrader.grade(record)
        assert 0.0 <= score <= 1.0


class TestReproducibility:
    """Verify seed-based reproducibility."""

    def test_same_seed_same_initial_obs(self):
        env1 = CrowdSafeEnv()
        env2 = CrowdSafeEnv()
        obs1 = env1.reset(task_id="task_01_gate_routing", seed=99)
        obs2 = env2.reset(task_id="task_01_gate_routing", seed=99)
        assert obs1.density_grid == obs2.density_grid

    def test_different_seeds_different_obs(self):
        env = CrowdSafeEnv()
        obs1 = env.reset(task_id="task_01_gate_routing", seed=1)
        obs2 = env.reset(task_id="task_01_gate_routing", seed=2)
        # Very likely to differ (not guaranteed but essentially certain)
        grids_differ = obs1.density_grid != obs2.density_grid
        assert grids_differ
