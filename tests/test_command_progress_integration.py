"""CPU checks of production RobotEnv hooks and the real progress tracker.

The lightweight parent only follows the documented Isaac Lab transition order;
it does not construct or emulate a physics scene. RobotEnv methods are extracted
unchanged with AST to omit simulator imports.
"""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"
SPEC = importlib.util.spec_from_file_location("h2_command_progress_integration", TASK / "command_progress.py")
PROGRESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROGRESS
SPEC.loader.exec_module(PROGRESS)


class _Asset:
    root_physx_view = SimpleNamespace(get_masses=lambda: torch.ones(2, 3))

    def __init__(self):
        self.data = SimpleNamespace(
            root_pos_w=torch.zeros(2, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        )

    def find_bodies(self, names, preserve_order=False):
        assert preserve_order
        return [0, 1], ["left", "right"]


class _Parent:
    """Supply data and the documented callback order, without physics."""

    def __init__(self, cfg, *args, **kwargs):
        self.cfg, self.num_envs, self.device, self.step_dt = cfg, 2, "cpu", 0.1
        self.scene = {"robot": _Asset()}
        self.command = torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
        self.command_manager = SimpleNamespace(get_command=lambda name: self.command)
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.common_step_counter = 0
        self.next_delta = torch.tensor([[0.02, 0.0], [0.02, 0.0]])
        self.next_command = None
        self.next_quaternion = None
        self.next_reset_ids = []
        self.reset_positions = torch.tensor([[100.0, 0.0], [-100.0, 0.0]])
        self.curriculum_snapshots = []
        self.curriculum_callback = None
        self.raise_before_reset = False
        self.raise_after_reset = False
        self.invalidations = []
        self.extras = {"log": {}, "other": "preserved"}

    def step(self, action):
        self.asset.data.root_pos_w[:, :2] += self.next_delta
        if self.next_quaternion is not None:
            self.asset.data.root_quat_w.copy_(self.next_quaternion)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        if self.raise_before_reset:
            raise RuntimeError("synthetic parent failure")
        if self.next_reset_ids:
            self._reset_idx(torch.tensor(self.next_reset_ids, dtype=torch.long))
        if self.raise_after_reset:
            raise RuntimeError("synthetic post-reset failure")
        if self.next_command is not None:
            self.command.copy_(self.next_command)
        return ("observation", "reward", "terminated", "timeout", "extras")

    def _reset_idx(self, env_ids):
        tracker = getattr(self, "command_progress_tracker", None)
        if tracker is not None:
            self.curriculum_snapshots.append({
                "ids": torch.as_tensor(env_ids).clone(),
                "target": tracker.target_distance.clone(),
                "progress": tracker.signed_progress.clone(),
                "ready": tracker.reset_ready.clone(),
                "elapsed": tracker.elapsed_time.clone(),
            })
        if self.curriculum_callback is not None:
            self.curriculum_callback(self, env_ids)
        self.extras["log"] = {"manager_metric": 17.0}
        self.asset.data.root_pos_w[env_ids, :2] = self.reset_positions[env_ids]
        self.episode_length_buf[env_ids] = 0


@pytest.fixture
def env_type():
    source = TASK / "robot_env.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RobotEnv")
    names = {"__init__", "step", "_reset_idx"}
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    cls = ast.ClassDef(name="RobotEnv", bases=[ast.Name(id="_Parent", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {
        "torch": torch, "_Parent": _Parent,
        "CommandProgressTracker": PROGRESS.CommandProgressTracker,
        "invalidate_lidar_observation": lambda env, ids: env.invalidations.append(torch.as_tensor(ids).clone()),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), str(source), "exec"), scope)
    # Foot kinematics are unrelated to this contract; only their existing call
    # position is preserved. No fake simulator or substitute tracker is used.
    scope["RobotEnv"].update_feet_state = lambda self: None
    return scope["RobotEnv"]


def _env(env_type, **cfg):
    return env_type(SimpleNamespace(feet_body_names=["left", "right"], **cfg))


def test_normal_transition_is_consumed_once_and_parent_result_is_preserved(env_type):
    env = _env(env_type)
    assert env.step(torch.zeros(2, 1)) == ("observation", "reward", "terminated", "timeout", "extras")
    tracker = env.command_progress_tracker
    torch.testing.assert_close(tracker.target_distance, torch.full((2,), 0.02))
    torch.testing.assert_close(tracker.signed_progress, torch.full((2,), 0.02))
    torch.testing.assert_close(tracker.elapsed_time, torch.full((2,), 0.1))
    assert not tracker.pending.any()
    assert not tracker.reset_ready.any()
    tracker.finish_step([0, 1], env.asset.data.root_pos_w[:, :2], for_reset=True)
    torch.testing.assert_close(tracker.signed_progress, torch.full((2,), 0.02))
    assert not tracker.reset_ready.any()


def test_resampled_command_does_not_relabel_the_previous_transition(env_type):
    env = _env(env_type)
    env.next_command = torch.tensor([[-0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    env.step(torch.zeros(2, 1))
    torch.testing.assert_close(env.command_progress_tracker.signed_progress, torch.full((2,), 0.02))
    env.next_delta *= -1
    env.next_command = None
    env.step(torch.zeros(2, 1))
    torch.testing.assert_close(env.asset.data.root_pos_w[:, :2], torch.zeros(2, 2))
    torch.testing.assert_close(env.command_progress_tracker.target_distance, torch.full((2,), 0.04))
    torch.testing.assert_close(env.command_progress_tracker.signed_progress, torch.full((2,), 0.04))


def test_start_yaw_rotates_command_into_the_world_displacement_frame(env_type):
    env = _env(env_type)
    env.asset.data.root_quat_w[:] = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    env.next_delta[:] = torch.tensor([0.0, 0.02])
    env.step(torch.zeros(2, 1))
    torch.testing.assert_close(env.command_progress_tracker.signed_progress, torch.full((2,), 0.02))


def test_heading_changes_after_physics_do_not_change_the_captured_command_frame(env_type):
    env = _env(env_type)
    env.next_quaternion = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]]).repeat(2, 1)
    env.step(torch.zeros(2, 1))
    torch.testing.assert_close(env.command_progress_tracker.signed_progress, torch.full((2,), 0.02))


def test_terminal_transition_is_available_to_curriculum_before_reset_teleport(env_type):
    env = _env(env_type)
    env.next_reset_ids = [0]
    env.step(torch.zeros(2, 1))
    snapshot = env.curriculum_snapshots[0]
    torch.testing.assert_close(snapshot["target"], torch.tensor([0.02, 0.0]))
    torch.testing.assert_close(snapshot["progress"], torch.tensor([0.02, 0.0]))
    assert snapshot["ready"].tolist() == [True, False]
    tracker = env.command_progress_tracker
    torch.testing.assert_close(tracker.target_distance, torch.tensor([0.0, 0.02]))
    torch.testing.assert_close(tracker.signed_progress, torch.tensor([0.0, 0.02]))
    assert env.asset.data.root_pos_w[0, 0] == 100.0
    assert not tracker.pending.any()
    assert not tracker.reset_ready.any()


def test_reset_reads_generation_after_curriculum_update(env_type):
    env = _env(env_type)
    env.next_reset_ids = [1]
    env.curriculum_callback = lambda env, ids: setattr(env, "_command_progress_generation", 7)
    env.step(torch.zeros(2, 1))
    assert env.command_progress_tracker.episode_generation.tolist() == [0, 7]


def test_curriculum_diagnostics_survive_parent_log_replacement(env_type):
    env = _env(env_type)
    env.next_reset_ids = [0]
    env.curriculum_callback = lambda env, ids: setattr(
        env, "_command_progress_pending_log", {"Curriculum/progress_ratio": 0.9}
    )
    env.step(torch.zeros(2, 1))
    assert env.extras["log"] == {"manager_metric": 17.0, "Curriculum/progress_ratio": 0.9}
    assert env.extras["other"] == "preserved"
    assert not hasattr(env, "_command_progress_pending_log")


def test_manual_reset_has_no_fabricated_finished_transition(env_type):
    env = _env(env_type)
    env.step(torch.zeros(2, 1))
    env._reset_idx([0])
    assert not env.curriculum_snapshots[0]["ready"].any()
    torch.testing.assert_close(env.command_progress_tracker.target_distance, torch.tensor([0.0, 0.02]))
    env.next_reset_ids = []
    env.step(torch.zeros(2, 1))
    # Float32 subtraction at the synthetic 100 m reset origin loses a few um.
    torch.testing.assert_close(env.command_progress_tracker.signed_progress, torch.tensor([0.02, 0.04]), atol=5e-6, rtol=2e-6)


def test_initial_reset_cannot_be_success_evidence(env_type):
    env = _env(env_type)
    env._reset_idx([0, 1])
    assert not env.curriculum_snapshots[0]["ready"].any()
    assert not env.command_progress_tracker.had_translation.any()
    torch.testing.assert_close(env.command_progress_tracker.target_distance, torch.zeros(2))


def test_reset_guard_allows_existing_behavior_before_tracker_creation(env_type):
    env = _env(env_type)
    del env.command_progress_tracker
    env._reset_idx([0])
    assert len(env.invalidations) == 1
    assert env.asset.data.root_pos_w[0, 0] == 100.0


def test_parent_failure_aborts_pending_evidence_and_propagates(env_type):
    env = _env(env_type)
    env.raise_before_reset = True
    with pytest.raises(RuntimeError, match="synthetic parent failure"):
        env.step(torch.zeros(2, 1))
    tracker = env.command_progress_tracker
    assert not tracker.pending.any()
    assert not tracker.valid.any()
    assert not tracker.reset_ready.any()


def test_constructor_uses_progress_threshold_configuration(env_type):
    env = _env(env_type, command_progress_min_speed=0.3, command_progress_max_step_speed=1.0)
    env.step(torch.zeros(2, 1))
    assert not env.command_progress_tracker.had_translation.any()
    env.next_delta[:] = torch.tensor([0.2, 0.0])
    env.step(torch.zeros(2, 1))
    assert not env.command_progress_tracker.valid.any()
