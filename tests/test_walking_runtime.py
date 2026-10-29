"""Exercise reset-before-teleport diagnostics through CPU data containers."""

from __future__ import annotations

import importlib.util
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as Box

import pytest
import torch

from test_command_progress_integration import env_type


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"
PACKAGE = "h2_walking_runtime_test"
package = ModuleType(PACKAGE)
package.__path__ = [str(TASK)]
sys.modules[PACKAGE] = package


def load_runtime():
    spec = importlib.util.spec_from_file_location(PACKAGE + ".walking_runtime", TASK / "walking_runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sensor(name, filters=True):
    return Box(body_names=[name], cfg=Box(history_length=3, update_period=.005, filter_prim_paths_expr=["one_collider"]), data=Box(
        net_forces_w_history=torch.zeros(2, 3, 1, 3),
        force_matrix_w_history=torch.zeros(2, 3, 1, 1, 3) if filters else None,
    ))


class Scene:
    """Match InteractiveScene lookup: no membership/iterator protocol."""

    def __init__(self, objects):
        self.objects = objects
        self.sensors = {k: v for k, v in objects.items() if k != "robot"}

    def __getitem__(self, name):
        return self.objects[name]


def environment(**overrides):
    sensors = {
        "pelvis_ground": sensor("pelvis"), "pelvis_robot": sensor("pelvis"),
        "left_foot_ground": sensor("left_foot"), "right_foot_ground": sensor("right_foot"),
    }
    sensors["contact_forces"] = Box(
        body_names=["pelvis", "left_foot", "right_foot"],
        cfg=Box(history_length=3, update_period=.005),
        data=Box(net_forces_w_history=torch.zeros(2, 3, 3, 3), net_forces_w=torch.zeros(2, 3, 3)),
        find_bodies=lambda names, preserve_order: ([0] if names == ["pelvis"] else [1, 2], names),
    )
    data = Box(root_pos_w=torch.tensor([[0., 0., 1.], [0., 0., 1.]]),
               projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
               joint_pos=torch.zeros(2, 2), default_joint_pos=torch.zeros(2, 2),
               joint_vel=torch.zeros(2, 2), applied_torque=torch.zeros(2, 2),
               body_pos_w=torch.zeros(2, 3, 3), body_lin_vel_w=torch.zeros(2, 3, 3),
               body_quat_w=torch.tensor([1., 0., 0., 0.]).expand(2, 3, 4).clone())
    data.body_pos_w[:, 1:, 2] = .0543
    robot = Box(data=data, find_joints=lambda names, preserve_order: ([0, 1], names))
    scene = {"robot": robot, **sensors}
    scene["height_scanner"] = Box(data=Box(ray_hits_w=torch.tensor([
        [[0., 0., 0.], [.01, 0., 0.]], [[0., 0., 0.], [.01, 0., 0.]],
    ])))
    scene = Scene(scene)
    cfg = Box(contact_source_diagnostics=True, staged_evidence=False,
              contact_diagnostic_max_samples=1, contact_diagnostic_threshold=1.,
              walking_diagnostic_joint_names=["j0", "j1"], feet_body_names=["left_foot", "right_foot"])
    cfg.__dict__.update(overrides)
    env = Box(cfg=cfg, scene=scene, asset=robot, num_envs=2, device="cpu", step_dt=.02,
              physics_dt=.005, feet_id_b=[1, 2], common_step_counter=10,
              max_episode_length_s=20., extras={"log": {}}, episode_length_buf=torch.tensor([10, 20]),
              action_manager=Box(action=torch.zeros(2, 2), get_term=lambda name: Box(processed_actions=torch.ones(2, 2))),
              termination_manager=Box(terminated=torch.tensor([True, False]), time_outs=torch.zeros(2, dtype=torch.bool),
                                      active_terms=["base_contact"], get_term=lambda name: torch.tensor([True, False]),
                                      get_term_cfg=lambda name: Box(func=lambda env: None, params={})))
    env.command_progress_tracker = Box(reset_ready=torch.tensor([True, False]))
    return env


def test_terminal_capture_is_bounded_detached_and_pre_reset():
    module = load_runtime()
    env = environment()
    env.scene["contact_forces"].data.net_forces_w_history[0, 0, 0, 2] = 2.
    env.scene["pelvis_robot"].data.force_matrix_w_history[0, 0, 0, 0, 2] = 2.
    runtime = module.WalkingRuntime(env)
    logs, sample = runtime.before_reset([0, 1])
    assert logs["ContactDiagnostics/natural_episodes_total"] == 1
    assert logs["ContactDiagnostics/self_at_trigger_total"] == 1
    assert sample["env_ids"].tolist() == [0]
    assert sample["source_code"].tolist() == [2]
    env.asset.data.root_pos_w[:] = 100.
    assert sample["pelvis_position_w"][0, 2] == 1.
    assert not sample["raw_action"].requires_grad


def test_manual_initial_reset_does_not_provide_terminal_samples():
    module = load_runtime()
    env = environment()
    env.command_progress_tracker.reset_ready[:] = False
    logs, sample = module.WalkingRuntime(env).before_reset([0, 1])
    assert logs["ContactDiagnostics/natural_episodes_total"] == 0
    assert sample is None


@pytest.mark.parametrize("field", ["pelvis_ground", "pelvis_robot", "left_foot_ground", "right_foot_ground"])
def test_missing_filtered_history_is_a_configuration_error(field):
    module = load_runtime()
    env = environment()
    env.scene[field].data.force_matrix_w_history = None
    with pytest.raises(RuntimeError, match="filtered.*history"):
        module.WalkingRuntime(env)


def test_mismatched_sensor_period_fails_before_any_attribution():
    module = load_runtime()
    env = environment()
    env.scene["pelvis_robot"].cfg.update_period = .02
    with pytest.raises(ValueError, match="physics"):
        module.WalkingRuntime(env)


def test_filter_count_mismatch_fails_before_attribution():
    module = load_runtime()
    env = environment()
    env.scene["pelvis_robot"].cfg.filter_prim_paths_expr.append("second_collider")
    with pytest.raises(RuntimeError, match="configured collider filters"):
        module.WalkingRuntime(env)


def test_filter_only_support_cannot_use_whole_body_net_force():
    module = load_runtime()
    env = environment()
    env.scene["contact_forces"].data.net_forces_w[:, 1:, 2] = 500.
    runtime = module.WalkingRuntime(env)
    _, contacts = runtime.ground_support()
    assert not contacts.any()
    env.scene["left_foot_ground"].data.force_matrix_w_history[:, 0, 0, 0, 2] = 100.
    _, contacts = runtime.ground_support()
    assert contacts.tolist() == [[True, False], [True, False]]


def test_horizontal_riser_collision_and_airborne_sole_are_not_support():
    module = load_runtime()
    env = environment()
    pairs = env.scene["left_foot_ground"].data.force_matrix_w_history
    pairs[:, 0, 0, 0, 0] = 500.
    runtime = module.WalkingRuntime(env)
    assert not runtime.ground_support()[1].any()
    pairs[:, 0, 0, 0, 2] = 100.
    env.asset.data.body_pos_w[:, 1, 2] = 1.
    assert not runtime.ground_support()[1].any()


def test_nonfinite_force_is_invalid_instead_of_normal_air_time():
    module = load_runtime()
    env = environment()
    env.scene["left_foot_ground"].data.force_matrix_w_history[0, 0, 0, 0, 2] = torch.nan
    _, contacts, valid = module.WalkingRuntime(env)._support_state()
    assert not contacts[0, 0]
    assert not valid[0, 0]


def test_adjacent_two_centimeter_tread_does_not_certify_a_lower_sole():
    module = load_runtime()
    env = environment(stage_support_height_tolerance=.015)
    env.scene["height_scanner"].data.ray_hits_w[..., 2] = .02
    env.scene["left_foot_ground"].data.force_matrix_w_history[:, 0, 0, 0, 2] = 100.
    assert not module.WalkingRuntime(env).ground_support()[1].any()


def test_stale_detailed_sample_is_removed_at_next_step():
    module = load_runtime()
    env = environment()
    env.extras["walking_diagnostics"] = {"stale": True}
    module.WalkingRuntime(env).begin_step()
    assert "walking_diagnostics" not in env.extras


def test_original_environment_hooks_capture_before_teleport_and_merge_after_reset(env_type):
    """Use the existing real RobotEnv AST fixture and its documented parent order."""
    class Recorder:
        def __init__(self, env):
            self.env = env
            self.calls = []

        def begin_step(self):
            self.calls.append("begin")

        def before_reset(self, ids):
            assert self.env.command_progress_tracker.reset_ready[ids].all()
            self.calls.append("before_reset")
            return {"ContactDiagnostics/test": 3.}, {"root": self.env.asset.data.root_pos_w[ids].clone()}

        def reset(self, ids):
            self.calls.append("reset")
            assert self.env.asset.data.root_pos_w[ids, 0].abs().min() == 100.

        def finish_step(self):
            self.calls.append("finish")

        def abort_step(self):
            self.calls.append("abort")

    env_type.__init__.__globals__["WalkingRuntime"] = Recorder
    env = env_type(Box(feet_body_names=["left", "right"], contact_source_diagnostics=True))
    env.next_reset_ids = [0]
    env.step(torch.zeros(2, 1))
    assert env.walking_runtime.calls == ["begin", "before_reset", "reset", "finish"]
    assert env.extras["log"]["manager_metric"] == 17.
    assert env.extras["log"]["ContactDiagnostics/test"] == 3.
    assert env.extras["walking_diagnostics"]["root"][0, 0] == pytest.approx(.02)
    env.next_reset_ids = []
    env.raise_before_reset = True
    with pytest.raises(RuntimeError, match="synthetic parent failure"):
        env.step(torch.zeros(2, 1))
    assert env.walking_runtime.calls[-1] == "abort"


def test_stale_profile_is_rejected_before_parent_environment_construction(env_type):
    calls = []

    def reject(cfg):
        calls.append(cfg)
        raise ValueError("construct a fresh profile with constructor arguments")

    env_type.__init__.__globals__["validate_walking_profile"] = reject
    cfg = Box(staged_evidence=True)
    with pytest.raises(ValueError, match="construct a fresh profile"):
        env_type(cfg)
    assert calls == [cfg]


@pytest.mark.parametrize("bad_evidence", [None, "horizontal", "nonfinite", "outside"])
def test_real_command_and_stage_trackers_receive_filtered_evidence_before_reset(bad_evidence):
    module = load_runtime()
    progress_type = importlib.import_module(PACKAGE + ".command_progress").CommandProgressTracker
    env = environment(contact_source_diagnostics=False, staged_evidence=True, walking_stage="shallow",
                      walking_stage_column_families=("flat",) * 6 + ("up",) * 2 + ("down",) * 2,
                      stage_min_group_episodes=1)
    # Extend the same strict data protocol to one environment in each family.
    containers = [env.asset.data, *[sensor.data for sensor in env.scene.sensors.values()]]
    seen = set()
    for data in containers:
        if id(data) in seen:
            continue
        seen.add(id(data))
        for key, value in vars(data).items():
            if isinstance(value, torch.Tensor) and value.shape[0] == 2:
                setattr(data, key, torch.cat((value, value[:1].clone()), dim=0))
    env.num_envs, env.step_dt, env.max_episode_length_s, env.max_episode_length = 3, 1., 3., 3
    env.termination_manager.terminated = torch.zeros(3, dtype=torch.bool)
    env.termination_manager.time_outs = torch.zeros(3, dtype=torch.bool)
    env.scene.terrain = Box(terrain_types=torch.tensor([0, 6, 8]), terrain_levels=torch.zeros(3, dtype=torch.long))
    env.scene.env_origins = torch.zeros(3, 3)
    env.cfg.scene = Box(terrain=Box(terrain_generator=Box(size=(10., 10.))))
    env.command_progress_tracker = progress_type(3, "cpu")
    runtime = module.WalkingRuntime(env)
    command = torch.tensor([[.2, 0., 0.]]).expand(3, 3)
    quaternion = torch.tensor([[1., 0., 0., 0.]]).expand(3, 4)
    for step in range(3):
        env.common_step_counter = step + 1
        env.command_progress_tracker.begin_step(env.asset.data.root_pos_w[:, :2], quaternion, command, 1.)
        runtime.begin_step()
        env.asset.data.root_pos_w[:, 0] += .2
        if bad_evidence == "outside":
            # No single transition exceeds the teleport threshold. Correct X
            # progress must still fail if it was collected on another tile.
            env.asset.data.root_pos_w[1, 1] += 2.
        heights = torch.tensor([0., .02 * step, -.02 * step])
        env.asset.data.body_pos_w[:, 1:, 0] = env.asset.data.root_pos_w[:, :1]
        env.asset.data.body_pos_w[:, 1:, 1] = env.asset.data.root_pos_w[:, 1:2]
        env.asset.data.body_pos_w[:, 1:, 2] = heights[:, None] + .0543
        env.scene["height_scanner"].data.ray_hits_w[..., 0] = env.asset.data.root_pos_w[:, :1]
        env.scene["height_scanner"].data.ray_hits_w[..., 1] = env.asset.data.root_pos_w[:, 1:2]
        env.scene["height_scanner"].data.ray_hits_w[..., 2] = heights[:, None]
        for name in ("left_foot_ground", "right_foot_ground"):
            forces = env.scene[name].data.force_matrix_w_history
            forces[:, 0, 0, 0] = torch.tensor([0., 0., 100.])
            if bad_evidence == "horizontal":
                forces[1, 0, 0, 0] = torch.tensor([100., 0., 0.])
            if bad_evidence == "nonfinite" and step == 0:
                forces[1, 0, 0, 0, 2] = torch.nan
        terminal = step == 2
        env.command_progress_tracker.finish_step([0, 1, 2], env.asset.data.root_pos_w[:, :2], for_reset=terminal)
        if terminal:
            env.termination_manager.time_outs[:] = True
            logs, _ = runtime.before_reset([0, 1, 2])
        else:
            runtime.finish_step()
    assert bool(logs["Stage/passed"]) is (bad_evidence is None)
    assert logs["Stage/flat_success_rate"] == 1.
    assert logs["Stage/down_success_rate"] == 1.
    assert logs["Stage/up_success_rate"] == (1. if bad_evidence is None else 0.)
    if bad_evidence is None:
        assert logs["Stage/next_index"] == 2
