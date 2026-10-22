"""CPU tests of the production stair reward, with plain tensor data containers.

AST extraction excludes simulator imports only. No Isaac environment, policy,
checkpoint or optimizer is constructed.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"


def _quat_inverse(q, v):
    xyz = -q[..., 1:]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def _yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = torch.zeros_like(q)
    out[..., 0], out[..., 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return out


def _load_reward():
    tree = ast.parse((TASK / "mdp/rewards.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "feet_stair_clearance")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    scope = {"torch": torch, "quat_apply_inverse": _quat_inverse, "yaw_quat": _yaw_quat}
    exec(compile(ast.fix_missing_locations(module), "production_stair_reward", "exec"), scope)
    return scope[function.name]


reward = _load_reward()


class Scene(dict):
    @property
    def sensors(self):
        return self


def _env(near=0.08, far=0.14, foot_z=0.115, right_near=None, shift=0.0, yaw=0.0):
    points = []
    for x in (-0.10, -0.05, 0.0, 0.10, 0.15, 0.30, 0.35):
        for y in (-0.16, -0.12, -0.08, 0.08, 0.12, 0.16):
            height = 0.0 if x < 0.08 else (near if x < 0.25 else far)
            if right_near is not None and y < 0 and 0.08 <= x < 0.25:
                height = right_near
            points.append([x, y, height])
    hits = torch.tensor([points])
    feet = torch.tensor([[[0.0, 0.12, foot_z], [0.0, -0.12, foot_z]]])
    rotation = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    hits[..., :2] = hits[..., :2] @ rotation.T
    feet[..., :2] = feet[..., :2] @ rotation.T
    hits[..., 2] += shift
    feet[..., 2] += shift
    data = SimpleNamespace(
        root_pos_w=torch.tensor([[0.0, 0.0, 1.0 + shift]]),
        root_quat_w=torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]),
        body_pos_w=feet,
    )
    command = torch.tensor([[0.4, 0.0, 0.0]])
    forces = torch.zeros(1, 3, 2, 3)
    scene = Scene(
        height_scanner=SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits)),
        contact_forces=SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces)),
    )
    return SimpleNamespace(
        scene=scene, asset=SimpleNamespace(data=data), feet_id_b=[0, 1],
        command=command, command_manager=SimpleNamespace(get_command=lambda name: command),
    )


def _score(env, mode="nearest", **params):
    return reward(
        env, command_name="base_velocity",
        foot_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 1]),
        terrain_sensor_cfg=SimpleNamespace(name="height_scanner"),
        clearance=0.035, min_rise=0.02, max_rise=0.16,
        height_std=0.08, target_mode=mode, **params,
    )


def test_nearest_target_ignores_distant_higher_steps():
    near = _score(_env(far=0.14))
    far = _score(_env(far=0.24))
    torch.testing.assert_close(near, torch.tensor([0.5]))
    torch.testing.assert_close(far, near)
    assert _score(_env(far=0.24), "highest").item() < _score(_env(far=0.14), "highest").item()


@pytest.mark.parametrize("yaw", [math.pi / 2, -math.pi / 2, math.pi])
def test_stair_reward_invariant_to_world_yaw(yaw):
    torch.testing.assert_close(_score(_env(yaw=yaw)), _score(_env()), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["nearest", "highest"])
@pytest.mark.parametrize("shift", [-4.0, 7.0])
def test_stair_reward_invariant_to_world_height(mode, shift):
    torch.testing.assert_close(_score(_env(shift=shift), mode), _score(_env(), mode), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("mode", ["nearest", "highest"])
@pytest.mark.parametrize("missing", ["ground", "front", "all"])
def test_missing_hits_never_create_a_floor_or_reward(mode, missing):
    env = _env(shift=5.0)
    hits = env.scene["height_scanner"].data.ray_hits_w
    mask = hits[..., 0] < 0.02 if missing == "ground" else hits[..., 0] > 0.08
    if missing == "all":
        mask[:] = True
    hits[mask] = torch.tensor([float("nan"), float("inf"), float("nan")])
    torch.testing.assert_close(_score(env, mode), torch.zeros(1))


def test_minimum_two_centimeter_step_receives_nonzero_shaping():
    torch.testing.assert_close(_score(_env(near=0.02, far=0.02, foot_z=0.055)), torch.tensor([0.125]))


def test_too_high_near_obstacle_cannot_be_skipped_for_far_lower_step():
    torch.testing.assert_close(_score(_env(near=0.20, far=0.08)), torch.zeros(1))


def test_each_foot_uses_its_own_lane():
    env = _env(right_near=0.14)
    # The right foot is in contact. Its high target must not affect the left.
    env.scene["contact_forces"].data.net_forces_w_history[:, :, 1, 2] = 10
    torch.testing.assert_close(_score(env), torch.tensor([0.5]))


def test_nonidentity_asset_indices_still_match_ordered_contacts():
    env = _env(right_near=0.14)
    env.asset.data.body_pos_w = env.asset.data.body_pos_w.flip(1)
    env.feet_id_b = [1, 0]
    env.scene["contact_forces"].data.net_forces_w_history[:, :, 1, 2] = 10
    torch.testing.assert_close(_score(env), torch.tensor([0.5]))


def test_robot_resolves_feet_in_the_same_requested_order_as_contacts():
    tree = ast.parse((TASK / "robot_env.py").read_text(encoding="utf-8"))
    robot = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RobotEnv")
    initializer = next(n for n in robot.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")

    class Asset:
        root_physx_view = SimpleNamespace(get_masses=lambda: torch.ones(1, 3))

        def find_bodies(self, names, preserve_order=False):
            assert names == ["left", "right"]
            assert preserve_order, "Asset ordering must match the configured contact ordering"
            return [2, 0], names

    class Parent:
        def __init__(self, *args, **kwargs):
            self.scene = {"robot": Asset()}
            self.device, self.num_envs = "cpu", 1

    extracted = ast.ClassDef(name="RobotEnv", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[initializer], decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {"torch": torch, "Parent": Parent}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, extracted], type_ignores=[])), "production_robot_initializer", "exec"), scope)
    env = scope["RobotEnv"](SimpleNamespace(feet_body_names=["left", "right"]))
    assert env.feet_id_b == [2, 0]
    assert env.feet_pos_b.shape == (1, 2, 3)


@pytest.mark.parametrize("command", [[-0.4, 0.0, 0.0], [0.4, 0.3, 0.0], [0.4, 0.0, 0.5], [0.0, 0.0, 0.0]])
def test_nearest_experiment_excludes_unsupported_commands(command):
    env = _env()
    env.command[:] = torch.tensor(command)
    torch.testing.assert_close(_score(env), torch.zeros(1))


def test_flat_ground_and_all_stance_have_no_stair_reward():
    torch.testing.assert_close(_score(_env(near=0.0, far=0.0)), torch.zeros(1))
    env = _env()
    env.scene["contact_forces"].data.net_forces_w_history[..., 2] = 10
    torch.testing.assert_close(_score(env), torch.zeros(1))


def test_mode_and_parameter_errors_are_explicit():
    with pytest.raises(ValueError, match="target_mode"):
        _score(_env(), mode="typo")
    with pytest.raises(ValueError, match="min_rise"):
        reward(_env(), "base_velocity", None, None, min_rise=0.2, max_rise=0.1)


@pytest.mark.parametrize("name", ["clearance", "min_rise", "max_rise", "height_std"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_parameters_are_rejected(name, value):
    with pytest.raises(ValueError):
        reward(_env(), "base_velocity", None, None, **{name: value})
