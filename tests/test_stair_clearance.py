"""CPU tests of the production stair reward, with plain tensor data containers.

AST extraction excludes simulator imports only. No Isaac environment, policy,
checkpoint or optimizer is constructed.
"""

from __future__ import annotations

import ast
import importlib.util
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
    spec = importlib.util.spec_from_file_location("h2_foot_geometry_cpu_tests", TASK / "foot_geometry.py")
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    scope["lowest_sole_height_world"] = geometry.lowest_sole_height_world
    exec(compile(ast.fix_missing_locations(module), "production_stair_reward", "exec"), scope)
    return scope[function.name]


reward = _load_reward()


@pytest.fixture(scope="module")
def geometry():
    spec = importlib.util.spec_from_file_location("h2_sole_surface_cpu_tests", TASK / "foot_geometry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _sole_env(**params):
    env = _env(**params)
    env.asset.data.body_quat_w = env.asset.data.root_quat_w[:, None, :].repeat(1, 2, 1)
    return env


def test_sole_reference_peak_accounts_for_link_to_collision_surface_offset():
    # USD sole surface is 5.43 cm below ankle_pitch_link in a level foot.
    # The old link-height peak .115 leaves that surface 1.93 cm below an 8 cm step.
    torch.testing.assert_close(_score(_env(foot_z=0.115)), torch.tensor([0.5]))
    old_peak = _score(_sole_env(foot_z=0.115), height_reference="sole")
    actual_clearance_peak = _score(_sole_env(foot_z=0.1693), height_reference="sole")
    assert old_peak.item() < actual_clearance_peak.item()
    torch.testing.assert_close(actual_clearance_peak, torch.tensor([0.5]))


def _rpy_quat(roll=0.0, pitch=0.0, yaw=0.0, *, dtype=torch.float32):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return torch.tensor([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], dtype=dtype)


def _rotated_points(q, points):
    # Independent full 3D quaternion-vector product, as opposed to the helper's Z row.
    xyz = q[..., 1:]
    cross = 2 * torch.cross(xyz, points, dim=-1)
    return points + q[..., :1] * cross + torch.cross(xyz, cross, dim=-1)


@pytest.mark.parametrize("roll,pitch,yaw", [(0.3, 0.0, 0.0), (-0.3, 0.0, 0.0),
                                          (0.0, 0.3, 0.0), (0.0, -0.3, 0.0),
                                          (0.0, 0.0, 1.4), (0.2, -0.3, 1.1)])
def test_sole_height_follows_foot_orientation_not_root_yaw(geometry, roll, pitch, yaw):
    env = _sole_env(foot_z=0.2)
    env.asset.data.body_quat_w[:] = _rpy_quat(roll, pitch, yaw)
    points = torch.tensor(geometry.H2_SOLE_SURFACE_POINTS_B)
    q = env.asset.data.body_quat_w.unsqueeze(2).expand(-1, -1, points.shape[1], -1)
    rotated = _rotated_points(q, points.unsqueeze(0).expand(1, -1, -1, -1))
    sole_z = 0.2 + rotated[..., 2].amin(dim=-1)
    expected = (0.5 * torch.exp(-torch.square((sole_z - 0.115) / 0.08))).mean(dim=1)
    torch.testing.assert_close(_score(env, height_reference="sole"), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sole_helper_preserves_dtype_device_and_normalizes_quaternion(geometry, dtype):
    positions = torch.tensor([[[0.0, 0.0, 0.2], [0.0, 0.0, 0.3]]], dtype=dtype)
    quats = _rpy_quat(0.2, -0.1, 0.7, dtype=dtype).repeat(1, 2, 1)
    result = geometry.lowest_sole_height_world(positions, quats)
    torch.testing.assert_close(geometry.lowest_sole_height_world(positions, -3.0 * quats), result)
    assert result.dtype == dtype and result.device == positions.device and result.shape == (1, 2)


def test_h2_sole_geometry_is_left_right_mirrored_and_in_link_metres(geometry):
    left, right = torch.tensor(geometry.H2_SOLE_SURFACE_POINTS_B, dtype=torch.float64)
    assert left.shape == right.shape == (57, 3)
    torch.testing.assert_close(right, left * torch.tensor([1.0, -1.0, 1.0]))
    torch.testing.assert_close(left[:, 2], torch.full((57,), -0.05429999902844429, dtype=torch.float64))
    assert left[:, 0].min().item() == pytest.approx(-0.07, abs=1e-7)
    assert left[:, 0].max().item() == pytest.approx(0.15, abs=1e-7)


def test_sole_helper_mirrored_roll_and_multiple_environments_are_independent(geometry):
    pos = torch.zeros(2, 2, 3, dtype=torch.float64)
    pos[..., 2] = 0.2
    quat = torch.stack((_rpy_quat(roll=0.3, dtype=torch.float64),
                        _rpy_quat(roll=-0.3, dtype=torch.float64))).unsqueeze(0).repeat(2, 1, 1)
    heights = geometry.lowest_sole_height_world(pos, quat)
    torch.testing.assert_close(heights[:, 0], heights[:, 1])
    pos[1, 0, 2] += 0.1
    changed = geometry.lowest_sole_height_world(pos, quat)
    expected = heights.clone()
    expected[1, 0] += 0.1
    torch.testing.assert_close(changed, expected)


@pytest.mark.parametrize("which", ["position", "quaternion"])
def test_sole_helper_marks_only_invalid_pose_as_nan(geometry, which):
    positions = torch.zeros(2, 2, 3)
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(2, 2, 1)
    if which == "position":
        positions[0, 0, 1] = float("nan")
    else:
        quats[0, 0] = 0.0
    result = geometry.lowest_sole_height_world(positions, quats)
    assert torch.isnan(result).tolist() == [[True, False], [False, False]]


@pytest.mark.parametrize("bad_pos,bad_quat", [
    (torch.zeros(2, 3), torch.zeros(2, 4)),
    (torch.zeros(1, 2, 3), torch.zeros(1, 3, 4)),
    (torch.zeros(1, 2, 3, dtype=torch.int64), torch.zeros(1, 2, 4)),
    (torch.zeros(1, 2, 3), torch.zeros(1, 2, 4, dtype=torch.float64)),
])
def test_sole_helper_rejects_incompatible_pose_inputs(geometry, bad_pos, bad_quat):
    with pytest.raises(ValueError, match="body"):
        geometry.lowest_sole_height_world(bad_pos, bad_quat)


@pytest.mark.parametrize("shift,yaw", [(-4.0, 0.0), (7.0, 0.0), (0.0, math.pi / 2), (3.0, -math.pi / 2)])
@pytest.mark.parametrize("mode", ["nearest", "highest"])
def test_sole_reference_keeps_world_height_and_yaw_invariance(mode, shift, yaw):
    reference = _score(_sole_env(foot_z=0.1693), mode, height_reference="sole")
    transformed = _score(_sole_env(foot_z=0.1693, shift=shift, yaw=yaw), mode, height_reference="sole")
    # World-height subtraction at a 7 m offset has float32 quantization.
    torch.testing.assert_close(transformed, reference, atol=1e-5, rtol=1e-5)


def test_sole_pose_uses_same_nonidentity_asset_indices_as_foot_positions():
    env = _sole_env(foot_z=0.1693)
    env.asset.data.body_quat_w[:, 1] = _rpy_quat(pitch=0.5)
    # Isolate the level left foot so a misindexed tilted pose would lower the score.
    env.scene["contact_forces"].data.net_forces_w_history[:, :, 1, 2] = 10
    expected = _score(env, height_reference="sole")
    env.asset.data.body_pos_w = env.asset.data.body_pos_w.flip(1)
    env.asset.data.body_quat_w = env.asset.data.body_quat_w.flip(1)
    env.feet_id_b = [1, 0]
    torch.testing.assert_close(_score(env, height_reference="sole"), expected)
    torch.testing.assert_close(expected, torch.tensor([0.5]))


@pytest.mark.parametrize("bad_quat", [[0.0, 0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0, 0.0],
                                     [1.0, float("inf"), 0.0, 0.0]])
def test_invalid_sole_quaternion_excludes_only_that_foot(bad_quat):
    env = _sole_env(foot_z=0.1693)
    env.asset.data.body_quat_w[:, 0] = torch.tensor(bad_quat)
    torch.testing.assert_close(_score(env, height_reference="sole"), torch.tensor([0.25]))
    env.asset.data.body_quat_w[:, 1] = torch.tensor(bad_quat)
    torch.testing.assert_close(_score(env, height_reference="sole"), torch.zeros(1))


def test_sole_requires_foot_pose_but_link_baseline_never_reads_it():
    env = _env()
    torch.testing.assert_close(_score(env), _score(env, height_reference="link"))
    with pytest.raises(ValueError, match="body_quat_w"):
        _score(env, height_reference="sole")


def test_custom_sole_points_replace_h2_geometry_and_must_match_feet():
    env = _sole_env(foot_z=0.115)
    points = torch.zeros(2, 1, 3)  # A surface coincident with the link reproduces the link baseline.
    torch.testing.assert_close(_score(env, height_reference="sole", sole_points_b=points), _score(env))
    with pytest.raises(ValueError, match="sole_points_b"):
        _score(env, height_reference="sole", sole_points_b=torch.zeros(1, 1, 3))
    with pytest.raises(ValueError, match="sole_points_b"):
        _score(env, height_reference="link", sole_points_b=points)


@pytest.mark.parametrize("bad_points", [torch.zeros(2, 0, 3), torch.zeros(2, 3),
                                       torch.full((2, 1, 3), float("nan")),
                                       torch.full((2, 1, 3), float("inf"))])
def test_invalid_custom_sole_surface_is_explicit(bad_points):
    with pytest.raises(ValueError, match="sole_points_b"):
        _score(_sole_env(), height_reference="sole", sole_points_b=bad_points)


def test_unknown_height_reference_is_explicit():
    with pytest.raises(ValueError, match="height_reference"):
        _score(_env(), height_reference="typo")


@pytest.mark.parametrize("mode", ["nearest", "highest"])
def test_sole_reference_has_no_reward_for_missing_rays_or_all_stance(mode):
    env = _sole_env(foot_z=0.1693)
    env.scene["height_scanner"].data.ray_hits_w[:] = float("nan")
    torch.testing.assert_close(_score(env, mode, height_reference="sole"), torch.zeros(1))
    env = _sole_env(foot_z=0.1693)
    env.scene["contact_forces"].data.net_forces_w_history[..., 2] = 10
    torch.testing.assert_close(_score(env, mode, height_reference="sole"), torch.zeros(1))


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
    spec = importlib.util.spec_from_file_location("actual_command_progress", TASK / "command_progress.py")
    progress = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(progress)
    scope = {"torch": torch, "Parent": Parent, "CommandProgressTracker": progress.CommandProgressTracker}
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
