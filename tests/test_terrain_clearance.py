"""CPU checks of terrain-relative walking rewards at their production call sites.

No simulator, policy, checkpoint, pickle or optimizer is constructed. Function
ASTs omit simulator imports only; shared tensor helpers execute unchanged.
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


def _load_module(name):
    spec = importlib.util.spec_from_file_location(f"h2_{name}_terrain_tests", TASK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_rewards():
    names = {"feet_swing_height", "feet_clearance_l2", "feet_stair_clearance", "_terrain_clearance_shortfall"}
    tree = ast.parse((TASK / "mdp/rewards.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    scope = {
        "torch": torch, "lowest_sole_height_world": _load_module("foot_geometry").lowest_sole_height_world,
        "quat_apply_inverse": _quat_inverse, "yaw_quat": _yaw_quat,
    }
    if (TASK / "terrain_clearance.py").exists():
        scope["local_ground_height"] = _load_module("terrain_clearance").local_ground_height
    exec(compile(ast.fix_missing_locations(module), "production_walking_rewards", "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names if name in scope})


def _quat_inverse(q, vector):
    axis = -q[..., 1:]
    twice_cross = 2 * torch.cross(axis, vector, dim=-1)
    return vector + q[..., :1] * twice_cross + torch.cross(axis, twice_cross, dim=-1)


def _yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    result = torch.zeros_like(q)
    result[..., 0], result[..., 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return result


REWARDS = _load_rewards()


class Scene(dict):
    @property
    def sensors(self):
        return self


def _env(clearance=0.03, ground=(0.0, 0.0), foot_y=(0.25, -0.25), shift=0.0):
    if isinstance(clearance, (int, float)):
        clearance = (clearance, clearance)
    points = []
    for y, z in zip(foot_y, ground):
        for dx in (-0.10, -0.05, 0.0, 0.05, 0.10):
            for dy in (-0.10, -0.05, 0.0, 0.05, 0.10):
                points.append((dx, y + dy, z + shift))
    feet = torch.tensor([[[0.0, y, z + 0.05429999902844429 + c + shift]
                          for y, z, c in zip(foot_y, ground, clearance)]], dtype=torch.float64)
    root = torch.tensor([[0.0, 0.0, 0.92 + shift]], dtype=torch.float64)
    command = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64)
    forces = torch.zeros(1, 3, 2, 3, dtype=torch.float64)
    asset = SimpleNamespace(data=SimpleNamespace(
        body_pos_w=feet,
        body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2], dtype=torch.float64),
        root_pos_w=root,
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
    ))
    env = SimpleNamespace(
        scene=Scene(height_scanner=SimpleNamespace(data=SimpleNamespace(
            ray_hits_w=torch.tensor([points], dtype=torch.float64))),
            contact_forces=SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))),
        asset=asset, feet_id_b=[0, 1], command=command,
        command_manager=SimpleNamespace(get_command=lambda name: command),
        feet_pos_b=feet - root[:, None],
        feet_vel_b=torch.tensor([[[0.3, 0.0, 0.0]] * 2], dtype=torch.float64),
    )
    env.update_feet_state = lambda: None
    return env


def _call(name, env, **params):
    return getattr(REWARDS, name)(env, "base_velocity", SimpleNamespace(name="contact_forces", body_ids=[0, 1]), **params)


def test_flat_walking_saturates_without_rewarding_extra_high_steps():
    scores = [_call("feet_swing_height", _env(height), height_mode="terrain") for height in (0.03, 0.04, 0.08, 0.15)]
    for score in scores:
        torch.testing.assert_close(score, torch.ones(1, dtype=torch.float64))


def test_terrain_penalty_has_zero_gradient_at_the_stair_clearance_target():
    env = _env(0.035, ground=(0.08, 0.08))
    env.asset.data.body_pos_w.requires_grad_()
    score = _call("feet_clearance_l2", env, height_mode="terrain")
    gradient = torch.autograd.grad(score.sum(), env.asset.data.body_pos_w)[0]
    torch.testing.assert_close(score, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("name", ["feet_swing_height", "feet_clearance_l2"])
def test_shortfall_improves_until_the_minimum_clearance(name):
    scores = [_call(name, _env(height), height_mode="terrain").item() for height in (-0.01, 0.0, 0.02, 0.03, 0.08)]
    if name == "feet_swing_height":
        assert scores[0] < scores[1] < scores[2] < scores[3] == scores[4]
    else:
        assert scores[0] > scores[1] > scores[2] > scores[3] == scores[4] == 0.0


def _stair_env(link_z):
    env = _env(0.035)
    env.asset.data.body_pos_w[..., 2] = link_z
    points = [[x, y, 0.0 if x < 0.08 else (0.08 if x < 0.25 else 0.14)]
              for x in (-0.10, -0.05, 0.0, 0.10, 0.15, 0.30, 0.35)
              for y in (-0.29, -0.25, -0.21, 0.21, 0.25, 0.29)]
    env.scene["height_scanner"].data.ray_hits_w = torch.tensor([points], dtype=torch.float64)
    return env


def _stair_score(env):
    return REWARDS.feet_stair_clearance(
        env, "base_velocity", SimpleNamespace(name="contact_forces", body_ids=[0, 1]),
        SimpleNamespace(name="height_scanner"), clearance=0.035, min_rise=0.02,
        max_rise=0.16, height_std=0.08, target_mode="nearest", height_reference="sole",
    )


def test_common_walking_terms_do_not_move_the_existing_stair_reward_peak():
    heights = [0.1592999990284443, 0.1692999990284443, 0.1792999990284443]
    original, combined = [], []
    for height in heights:
        env = _stair_env(height)
        stair = _stair_score(env)
        swing = _call("feet_swing_height", env, height_mode="terrain")
        penalty = _call("feet_clearance_l2", env, height_mode="terrain")
        torch.testing.assert_close(swing, torch.ones_like(swing))
        torch.testing.assert_close(penalty, torch.zeros_like(penalty))
        original.append(stair.item())
        combined.append((2.5 * stair + 0.75 * swing - 4.0 * penalty).item())
    assert original.index(max(original)) == combined.index(max(combined)) == 1


@pytest.mark.parametrize("name", ["feet_swing_height", "feet_clearance_l2"])
def test_terrain_height_component_does_not_use_root_height_or_orientation(name):
    env = _env(0.02)
    before = _call(name, env, height_mode="terrain")
    env.asset.data.root_pos_w[:, 2] += 0.4
    env.asset.data.root_quat_w[:] = torch.tensor([[math.cos(0.4), math.sin(0.4), 0.0, 0.0]])
    env.feet_pos_b[..., 2] -= 0.4  # Legacy body-frame height changes; foot world geometry stays fixed.
    torch.testing.assert_close(_call(name, env, height_mode="terrain"), before)


@pytest.mark.parametrize("name", ["feet_swing_height", "feet_clearance_l2"])
@pytest.mark.parametrize("shift", [-4.0, 7.0])
def test_world_height_translation_changes_neither_terrain_term(name, shift):
    torch.testing.assert_close(
        _call(name, _env(0.02, shift=shift), height_mode="terrain"),
        _call(name, _env(0.02), height_mode="terrain"), atol=1e-12, rtol=1e-12,
    )


@pytest.mark.parametrize("name", ["feet_swing_height", "feet_clearance_l2"])
@pytest.mark.parametrize("yaw", [math.pi / 2, -math.pi / 2, math.pi])
def test_world_yaw_rotation_changes_neither_terrain_term(name, yaw):
    env = _env(0.02)
    expected = _call(name, env, height_mode="terrain")
    rotation = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]], dtype=torch.float64)
    env.asset.data.body_pos_w[..., :2] = env.asset.data.body_pos_w[..., :2] @ rotation.T
    hits = env.scene["height_scanner"].data.ray_hits_w
    hits[..., :2] = hits[..., :2] @ rotation.T
    env.asset.data.body_quat_w[:] = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    torch.testing.assert_close(_call(name, env, height_mode="terrain"), expected, atol=1e-12, rtol=1e-12)


def test_each_foot_queries_its_own_terrain_and_keeps_resolved_asset_order():
    env = _env((0.02, 0.05), ground=(0.08, -0.2))
    before = [_call(name, env, height_mode="terrain") for name in ("feet_swing_height", "feet_clearance_l2")]
    torch.testing.assert_close(before[0], torch.tensor([(math.exp(-0.01) + 1) / 2], dtype=torch.float64))
    torch.testing.assert_close(before[1], torch.tensor([0.0001 * 0.3], dtype=torch.float64))
    positions, quaternions = env.asset.data.body_pos_w, env.asset.data.body_quat_w
    env.asset.data.body_pos_w = torch.stack((positions[:, 1], torch.zeros_like(positions[:, 0]), positions[:, 0]), dim=1)
    env.asset.data.body_quat_w = torch.stack((quaternions[:, 1], quaternions[:, 0], quaternions[:, 0]), dim=1)
    env.feet_id_b = [2, 0]
    for name, expected in zip(("feet_swing_height", "feet_clearance_l2"), before):
        torch.testing.assert_close(_call(name, env, height_mode="terrain"), expected)


@pytest.mark.parametrize("name", ["feet_swing_height", "feet_clearance_l2"])
def test_downstairs_does_not_query_the_lower_surface_far_ahead(name):
    env = _env(0.02)
    expected = _call(name, env, height_mode="terrain")
    lower = torch.tensor([[[x, y, -0.2] for x in (0.3, 0.35, 0.4) for y in (-0.25, 0.25)]], dtype=torch.float64)
    env.scene["height_scanner"].data.ray_hits_w = torch.cat((env.scene["height_scanner"].data.ray_hits_w, lower), dim=1)
    torch.testing.assert_close(_call(name, env, height_mode="terrain"), expected)
    # The local reference can descend after reaching the next tread, without a
    # preference for lowering the foot while it is still over the upper tread.
    torch.testing.assert_close(_call(name, _env(0.02, ground=(-0.2, -0.2)), height_mode="terrain"), expected)


@pytest.mark.parametrize("command,swing_active,penalty_active", [
    ([0.3, 0.0, 0.0], True, True), ([-0.3, 0.0, 0.0], True, True),
    ([0.0, 0.3, 0.0], True, True), ([0.0, 0.0, 0.3], False, True),
    ([0.1, 0.0, 0.0], False, True), ([0.0, 0.0, 0.0], False, False),
])
def test_original_movement_gates_are_retained(command, swing_active, penalty_active):
    env = _env(0.0)
    env.command[:] = torch.tensor(command)
    assert bool((_call("feet_swing_height", env, height_mode="terrain") > 0).item()) == swing_active
    assert bool((_call("feet_clearance_l2", env, height_mode="terrain") > 0).item()) == penalty_active


@pytest.mark.parametrize("name", ["feet_swing_height", "feet_clearance_l2"])
@pytest.mark.parametrize("missing", ["nan", "inf", "empty", "out_of_range"])
def test_missing_ground_never_creates_a_floor_or_reward(name, missing):
    env = _env(0.02, shift=5.0)
    sensor = env.scene["height_scanner"]
    if missing == "empty":
        sensor.data.ray_hits_w = sensor.data.ray_hits_w[:, :0]
    elif missing == "out_of_range":
        sensor.data.ray_hits_w[..., :2] += 100.0
    else:
        sensor.data.ray_hits_w[:] = float(missing)
    torch.testing.assert_close(_call(name, env, height_mode="terrain"), torch.zeros(1, dtype=torch.float64))


@pytest.mark.parametrize("name", ["feet_swing_height", "feet_clearance_l2"])
def test_bad_pose_invalidates_only_that_foot_and_all_stance_is_zero(name):
    env = _env(0.0)
    before = _call(name, env, height_mode="terrain")
    env.asset.data.body_quat_w[:, 0] = 0.0
    torch.testing.assert_close(_call(name, env, height_mode="terrain"), before / 2)
    env.asset.data.body_pos_w[:, 1, 2] = float("nan")
    env.feet_vel_b[:, 1] = float("nan")
    torch.testing.assert_close(_call(name, env, height_mode="terrain"), torch.zeros(1, dtype=torch.float64))
    env = _env(0.0)
    env.scene["contact_forces"].data.net_forces_w_history[..., 2] = 10.0
    torch.testing.assert_close(_call(name, env, height_mode="terrain"), torch.zeros(1, dtype=torch.float64))


@pytest.mark.parametrize("invalid_xy", [[float("nan"), 0.2], [0.2, float("inf")],
                                       [float("inf"), -float("inf")]])
def test_invalid_swing_velocity_is_isolated_before_the_fractional_norm_backward(invalid_xy):
    env = _env(0.0)
    env.feet_vel_b = torch.tensor([[[0.3, 0.1, 0.0], [*invalid_xy, 0.0]]],
                                 dtype=torch.float64, requires_grad=True)
    score = _call("feet_clearance_l2", env, height_mode="terrain")
    gradient = torch.autograd.grad(score.sum(), env.feet_vel_b)[0]
    root_x, root_y = math.sqrt(0.3), math.sqrt(0.1)
    expected_score = 0.03 ** 2 * (root_x + root_y) ** 2
    expected_gradient = torch.tensor([[[0.03 ** 2 * (root_x + root_y) / root_x,
                                        0.03 ** 2 * (root_x + root_y) / root_y, 0.0],
                                       [0.0, 0.0, 0.0]]], dtype=torch.float64)
    torch.testing.assert_close(score, torch.tensor([expected_score], dtype=torch.float64))
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("name", ["feet_swing_height", "feet_clearance_l2"])
def test_missing_sensor_or_pose_is_an_explicit_configuration_error(name):
    env = _env()
    del env.asset.data.body_quat_w
    with pytest.raises(ValueError, match="body_quat_w"):
        _call(name, env, height_mode="terrain")
    env = _env()
    del env.scene["height_scanner"]
    with pytest.raises(ValueError, match="terrain sensor"):
        _call(name, env, height_mode="terrain")
    # The legacy path neither requires nor reads those new data sources.
    del env.asset.data.body_quat_w
    torch.testing.assert_close(_call(name, env), _call(name, env, height_mode="body_legacy"))


def test_legacy_paths_match_the_original_body_frame_formulas():
    env = _env((0.0, 0.1))
    env.feet_pos_b = torch.tensor([[[-0.1, 0.2, -0.80], [0.1, -0.2, -0.65]]], dtype=torch.float64)
    env.feet_vel_b = torch.tensor([[[0.2, -0.1, 0.0], [0.3, 0.4, 0.0]]], dtype=torch.float64)
    expected_swing = torch.exp(-((env.feet_pos_b[..., 2] + 0.67) / 0.10).square()).mean(dim=1)
    expected_penalty = ((env.feet_pos_b[..., 2] + 0.70).clamp(max=0).square()
                        * env.feet_vel_b[..., :2].norm(p=0.5, dim=-1)).sum(dim=1)
    torch.testing.assert_close(_call("feet_swing_height", env), expected_swing, rtol=0, atol=0)
    torch.testing.assert_close(_call("feet_clearance_l2", env, target_feet_height=-0.70), expected_penalty, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["feet_swing_height", "feet_clearance_l2"])
@pytest.mark.parametrize("parameter", ["min_clearance", "ground_query_radius"])
@pytest.mark.parametrize("value", [0.0, -0.01, float("nan"), float("inf")])
def test_invalid_terrain_parameters_are_rejected(name, parameter, value):
    with pytest.raises(ValueError, match="min_clearance|radius"):
        _call(name, _env(), height_mode="terrain", **{parameter: value})


@pytest.mark.parametrize("value", [0.0, -0.01, float("nan"), float("inf")])
def test_invalid_terrain_height_std_is_rejected(value):
    with pytest.raises(ValueError, match="height_std"):
        _call("feet_swing_height", _env(), height_mode="terrain", height_std=value)


@pytest.mark.parametrize("name", ["feet_swing_height", "feet_clearance_l2"])
def test_unknown_height_mode_is_rejected(name):
    with pytest.raises(ValueError, match="height_mode"):
        _call(name, _env(), height_mode="unknown")


def test_custom_surface_points_are_used_by_both_production_terms():
    env = _env(0.0)
    custom = torch.tensor([[[0.0, 0.0, 0.0]]] * 2, dtype=torch.float64)
    assert _call("feet_clearance_l2", env, height_mode="terrain").item() > 0
    torch.testing.assert_close(
        _call("feet_clearance_l2", env, height_mode="terrain", sole_points_b=custom),
        torch.zeros(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        _call("feet_swing_height", env, height_mode="terrain", sole_points_b=custom),
        torch.ones(1, dtype=torch.float64),
    )


@pytest.mark.parametrize("name", ["feet_swing_height", "feet_clearance_l2"])
def test_multiple_environments_do_not_share_ground_or_pose_validity(name):
    env = _env(0.02)
    for key, value in vars(env.asset.data).items():
        setattr(env.asset.data, key, value.repeat(3, *([1] * (value.ndim - 1))))
    env.feet_pos_b = env.feet_pos_b.repeat(3, 1, 1)
    env.feet_vel_b = env.feet_vel_b.repeat(3, 1, 1)
    env.command = env.command.repeat(3, 1)
    env.command_manager.get_command = lambda command_name: env.command
    scanner, sensor = env.scene["height_scanner"], env.scene["contact_forces"]
    scanner.data.ray_hits_w = scanner.data.ray_hits_w.repeat(3, 1, 1)
    sensor.data.net_forces_w_history = sensor.data.net_forces_w_history.repeat(3, 1, 1, 1)
    scanner.data.ray_hits_w[1] = float("nan")
    env.asset.data.body_pos_w[2, :, 2] += 0.1
    result = _call(name, env, height_mode="terrain")
    expected = [math.exp(-0.01), 0.0, 1.0] if name == "feet_swing_height" else [0.00006, 0.0, 0.0]
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["feet_swing_height", "feet_clearance_l2"])
def test_terrain_rewards_preserve_float_dtype_and_device(name, dtype):
    env = _env(0.02)
    for key, value in vars(env.asset.data).items():
        setattr(env.asset.data, key, value.to(dtype))
    env.feet_pos_b = env.feet_pos_b.to(dtype)
    env.feet_vel_b = env.feet_vel_b.to(dtype)
    env.command = env.command.to(dtype)
    env.command_manager.get_command = lambda command_name: env.command
    env.scene["height_scanner"].data.ray_hits_w = env.scene["height_scanner"].data.ray_hits_w.to(dtype)
    env.scene["contact_forces"].data.net_forces_w_history = env.scene["contact_forces"].data.net_forces_w_history.to(dtype)
    result = _call(name, env, height_mode="terrain")
    assert result.dtype == dtype and result.device == env.asset.data.body_pos_w.device
    assert torch.isfinite(result).all()


def test_local_ground_query_matches_a_separate_loop_reference_without_mutation():
    query = _load_module("terrain_clearance").local_ground_height
    env = _env(0.02, ground=(0.2, -0.3))
    rays = env.scene["height_scanner"].data.ray_hits_w
    feet = env.asset.data.body_pos_w
    rays_before, feet_before = rays.clone(), feet.clone()
    expected = []
    for foot in feet[0].tolist():
        values = sorted(ray[2] for ray in rays[0].tolist()
                        if math.hypot(ray[0] - foot[0], ray[1] - foot[1]) <= 0.1)
        expected.append(values[(len(values) - 1) // 2])
    torch.testing.assert_close(query(rays, feet), torch.tensor([expected], dtype=torch.float64))
    torch.testing.assert_close(rays, rays_before, rtol=0, atol=0)
    torch.testing.assert_close(feet, feet_before, rtol=0, atol=0)


@pytest.mark.parametrize("rays,feet", [
    (torch.zeros(1, 2, 2), torch.zeros(1, 2, 3)),
    (torch.zeros(1, 2, 3), torch.zeros(1, 2, 4)),
    (torch.zeros(1, 2, 3), torch.zeros(2, 2, 3)),
    (torch.zeros(1, 2, 3, dtype=torch.int64), torch.zeros(1, 2, 3)),
    (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3, dtype=torch.float64)),
])
def test_ground_query_shape_and_dtype_errors_are_explicit(rays, feet):
    with pytest.raises(ValueError):
        _load_module("terrain_clearance").local_ground_height(rays, feet)


def test_reward_imports_the_shared_ground_query():
    tree = ast.parse((TASK / "mdp/rewards.py").read_text(encoding="utf-8"))
    assert any(isinstance(node, ast.ImportFrom) and node.level == 2 and node.module == "terrain_clearance"
               and any(alias.name == "local_ground_height" for alias in node.names) for node in tree.body)
