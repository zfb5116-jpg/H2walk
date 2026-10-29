"""CPU synthetic checks of the real reward functions, without importing Isaac Lab."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REWARDS_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/mdp/rewards.py"
)


class SceneEntityCfg:
    """Only the resolved fields read by these reward functions."""

    def __init__(self, name="robot", joint_ids=slice(None), body_ids=slice(None)):
        self.name = name
        self.joint_ids = joint_ids
        self.body_ids = body_ids


def load_reward_functions():
    """Compile selected source AST nodes, avoiding simulator imports and setup."""
    names = {
        "track_forward_velocity",
        "track_forward_world_velocity",
        "base_lateral_velocity_l2",
        "joint_tor_limits",
        "air_time_variance_penalty",
    }
    source = ast.parse(REWARDS_PATH.read_text(encoding="utf-8"), filename=str(REWARDS_PATH))
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    namespace = {"torch": torch, "SceneEntityCfg": SceneEntityCfg}
    exec(compile(ast.fix_missing_locations(module), str(REWARDS_PATH), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


REWARDS = load_reward_functions()


def make_env(command=None, velocity=None, **data):
    if velocity is not None:
        data["root_lin_vel_b"] = torch.tensor(velocity, dtype=torch.float64)
    command_tensor = torch.tensor(command, dtype=torch.float64) if command is not None else None
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(**data))},
        command_manager=SimpleNamespace(get_command=lambda name: command_tensor),
    )


def test_forward_tracking_is_invariant_to_world_yaw():
    yaws = torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2], dtype=torch.float64)
    world_velocity = torch.stack((0.3 * yaws.cos(), 0.3 * yaws.sin(), torch.zeros_like(yaws)), dim=1)
    env = make_env(
        command=[[0.3, 0.0, 0.0]] * 4,
        velocity=[[0.3, 0.0, 0.0]] * 4,
        root_lin_vel_w=world_velocity,
    )
    score = REWARDS.track_forward_velocity(env, "base_velocity")
    torch.testing.assert_close(score, torch.ones(4, dtype=torch.float64))
    torch.testing.assert_close(REWARDS.track_forward_world_velocity(env, "base_velocity"), score)


def test_forward_reward_retains_command_gate_and_error_sensitivity():
    env = make_env(
        command=[[0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.2, 0.0]],
        velocity=[[0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.2, 0.0]],
    )
    score = REWARDS.track_forward_velocity(env, "base_velocity")
    torch.testing.assert_close(score, torch.tensor([1.0, math.exp(-0.36), 0.0, 0.0], dtype=torch.float64))


def test_lateral_tracking_is_zero_at_commanded_velocity_and_keeps_gate():
    env = make_env(
        command=[[0.3, 0.08, 0.0], [0.3, -0.08, 0.0], [0.3, 0.08, 0.0], [0.3, 0.2, 0.0]],
        velocity=[[0.3, 0.08, 0.0], [0.3, -0.08, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.0]],
    )
    result = REWARDS.base_lateral_velocity_l2(env, "base_velocity")
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, 0.0064, 0.0], dtype=torch.float64))


def test_joint_torque_limits_use_each_joint_and_each_environment():
    env = make_env(
        applied_torque=torch.tensor([[15.0, -67.0, 125.0, 53.0], [15.0, 50.0, 90.0, 40.0]]),
        joint_effort_limits=torch.tensor([[19.0, 66.0, 120.0, 54.0], [20.0, 66.0, 120.0, 54.0]]),
    )
    torch.testing.assert_close(REWARDS.joint_tor_limits(env, 0.75), torch.tensor([3.75, 0.5]))
    subset = SceneEntityCfg(joint_ids=[2, 0])
    torch.testing.assert_close(REWARDS.joint_tor_limits(env, 0.75, asset_cfg=subset), torch.tensor([1.75, 0.0]))


def test_joint_torque_limits_accept_slice_and_preserve_scalar_baseline():
    env = make_env(applied_torque=torch.tensor([[269.5, -270.5, 272.0]]))
    torch.testing.assert_close(REWARDS.joint_tor_limits(env, 0.75, effort_limit=360.0), torch.tensor([1.5]))
    torch.testing.assert_close(
        REWARDS.joint_tor_limits(env, 0.75, effort_limit=360.0, asset_cfg=SceneEntityCfg(joint_ids=slice(1, 3))),
        torch.tensor([1.5]),
    )


@pytest.mark.parametrize("soft_ratio", [0.0, -0.1, 1.01, float("nan"), float("inf")])
def test_joint_torque_limits_reject_invalid_soft_ratio(soft_ratio):
    with pytest.raises(ValueError, match="soft_ratio"):
        REWARDS.joint_tor_limits(make_env(), soft_ratio)


@pytest.mark.parametrize("effort_limit", [0.0, -1.0, float("nan"), float("inf")])
def test_joint_torque_limits_reject_invalid_scalar_limit(effort_limit):
    with pytest.raises(ValueError, match="effort_limit"):
        REWARDS.joint_tor_limits(make_env(), 0.75, effort_limit=effort_limit)


@pytest.mark.parametrize("std", [0.0, -1.0, float("nan"), float("inf")])
def test_forward_reward_rejects_invalid_std(std):
    with pytest.raises(ValueError, match="std"):
        REWARDS.track_forward_velocity(make_env(), "base_velocity", std=std)


@pytest.mark.parametrize("command_threshold", [-1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("name", ["track_forward_velocity", "base_lateral_velocity_l2"])
def test_velocity_rewards_reject_invalid_threshold(name, command_threshold):
    with pytest.raises(ValueError, match="command_threshold"):
        getattr(REWARDS, name)(make_env(), "base_velocity", command_threshold=command_threshold)


def _timing_env(air, contact, command=(.2, 0., 0.)):
    sensor = SimpleNamespace(data=SimpleNamespace(
        last_air_time=torch.tensor(air, dtype=torch.float64),
        last_contact_time=torch.tensor(contact, dtype=torch.float64),
    ))
    commands = torch.tensor([command] * len(air), dtype=torch.float64)
    return SimpleNamespace(
        scene=SimpleNamespace(sensors={"contact": sensor}),
        command_manager=SimpleNamespace(get_command=lambda name: commands),
    )


def test_bounded_timing_removes_late_first_lift_quadratic_amplification():
    env = _timing_env([[0., 0.]], [[19., 0.]])
    sensor = SceneEntityCfg("contact")
    baseline = REWARDS.air_time_variance_penalty(env, "base_velocity", sensor)
    explicit_legacy = REWARDS.air_time_variance_penalty(
        env, "base_velocity", sensor, cross_time_mode="legacy",
    )
    bounded = REWARDS.air_time_variance_penalty(
        env, "base_velocity", sensor, cross_time_mode="bounded",
    )
    torch.testing.assert_close(baseline, torch.tensor([45.1875], dtype=torch.float64))
    torch.testing.assert_close(explicit_legacy, baseline)
    # Contact variance remains .0625; only the cross-time term falls to .03125.
    torch.testing.assert_close(bounded, torch.tensor([.09375], dtype=torch.float64))


def test_bounded_timing_clips_difference_instead_of_individual_durations():
    env = _timing_env([[5., 5.], [10., 10.]], [[5.2, 5.2], [11., 11.]])
    bounded = REWARDS.air_time_variance_penalty(
        env, "base_velocity", SceneEntityCfg("contact"), cross_time_mode="bounded",
    )
    torch.testing.assert_close(bounded, torch.tensor([.01, .0625], dtype=torch.float64))


def test_bounded_timing_preserves_normal_small_differences_and_command_gate():
    env = _timing_env([[.2, .3]], [[.4, .5]])
    sensor = SceneEntityCfg("contact")
    torch.testing.assert_close(
        REWARDS.air_time_variance_penalty(env, "base_velocity", sensor),
        REWARDS.air_time_variance_penalty(env, "base_velocity", sensor, cross_time_mode="bounded"),
    )
    quiet = _timing_env([[0., 0.]], [[19., 0.]], command=(0., 0., 0.))
    torch.testing.assert_close(
        REWARDS.air_time_variance_penalty(quiet, "base_velocity", sensor, cross_time_mode="bounded"),
        torch.zeros(1, dtype=torch.float64),
    )


def test_bounded_timing_uses_the_explicit_difference_limit():
    env = _timing_env([[0., 0.]], [[19., 19.]])
    result = REWARDS.air_time_variance_penalty(
        env, "base_velocity", SceneEntityCfg("contact"),
        cross_time_mode="bounded", max_time_difference=.3,
    )
    # Equal per-foot timers give zero inter-foot variance: cross term only.
    torch.testing.assert_close(result, torch.tensor([.0225], dtype=torch.float64))


def test_bounded_timing_preserves_sensor_subsets_and_difference_sign_symmetry():
    env = _timing_env(
        [[0., 100., 0.], [19., 100., 19.]],
        [[19., 0., 19.], [0., 0., 0.]],
    )
    result = REWARDS.air_time_variance_penalty(
        env, "base_velocity", SceneEntityCfg("contact", body_ids=[2, 0]), cross_time_mode="bounded",
    )
    torch.testing.assert_close(result, torch.full((2,), .0625, dtype=torch.float64))


@pytest.mark.parametrize("mode", [None, "clip", True, [], {}])
def test_timing_rejects_unknown_modes(mode):
    with pytest.raises(ValueError, match="cross_time_mode"):
        REWARDS.air_time_variance_penalty(
            _timing_env([[0., 0.]], [[0., 0.]]), "base_velocity", SceneEntityCfg("contact"),
            cross_time_mode=mode,
        )


@pytest.mark.parametrize("limit", [0., -.5, float("nan"), float("inf"), True, "0.5", None])
@pytest.mark.parametrize("mode", ["legacy", "bounded"])
def test_timing_rejects_invalid_difference_limit(limit, mode):
    with pytest.raises(ValueError, match="max_time_difference"):
        REWARDS.air_time_variance_penalty(
            _timing_env([[0., 0.]], [[0., 0.]]), "base_velocity", SceneEntityCfg("contact"),
            cross_time_mode=mode, max_time_difference=limit,
        )
