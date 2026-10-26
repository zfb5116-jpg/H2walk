"""CPU checks of the production command-relative tracking reward and baselines.

AST extraction omits simulator imports only. No environment, checkpoint,
training loop, optimizer or pickle data is constructed or loaded.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"


class SceneEntityCfg:
    def __init__(self, name="robot"):
        self.name = name


def _load_rewards():
    names = {"track_command_lin_vel_xy_exp", "track_forward_velocity", "track_forward_world_velocity"}
    tree = ast.parse((TASK / "mdp/rewards.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    namespace = {"torch": torch, "SceneEntityCfg": SceneEntityCfg}
    exec(compile(ast.fix_missing_locations(module), "production_command_tracking", "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names if name in namespace})


REWARDS = _load_rewards()


def _env(command, velocity, dtype=torch.float64):
    command = torch.as_tensor(command, dtype=dtype)
    velocity = torch.as_tensor(velocity, dtype=dtype)
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=velocity))},
        command_manager=SimpleNamespace(get_command=lambda name: command),
        command=command, velocity=velocity,
    )


def _xy(env, **params):
    assert hasattr(REWARDS, "track_command_lin_vel_xy_exp"), "Production XY tracker has not been implemented"
    return REWARDS.track_command_lin_vel_xy_exp(env, **{"std": 0.5, "command_name": "base_velocity", **params})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("std", [0.1, 0.5, 1.3])
def test_absolute_xy_matches_the_official_ungated_formula_exactly(dtype, std):
    generator = torch.Generator().manual_seed(42)
    env = _env(torch.randn(128, 3, generator=generator), torch.randn(128, 3, generator=generator), dtype)
    expected = torch.exp(-torch.square(env.command[:, :2] - env.velocity[:, :2]).sum(dim=1) / std ** 2)
    torch.testing.assert_close(_xy(env, std=std), expected, rtol=0, atol=0)


def test_relative_bandwidth_distinguishes_standing_from_tracking_a_slow_command():
    env = _env([[0.2, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    torch.testing.assert_close(_xy(env), torch.tensor([math.exp(-0.16)], dtype=torch.float64))
    torch.testing.assert_close(
        _xy(env, tracking_mode="command_relative"), torch.tensor([math.exp(-4.0)], dtype=torch.float64),
    )


def test_mixed_tracking_standing_score_keeps_a_broad_component():
    env = _env([[0.2, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    expected = 0.25 * math.exp(-0.16) + 0.75 * math.exp(-4.0)
    torch.testing.assert_close(
        _xy(env, tracking_mode="command_relative_mixed"), torch.tensor([expected], dtype=torch.float64),
    )


def test_mixed_tracking_retains_float32_signal_when_relative_underflows():
    env = _env([[0.1, 0.0, 0.0]], [[1.1, 0.0, 0.0]], dtype=torch.float32)
    relative = _xy(env, tracking_mode="command_relative")
    mixed = _xy(env, tracking_mode="command_relative_mixed")
    assert relative.item() == 0.0
    assert mixed.item() > 0.0


@pytest.mark.parametrize("name", ["track_command_lin_vel_xy_exp", "track_forward_velocity", "track_forward_world_velocity"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("relative_weight", [0.0, 0.25, 0.75, 1.0])
def test_mixed_tracking_is_the_existing_score_mixture_with_exact_endpoints(name, dtype, relative_weight):
    generator = torch.Generator().manual_seed(21)
    env = _env(torch.randn(48, 3, generator=generator), torch.randn(48, 3, generator=generator), dtype)
    function = getattr(REWARDS, name)
    params = {"std": 0.5, "command_name": "base_velocity"}
    absolute = function(env, **params, tracking_mode="absolute")
    relative = function(env, **params, tracking_mode="command_relative")
    mixed = function(env, **params, tracking_mode="command_relative_mixed", relative_weight=relative_weight)
    if relative_weight == 0.0:
        expected = absolute
    elif relative_weight == 1.0:
        expected = relative
    else:
        expected = (1 - relative_weight) * absolute + relative_weight * relative
    torch.testing.assert_close(mixed, expected, rtol=0, atol=0)
    assert mixed.dtype == dtype and mixed.device == env.velocity.device


@pytest.mark.parametrize("name", ["track_forward_velocity", "track_forward_world_velocity"])
def test_forward_mixture_retains_float32_signal_without_opening_its_gate(name):
    env = _env([[0.2, 0.0, 0.0], [0.2, 0.15, 0.0], [-0.2, 0.0, 0.0]],
               [[1.3, 0.0, 0.0]] * 3, dtype=torch.float32)
    function = getattr(REWARDS, name)
    relative = function(env, "base_velocity", tracking_mode="command_relative")
    mixed = function(env, "base_velocity", tracking_mode="command_relative_mixed")
    assert relative[0].item() == 0.0 and mixed[0].item() > 0.0
    torch.testing.assert_close(mixed[1:], torch.zeros(2, dtype=torch.float32))


def test_forward_alias_passes_the_mixture_weight_as_its_last_parameter():
    env = _env([[0.2, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    direct = REWARDS.track_forward_velocity(env, "base_velocity", 0.5, 0.15, "command_relative_mixed", 0.1, 0.25)
    alias = REWARDS.track_forward_world_velocity(env, "base_velocity", 0.5, 0.15, "command_relative_mixed", 0.1, 0.25)
    torch.testing.assert_close(alias, direct, rtol=0, atol=0)
    expected = 0.75 * math.exp(-0.16) + 0.25 * math.exp(-4.0)
    torch.testing.assert_close(alias, torch.tensor([expected], dtype=torch.float64))


@pytest.mark.parametrize("name", ["track_command_lin_vel_xy_exp", "track_forward_velocity", "track_forward_world_velocity"])
@pytest.mark.parametrize("weight", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_mixture_weight_is_an_explicit_error(name, weight):
    env = _env([[0.2, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="relative_weight"):
        getattr(REWARDS, name)(env, std=0.5, command_name="base_velocity",
                               tracking_mode="command_relative_mixed", relative_weight=weight)


@pytest.mark.parametrize("name", ["track_command_lin_vel_xy_exp", "track_forward_velocity", "track_forward_world_velocity"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pure_relative_mode_stays_bitwise_equal_to_the_original_expression(name, dtype):
    generator = torch.Generator().manual_seed(22)
    env = _env(torch.randn(48, 3, generator=generator), torch.randn(48, 3, generator=generator), dtype)
    if name == "track_command_lin_vel_xy_exp":
        sigma = 0.5 * torch.linalg.vector_norm(env.command[:, :2], dim=1).clamp(min=0.1)
        error = (env.command[:, :2] - env.velocity[:, :2]).square().sum(dim=1)
        expected = torch.exp(-error / sigma.square())
    else:
        gate = (env.command[:, 0] > 0.15) & (env.command[:, 1].abs() < 0.15)
        sigma = 0.5 * env.command[:, 0].abs().clamp(min=0.1)
        expected = torch.exp(-((env.velocity[:, 0] - env.command[:, 0]) / sigma).square()) * gate
    actual = getattr(REWARDS, name)(env, std=0.5, command_name="base_velocity", tracking_mode="command_relative")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["absolute", "command_relative"])
def test_exact_tracking_is_one_for_forward_backward_sideways_and_stop(mode):
    command = [[0.2, 0.0, 0.8], [-0.2, 0.0, -0.8], [0.0, 0.2, 0.5],
               [0.2, -0.1, -0.5], [0.0, 0.0, 0.0]]
    env = _env(command, command)
    # Only XY is tracked here: velocity Z and yaw-rate commands are irrelevant.
    env.velocity[:, 2] += 10.0
    torch.testing.assert_close(_xy(env, tracking_mode=mode), torch.ones(5, dtype=torch.float64), rtol=0, atol=0)


def test_relative_xy_rejects_reverse_motion_and_lateral_drift():
    env = _env([[0.2, 0.0, 0.0]] * 4,
               [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.2, 0.2, 0.0]])
    expected = torch.tensor([1.0, math.exp(-4.0), math.exp(-16.0), math.exp(-4.0)], dtype=torch.float64)
    torch.testing.assert_close(_xy(env, tracking_mode="command_relative"), expected)
    assert _xy(env, tracking_mode="command_relative")[2] < _xy(env)[2]


@pytest.mark.parametrize("mode", ["absolute", "command_relative"])
def test_tracking_does_not_require_or_read_contact_data(mode):
    env = _env([[0.2, 0.0, 0.0]], [[0.2, 0.0, 0.0]])
    assert not hasattr(env.scene, "sensors")
    torch.testing.assert_close(_xy(env, tracking_mode=mode), torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(
        REWARDS.track_forward_velocity(env, "base_velocity", tracking_mode=mode),
        torch.ones(1, dtype=torch.float64),
    )


def test_zero_command_uses_a_finite_speed_floor_and_rejects_motion():
    env = _env([[0.0, 0.0, 0.0], [0.0, 0.0, 0.8], [0.0, 0.0, -0.8]],
               [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, -0.1, 0.0]])
    expected = torch.tensor([1.0, math.exp(-4.0), math.exp(-4.0)], dtype=torch.float64)
    torch.testing.assert_close(_xy(env, tracking_mode="command_relative"), expected)
    assert torch.isfinite(_xy(env, tracking_mode="command_relative")).all()


def test_relative_bandwidth_is_continuous_at_the_speed_floor():
    speeds = torch.tensor([0.1 - 1e-8, 0.1, 0.1 + 1e-8], dtype=torch.float64)
    command = torch.zeros(3, 3, dtype=torch.float64)
    command[:, 0] = speeds
    env = _env(command, torch.zeros_like(command))
    actual = _xy(env, tracking_mode="command_relative")
    torch.testing.assert_close(actual, torch.full_like(actual, math.exp(-4.0)), rtol=1e-6, atol=1e-12)
    assert actual[0] >= actual[1]


def test_relative_xy_tracking_preserves_fractional_error_across_command_scales():
    factors = torch.tensor([0.5, 1.0, 3.0, 8.0], dtype=torch.float64)
    command = factors[:, None] * torch.tensor([[0.2, -0.1, 0.0]], dtype=torch.float64)
    env = _env(command, 0.75 * command)
    expected = torch.full((4,), math.exp(-0.25), dtype=torch.float64)
    torch.testing.assert_close(_xy(env, tracking_mode="command_relative"), expected)


@pytest.mark.parametrize("std", [0.1, 0.5, 1.3])
def test_absolute_forward_retains_the_existing_formula_and_gate_exactly(std):
    generator = torch.Generator().manual_seed(9)
    env = _env(torch.randn(128, 3, generator=generator), torch.randn(128, 3, generator=generator))
    gate = (env.command[:, 0] > 0.15) & (env.command[:, 1].abs() < 0.15)
    expected = torch.exp(-((env.velocity[:, 0] - env.command[:, 0]) / std).square()) * gate
    actual = REWARDS.track_forward_velocity(env, "base_velocity", std=std)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(REWARDS.track_forward_world_velocity(env, "base_velocity", std=std), expected,
                               rtol=0, atol=0)


def test_relative_forward_uses_x_speed_and_preserves_its_gate():
    env = _env([[0.2, 0.1, 0.8], [0.4, 0.0, -0.8], [0.8, -0.1, 0.0],
                [0.15, 0.0, 0.0], [0.2, 0.15, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
               [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0],
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    actual = REWARDS.track_forward_velocity(env, "base_velocity", tracking_mode="command_relative")
    torch.testing.assert_close(actual, torch.tensor([math.exp(-1)] * 3 + [0.0] * 4, dtype=torch.float64))


@pytest.mark.parametrize("mode", ["absolute", "command_relative"])
def test_forward_world_alias_forwards_all_tail_parameters(mode):
    env = _env([[0.05, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    direct = REWARDS.track_forward_velocity(env, "base_velocity", 0.5, 0.01, mode, 0.2)
    alias = REWARDS.track_forward_world_velocity(env, "base_velocity", 0.5, 0.01, mode, 0.2)
    torch.testing.assert_close(alias, direct, rtol=0, atol=0)
    if mode == "command_relative":
        torch.testing.assert_close(alias, torch.tensor([math.exp(-0.25)], dtype=torch.float64))


@pytest.mark.parametrize("mode", ["absolute", "command_relative"])
def test_xy_asset_selection_and_base_frame_ignore_world_velocity(mode):
    env = _env([[0.2, 0.0, 0.0]] * 4, [[0.1, 0.0, 0.0]] * 4)
    body = env.scene.pop("robot")
    yaws = torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2], dtype=torch.float64)
    body.data.root_lin_vel_w = torch.stack((0.1 * yaws.cos(), 0.1 * yaws.sin(), torch.zeros_like(yaws)), dim=1)
    env.scene["alternate_robot"] = body
    actual = _xy(env, asset_cfg=SceneEntityCfg("alternate_robot"), tracking_mode=mode)
    torch.testing.assert_close(actual, actual[:1].expand_as(actual), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["absolute", "command_relative"])
def test_tracking_keeps_input_tensors_unchanged(mode):
    env = _env([[0.2, 0.1, 0.8], [0.0, 0.0, 0.0]], [[0.1, -0.1, 0.0], [0.01, 0.0, 0.0]])
    commands_before, velocities_before = env.command.clone(), env.velocity.clone()
    _xy(env, tracking_mode=mode)
    REWARDS.track_forward_velocity(env, "base_velocity", tracking_mode=mode)
    torch.testing.assert_close(env.command, commands_before, rtol=0, atol=0)
    torch.testing.assert_close(env.velocity, velocities_before, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["track_command_lin_vel_xy_exp", "track_forward_velocity", "track_forward_world_velocity"])
@pytest.mark.parametrize("parameter", ["std", "min_command_speed"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_bandwidth_scalars_raise_explicit_errors(name, parameter, value):
    env = _env([[0.2, 0.0, 0.0]], [[0.1, 0.0, 0.0]])
    params = {"std": 0.5, "command_name": "base_velocity", parameter: value}
    with pytest.raises(ValueError, match=parameter):
        getattr(REWARDS, name)(env, **params)


@pytest.mark.parametrize("name", ["track_command_lin_vel_xy_exp", "track_forward_velocity", "track_forward_world_velocity"])
@pytest.mark.parametrize("mode", ["relative", "", None])
def test_unknown_tracking_mode_is_explicit(name, mode):
    env = _env([[0.2, 0.0, 0.0]], [[0.1, 0.0, 0.0]])
    with pytest.raises(ValueError, match="tracking_mode"):
        getattr(REWARDS, name)(env, std=0.5, command_name="base_velocity", tracking_mode=mode)


def _configured_term(name):
    tree = ast.parse((TASK / "config/rewards_cfg.py").read_text(encoding="utf-8"))
    config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg")
    assignment = next(node for node in config.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    scope = {"math": math, "RewTerm": lambda **kwargs: SimpleNamespace(**kwargs), "h2_mdp": REWARDS,
             "mdp": SimpleNamespace(track_lin_vel_xy_exp=object(), track_ang_vel_z_exp=object())}
    return eval(compile(ast.Expression(body=assignment.value), "production_tracking_reward_config", "eval"), scope)


def test_shared_xy_and_forward_configs_keep_the_absolute_baseline_and_weights():
    for name, function, weight in (("track_lin_vel_xy", REWARDS.track_command_lin_vel_xy_exp, 8.0),
                                   ("track_forward_progress", REWARDS.track_forward_velocity, 2.0)):
        term = _configured_term(name)
        assert term.func is function and term.weight == weight
        assert term.params == {"command_name": "base_velocity", "std": 0.5,
                               "tracking_mode": "absolute", "min_command_speed": 0.1}
    env = _env([[0.2, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    term = _configured_term("track_lin_vel_xy")
    torch.testing.assert_close(term.func(env, **term.params), torch.tensor([math.exp(-0.16)], dtype=torch.float64))


def test_yaw_config_keeps_its_original_parameters_and_weight():
    term = _configured_term("track_ang_vel_z")
    assert term.weight == 2.0
    assert term.params == {"command_name": "base_velocity", "std": 0.5}
