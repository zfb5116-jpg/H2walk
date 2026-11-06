"""Synthetic budget properties only; no policy, optimizer or simulator."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import math

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"
spec = spec_from_file_location("budget_progress_cpu", TASK / "budget_progress.py")
budget = module_from_spec(spec)
spec.loader.exec_module(budget)


def state(n=2, distance=4.):
    result = budget.BudgetProgressState(n, "cpu", distance)
    result.test_position = torch.zeros(n, 2)
    return result


def advance(s, dx=0., dy=0., *, dt=.02, command=None, quaternion=None, velocity=None):
    n = s.num_envs
    cmd = torch.tensor([[.2, 0., 0.]] * n) if command is None else command
    quat = torch.tensor([[1., 0., 0., 0.]] * n) if quaternion is None else quaternion
    s.begin_transition(s.test_position.clone(), quat, cmd, dt)
    s.test_position[:, 0] += dx
    s.test_position[:, 1] += dy
    vel = torch.tensor([[dx / dt, dy / dt, 0.]] * n) if velocity is None else velocity
    s.finish_transition(s.test_position.clone(), vel)
    return s


def weighted_step_return(s, dt):
    p = budget.BUDGET_PROGRESS_PARAMETERS
    return dt * (p["progress_budget"] * s.advance_rate
                 - p["retreat_weight"] * s.retreat_rate
                 - p["overspeed_weight"] * s.overspeed_cost
                 - p["lateral_weight"] * s.lateral_cost)


@pytest.mark.parametrize("dt", [.01, .02, .05])
def test_normal_route_has_fixed_twenty_point_budget_across_control_dt(dt):
    s = state()
    integrated = torch.zeros(2)
    for _ in range(round(20. / dt)):
        advance(s, dx=.2 * dt, dt=dt)
        integrated += weighted_step_return(s, dt)
    torch.testing.assert_close(integrated, torch.full((2,), 20.), atol=.002, rtol=0.)
    torch.testing.assert_close(s.paid_distance, torch.full((2,), 4.), atol=.0005, rtol=0.)
    assert s.route_valid.all()
    assert (s.highwater <= s.goal_distance).all()


def test_same_route_four_metres_per_second_no_longer_outscores_command_speed():
    s = state()
    integrated = torch.zeros(2)
    positive = torch.zeros(2)
    for _ in range(50):
        advance(s, dx=.08)
        integrated += weighted_step_return(s, .02)
        positive += 20. * s.advance_rate * .02
    torch.testing.assert_close(positive, torch.ones(2), atol=.0001, rtol=0.)
    assert (integrated < -80.).all()
    torch.testing.assert_close(s.paid_distance, torch.full((2,), .2), atol=.0001, rtol=0.)
    assert s.route_valid.all()


def test_overspeed_cost_is_unclipped_and_progress_is_rate_limited():
    slow = advance(state(), dx=.004)
    fast = advance(state(), dx=.08)
    torch.testing.assert_close(slow.advance_rate, fast.advance_rate, atol=1e-6, rtol=0.)
    torch.testing.assert_close(fast.overspeed_cost, torch.full((2,), (20. - 1.2)**2), atol=.0001, rtol=0.)
    assert (fast.overspeed_cost > 300.).all()


def test_frontier_is_fully_consumed_and_standing_cannot_collect_backlog():
    s = advance(state(), dx=.08)
    torch.testing.assert_close(s.highwater, torch.full((2,), .08))
    torch.testing.assert_close(s.paid_distance, torch.full((2,), .004))
    for _ in range(25):
        advance(s)
        assert not s.advance_rate.any()
    torch.testing.assert_close(s.paid_distance, torch.full((2,), .004))


def test_closed_route_is_not_positive_and_old_peak_cannot_be_farmed():
    s = state()
    total = torch.zeros(2)
    advance(s, dx=.004)
    total += weighted_step_return(s, .02)
    advance(s, dx=-.004)
    total += weighted_step_return(s, .02)
    assert (total <= 1e-7).all()
    for _ in range(4):
        advance(s, dx=.004)
        assert not s.advance_rate.any()
        advance(s, dx=-.004)
    assert not s.raw_route_progress.any()


def test_regress_below_start_is_real_retreat_without_negative_frontier():
    s = advance(state(), dx=-.004)
    assert (s.retreat_rate > 0.).all()
    assert not s.highwater.any()
    torch.testing.assert_close(s.raw_route_progress, torch.full((2,), -.004))


@pytest.mark.parametrize("command", [[.4, 0., 0.], [-.2, 0., 0.], [0., .2, 0.], [.2, 0., .1], [0., 0., 0.]])
def test_command_change_invalidates_episode_and_recovery_does_not_refill(command):
    s = advance(state(), dx=.004)
    advance(s, dx=.004, command=torch.tensor([command] * 2))
    assert not s.route_valid.any()
    assert not s.command_valid.any()
    assert not s.advance_rate.any()
    old_paid = s.paid_distance.clone()
    advance(s, dx=.004)
    assert not s.route_valid.any()
    assert not s.advance_rate.any()
    assert (s.highwater >= .012 - 1e-7).all()
    torch.testing.assert_close(s.paid_distance, old_paid)


def test_robot_yaw_drift_does_not_redefine_route_or_command():
    s = advance(state(), dx=.004)
    quat = torch.tensor([[math.cos(.4), 0., 0., math.sin(.4)]] * 2)
    advance(s, dx=.004, quaternion=quat)
    assert s.route_valid.all() and s.command_valid.all()
    torch.testing.assert_close(s.raw_route_progress, torch.full((2,), .008))


def test_initial_yaw_and_nonunit_quaternion_define_world_route():
    s = state()
    quat = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]] * 2) * 3.
    advance(s, dy=.004, quaternion=quat)
    torch.testing.assert_close(s.advance_rate, torch.full((2,), .05), atol=1e-6, rtol=0.)
    assert s.route_valid.all()
    assert (s.lateral_cost < 1e-10).all()


def test_public_route_and_displacement_fields_use_fixed_world_heading():
    s = state()
    quat = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]] * 2)
    advance(s, dx=-.002, dy=.004, quaternion=quat)
    torch.testing.assert_close(s.direction, torch.tensor([[0., 1.]] * 2), atol=1e-6, rtol=0.)
    torch.testing.assert_close(s.command_speed, torch.full((2,), .2))
    torch.testing.assert_close(s.forward_velocity, torch.full((2,), .2), atol=1e-6, rtol=0.)
    torch.testing.assert_close(s.lateral_velocity, torch.full((2,), .1), atol=1e-6, rtol=0.)
    torch.testing.assert_close(s.lateral_distance, torch.full((2,), .002), atol=1e-6, rtol=0.)
    # Public reference values are copies, so a caller cannot replace the anchor.
    s.direction.zero_()
    s.command_speed.zero_()
    torch.testing.assert_close(s.command_speed, torch.full((2,), .2))
    assert (s.direction[:, 1] > .99).all()


def test_public_unknowns_are_nan_and_partial_reset_preserves_other_route():
    s = state()
    for name in ("direction", "command_speed", "forward_velocity", "lateral_velocity", "lateral_distance"):
        assert getattr(s, name).isnan().all()
    advance(s, dx=.004, dy=.002)
    s.reset([0])
    for name in ("direction", "command_speed", "forward_velocity", "lateral_velocity", "lateral_distance"):
        value = getattr(s, name)
        assert value[0].isnan().all()
        assert value[1].isfinite().all()


def test_public_displacement_fields_do_not_turn_missing_position_into_zero():
    s = advance(state(), dx=.004)
    s.begin_transition(s.test_position, torch.tensor([[1., 0., 0., 0.]] * 2),
                       torch.tensor([[.2, 0., 0.]] * 2), .02)
    s.test_position[0, 0] = torch.nan
    s.finish_transition(s.test_position, torch.zeros(2, 3))
    assert not s.route_valid[0]
    for name in ("forward_velocity", "lateral_velocity", "lateral_distance"):
        assert getattr(s, name)[0].isnan()
        assert torch.isfinite(getattr(s, name)[1])


def test_velocity_position_disagreement_is_diagnostic_not_a_reward_source():
    honest = advance(state(), dx=.004)
    mismatch = advance(state(), dx=.004, velocity=torch.tensor([[4., 0., 0.]] * 2))
    torch.testing.assert_close(honest.advance_rate, mismatch.advance_rate)
    torch.testing.assert_close(honest.overspeed_cost, mismatch.overspeed_cost)
    torch.testing.assert_close(mismatch.displacement_velocity_error, torch.full((2,), 3.8))
    assert mismatch.route_valid.all() and mismatch.velocity_diagnostic_valid.all()
    invalid = advance(state(), dx=.004, velocity=torch.full((2, 3), torch.nan))
    assert invalid.route_valid.all()
    assert not invalid.velocity_diagnostic_valid.any()
    torch.testing.assert_close(invalid.advance_rate, honest.advance_rate)


def test_lateral_displacement_has_cost_without_forward_credit():
    s = advance(state(), dy=.004)
    assert not s.advance_rate.any()
    torch.testing.assert_close(s.lateral_cost, torch.ones(2))


def test_position_jump_invalidates_persistently_and_consumes_finite_frontier():
    s = advance(state(), dx=.004)
    advance(s, dx=.2)
    assert not s.route_valid.any() and not s.advance_rate.any()
    torch.testing.assert_close(s.highwater, torch.full((2,), .204))
    old_paid = s.paid_distance.clone()
    advance(s, dx=.004)
    assert not s.advance_rate.any() and not s.route_valid.any()
    torch.testing.assert_close(s.paid_distance, old_paid)


def test_unobserved_between_step_position_change_invalidates_episode():
    s = advance(state(), dx=.004)
    s.test_position[:, 0] += .01
    advance(s, dx=.004)
    assert not s.route_valid.any() and not s.advance_rate.any()
    assert (s.highwater >= .018 - 1e-6).all()


@pytest.mark.parametrize("fault", ["position", "quaternion", "zero_quaternion", "command"])
def test_nonfinite_or_invalid_input_cannot_recover_positive_reward_until_reset(fault):
    s = state()
    command = torch.tensor([[.2, 0., 0.]] * 2)
    quat = torch.tensor([[1., 0., 0., 0.]] * 2)
    if fault == "position":
        s.test_position[0, 0] = torch.nan
    elif fault == "quaternion":
        quat[0, 0] = torch.nan
    elif fault == "zero_quaternion":
        quat[0] = 0.
    else:
        command[0, 0] = torch.nan
    advance(s, dx=.004, command=command, quaternion=quat)
    assert not s.route_valid[0] and not s.advance_rate[0]
    assert s.route_valid[1] and s.advance_rate[1] > 0
    s.test_position[0] = 0.
    advance(s, dx=.004)
    assert not s.route_valid[0] and not s.advance_rate[0]


def test_nonfinite_finish_position_stays_invalid_after_recovery():
    s = state()
    s.begin_transition(s.test_position, torch.tensor([[1., 0., 0., 0.]] * 2),
                       torch.tensor([[.2, 0., 0.]] * 2), .02)
    s.test_position[0, 0] = torch.nan
    s.finish_transition(s.test_position, torch.zeros(2, 3))
    assert not s.route_valid[0]
    s.test_position[0] = 0.
    advance(s, dx=.004)
    assert not s.route_valid[0] and not s.advance_rate[0]


@pytest.mark.parametrize("distance", [1e38, -1e38])
def test_extreme_finite_position_invalidates_without_nonfinite_reward_outputs(distance):
    s = advance(state(), dx=distance)
    assert not s.route_valid.any() and not s.advance_rate.any()
    for name in ("advance_rate", "retreat_rate", "overspeed_cost", "lateral_cost"):
        assert torch.isfinite(getattr(s, name)).all()


def test_partial_reset_reanchors_only_requested_row():
    s = advance(state(), dx=.08)
    s.reset([0])
    advance(s, dx=.004)
    torch.testing.assert_close(s.highwater, torch.tensor([.004, .084]), atol=1e-6, rtol=0.)
    torch.testing.assert_close(s.paid_distance, torch.tensor([.004, .008]), atol=1e-6, rtol=0.)
    assert s.route_valid.all()


def test_abort_cannot_refill_a_consumed_episode_budget():
    s = advance(state(), dx=.08)
    s.begin_transition(s.test_position, torch.tensor([[1., 0., 0., 0.]] * 2),
                       torch.tensor([[.2, 0., 0.]] * 2), .02)
    s.abort_step()
    old_paid = s.paid_distance.clone()
    advance(s, dx=.004)
    assert not s.route_valid.any() and not s.advance_rate.any()
    torch.testing.assert_close(s.paid_distance, old_paid)


def test_frontier_and_total_positive_budget_are_bounded_at_goal():
    s = state(distance=.01)
    total = torch.zeros(2)
    for _ in range(10):
        advance(s, dx=.004)
        total += 20. * s.advance_rate * .02
    torch.testing.assert_close(total, torch.full((2,), 20.), atol=1e-5, rtol=0.)
    torch.testing.assert_close(s.highwater, torch.full((2,), .01))
    assert (s.raw_route_progress > s.goal_distance).all()


def test_transition_lifecycle_rejects_misordered_calls():
    s = state()
    with pytest.raises(RuntimeError, match="begin"):
        s.finish_transition(torch.zeros(2, 2), torch.zeros(2, 3))
    s.begin_transition(s.test_position, torch.tensor([[1., 0., 0., 0.]] * 2),
                       torch.tensor([[.2, 0., 0.]] * 2), .02)
    with pytest.raises(RuntimeError, match="pending"):
        s.begin_transition(s.test_position, torch.tensor([[1., 0., 0., 0.]] * 2),
                           torch.tensor([[.2, 0., 0.]] * 2), .02)


@pytest.mark.parametrize("distance", [0., -1., float("nan"), float("inf"), True])
def test_goal_distance_must_be_positive_finite(distance):
    with pytest.raises(ValueError, match="goal_distance"):
        state(distance=distance)
