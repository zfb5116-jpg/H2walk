"""CPU trajectory checks for the production command-progress tracker.

Loads the pure tensor module directly. No Isaac environment, simulator,
checkpoint or optimizer is constructed.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/command_progress.py"
)


@pytest.fixture(scope="module")
def tracker_class():
    name = "h2_command_progress_cpu_tests"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.CommandProgressTracker


def _begin(tracker, positions, commands=None, *, yaw=0.0, dt=0.5):
    n = len(positions)
    position = torch.tensor(positions, dtype=torch.float32)
    quaternion = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]).repeat(n, 1)
    command = torch.tensor(commands if commands is not None else [[0.4, 0.0, 0.0]] * n)
    tracker.begin_step(position, quaternion, command, dt)


def _finish(tracker, positions, ids=None, *, for_reset=False):
    if ids is None:
        ids = list(range(len(positions)))
    tracker.finish_step(ids, torch.tensor(positions, dtype=torch.float32), for_reset=for_reset)


def test_stationary_robot_accumulates_requested_distance_but_no_progress(tracker_class):
    tracker = tracker_class(1, "cpu")
    for _ in range(40):
        _begin(tracker, [[0.0, 0.0]], [[0.2, 0.0, 0.0]], dt=0.1)
        _finish(tracker, [[0.0, 0.0]])
    torch.testing.assert_close(tracker.target_distance, torch.tensor([0.8]))
    torch.testing.assert_close(tracker.signed_progress, torch.zeros(1))
    torch.testing.assert_close(tracker.active_time, torch.tensor([4.0]))
    assert tracker.had_translation.tolist() == [True]
    assert tracker.valid.tolist() == [True]


@pytest.mark.parametrize(
    ("command", "endpoint", "expected"),
    [
        ([0.4, 0.0, 0.0], [0.2, 0.0], 0.2),
        ([0.4, 0.0, 0.0], [-0.2, 0.0], -0.2),
        ([-0.4, 0.0, 0.0], [-0.2, 0.0], 0.2),
        ([0.0, 0.4, 0.0], [0.0, 0.2], 0.2),
        ([0.4, 0.0, 0.0], [0.0, 0.2], 0.0),
    ],
)
def test_actual_motion_is_signed_and_aligned_with_command(tracker_class, command, endpoint, expected):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], [command])
    _finish(tracker, [endpoint])
    torch.testing.assert_close(tracker.target_distance, torch.tensor([0.2]))
    torch.testing.assert_close(tracker.signed_progress, torch.tensor([expected]))


@pytest.mark.parametrize("yaw", [math.pi / 2, -math.pi / 2, math.pi])
def test_rotating_world_heading_preserves_command_progress(tracker_class, yaw):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], yaw=yaw)
    _finish(tracker, [[0.2 * math.cos(yaw), 0.2 * math.sin(yaw)]])
    torch.testing.assert_close(tracker.signed_progress, torch.tensor([0.2]))
    torch.testing.assert_close(tracker.target_distance, torch.tensor([0.2]))


def test_begin_snapshots_command_heading_and_position_before_they_change(tracker_class):
    tracker = tracker_class(1, "cpu")
    position = torch.zeros(1, 2)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    command = torch.tensor([[0.4, 0.0, 0.0]])
    tracker.begin_step(position, quaternion, command, 0.5)
    position[:] = 100.0
    quaternion[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    command[:] = torch.tensor([-1.0, 0.0, 0.0])
    _finish(tracker, [[0.2, 0.0]])
    torch.testing.assert_close(tracker.signed_progress, torch.tensor([0.2]))
    torch.testing.assert_close(tracker.target_distance, torch.tensor([0.2]))
    assert tracker.valid.item()


def test_fixed_command_forward_then_backward_motion_cancels(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]])
    _finish(tracker, [[0.2, 0.0]])
    _begin(tracker, [[0.2, 0.0]])
    _finish(tracker, [[0.0, 0.0]])
    torch.testing.assert_close(tracker.signed_progress, torch.zeros(1))
    torch.testing.assert_close(tracker.target_distance, torch.tensor([0.4]))


def test_reversing_command_counts_each_matching_leg_not_net_displacement(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]])
    _finish(tracker, [[0.2, 0.0]])
    _begin(tracker, [[0.2, 0.0]], [[-0.4, 0.0, 0.0]])
    _finish(tracker, [[0.0, 0.0]])
    torch.testing.assert_close(tracker.signed_progress, torch.tensor([0.4]))
    torch.testing.assert_close(tracker.target_distance, torch.tensor([0.4]))


@pytest.mark.parametrize("command", [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.05, 0.0, 0.0], [0.06, 0.06, 0.0]])
def test_standing_turning_and_tiny_commands_provide_no_translation_evidence(tracker_class, command):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], [command])
    _finish(tracker, [[0.1, 0.0]])
    assert tracker.had_translation.tolist() == [False]
    torch.testing.assert_close(tracker.elapsed_time, torch.tensor([0.5]))
    torch.testing.assert_close(tracker.active_time, torch.zeros(1))
    torch.testing.assert_close(tracker.target_distance, torch.zeros(1))
    torch.testing.assert_close(tracker.signed_progress, torch.zeros(1))


def test_partial_terminal_reset_does_not_count_teleport_or_drop_other_environment(tracker_class):
    tracker = tracker_class(2, "cpu")
    _begin(tracker, [[0.0, 0.0], [0.0, 0.0]])
    _finish(tracker, [[0.2, 0.0], [0.2, 0.0]], [0], for_reset=True)
    assert tracker.reset_ready.tolist() == [True, False]
    assert tracker.pending.tolist() == [False, True]
    tracker.reported[0] = True
    tracker.reset([0], generation=3)
    # Same global step: env0 is at its new spawn, env1 still needs settlement.
    _finish(tracker, [[100.0, 0.0], [0.2, 0.0]])
    assert tracker.pending.tolist() == [False, False]
    assert tracker.reset_ready.tolist() == [False, False]
    assert tracker.reported.tolist() == [False, False]
    assert tracker.episode_generation.tolist() == [3, 0]
    assert tracker.had_translation.tolist() == [False, True]
    torch.testing.assert_close(tracker.elapsed_time, torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(tracker.target_distance, torch.tensor([0.0, 0.2]))
    torch.testing.assert_close(tracker.signed_progress, torch.tensor([0.0, 0.2]))


def test_repeated_settlement_is_idempotent(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]])
    _finish(tracker, [[0.2, 0.0]], for_reset=True)
    _finish(tracker, [[0.4, 0.0]], for_reset=True)
    _finish(tracker, [[99.0, 0.0]])
    torch.testing.assert_close(tracker.elapsed_time, torch.tensor([0.5]))
    torch.testing.assert_close(tracker.signed_progress, torch.tensor([0.2]))
    assert tracker.reset_ready.item()


def test_manual_reset_without_pending_step_does_not_become_a_completed_episode(tracker_class):
    tracker = tracker_class(1, "cpu")
    _finish(tracker, [[50.0, 0.0]], for_reset=True)
    assert not tracker.reset_ready.item()
    tracker.reset([0], generation=2)
    _begin(tracker, [[50.0, 0.0]])
    _finish(tracker, [[50.2, 0.0]])
    assert tracker.valid.item()
    assert tracker.episode_generation.item() == 2
    torch.testing.assert_close(tracker.signed_progress, torch.tensor([0.2]), atol=4e-6, rtol=1e-6)


@pytest.mark.parametrize("field", ["position", "quaternion", "command", "endpoint"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_transition_invalidates_episode_without_nonfinite_progress(tracker_class, field, value):
    tracker = tracker_class(1, "cpu")
    position = torch.zeros(1, 2)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    command = torch.tensor([[0.4, 0.0, 0.0]])
    endpoint = torch.tensor([[0.2, 0.0]])
    {"position": position, "quaternion": quaternion, "command": command, "endpoint": endpoint}[field][0, 0] = value
    tracker.begin_step(position, quaternion, command, 0.5)
    tracker.finish_step([0], endpoint)
    assert not tracker.valid.item()
    assert torch.isfinite(tracker.signed_progress).all()
    torch.testing.assert_close(tracker.signed_progress, torch.zeros(1))


def test_implausible_displacement_invalidates_whole_episode(tracker_class):
    tracker = tracker_class(1, "cpu", max_step_speed=5.0)
    _begin(tracker, [[0.0, 0.0]], dt=0.1)
    _finish(tracker, [[0.04, 0.0]])
    _begin(tracker, [[0.04, 0.0]], dt=0.1)
    _finish(tracker, [[1.0, 0.0]])
    assert not tracker.valid.item()
    torch.testing.assert_close(tracker.signed_progress, torch.tensor([0.04]))


def test_pose_change_between_steps_invalidates_episode_until_reset(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], dt=0.1)
    _finish(tracker, [[0.04, 0.0]])
    _begin(tracker, [[0.10, 0.0]], dt=0.1)
    _finish(tracker, [[0.14, 0.0]])
    assert not tracker.valid.item()
    tracker.reset([0])
    _begin(tracker, [[100.0, 0.0]], dt=0.1)
    _finish(tracker, [[100.04, 0.0]])
    assert tracker.valid.item()


def test_abort_invalidates_only_unsettled_rows_and_clears_pending(tracker_class):
    tracker = tracker_class(2, "cpu")
    _begin(tracker, [[0.0, 0.0], [0.0, 0.0]])
    _finish(tracker, [[0.2, 0.0], [0.0, 0.0]], [0])
    tracker.abort_step()
    assert tracker.pending.tolist() == [False, False]
    assert tracker.valid.tolist() == [True, False]
    torch.testing.assert_close(tracker.signed_progress, torch.tensor([0.2, 0.0]))


def test_begin_requires_previous_transition_to_be_finished_or_aborted(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]])
    with pytest.raises(RuntimeError):
        _begin(tracker, [[0.0, 0.0]])
    tracker.abort_step()
    _begin(tracker, [[0.0, 0.0]])


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_step_duration_must_be_finite_and_positive(tracker_class, value):
    tracker = tracker_class(1, "cpu")
    with pytest.raises(ValueError):
        _begin(tracker, [[0.0, 0.0]], dt=value)


@pytest.mark.parametrize("name", ["min_command_speed", "max_step_speed"])
@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_tracker_thresholds_must_be_finite_and_positive(tracker_class, name, value):
    with pytest.raises(ValueError):
        tracker_class(1, "cpu", **{name: value})


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_environment_count_is_a_positive_integer(tracker_class, count):
    with pytest.raises((ValueError, TypeError)):
        tracker_class(count, "cpu")


@pytest.mark.parametrize("field", ["position", "quaternion", "command"])
def test_input_shapes_are_checked_before_beginning_a_transition(tracker_class, field):
    tracker = tracker_class(1, "cpu")
    inputs = {"position": torch.zeros(1, 2), "quaternion": torch.tensor([[1.0, 0.0, 0.0, 0.0]]), "command": torch.tensor([[0.4, 0.0, 0.0]])}
    inputs[field] = inputs[field][:, :-1]
    with pytest.raises(ValueError):
        tracker.begin_step(inputs["position"], inputs["quaternion"], inputs["command"], 0.5)
    assert not tracker.pending.any()


@pytest.mark.parametrize("ids", [[-1], [1]])
def test_invalid_environment_indices_are_rejected(tracker_class, ids):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]])
    with pytest.raises((ValueError, IndexError)):
        _finish(tracker, [[0.2, 0.0]], ids)


def test_empty_environment_subset_is_a_noop(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]])
    _finish(tracker, [[0.2, 0.0]], [])
    assert tracker.pending.item()
    torch.testing.assert_close(tracker.elapsed_time, torch.zeros(1))


def test_pure_lateral_tracking_does_not_prove_forward_motion(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], [[0.0, 0.4, 0.0]])
    _finish(tracker, [[0.0, 0.2]])
    torch.testing.assert_close(tracker.signed_progress, torch.tensor([0.2]))
    torch.testing.assert_close(tracker.target_distance, torch.tensor([0.2]))
    torch.testing.assert_close(tracker.forward_signed_progress, torch.zeros(1))
    torch.testing.assert_close(tracker.forward_target_distance, torch.zeros(1))
    torch.testing.assert_close(tracker.forward_active_time, torch.zeros(1))
    assert tracker.had_forward_command.tolist() == [False]


def test_lateral_drift_can_match_diagonal_projection_without_any_forward_progress(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], [[0.2, 0.2, 0.0]])
    _finish(tracker, [[0.0, 0.2]])
    # Total commanded-direction distance alone would incorrectly pass this case.
    torch.testing.assert_close(tracker.signed_progress, tracker.target_distance)
    torch.testing.assert_close(tracker.forward_target_distance, torch.tensor([0.1]))
    torch.testing.assert_close(tracker.forward_signed_progress, torch.zeros(1))
    torch.testing.assert_close(tracker.forward_active_time, torch.tensor([0.5]))
    assert tracker.had_forward_command.tolist() == [True]


def test_forward_progress_uses_local_x_rotated_to_world_heading(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], yaw=math.pi / 2)
    _finish(tracker, [[0.0, 0.2]])
    torch.testing.assert_close(tracker.forward_target_distance, torch.tensor([0.2]))
    torch.testing.assert_close(tracker.forward_signed_progress, torch.tensor([0.2]))
    torch.testing.assert_close(tracker.forward_active_time, torch.tensor([0.5]))


def test_negative_x_command_requires_matching_backward_displacement(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], [[-0.4, 0.0, 0.0]])
    _finish(tracker, [[-0.2, 0.0]])
    torch.testing.assert_close(tracker.forward_target_distance, torch.tensor([0.2]))
    torch.testing.assert_close(tracker.forward_signed_progress, torch.tensor([0.2]))
    assert tracker.had_forward_command.tolist() == [True]


def test_forward_exposure_threshold_depends_on_x_component_not_total_speed(tracker_class):
    tracker = tracker_class(2, "cpu")
    _begin(tracker, [[0.0, 0.0], [0.0, 0.0]], [[0.09, 0.4, 0.0], [0.1, 0.4, 0.0]])
    _finish(tracker, [[0.045, 0.2], [0.05, 0.2]])
    assert tracker.had_translation.tolist() == [True, True]
    assert tracker.had_forward_command.tolist() == [False, True]
    torch.testing.assert_close(tracker.forward_active_time, torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(tracker.forward_target_distance, torch.tensor([0.0, 0.05]))
    torch.testing.assert_close(tracker.forward_signed_progress, torch.tensor([0.0, 0.05]))


def test_partial_reset_clears_forward_evidence_only_for_selected_environment(tracker_class):
    tracker = tracker_class(2, "cpu")
    _begin(tracker, [[0.0, 0.0], [0.0, 0.0]])
    _finish(tracker, [[0.2, 0.0], [0.2, 0.0]])
    tracker.reset([0], generation=4)
    assert tracker.had_forward_command.tolist() == [False, True]
    torch.testing.assert_close(tracker.forward_active_time, torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(tracker.forward_target_distance, torch.tensor([0.0, 0.2]))
    torch.testing.assert_close(tracker.forward_signed_progress, torch.tensor([0.0, 0.2]))


def test_active_forward_command_does_not_add_inactive_error(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]])
    _finish(tracker, [[0.2, 0.0]])
    torch.testing.assert_close(tracker.inactive_forward_error, torch.zeros(1))
    torch.testing.assert_close(tracker.forward_signed_progress, torch.tensor([0.2]))


def test_backward_drift_during_stop_is_recorded_as_inactive_error(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], [[0.0, 0.0, 0.0]])
    _finish(tracker, [[-0.1, 0.0]])
    torch.testing.assert_close(tracker.inactive_forward_error, torch.tensor([0.1]))
    torch.testing.assert_close(tracker.forward_signed_progress, torch.zeros(1))
    torch.testing.assert_close(tracker.forward_target_distance, torch.zeros(1))


def test_forward_drift_during_stop_adds_error_without_earning_progress(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], [[0.0, 0.0, 0.0]])
    _finish(tracker, [[0.1, 0.0]])
    torch.testing.assert_close(tracker.inactive_forward_error, torch.tensor([0.1]))
    torch.testing.assert_close(tracker.forward_signed_progress, torch.zeros(1))
    torch.testing.assert_close(tracker.forward_target_distance, torch.zeros(1))


def test_correct_tracking_of_small_x_command_has_no_inactive_error(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], [[0.05, 0.0, 0.0]])
    _finish(tracker, [[0.025, 0.0]])
    torch.testing.assert_close(tracker.inactive_forward_error, torch.zeros(1))
    assert tracker.had_forward_command.tolist() == [False]


def test_inactive_error_uses_yaw_heading_and_resets_only_selected_rows(tracker_class):
    tracker = tracker_class(2, "cpu")
    _begin(tracker, [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], yaw=math.pi / 2)
    _finish(tracker, [[0.0, -0.1], [0.0, -0.2]])
    torch.testing.assert_close(tracker.inactive_forward_error, torch.tensor([0.1, 0.2]))
    tracker.reset([0])
    torch.testing.assert_close(tracker.inactive_forward_error, torch.tensor([0.0, 0.2]))


def test_yaw_change_during_stop_cannot_hide_return_to_start_as_lateral_motion(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]])
    _finish(tracker, [[0.2, 0.0]])
    # Turning 90 degrees makes world-X return motion local lateral drift.
    # It must still cancel the evidence for sustained forward command tracking.
    _begin(tracker, [[0.2, 0.0]], [[0.0, 0.0, 0.0]], yaw=math.pi / 2)
    _finish(tracker, [[0.0, 0.0]])
    torch.testing.assert_close(tracker.forward_signed_progress, torch.tensor([0.2]))
    torch.testing.assert_close(tracker.inactive_forward_error, torch.tensor([0.2]))
    torch.testing.assert_close(tracker.forward_signed_progress - tracker.inactive_forward_error, torch.zeros(1))


def test_correct_lateral_command_tracking_has_no_inactive_error(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], [[0.0, 0.2, 0.0]])
    _finish(tracker, [[0.0, 0.1]])
    torch.testing.assert_close(tracker.inactive_forward_error, torch.zeros(1))
    assert tracker.had_forward_command.tolist() == [False]


def test_correct_lateral_tracking_after_yaw_rotation_has_no_inactive_error(tracker_class):
    tracker = tracker_class(1, "cpu")
    _begin(tracker, [[0.0, 0.0]], [[0.0, 0.2, 0.0]], yaw=math.pi / 2)
    _finish(tracker, [[-0.1, 0.0]])
    torch.testing.assert_close(tracker.inactive_forward_error, torch.zeros(1), atol=1e-6, rtol=1e-6)
    assert tracker.had_forward_command.tolist() == [False]
