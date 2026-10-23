"""CPU checks of completed-episode X curriculum and its actual MDP callback.

The evaluator imports normally without Isaac Lab. The production callback is
extracted with AST only to skip simulator imports, and is run on containers;
this is not an Isaac environment integration or locomotion performance test.
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
spec = importlib.util.spec_from_file_location("h2_progress_evaluator_test", TASK / "progress_curriculum.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ProgressCurriculum = module.ProgressCurriculum


def tracker(n=4, ratio=1.0):
    return SimpleNamespace(
        elapsed_time=torch.full((n,), 20.0),
        active_time=torch.full((n,), 10.0),
        target_distance=torch.full((n,), 2.0),
        signed_progress=torch.full((n,), 2.0 * ratio),
        forward_active_time=torch.full((n,), 10.0),
        forward_target_distance=torch.full((n,), 2.0),
        forward_signed_progress=torch.full((n,), 2.0 * ratio),
        inactive_forward_error=torch.zeros(n),
        had_translation=torch.ones(n, dtype=torch.bool),
        had_forward_command=torch.ones(n, dtype=torch.bool),
        valid=torch.ones(n, dtype=torch.bool),
        reset_ready=torch.ones(n, dtype=torch.bool),
        reported=torch.zeros(n, dtype=torch.bool),
        episode_generation=torch.zeros(n, dtype=torch.long),
    )


def evaluator(**kwargs):
    params = {"interval_steps": 1000, "min_episodes": 4}
    params.update(kwargs)
    return ProgressCurriculum(**params)


def observe(e, t, ids=None, step=1000, terminated=None, time_outs=None):
    n = t.elapsed_time.numel()
    return e.observe(
        t, list(range(n)) if ids is None else ids,
        terminated=torch.zeros(n, dtype=torch.bool) if terminated is None else terminated,
        time_outs=torch.ones(n, dtype=torch.bool) if time_outs is None else time_outs,
        max_episode_length_s=20.0, current_step=step,
    )


def test_stationary_complete_episodes_never_pass():
    result = observe(evaluator(), tracker(ratio=0.0))
    assert result["episodes"] == 4
    assert result["successes"] == 0
    assert result["passed"] is False


def test_actual_commanded_progress_passes_after_minimum_window():
    e, t = evaluator(), tracker()
    assert observe(e, t, step=999) is None
    result = observe(e, t, step=1000)
    assert result["success_rate"] == 1.0
    assert result["mean_progress_ratio"] == 1.0
    assert result["passed"] is True
    assert e.last_evaluation_step == 1000


def test_asynchronous_resets_do_not_have_to_hit_an_exact_step_multiple():
    e, t = evaluator(), tracker()
    assert observe(e, t, ids=[0, 1], step=997) is None
    result = observe(e, t, ids=[2, 3], step=1003)
    assert result["episodes"] == 4
    assert result["passed"] is True
    assert e.last_evaluation_step == 1003


def test_repeated_reset_batch_and_duplicate_ids_are_counted_once():
    e, t = evaluator(min_episodes=3), tracker(2)
    assert observe(e, t, ids=[0, 0, 1], step=1000) is None
    assert e.episode_count == 2
    assert observe(e, t, step=2000) is None
    assert e.episode_count == 2


def test_minimum_sample_count_keeps_a_late_window_open():
    e, t = evaluator(), tracker()
    assert observe(e, t, ids=[0], step=1200) is None
    assert e.last_evaluation_step == 0
    assert observe(e, t, ids=[1, 2, 3], step=1800)["passed"] is True


def test_step_counter_cannot_rewind_before_a_window_has_been_evaluated():
    e, t = evaluator(), tracker()
    assert observe(e, t, ids=[0], step=900) is None
    with pytest.raises(ValueError, match="nondecreasing"):
        observe(e, t, ids=[1], step=899)


def test_success_is_per_episode_not_the_average_distance_of_fast_outliers():
    t = tracker()
    t.forward_signed_progress[:] = torch.tensor([0.0, 0.0, 0.0, 8.0])
    result = observe(evaluator(), t)
    assert result["mean_progress_ratio"] == 1.0
    assert result["success_rate"] == 0.0
    assert result["passed"] is False


@pytest.mark.parametrize("ratio", [-1.0, 0.79, 1.21, 4.0])
def test_wrong_direction_insufficient_or_excessive_progress_does_not_pass(ratio):
    assert observe(evaluator(), tracker(ratio=ratio))["passed"] is False


@pytest.mark.parametrize("field,value", [
    ("elapsed_time", 5.0), ("forward_active_time", 0.5),
    ("forward_target_distance", 0.1), ("valid", False),
])
def test_short_or_invalid_natural_episodes_remain_failures(field, value):
    t = tracker()
    getattr(t, field)[:] = value
    result = observe(evaluator(), t)
    assert result["episodes"] == 4
    assert result["successes"] == 0


def test_timeout_and_failure_together_are_failures():
    result = observe(evaluator(), tracker(), terminated=torch.ones(4, dtype=torch.bool))
    assert result["episodes"] == 4
    assert result["passed"] is False


def test_early_falls_are_not_removed_from_the_denominator():
    t = tracker()
    t.elapsed_time[:3] = 0.1
    t.forward_target_distance[:3] = 0.02
    result = observe(evaluator(), t, terminated=torch.tensor([True, True, True, False]))
    assert result["episodes"] == 4
    assert result["success_rate"] == 0.25
    assert result["passed"] is False


def test_standing_sideways_and_initial_manual_resets_are_excluded():
    t = tracker(5)
    t.had_forward_command[:2] = False  # Valid standing/sideways-only commands.
    t.reset_ready[2] = False
    done = torch.tensor([True, True, True, False, True])
    e = evaluator(min_episodes=2)
    assert observe(e, t, time_outs=done) is None
    assert e.episode_count == 1


def test_pure_sideways_motion_never_qualifies_for_x_expansion():
    t = tracker()
    t.had_forward_command[:] = False
    t.forward_active_time[:] = 0
    t.forward_target_distance[:] = 0
    t.forward_signed_progress[:] = 0
    e = evaluator()
    assert observe(e, t) is None
    assert e.episode_count == 0


def test_good_xy_projection_cannot_mask_no_actual_x_progress():
    t = tracker()
    t.signed_progress[:] = t.target_distance  # Full XY score alone is good.
    t.forward_signed_progress[:] = 0
    assert observe(evaluator(), t)["passed"] is False


def test_invalid_command_without_recorded_exposure_counts_as_failure():
    t = tracker()
    t.had_forward_command[:] = False
    t.valid[:] = False
    result = observe(evaluator(), t)
    assert result["episodes"] == 4
    assert result["success_rate"] == 0.0


def test_evaluation_clears_the_window_even_when_it_fails():
    e = evaluator()
    assert observe(e, tracker(ratio=0.0), step=1000)["passed"] is False
    assert e.episode_count == e.success_count == 0
    assert observe(e, tracker(), step=1500) is None
    assert observe(e, tracker(), step=2000)["passed"] is True


def test_old_generation_evidence_is_not_reused_after_expansion():
    e = evaluator()
    assert observe(e, tracker())["passed"] is True
    assert e.advance_generation() == 1
    assert observe(e, tracker(), step=2000) is None
    assert e.episode_count == 0
    fresh = tracker()
    fresh.episode_generation[:] = 1
    assert observe(e, fresh, step=2001)["passed"] is True


@pytest.mark.parametrize("field", ["elapsed_time", "forward_active_time", "forward_target_distance", "forward_signed_progress"])
def test_nonfinite_telemetry_cannot_pass_even_if_valid_flag_is_incorrect(field):
    t = tracker()
    getattr(t, field)[:] = float("nan")
    assert observe(evaluator(), t)["passed"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_inactive_tracking_error_is_a_failure(value):
    t = tracker()
    t.inactive_forward_error[:] = value
    assert observe(evaluator(), t)["passed"] is False


def real_trajectory(first_steps=20, inactive_command=0.0, inactive_velocity=0.0,
                    *, inactive_yaw=0.0, inactive_command_y=0.0, inactive_velocity_y=0.0):
    """Run the actual tracker on prescribed CPU positions, not a simulator."""
    spec = importlib.util.spec_from_file_location("h2_progress_real_trajectory", TASK / "command_progress.py")
    tracking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracking)
    t = tracking.CommandProgressTracker(4, "cpu")
    positions = torch.zeros(4, 2)
    for step in range(40):
        command_x = 0.2 if step < first_steps else inactive_command
        command_y = 0.0 if step < first_steps else inactive_command_y
        velocity_x = 0.2 if step < first_steps else inactive_velocity
        velocity_y = 0.0 if step < first_steps else inactive_velocity_y
        yaw = 0.0 if step < first_steps else inactive_yaw
        quaternion = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]).repeat(4, 1)
        command = torch.tensor([[command_x, command_y, 0.0]]).repeat(4, 1)
        t.begin_step(positions, quaternion, command, dt=0.5)
        end = positions + torch.tensor([[velocity_x * 0.5, velocity_y * 0.5]])
        t.finish_step(None, end, for_reset=(step == 39))
        positions = end
    return t, positions


@pytest.mark.parametrize("inactive_command,inactive_velocity,passed", [
    (0.0, -0.2, False),  # Uncommanded return to the starting point.
    (0.0, 0.0, True),    # Stop command is followed after real progress.
    (-0.05, -0.05, True),  # Correct subthreshold command is not drift.
])
def test_real_trajectory_inactive_tracking_cannot_hide_a_return_leg(inactive_command, inactive_velocity, passed):
    t, positions = real_trajectory(inactive_command=inactive_command, inactive_velocity=inactive_velocity)
    result = observe(evaluator(), t)
    assert result["passed"] is passed
    if inactive_command == 0.0 and inactive_velocity < 0.0:
        torch.testing.assert_close(positions, torch.zeros(4, 2), atol=1e-6, rtol=0)
        torch.testing.assert_close(t.forward_signed_progress, torch.full((4,), 2.0))
        torch.testing.assert_close(t.inactive_forward_error, torch.full((4,), 2.0))
        assert result["mean_progress_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_real_trajectory_legitimate_small_command_return_is_command_tracking():
    t, positions = real_trajectory(first_steps=8, inactive_command=-0.05, inactive_velocity=-0.05)
    torch.testing.assert_close(positions, torch.zeros(4, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(t.inactive_forward_error, torch.zeros(4), atol=1e-6, rtol=0)
    assert observe(evaluator(), t)["passed"] is True


def test_real_trajectory_sideways_stop_drift_after_yaw_change_cannot_hide_return():
    t, positions = real_trajectory(inactive_yaw=math.pi / 2, inactive_velocity=-0.2)
    torch.testing.assert_close(positions, torch.zeros(4, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(t.forward_signed_progress, torch.full((4,), 2.0))
    torch.testing.assert_close(t.inactive_forward_error, torch.full((4,), 2.0))
    assert observe(evaluator(), t)["passed"] is False


def test_real_trajectory_correct_side_command_after_yaw_change_is_not_drift():
    t, positions = real_trajectory(
        inactive_yaw=math.pi / 2, inactive_command_y=0.2, inactive_velocity=-0.2,
    )
    # Returning to the world origin is legitimate when the rotated sideways
    # command actually requests that motion; this is not terrain completion.
    torch.testing.assert_close(positions, torch.zeros(4, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(t.inactive_forward_error, torch.zeros(4), atol=1e-6, rtol=0)
    assert observe(evaluator(), t)["passed"] is True


@pytest.mark.parametrize("kwargs", [
    {"interval_steps": 0}, {"min_episodes": 0}, {"min_episodes": 1.5},
    {"success_rate": 0.0}, {"success_rate": 1.1},
    {"min_progress_ratio": 0.0}, {"min_progress_ratio": 1.3},
    {"min_episode_fraction": 1.1}, {"min_active_time": 0.0},
    {"min_target_distance": 0.0}, {"generation": -1},
])
def test_invalid_evaluator_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        evaluator(**kwargs)


@pytest.mark.parametrize("field", ["success_rate", "min_progress_ratio", "max_progress_ratio", "min_episode_fraction", "min_active_time", "min_target_distance"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_thresholds_are_rejected(field, value):
    with pytest.raises(ValueError):
        evaluator(**{field: value})


def load_callback():
    path = TASK / "mdp/curriculums.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "lin_vel_cmd_levels"]
    assert len(functions) == 1
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {"torch": torch, "ProgressCurriculum": ProgressCurriculum, "expanded_x_range": module.expanded_x_range}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[])), str(path), "exec"), scope)
    return scope["lin_vel_cmd_levels"]


callback = load_callback()


def env_container():
    command = SimpleNamespace(cfg=SimpleNamespace(
        ranges=SimpleNamespace(lin_vel_x=(-0.1, 0.3), lin_vel_y=(-0.08, 0.08), ang_vel_z=(-0.3, 0.3)),
        limit_ranges=SimpleNamespace(lin_vel_x=(-1.0, 1.2)),
    ))
    return SimpleNamespace(
        command_progress_tracker=tracker(),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        termination_manager=SimpleNamespace(terminated=torch.zeros(4, dtype=torch.bool), time_outs=torch.ones(4, dtype=torch.bool)),
        max_episode_length=1000, max_episode_length_s=20.0, common_step_counter=1000,
        device="cpu", extras={},
    ), command


def test_production_callback_expands_only_x_and_preserves_return_contract():
    env, cmd = env_container()
    result = callback(env, [0, 1, 2, 3], min_episodes=4)
    assert cmd.cfg.ranges.lin_vel_x == pytest.approx((-0.2, 0.4))
    assert cmd.cfg.ranges.lin_vel_y == (-0.08, 0.08)
    assert cmd.cfg.ranges.ang_vel_z == (-0.3, 0.3)
    torch.testing.assert_close(result, torch.tensor(0.4))
    assert env._command_progress_generation == 1
    assert env.extras["log"]["Curriculum/command_progress_expanded"] == 1.0
    assert env._command_progress_pending_log == env.extras["log"]


def test_callback_stationary_does_not_expand_despite_complete_survival():
    env, cmd = env_container()
    env.command_progress_tracker.forward_signed_progress[:] = 0
    callback(env, [0, 1, 2, 3], min_episodes=4)
    assert cmd.cfg.ranges.lin_vel_x == (-0.1, 0.3)
    assert getattr(env, "_command_progress_generation", 0) == 0


def test_static_high_reward_passes_legacy_but_not_progress_curriculum():
    source = ast.parse((TASK / "mdp/curriculums.py").read_text(encoding="utf-8"))
    legacy = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                  and node.name == "lin_vel_cmd_levels_legacy")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, legacy], type_ignores=[])), "production_legacy_curriculum", "exec"), scope)

    old_env, old_command = env_container()
    old_command.cfg.limit_ranges.lin_vel_y = (-0.25, 0.25)
    old_env.command_progress_tracker.forward_signed_progress[:] = 0.0
    # A full episode at 95% normalized reward crosses the old 80% gate.
    old_env.reward_manager = SimpleNamespace(
        get_term_cfg=lambda name: SimpleNamespace(weight=8.0),
        _episode_sums={"track_lin_vel_xy": torch.full((4,), 8.0 * 0.95 * 20.0)},
    )
    scope[legacy.name](old_env, [0, 1, 2, 3])
    assert old_command.cfg.ranges.lin_vel_x == pytest.approx((-0.2, 0.4))

    new_env, new_command = env_container()
    new_env.command_progress_tracker.forward_signed_progress[:] = 0.0
    new_env.reward_manager = old_env.reward_manager
    callback(new_env, [0, 1, 2, 3], min_episodes=4)
    assert new_command.cfg.ranges.lin_vel_x == (-0.1, 0.3)


def test_callback_range_cap_does_not_advance_generation():
    env, cmd = env_container()
    cmd.cfg.ranges.lin_vel_x = cmd.cfg.limit_ranges.lin_vel_x
    callback(env, [0, 1, 2, 3], min_episodes=4)
    assert cmd.cfg.ranges.lin_vel_x == (-1.0, 1.2)
    assert getattr(env, "_command_progress_generation", 0) == 0
    assert env._command_progress_curriculum.generation == 0


def test_callback_missing_tracker_is_an_error_not_reward_fallback():
    env, _ = env_container()
    del env.command_progress_tracker
    with pytest.raises(RuntimeError, match="command_progress_tracker"):
        callback(env, [0])


def test_callback_rejects_live_settings_changes():
    env, _ = env_container()
    callback(env, [0], min_episodes=4)
    with pytest.raises(ValueError, match="settings"):
        callback(env, [1], min_episodes=5)


@pytest.mark.parametrize("current,limits,increment", [
    ((0.0, float("inf")), (-1.0, 1.0), 0.1),
    ((-0.1, 0.3), (-1.0, 1.0), float("nan")),
    ((-0.1, 0.3), (-1.0, 1.0), 0.0),
    ((-2.0, 0.3), (-1.0, 1.0), 0.1),
    ((0.3, -0.1), (-1.0, 1.0), 0.1),
])
def test_invalid_range_expansions_are_rejected(current, limits, increment):
    with pytest.raises(ValueError):
        module.expanded_x_range(current, limits, increment)


def test_range_expansion_clamps_each_endpoint_without_mutating_limits():
    limits = [-0.15, 0.35]
    assert module.expanded_x_range((-0.1, 0.3), limits, 0.1) == (-0.15, 0.35)
    assert limits == [-0.15, 0.35]
