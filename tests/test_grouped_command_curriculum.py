"""CPU grouped-progress and fixed callback diagnostics; no simulator imports."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from test_command_curriculum import ProgressCurriculum, callback, env_container, tracker


def _observe(evaluator, state, groups, ids=None, step=1000, **kwargs):
    count = state.elapsed_time.numel()
    return evaluator.observe(
        state, list(range(count)) if ids is None else ids,
        terminated=kwargs.get("terminated", torch.zeros(count, dtype=torch.bool)),
        time_outs=kwargs.get("time_outs", torch.ones(count, dtype=torch.bool)),
        max_episode_length_s=20.0, current_step=step, group_ids=groups,
    )


def _evaluator(**kwargs):
    settings = dict(interval_steps=1000, min_episodes=4, required_group_ids=(0, 1), min_group_episodes=2)
    settings.update(kwargs)
    return ProgressCurriculum(**settings)


def _terrain_env():
    env, command = env_container()
    env.scene = SimpleNamespace(terrain=SimpleNamespace(
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(curriculum=True, num_cols=2)),
        terrain_types=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    ))
    return env, command


def test_global_success_cannot_hide_a_failed_terrain_column():
    groups = torch.arange(10).repeat_interleave(10)
    state = tracker(100)
    state.forward_signed_progress[-10:] = 0.0
    evaluator = _evaluator(required_group_ids=tuple(range(10)), min_group_episodes=10)
    result = _observe(evaluator, state, groups)
    assert result["success_rate"] == 0.9
    assert result["passed"] is False
    assert result["group_success_rates"][9] == 0.0


def test_missing_group_samples_keep_the_window_open_across_intervals():
    groups = torch.tensor([0, 0, 1, 1])
    state = tracker()
    evaluator = _evaluator()
    assert _observe(evaluator, state, groups, [0, 1], 1200) is None
    assert evaluator.episode_count == 2
    assert evaluator.last_evaluation_step == 0
    result = _observe(evaluator, state, groups, [3, 2, 2], 2400)
    assert result["passed"] is True
    assert result["group_episodes"] == {0: 2, 1: 2}


def test_callback_emits_complete_keys_before_and_after_an_evaluation():
    env, command = _terrain_env()
    env.extras["log"] = {"other_manager": 9.0}
    callback(env, [0], min_episodes=4, grouping="terrain_columns", min_group_episodes=2)
    first = dict(env.extras["log"])
    assert first["Curriculum/command_progress_has_evaluation"] == 0.0
    callback(env, [1, 2, 3], min_episodes=4, grouping="terrain_columns", min_group_episodes=2)
    second = dict(env.extras["log"])
    assert first.keys() == second.keys()
    assert second["Curriculum/command_progress_evaluation_id"] == 1.0
    assert second["Curriculum/command_progress_expanded"] == 1.0
    callback(env, [], min_episodes=4, grouping="terrain_columns", min_group_episodes=2)
    assert env.extras["log"]["Curriculum/command_progress_expanded"] == 0.0
    assert env.extras["log"]["other_manager"] == 9.0
    assert command.cfg.ranges.lin_vel_x == pytest.approx((-0.2, 0.4))


def test_result_separates_active_progress_from_inactive_residual():
    state = tracker()
    state.forward_signed_progress[:] = torch.tensor([2.0, 1.0, 3.0, 2.0])
    state.inactive_forward_error[:] = torch.tensor([0.2, 0.4, 0.0, 0.2])
    result = _observe(_evaluator(), state, torch.tensor([0, 1, 0, 1]))
    assert result["active_progress_sum"] == pytest.approx(8.0)
    assert result["inactive_error_sum"] == pytest.approx(0.8)
    assert result["target_distance_sum"] == pytest.approx(8.0)
    assert result["distance_weighted_effective_ratio"] == pytest.approx(0.9)


def test_each_group_must_meet_the_exact_success_boundary():
    state = tracker(10)
    groups = torch.tensor([0] * 5 + [1] * 5)
    state.forward_signed_progress[[0, 5]] = 0.0
    result = _observe(_evaluator(min_group_episodes=5), state, groups)
    assert result["group_success_rates"] == {0: 0.8, 1: 0.8}
    assert result["passed"] is True


def test_ordinary_success_cannot_compensate_for_sixty_percent_stair_success():
    state = tracker(100)
    state.forward_signed_progress[-20:] = 0.0
    groups = torch.tensor([0] * 50 + [1] * 50)
    result = _observe(_evaluator(min_group_episodes=32), state, groups)
    assert result["success_rate"] == 0.8
    assert result["group_success_rates"] == {0: 1.0, 1: 0.6}
    assert result["passed"] is False


def test_group_assignment_indexes_environment_ids_not_reset_order():
    state = tracker(6)
    groups = torch.tensor([7, 2, 7, 2, 7, 2])
    state.forward_signed_progress[[0, 2, 4]] = 0.0
    result = _observe(_evaluator(required_group_ids=(7, 2)), state, groups, [4, 1, 3, 2, 2])
    assert result["group_episodes"] == {2: 2, 7: 2}
    assert result["group_success_rates"] == {2: 1.0, 7: 0.0}


def test_duplicate_resets_and_missing_samples_do_not_double_count_groups():
    state = tracker()
    groups = torch.tensor([0, 0, 1, 1])
    evaluator = _evaluator()
    assert _observe(evaluator, state, groups, [0, 0, 1], 1000) is None
    assert _observe(evaluator, state, groups, [1, 0], 2000) is None
    assert evaluator.group_episode_counts == {0: 2, 1: 0}
    result = _observe(evaluator, state, groups, [2, 3], 2100)
    assert result["episodes"] == 4
    assert result["evaluation_id"] == 1
    assert evaluator.group_episode_counts == {0: 0, 1: 0}


@pytest.mark.parametrize("failure", ["early", "invalid", "terminated"])
def test_bad_natural_episodes_stay_in_their_own_group_denominator(failure):
    state = tracker()
    groups = torch.tensor([0, 0, 1, 1])
    terminated = torch.zeros(4, dtype=torch.bool)
    if failure == "early":
        state.elapsed_time[2:] = 1.0
    elif failure == "invalid":
        state.valid[2:] = False
        state.had_forward_command[2:] = False
    else:
        terminated[2:] = True
    result = _observe(_evaluator(), state, groups, terminated=terminated)
    assert result["group_episodes"] == {0: 2, 1: 2}
    assert result["group_success_rates"] == {0: 1.0, 1: 0.0}


def test_old_generation_and_nontranslation_resets_are_not_group_evidence():
    state = tracker(8)
    groups = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    state.episode_generation[:2] = 1
    state.had_forward_command[2:4] = False
    time_outs = torch.ones(8, dtype=torch.bool)
    time_outs[4] = False  # External/manual reset, not a natural episode.
    evaluator = _evaluator(min_group_episodes=1, min_episodes=3)
    result = _observe(evaluator, state, groups, time_outs=time_outs)
    assert result["group_episodes"] == {0: 1, 1: 2}
    assert evaluator.advance_generation() == 1
    assert _observe(evaluator, tracker(8), groups, step=2000) is None
    fresh = tracker(8)
    fresh.episode_generation[:] = 1
    assert _observe(evaluator, fresh, groups, step=2100)["evaluation_id"] == 2


def test_distance_weighted_decomposition_is_not_mean_of_episode_ratios():
    state = tracker()
    state.forward_target_distance[:] = torch.tensor([1.0, 1.0, 3.0, 3.0])
    state.forward_signed_progress[:] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    state.inactive_forward_error[:] = torch.tensor([0.0, 0.0, 0.6, 0.6])
    result = _observe(_evaluator(), state, torch.tensor([0, 1, 0, 1]))
    assert result["mean_progress_ratio"] == pytest.approx(0.4)
    assert result["distance_weighted_active_ratio"] == pytest.approx(0.25)
    assert result["distance_weighted_inactive_error_ratio"] == pytest.approx(0.15)
    assert result["distance_weighted_effective_ratio"] == pytest.approx(0.1)


@pytest.mark.parametrize("field", ["forward_target_distance", "forward_signed_progress", "inactive_forward_error"])
def test_invalid_distance_samples_are_excluded_from_diagnostic_sums(field):
    state = tracker()
    getattr(state, field)[0] = float("nan")
    result = _observe(_evaluator(), state, torch.tensor([0, 0, 1, 1]))
    assert result["episodes"] == 4
    assert result["progress_ratio_samples"] == 3
    assert result["active_progress_sum"] == pytest.approx(6.0)
    assert result["target_distance_sum"] == pytest.approx(6.0)


@pytest.mark.parametrize("groups", [
    None, torch.zeros(4, dtype=torch.int32), torch.zeros(4, dtype=torch.bool),
    torch.zeros(4, 1, dtype=torch.long), torch.zeros(3, dtype=torch.long),
    torch.tensor([0, 0, -1, 1]), torch.tensor([0, 0, 2, 1]),
    torch.empty(4, dtype=torch.long, device="meta"),
])
def test_bad_group_ids_fail_before_consuming_reset_evidence(groups):
    state = tracker()
    with pytest.raises(ValueError, match="group_ids"):
        _observe(_evaluator(), state, groups, ids=[0])
    assert not state.reported.any()


@pytest.mark.parametrize("required", [(), (0, 0), (-1, 0), (True, 1), (0.5, 1), "0", 1])
def test_invalid_required_groups_are_explicit_errors(required):
    with pytest.raises(ValueError, match="required_group_ids"):
        _evaluator(required_group_ids=required)


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5])
def test_invalid_per_group_minimum_is_rejected(minimum):
    with pytest.raises(ValueError, match="min_group_episodes"):
        _evaluator(min_group_episodes=minimum)


def test_per_group_minimum_defaults_to_the_original_minimum():
    evaluator = _evaluator(min_episodes=4, min_group_episodes=None)
    assert _observe(evaluator, tracker(), torch.tensor([0, 0, 1, 1])) is None
    assert evaluator.group_episode_counts == {0: 2, 1: 2}


@pytest.mark.parametrize("seed", range(12))
def test_single_group_and_original_global_have_identical_old_result_fields(seed):
    generator = torch.Generator().manual_seed(seed)
    state_a, state_b = tracker(40), tracker(40)
    progress = torch.randint(0, 4, (40,), generator=generator).float()
    for state in (state_a, state_b):
        state.forward_signed_progress[:] = progress
    plain = ProgressCurriculum(interval_steps=1000, min_episodes=4)
    grouped = _evaluator(required_group_ids=(0,), min_group_episodes=4)
    result_plain = _observe(plain, state_a, None)
    result_grouped = _observe(grouped, state_b, torch.zeros(40, dtype=torch.long))
    for key in ("episodes", "successes", "success_rate", "mean_progress_ratio", "progress_ratio_samples", "passed"):
        assert result_plain[key] == result_grouped[key]


@pytest.mark.parametrize("invalid", ["random_tiles", "missing_column", "negative_column", "wrong_dtype", "wrong_shape"])
def test_callback_rejects_invalid_fixed_column_topology(invalid):
    env, _ = _terrain_env()
    terrain = env.scene.terrain
    if invalid == "random_tiles":
        terrain.cfg.terrain_generator.curriculum = False
    elif invalid == "missing_column":
        terrain.terrain_types[:] = 0
    elif invalid == "negative_column":
        terrain.terrain_types[0] = -1
    elif invalid == "wrong_dtype":
        terrain.terrain_types = terrain.terrain_types.float()
    else:
        terrain.terrain_types = terrain.terrain_types[:, None]
    with pytest.raises(ValueError, match="terrain"):
        callback(env, [0], grouping="terrain_columns", min_episodes=4, min_group_episodes=2)
    assert not env.command_progress_tracker.reported.any()


@pytest.mark.parametrize("change", ["grouping", "columns", "minimum", "assignment_tensor"])
def test_callback_rejects_live_grouping_and_topology_changes(change):
    env, _ = _terrain_env()
    params = dict(grouping="terrain_columns", min_episodes=4, min_group_episodes=2)
    callback(env, [0], **params)
    if change == "grouping":
        params["grouping"] = "global"
    elif change == "columns":
        env.scene.terrain.cfg.terrain_generator.num_cols = 3
    elif change == "minimum":
        params["min_group_episodes"] = 3
    else:
        env.scene.terrain.terrain_types = env.scene.terrain.terrain_types.clone()
    with pytest.raises(ValueError, match="settings|replace"):
        callback(env, [1], **params)


def test_fixed_diagnostics_repeat_latest_window_and_keep_pulse_separate():
    env, _ = _terrain_env()
    params = dict(grouping="terrain_columns", min_episodes=4, min_group_episodes=2)
    callback(env, [0, 1, 2, 3], **params)
    first = dict(env._command_progress_pending_log)
    env.extras["log"] = {"new_manager_log": 7.0}
    env.common_step_counter += 1
    callback(env, [], **params)
    second = env._command_progress_pending_log
    assert first.keys() == second.keys()
    assert first["Curriculum/command_progress_expanded"] == 1.0
    assert second["Curriculum/command_progress_expanded"] == 0.0
    assert first["Curriculum/command_progress_evaluation_id"] == second["Curriculum/command_progress_evaluation_id"] == 1
    assert first["Curriculum/command_progress_group_0_success_rate"] == second["Curriculum/command_progress_group_0_success_rate"] == 1.0
    assert env.extras["log"]["new_manager_log"] == 7.0


@pytest.mark.parametrize("replacement", [[0, 1, 0, 1], [0, 0, 0, 0]])
def test_callback_rejects_in_place_column_changes_without_consuming_evidence(replacement):
    env, _ = _terrain_env()
    params = dict(grouping="terrain_columns", min_episodes=4, min_group_episodes=2)
    callback(env, [0], **params)
    evaluator = env._command_progress_curriculum
    counts_before = dict(evaluator.group_episode_counts)
    successes_before = dict(evaluator.group_success_counts)
    reported_before = env.command_progress_tracker.reported.clone()
    env.scene.terrain.terrain_types[:] = torch.tensor(replacement)
    env.common_step_counter += 1
    with pytest.raises(ValueError, match="fixed.*group_ids|group_ids.*fixed"):
        callback(env, [1], **params)
    assert evaluator.group_episode_counts == counts_before == {0: 1, 1: 0}
    assert evaluator.group_success_counts == successes_before
    assert evaluator.episode_count == 1
    assert evaluator._last_observation_step == 1000
    assert torch.equal(env.command_progress_tracker.reported, reported_before)


@pytest.mark.parametrize("in_place", [False, True])
def test_core_freezes_group_values_independently_of_tensor_identity(in_place):
    state = tracker()
    groups = torch.tensor([0, 0, 1, 1])
    evaluator = _evaluator()
    assert _observe(evaluator, state, groups, [0]) is None
    changed = groups if in_place else groups.clone()
    changed[:] = torch.tensor([0, 1, 0, 1])
    with pytest.raises(ValueError, match="group_ids.*fixed"):
        _observe(evaluator, state, changed, [1], 2000)
    assert evaluator.group_episode_counts == {0: 1, 1: 0}
    assert state.reported.tolist() == [True, False, False, False]


def test_group_map_is_frozen_even_before_the_first_candidate_reset():
    state = tracker()
    state.reset_ready[:] = False
    groups = torch.tensor([0, 0, 1, 1])
    evaluator = _evaluator()
    assert _observe(evaluator, state, groups, [], 1000) is None
    groups[:] = torch.tensor([0, 1, 0, 1])
    state.reset_ready[:] = True
    with pytest.raises(ValueError, match="group_ids.*fixed"):
        _observe(evaluator, state, groups, [1], 2000)
    assert evaluator.episode_count == 0
    assert not state.reported.any()


def test_same_fixed_values_in_a_new_core_tensor_are_safe():
    state = tracker()
    groups = torch.tensor([0, 0, 1, 1])
    evaluator = _evaluator()
    assert _observe(evaluator, state, groups, [0]) is None
    result = _observe(evaluator, state, groups.clone(), [1, 2, 3], 2000)
    assert result["passed"] is True
