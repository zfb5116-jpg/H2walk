"""CPU stage evidence tests; no Isaac import, environment or policy execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"
SPEC = importlib.util.spec_from_file_location("h2_staged_curriculum_test", TASK / "staged_curriculum.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
StairTraversalTracker = MODULE.StairTraversalTracker
StageEvaluator = MODULE.StageEvaluator


def command_state(n=3, ratio=1.0, generation=0):
    return SimpleNamespace(
        elapsed_time=torch.full((n,), 3.0), forward_active_time=torch.full((n,), 3.0),
        forward_target_distance=torch.full((n,), .6), forward_signed_progress=torch.full((n,), .6 * ratio),
        inactive_forward_error=torch.zeros(n), valid=torch.ones(n, dtype=torch.bool),
        reset_ready=torch.ones(n, dtype=torch.bool), episode_generation=torch.full((n,), generation, dtype=torch.long),
    )


def trace(heights=None, n=3, contacts=None, generation=0):
    tracker = StairTraversalTracker(n, "cpu")
    tracker.reset(list(range(n)), generation=generation)
    root = torch.zeros(n, 3)
    if heights is None:
        heights = [torch.zeros(n, 2), torch.tensor([[0., 0.], [.02, .02], [-.02, -.02]]),
                   torch.tensor([[0., 0.], [.04, .04], [-.04, -.04]])]
    for index, height in enumerate(heights):
        tracker.begin_step(root, 1.0)
        root = root.clone()
        root[:, 0] += .2
        tracker.finish_step(None, root, height, torch.ones(n, 2, dtype=torch.bool) if contacts is None else contacts,
                            for_reset=index == len(heights) - 1)
    return tracker


def evaluator(stage="shallow", **kwargs):
    params = {"min_group_episodes": 1, "interval_steps": 1}
    params.update(kwargs)
    return StageEvaluator(stage, **params)


def observe(e, t, command=None, ids=None, families=None, **kwargs):
    n = t.elapsed_time.numel()
    return e.observe(command_state(n) if command is None else command, t, list(range(n)) if ids is None else ids,
                     family_ids=torch.tensor([0, 1, 2]) if families is None else families,
                     terminated=kwargs.get("terminated", torch.zeros(n, dtype=torch.bool)),
                     time_outs=kwargs.get("time_outs", torch.ones(n, dtype=torch.bool)),
                     max_episode_length_s=3.0, current_step=kwargs.get("step", 3))


def test_shared_stage_names_heights_and_family_columns():
    assert MODULE.STAGE_ORDER == ("bootstrap", "shallow", "stairs_08", "stairs_12", "stairs_16", "stairs_20")
    assert MODULE.stage_column_families("bootstrap") == ("flat",)
    assert MODULE.stage_column_families("shallow") == ("flat",) * 6 + ("up",) * 2 + ("down",) * 2
    # A random range ending at .20 would almost surely never generate an
    # exact 20 cm tread in the single-row final stage.
    assert MODULE.STAGE_HEIGHT_RANGES["stairs_20"] == (.20, .20)
    with pytest.raises(ValueError, match="stage"):
        MODULE.stage_column_families("automatic")


def test_all_families_need_real_supported_traversal_then_propose_next_stage():
    result = observe(evaluator(), trace())
    assert result["passed"] is True
    assert result["next_stage"] == "stairs_08"
    assert result["group_success_rates"] == {0: 1.0, 1: 1.0, 2: 1.0}


def test_stationary_and_short_episodes_cannot_pass():
    state = command_state(ratio=0.0)
    assert observe(evaluator(), trace(), state)["passed"] is False
    state = command_state()
    state.elapsed_time[:] = 1.0
    assert observe(evaluator(), trace(), state)["passed"] is False


def test_one_failed_stair_family_cannot_be_hidden_by_flat_success():
    state = command_state()
    state.forward_signed_progress[2] = 0.
    result = observe(evaluator(success_rate=.6), trace(), state)
    assert result["success_rate"] == pytest.approx(2 / 3)
    assert result["passed"] is False
    assert result["next_stage"] is None


def test_no_filtered_ground_support_is_missing_evidence_not_success():
    t = trace(contacts=torch.zeros(3, 2, dtype=torch.bool))
    assert observe(evaluator(), t)["passed"] is False
    t = trace()
    t.support_seen[1, 1] = False
    assert observe(evaluator(), t)["group_success_rates"][1] == 0.


def test_one_foot_or_one_tall_step_is_not_two_supported_stair_events():
    heights = [torch.zeros(3, 2), torch.tensor([[0., 0.], [.04, .04], [-.04, -.04]]),
               torch.tensor([[0., 0.], [.04, .04], [-.04, -.04]])]
    assert observe(evaluator(), trace(heights))["passed"] is False
    heights[1][1] = torch.tensor([.02, 0.])
    heights[2][1] = torch.tensor([.04, 0.])
    assert observe(evaluator(), trace(heights))["group_success_rates"][1] == 0.


def test_returning_to_same_tread_or_wrong_vertical_direction_cannot_pass():
    t = trace()
    t.last_supported_height[1] = 0.
    assert observe(evaluator(), t)["group_success_rates"][1] == 0.
    heights = [torch.zeros(3, 2), torch.tensor([[0., 0.], [-.02, -.02], [.02, .02]]),
               torch.tensor([[0., 0.], [-.04, -.04], [.04, .04]])]
    assert observe(evaluator(), trace(heights))["passed"] is False


def test_repeated_old_treads_do_not_create_new_record_events():
    heights = [torch.zeros(1, 2), torch.full((1, 2), .02), torch.zeros(1, 2), torch.full((1, 2), .02)]
    t = trace(heights, n=1)
    assert t.up_events.tolist() == [[1, 1]]
    assert t.down_events.tolist() == [[0, 0]]


def test_initial_manual_reset_and_duplicate_final_consumption_are_ignored():
    t = StairTraversalTracker(3, "cpu")
    t.finish_step([0, 1, 2], torch.zeros(3, 3), torch.zeros(3, 2), torch.ones(3, 2, dtype=torch.bool), for_reset=True)
    e = evaluator()
    assert observe(e, t) is None
    t = trace()
    result = observe(e, t, ids=[0, 0, 1, 2])
    assert result["episodes"] == 3
    assert observe(e, t, step=4) is None
    t = trace()
    assert observe(evaluator(), t, time_outs=torch.zeros(3, dtype=torch.bool)) is None


def test_partial_terminal_finish_consumes_pending_once():
    t = StairTraversalTracker(2, "cpu")
    root = torch.zeros(2, 3)
    t.begin_step(root, .02)
    terminal = root.clone()
    terminal[:, 0] = .002
    t.finish_step([0], terminal, torch.zeros(2, 2), torch.ones(2, 2, dtype=torch.bool), for_reset=True)
    terminal[0, 0] = 100.
    t.finish_step(None, terminal, torch.zeros(2, 2), torch.ones(2, 2, dtype=torch.bool))
    assert t.elapsed_time.tolist() == pytest.approx([.02, .02])
    assert t.reset_ready.tolist() == [True, False]
    assert t.valid.tolist() == [True, True]


def test_teleport_and_nonfinite_ground_evidence_are_failures():
    t = StairTraversalTracker(1, "cpu")
    root = torch.zeros(1, 3)
    t.begin_step(root, .02)
    t.finish_step(None, torch.ones(1, 3), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))
    assert not t.valid.item()
    t = trace()
    t.valid[1] = False
    assert observe(evaluator(), t)["group_success_rates"][1] == 0.
    t = StairTraversalTracker(1, "cpu")
    t.begin_step(root, .02)
    t.finish_step(None, root, torch.full((1, 2), torch.nan), torch.ones(1, 2, dtype=torch.bool))
    assert not t.valid.item()


def test_failure_overrides_timeout_and_missing_family_keeps_window_open():
    result = observe(evaluator(), trace(), terminated=torch.tensor([False, True, False]))
    assert result["group_success_rates"][1] == 0.
    e, t = evaluator(), trace()
    assert observe(e, t, ids=[0, 1]) is None
    result = observe(e, t, ids=[2], step=4)
    assert result["passed"] is True


def test_bootstrap_still_requires_ground_support_and_real_progress():
    t = trace([torch.zeros(1, 2)] * 3, n=1)
    result = observe(evaluator("bootstrap"), t, families=torch.zeros(1, dtype=torch.long))
    assert result["passed"] is True
    assert result["next_stage"] == "shallow"
    t = trace([torch.zeros(1, 2)] * 3, n=1, contacts=torch.zeros(1, 2, dtype=torch.bool))
    assert not observe(evaluator("bootstrap"), t, families=torch.zeros(1, dtype=torch.long))["passed"]


def test_wrong_generation_and_mutated_family_assignments_do_not_pass():
    assert observe(evaluator(), trace(generation=1), command=command_state(generation=1)) is None
    e, t = evaluator(), trace()
    assert observe(e, t, ids=[0]) is None
    with pytest.raises(ValueError, match="fixed"):
        observe(e, t, ids=[1], families=torch.tensor([0, 2, 1]), step=4)


def test_final_stage_has_no_further_stage_and_evidence_is_not_checkpointed():
    heights = [torch.zeros(3, 2), torch.tensor([[0., 0.], [.16, .16], [-.16, -.16]]),
               torch.tensor([[0., 0.], [.32, .32], [-.32, -.32]])]
    result = observe(evaluator("stairs_20"), trace(heights))
    assert result["passed"] is True
    assert result["next_stage"] is None
    assert result["window_checkpointed"] is False


@pytest.mark.parametrize("settings", [
    {"min_group_episodes": 0}, {"min_group_episodes": True}, {"interval_steps": -1},
    {"min_stair_events": 0}, {"generation": -1}, {"success_rate": 0.}, {"success_rate": 1.1},
    {"min_episode_fraction": 0.}, {"min_progress_ratio": 1.3}, {"min_target_distance": 0.},
    {"min_active_time": 0.}, {"success_rate": float("nan")},
])
def test_invalid_evaluation_thresholds_are_rejected(settings):
    with pytest.raises(ValueError):
        StageEvaluator("shallow", **settings)


@pytest.mark.parametrize("settings", [{"num_envs": 0}, {"num_envs": True},
                                     {"min_level_change": 0.}, {"max_step_speed": float("inf")}])
def test_invalid_tracker_settings_are_rejected(settings):
    values = {"num_envs": 1, "device": "cpu"}
    values.update(settings)
    with pytest.raises(ValueError):
        StairTraversalTracker(**values)


def test_input_contracts_and_aborted_transitions_remain_conservative():
    t = StairTraversalTracker(1, "cpu")
    root = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="dt"):
        t.begin_step(root, 0.)
    with pytest.raises(ValueError, match="shape"):
        t.begin_step(torch.zeros(1, 2), .02)
    t.begin_step(root, .02)
    with pytest.raises(RuntimeError, match="pending"):
        t.begin_step(root, .02)
    with pytest.raises(ValueError, match="dtype"):
        t.finish_step(None, root, torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="out of range"):
        t.finish_step([1], root, torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))
    t.abort_step()
    assert not t.pending.item() and not t.reset_ready.item() and not t.valid.item()
    t.reset([0])
    assert t.valid.item()
    t.begin_step(root, .02)
    t.finish_step(None, root, torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))
    root[0, 0] = .5
    t.begin_step(root, .02)
    assert not t.valid.item()


def test_family_dtype_undeclared_family_and_backward_step_reject_before_consumption():
    t, e = trace(), evaluator()
    with pytest.raises(ValueError, match="dtype"):
        observe(e, t, families=torch.tensor([0., 1., 2.]))
    with pytest.raises(ValueError, match="undeclared"):
        observe(e, t, families=torch.tensor([0, 1, 3]))
    assert not t.reported.any()
    observe(e, t)
    with pytest.raises(ValueError, match="nondecreasing"):
        observe(e, t, step=2)
