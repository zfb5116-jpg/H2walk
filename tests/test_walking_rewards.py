"""Synthetic walking transitions; no simulator, model or optimizer is created."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"


@pytest.fixture
def state():
    spec = spec_from_file_location("walking_rewards_cpu", TASK / "walking_rewards.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WalkingRewardState(2, "cpu")


def transition(state, *, dx=0., foot=0, foot_x=0., support=(True, True), contact=None,
               dt=.02, command=.2, valid=(True, True)):
    if not hasattr(state, "test_root"):
        state.test_root = torch.zeros(2, 2)
        state.test_feet = torch.zeros(2, 2, 3)
    raw = torch.tensor([support if contact is None else contact] * 2)
    sup = torch.tensor([support] * 2)
    validity = torch.tensor([valid] * 2)
    start = state.test_root.clone()
    feet = state.test_feet.clone()
    state.begin_transition(start, torch.tensor([[1., 0., 0., 0.]] * 2),
                           torch.tensor([[command, 0., 0.]] * 2), feet,
                           raw, sup, validity, dt)
    state.test_root[:, 0] += dx
    state.test_feet[:, foot, 0] += foot_x
    state.finish_transition(state.test_root, state.test_feet, raw, sup, validity)
    return state


def test_signed_progress_distinguishes_rest_from_pulse_and_cancels_reversal(state):
    assert transition(state).signed_progress.tolist() == [0., 0.]
    forward = transition(state, dx=.008).signed_progress.clone()
    backward = transition(state, dx=-.008).signed_progress.clone()
    torch.testing.assert_close(forward + backward, torch.zeros(2))
    assert forward.mean() > 0


def test_signed_progress_has_no_support_gate_or_step_saturation(state):
    a = transition(state, dx=.02).signed_progress.clone()
    b = transition(state, dx=-.02, support=(False, False)).signed_progress.clone()
    torch.testing.assert_close(a + b, torch.zeros(2))
    assert a[0] == pytest.approx(5.)


@pytest.mark.parametrize("command,dx", [(-.2, -.004), (.2, .004)])
def test_progress_follows_command_direction(state, command, dx):
    torch.testing.assert_close(transition(state, command=command, dx=dx).signed_progress, torch.ones(2))


def test_initial_air_static_hold_and_vertical_motion_have_no_step_or_swing_credit(state):
    transition(state, support=(False, True))
    assert not state.swing.any()
    assert not state.alternating_steps.any()
    assert not state.moving_swing.any()


def perform_step(state, foot, *, root=.012, distance=.06, other=True, dt=.02, command=.2):
    import math

    transition(state, dt=dt, command=command)
    sup = [True, True]
    sup[foot] = False
    if not other:
        sup[1-foot] = False
    count = math.ceil(.1/dt)
    for _ in range(count):
        transition(state, dx=root/count, foot=foot, foot_x=distance/count, support=tuple(sup), dt=dt, command=command)
    transition(state, dt=dt, command=command)
    return state.alternating_steps.clone()


def test_real_alternating_steps_earn_but_repeat_same_foot_does_not(state):
    assert (perform_step(state, 0) > 0).all()
    assert not perform_step(state, 0).any()
    assert (perform_step(state, 1) > 0).all()


def test_repeated_zero_net_retreat_and_step_to_same_peak_cannot_farm_landings(state):
    assert perform_step(state, 0).any()
    for foot in (1, 0, 1):
        transition(state, dx=-.012)
        assert not perform_step(state, foot).any()


def test_deliberately_reversing_command_can_earn_real_backward_alternating_step(state):
    assert perform_step(state, 0).any()
    assert perform_step(state, 1, root=-.012, distance=-.06, command=-.2).any()


def test_partial_reset_clears_only_selected_frontier_and_alternation_sequence(state):
    assert perform_step(state, 0).any()
    state.reset([0])
    event = perform_step(state, 0)
    assert event[0] > 0 and event[1] == 0


@pytest.mark.parametrize("kwargs", [dict(root=0.), dict(distance=0.), dict(other=False)])
def test_march_in_place_vertical_toe_raise_and_jump_are_not_successful_steps(state, kwargs):
    assert not perform_step(state, 0, **kwargs).any()


def test_partial_reset_does_not_leak_cycle_or_touch_other_environment(state):
    transition(state)
    transition(state, support=(False, True), foot_x=.02)
    state.reset(torch.tensor([0]))
    assert state.swing.tolist() == [[False, False], [True, False]]


def test_invalid_support_and_riser_contact_do_not_create_landing(state):
    transition(state)
    transition(state, support=(False, True), foot_x=.06)
    transition(state, support=(False, True), contact=(True, True), valid=(False, True))
    assert not state.alternating_steps.any()


@pytest.mark.parametrize("dt", [.01, .02, .04])
def test_event_integral_is_independent_of_control_dt(state, dt):
    event = perform_step(state, 0, dt=dt)
    torch.testing.assert_close(event * dt, torch.ones(2))


def test_long_hover_stops_positive_swing_shaping(state):
    transition(state)
    for _ in range(70):
        transition(state, support=(False, True), foot_x=.001)
    assert not state.moving_swing.any()


def test_latched_world_target_does_not_change_mid_swing_and_clears_on_landing(state):
    transition(state)
    z = torch.full((2, 2), .15)
    weight = torch.ones_like(z)
    valid = torch.ones_like(z, dtype=torch.bool)
    state.hold_targets(z, weight, valid, torch.ones(2, dtype=torch.bool))
    transition(state, support=(False, True), foot_x=.02)
    first = state.hold_targets(z+.1, weight, valid, torch.ones(2, dtype=torch.bool))[0].clone()
    transition(state, support=(False, True), foot_x=.02)
    held = state.hold_targets(z+.2, weight, valid, torch.ones(2, dtype=torch.bool))[0]
    torch.testing.assert_close(first[:, 0], torch.full((2,), .15))
    torch.testing.assert_close(held[:, 0], first[:, 0])
    transition(state)
    assert not state.hold_targets(z, weight, valid, torch.ones(2, dtype=torch.bool))[2].any()


def test_missing_target_at_liftoff_is_not_acquired_in_midair(state):
    transition(state)
    z = torch.ones(2, 2)
    state.hold_targets(z, z, torch.zeros_like(z, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    transition(state, support=(False, True), foot_x=.02)
    state.hold_targets(z, z, torch.ones_like(z, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    transition(state, support=(False, True), foot_x=.02)
    assert not state.hold_targets(z*2, z, torch.ones_like(z, dtype=torch.bool), torch.ones(2, dtype=torch.bool))[2].any()


def test_stance_arming_survives_raw_contact_history_delay(state):
    transition(state)
    transition(state, support=(False, True), contact=(True, True))
    transition(state, support=(False, True), contact=(False, True), foot_x=.02)
    assert state.liftoff.tolist() == [[True, False], [True, False]]


def test_command_gate_exit_clears_lock_and_no_relock_in_air(state):
    transition(state)
    z = torch.ones(2, 2)
    valid = torch.ones_like(z, dtype=torch.bool)
    gate = torch.ones(2, dtype=torch.bool)
    state.hold_targets(z, z, valid, gate)
    transition(state, support=(False, True), foot_x=.02)
    assert state.hold_targets(z, z, valid, gate)[2][:, 0].all()
    state.hold_targets(z, z, valid, ~gate)
    transition(state, support=(False, True), foot_x=.02)
    assert not state.hold_targets(z, z, valid, gate)[2].any()


def test_held_target_survives_missing_current_scan_without_changing_world_height(state):
    transition(state)
    z = torch.full((2, 2), 7.15)
    valid = torch.ones_like(z, dtype=torch.bool)
    gate = torch.ones(2, dtype=torch.bool)
    state.hold_targets(z, z, valid, gate)
    transition(state, support=(False, True), foot_x=.02)
    state.hold_targets(z, z, valid, gate)
    transition(state, support=(False, True), foot_x=.02)
    held, _, usable = state.hold_targets(z*float("nan"), z, ~valid, gate)
    torch.testing.assert_close(held[:, 0], z[:, 0])
    assert usable[:, 0].all()


def test_invalid_channel_clears_stance_arm_until_new_qualified_support(state):
    transition(state)
    transition(state, support=(False, True), valid=(False, True))
    transition(state, support=(False, True), foot_x=.03)
    assert not state.swing.any()
    transition(state)
    transition(state, support=(False, True), foot_x=.03)
    assert state.swing[:, 0].all()


@pytest.mark.parametrize("pre_fault", ["channel", "opposite_support", "foot_pose"])
def test_pre_step_fault_invalidates_cycle_even_if_post_step_channels_recover(state, pre_fault):
    transition(state)
    for _ in range(4):
        transition(state, dx=.004, foot_x=.012, support=(False, True))
    root = state.test_root.clone()
    feet = state.test_feet.clone()
    contacts = torch.tensor([[False, True]]*2)
    support = contacts.clone()
    valid = torch.ones_like(support)
    if pre_fault == "channel":
        valid[0, 0] = False
    elif pre_fault == "opposite_support":
        support[0, 1] = False
    else:
        feet[0, 0, 2] = float("nan")
    state.begin_transition(root, torch.tensor([[1., 0., 0., 0.]]*2),
                           torch.tensor([[.2, 0., 0.]]*2), feet, contacts, support, valid, .02)
    state.test_root[:, 0] += .004
    state.test_feet[:, 0, 0] += .012
    state.finish_transition(state.test_root, state.test_feet, contacts, contacts,
                            torch.ones_like(support))
    transition(state, dx=.004)
    assert state.alternating_steps.tolist() == [0., 50.]


def test_landing_refreshes_next_stance_candidate_without_waiting_an_extra_step(state):
    transition(state)
    z = torch.full((2, 2), .15)
    valid = torch.ones_like(z, dtype=torch.bool)
    gate = torch.ones(2, dtype=torch.bool)
    state.hold_targets(z, z, valid, gate)
    transition(state, support=(False, True), foot_x=.02)
    state.hold_targets(z, z, valid, gate)
    transition(state)
    state.hold_targets(z+.2, z, valid, gate)
    transition(state, support=(False, True), foot_x=.02)
    held = state.hold_targets(z+.3, z, valid, gate)[0]
    torch.testing.assert_close(held[:, 0], torch.full((2,), .35))


def test_progress_uses_captured_yaw_and_zero_command_has_no_credit(state):
    root = torch.zeros(2, 2)
    feet = torch.zeros(2, 2, 3)
    both = torch.ones(2, 2, dtype=torch.bool)
    q = torch.tensor([[2**-.5, 0., 0., 2**-.5]]*2)
    state.begin_transition(root, q, torch.tensor([[.2, 0., 0.], [0., 0., .2]]),
                           feet, both, both, both, .02)
    end = root.clone()
    end[:, 1] = .004
    state.finish_transition(end, feet, both, both, both)
    torch.testing.assert_close(state.signed_progress, torch.tensor([1., 0.]))


def test_implausible_or_nonfinite_progress_is_zero_for_only_affected_row(state):
    root = torch.zeros(2, 2)
    feet = torch.zeros(2, 2, 3)
    both = torch.ones(2, 2, dtype=torch.bool)
    q = torch.tensor([[1., 0., 0., 0.]]*2)
    state.begin_transition(root, q, torch.tensor([[.2, 0., 0.]]*2), feet, both, both, both, .02)
    end = torch.tensor([[50., 0.], [.004, 0.]])
    state.finish_transition(end, feet, both, both, both)
    torch.testing.assert_close(state.signed_progress, torch.tensor([0., 1.]))


def test_real_adapter_and_reward_wrappers_share_terminal_transition_once(state):
    import ast
    from types import SimpleNamespace

    data = SimpleNamespace(root_pos_w=torch.zeros(2, 3),
                           root_quat_w=torch.tensor([[1., 0., 0., 0.]]*2),
                           body_pos_w=torch.zeros(2, 2, 3))
    support = torch.ones(2, 2, dtype=torch.bool)
    valid = torch.ones_like(support)
    force = torch.zeros(2, 3, 2, 3)
    force[..., 2] = 10.
    env = SimpleNamespace(asset=SimpleNamespace(data=data), feet_id_b=[0, 1], step_dt=.02,
        command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[.2, 0., 0.]]*2)),
        walking_runtime=SimpleNamespace(_support_state=lambda: (torch.zeros(2, 2), support, valid),
            net_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w_history=force)), feet_ids=[0, 1]),
        walking_reward_state=state)
    tree = ast.parse((TASK/"mdp/rewards.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"walk_signed_progress", "walk_alternating_steps", "walk_support_quality"}]
    scope = {}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[])),
                 "walking_wrappers", "exec"), scope)
    state.begin_step(env)
    data.root_pos_w[:, 0] += .004
    # Calling support first must still settle exactly one executed transition.
    torch.testing.assert_close(scope["walk_support_quality"](env), torch.zeros(2))
    before_reset = scope["walk_signed_progress"](env).clone()
    torch.testing.assert_close(scope["walk_signed_progress"](env), before_reset)
    torch.testing.assert_close(before_reset, torch.ones(2))
    assert not scope["walk_alternating_steps"](env).any()
    # Terminal row is cleared after reward, before a reset teleport can be read.
    state.reset([0])
    data.root_pos_w[0, 0] = 50.
    assert scope["walk_signed_progress"](env).tolist() == [0., 1.]
    state.begin_step(env)
    data.root_pos_w[:, 0] += .004
    torch.testing.assert_close(scope["walk_signed_progress"](env), torch.ones(2), atol=5e-4, rtol=0)
