"""Budget-v3 production state and pre-reset integration on CPU containers."""
import ast
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as Box

import pytest
import torch

from test_terrain_walking_rewards import flat_rays

TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"


@pytest.fixture
def module():
    package = ModuleType("budget_state_cpu")
    package.__path__ = [str(TASK)]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__+".budget_walking_rewards")


class Transition:
    def __init__(self, module):
        self.state = module.BudgetWalkingRewardState(2, "cpu", goal_distance=4.)
        self.root = torch.zeros(2, 2)
        self.feet = torch.tensor([[[0., .12, .0543], [0., -.12, .0543]]]*2)
        self.q = torch.tensor([[[1., 0., 0., 0.]]*2]*2)
        self.support = torch.ones(2, 2, dtype=torch.bool)
        self.valid = self.support.clone()
        self.command = torch.tensor([[.2, 0., 0.]]*2)
        self.gravity = torch.tensor([[0., 0., -1.]]*2)
        self.rays = flat_rays()

    def step(self, *, dx=.004, dy=0., foot=None, foot_dx=0., foot_dy=0., support=(True, True), dt=.02,
             velocity=None, gravity=None, unknown_scan=False):
        s = self.state
        before = self.root.clone()
        s.begin_transition(self.root, self.q[:, 0], self.command, self.feet,
                           self.support, self.support, self.valid, dt)
        s.prepare_terrain(self.rays, self.feet, self.q, self.support, self.valid)
        s.progress.begin_transition(self.root, self.q[:, 0], self.command, dt)
        self.root[:, 0] += dx
        self.root[:, 1] += dy
        if foot is not None:
            self.feet[:, foot, 0] += foot_dx
            self.feet[:, foot, 1] += foot_dy
        self.support = torch.tensor([support]*2)
        if velocity is None:
            velocity = torch.cat(((self.root-before)/dt, torch.zeros(2, 1)), dim=-1)
        rays = torch.full_like(self.rays, torch.nan) if unknown_scan else self.rays
        s.finish_budget_transition(self.root, self.feet, self.support, self.support, self.valid,
            rays, self.q, torch.ones(2, 2)*245.25, torch.ones(2)*50., velocity,
            self.q[:, 0], self.gravity if gravity is None else gravity)
        return s


def cycle(t, foot):
    t.step()
    support = [True, True]
    support[foot] = False
    for _ in range(6):
        t.step(foot=foot, foot_dx=.08/6, support=tuple(support))
    return t.step()


def test_actual_qualified_events_count_both_feet_and_repeated_foot_is_rejected(module):
    t = Transition(module)
    cycle(t, 0)
    cycle(t, 0)
    cycle(t, 1)
    torch.testing.assert_close(t.state.landings_per_foot, torch.tensor([[1, 1], [1, 1]]))
    assert (t.state._rejections["repeat_foot"] > 0).all()


def test_new_mode_has_no_bank_for_overspeed_then_standing(module):
    t = Transition(module)
    s = t.step(dx=.08)
    paid = s.progress.paid_distance.clone()
    for _ in range(10):
        t.step(dx=0.)
        assert not s.progress.advance_rate.any()
    torch.testing.assert_close(s.progress.paid_distance, paid)
    assert s.episode_logs([0, 1])["BudgetWalk/route_forward_velocity"] > 0


def test_yaw_sideways_round_trips_cannot_farm_alternating_events(module):
    t = Transition(module)
    t.step(dx=0.)  # Fixed +X route.
    for foot, sign in [(0, 1.), (1, -1.), (0, 1.), (1, -1.)]:
        t.q[:, 0] = torch.tensor([2**-.5, 0., 0., sign*2**-.5])
        t.step(dx=0.)
        support = [True, True]
        support[foot] = False
        for _ in range(6):
            t.step(dx=0., dy=sign*.004, foot=foot, foot_dy=sign*.08/6, support=tuple(support))
        s = t.step(dx=0.)
        assert not s.alternating_steps.any()
    assert not s.landings_per_foot.any()
    assert not s.progress.paid_distance.any()
    assert (s._rejections["route_frontier"] > 0).all()


@pytest.mark.parametrize("command", [0., float("nan")])
def test_uninitialized_route_has_finite_reward_outputs_and_unknown_heading(module, command):
    t = Transition(module)
    t.command[:, 0] = command
    s = t.step()
    assert not s.progress.route_valid.any()
    for value in (s.yaw_cost, s.stall_cost, s.progress.advance_rate, s.alternating_steps):
        assert torch.isfinite(value).all()
    assert not s.success_evidence.any()
    assert torch.isnan(torch.tensor(s.episode_logs([0, 1])["BudgetWalk/yaw_error"]))


def test_finite_quaternion_with_overflowed_norm_cannot_poison_rewards(module):
    t = Transition(module)
    t.q[:, 0] = 1e38
    s = t.step()
    assert not s.progress.route_valid.any()
    assert torch.isfinite(s.yaw_cost).all()
    assert not s.success_evidence.any()


def test_unknown_velocity_rows_do_not_dilute_known_error_mean(module):
    t = Transition(module)
    velocity = torch.tensor([[float("nan"), 0., 0.], [.3, 0., 0.]])
    s = t.step(velocity=velocity)
    logs = s.episode_logs([0, 1])
    assert logs["BudgetWalk/position_physics_disagreement"] == pytest.approx(.1)
    assert logs["BudgetWalk/position_physics_disagreement_known_fraction"] == .5


def test_unknown_scan_and_missing_physics_velocity_are_separate_diagnostics(module):
    t = Transition(module)
    s = t.step(unknown_scan=True, velocity=torch.full((2, 3), torch.nan))
    logs = s.episode_logs([0, 1])
    assert logs["BudgetWalk/coverage_known_fraction"] == 0.
    assert logs["BudgetWalk/velocity_diagnostic_valid_fraction"] == 0.
    assert not s.success_evidence.any()
    assert (s.progress.advance_rate > 0).all()  # Position evidence remains independent.
    assert torch.isnan(torch.tensor(logs["BudgetWalk/position_physics_disagreement"]))
    assert logs["BudgetWalk/position_physics_disagreement_known_fraction"] == 0.


def prepare_near_goal(t):
    s = t.state
    # A deterministic terminal fixture: earlier physical evidence is supplied,
    # then production code evaluates the final continuous stable interval.
    t.step()
    t.root[:, 0] = 3.8
    s.progress._expected_position.copy_(t.root)
    s.progress.highwater.fill_(3.8)
    s.landings_per_foot.fill_(2)
    s.elapsed.fill_(19.8)
    s.supported_time.fill_(19.8)
    return s


def test_success_evidence_requires_continuous_quality_hold_not_just_goal_x(module):
    t = Transition(module)
    s = prepare_near_goal(t)
    for _ in range(9):
        t.step()
        assert not s.success_evidence.any()
    t.step()
    assert s.success_evidence.all()
    t.step(support=(False, False))
    assert not s.success_evidence.any()
    assert not s.stable_hold.any()


@pytest.mark.parametrize("fault", ["no_landings", "fast", "tilted", "velocity_mismatch", "no_support", "unknown"])
def test_terminal_goal_fixture_cannot_pass_bad_gait_or_data(module, fault):
    t = Transition(module)
    s = prepare_near_goal(t)
    if fault == "no_landings":
        s.landings_per_foot.zero_()
    kwargs = {
        "fast": {"dx": .08},
        "tilted": {"gravity": torch.tensor([[1., 0., 0.]]*2)},
        "velocity_mismatch": {"velocity": torch.zeros(2, 3)},
        "no_support": {"support": (False, False)},
        "unknown": {"unknown_scan": True},
    }.get(fault, {})
    if fault == "velocity_mismatch":
        kwargs["velocity"][:, 0] = 2.
    for _ in range(11):
        t.step(**kwargs)
    assert not s.success_evidence.any()


def test_gait_stall_is_not_cleared_by_moving_without_qualified_steps(module):
    t = Transition(module)
    for _ in range(4):
        t.step(dx=.2, dt=1.)
    assert (t.state.stall_cost == 1).all()
    assert not t.state.landings_per_foot.any()


def test_partial_reset_clears_only_requested_gait_and_budget_rows(module):
    t = Transition(module)
    cycle(t, 0)
    t.state.reset([0])
    assert not t.state.landings_per_foot[0].any()
    assert t.state.landings_per_foot[1, 0] == 1
    assert t.state.progress.paid_distance[0] == 0
    assert t.state.progress.paid_distance[1] > 0


def test_abort_before_finish_keeps_ledger_and_does_not_mask_parent_error(module):
    t = Transition(module)
    t.step()
    s = t.state
    paid = s.progress.paid_distance.clone()
    s.begin_transition(t.root, t.q[:, 0], t.command, t.feet, t.support, t.support, t.valid, .02)
    s.progress.begin_transition(t.root, t.q[:, 0], t.command, .02)
    s.abort_step()
    torch.testing.assert_close(s.progress.paid_distance, paid)
    assert not s.progress.route_valid.any()
    assert not s.success_evidence.any()
    assert not s._pending


def production_adapters():
    tree = ast.parse((TASK/"mdp/rewards.py").read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("budget_walk_")]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[])), "adapters", "exec"), scope)
    return scope


@pytest.mark.parametrize("outcome", ["valid", "failure", "bad_gait", "early_timeout"])
def test_actual_adapter_success_once_and_stage_share_terminal_evidence(module, outcome):
    from test_walking_runtime import environment, load_runtime

    runtime_module = load_runtime()
    env = environment(staged_evidence=True, walking_stage="bootstrap",
        walking_stage_column_families=("flat",), stage_min_group_episodes=1)
    env.max_episode_length = 1000
    env.scene.objects["terrain"] = Box(terrain_types=torch.zeros(2, dtype=torch.long), terrain_levels=torch.zeros(2,dtype=torch.long))
    env.scene.terrain = env.scene.objects["terrain"]
    env.scene.env_origins = torch.zeros(2, 3)
    env.cfg.scene = Box(terrain=Box(terrain_generator=Box(size=(10., 10.))))
    env.asset.data.root_quat_w = torch.tensor([[1., 0., 0., 0.]]*2)
    env.asset.data.root_lin_vel_w = torch.tensor([[.2, 0., 0.]]*2)
    env.masses = torch.ones(2)*50
    env.command_manager = Box(get_command=lambda _: torch.tensor([[.2, 0., 0.]]*2))
    command_module = importlib.import_module(runtime_module.__package__+".command_progress")
    env.command_progress_tracker = command_module.CommandProgressTracker(2, "cpu")
    runtime = env.walking_runtime = runtime_module.WalkingRuntime(env)
    state = env.walking_reward_state = module.BudgetWalkingRewardState(2, "cpu", goal_distance=4.)
    data = env.asset.data
    data.root_pos_w[:, 0] = 3.8
    data.body_pos_w[:, 1:, 0] = 3.8
    env.scene["height_scanner"].data.ray_hits_w = flat_rays()+torch.tensor([3.8, 0., 0.])
    for name in ("left_foot_ground", "right_foot_ground"):
        env.scene[name].data.force_matrix_w_history[..., 2] = 245.25
    env.scene["contact_forces"].data.net_forces_w_history[:, :, 1:, 2] = 245.25
    command = env.command_manager.get_command("base_velocity")
    state.progress.begin_transition(data.root_pos_w[:, :2], data.root_quat_w, command, .02)
    state.progress.finish_transition(data.root_pos_w[:, :2], data.root_lin_vel_w)
    state.progress._origin[:, 0] = 0.
    state.progress.highwater.fill_(3.8)
    state.elapsed.fill_(19.98)
    state.supported_time.fill_(19.98)
    state.stable_hold.fill_(.2)
    state.landings_per_foot.fill_(2)
    tracker = env.command_progress_tracker
    tracker.elapsed_time.fill_(19.98)
    tracker.forward_active_time.fill_(19.98)
    tracker.forward_target_distance.fill_(3.996)
    tracker.forward_signed_progress.fill_(3.8)
    runtime.traversal.elapsed_time.fill_(19.98)
    runtime.traversal.support_seen.fill_(True)
    tracker.begin_step(data.root_pos_w[:, :2], data.root_quat_w, command, .02)
    state.begin_step(env)
    runtime.begin_step()
    data.root_pos_w[:, 0] += .004
    env.termination_manager.terminated[:] = False
    env.termination_manager.time_outs[:] = True
    if outcome == "failure":
        env.termination_manager.terminated[1] = True
    elif outcome == "bad_gait":
        state.landings_per_foot[1] = 0
    elif outcome == "early_timeout":
        tracker.elapsed_time[1] = 9.98
        runtime.traversal.elapsed_time[1] = 9.98
    env.common_step_counter = 1000
    adapters = production_adapters()
    expected = torch.tensor([1., 1. if outcome == "valid" else 0.])
    torch.testing.assert_close(adapters["budget_walk_success"](env)*.02, expected)
    before = state.progress.paid_distance.clone()
    adapters["budget_walk_advance"](env)
    adapters["budget_walk_success"](env)
    torch.testing.assert_close(state.progress.paid_distance, before)
    logs, _ = runtime.before_reset([0, 1])
    assert logs["Stage/passed"] == (1. if outcome == "valid" else 0.)
    assert logs["BudgetWalk/success"] == float(expected.mean())
    assert logs["Stage/flat_success_rate"] == float(expected.mean())
    assert not tracker.pending.any()
    assert not runtime.traversal.pending.any()
    state.reset([0, 1])
    data.root_pos_w[:, 0] = 100.
    assert not adapters["budget_walk_success"](env).any()
