"""Synthetic transitions against production terrain reward state; no simulator."""
import ast
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"


@pytest.fixture
def module():
    name = "terrain_state_cpu"
    package = ModuleType(name)
    package.__path__ = [str(TASK)]
    sys.modules[name] = package
    return importlib.import_module(name + ".terrain_walking_rewards")


def flat_rays(height=0.):
    x, y = torch.meshgrid(torch.arange(-.4, .81, .05), torch.arange(-.4, .41, .05), indexing="ij")
    return torch.stack((x.flatten(), y.flatten(), torch.full_like(x.flatten(), height)), -1)[None].repeat(2, 1, 1)


class Transition:
    def __init__(self, module, dt=.02):
        self.module = module
        self.state = module.TerrainWalkingRewardState(2, "cpu")
        self.dt = dt
        self.root = torch.zeros(2, 2)
        self.feet = torch.tensor([[[0., .12, .0543], [0., -.12, .0543]]]*2)
        self.q = torch.tensor([[[1., 0., 0., 0.]]*2]*2)
        self.support = torch.ones(2, 2, dtype=torch.bool)
        self.contacts = self.support.clone()
        self.valid = self.support.clone()
        self.rays = flat_rays()

    def step(self, *, dx=0., left_dx=0., left_z=None, support=(True, True), contact=None, rays=None, force=0., command=.2):
        state = self.state
        state.begin_transition(self.root, self.q[:, 0], torch.tensor([[command, 0., 0.]]*2), self.feet,
                               self.contacts, self.support, self.valid, self.dt)
        state.prepare_terrain(self.rays, self.feet, self.q, self.support, self.valid)
        self.root[:, 0] += dx
        self.feet[:, 0, 0] += left_dx
        if left_z is not None:
            self.feet[:, 0, 2] = left_z
        self.support = torch.tensor([support]*2)
        self.contacts = self.support.clone() if contact is None else torch.tensor([contact]*2)
        if rays is not None:
            self.rays = rays
        state.finish_terrain_transition(self.root, self.feet, self.contacts, self.support, self.valid,
                                        self.rays, self.q, torch.full((2, 2), force), torch.ones(2)*50.)
        return state


def test_liftoff_locks_xyz_and_apex_despite_changed_terrain(module):
    t = Transition(module)
    t.step()
    s = t.step(dx=.004, left_dx=.02, left_z=.09, support=(False, True))
    assert s.target_valid[:, 0].all()
    landing, apex = s.target_landing.clone(), s.target_apex.clone()
    t.step(dx=.004, left_dx=.02, support=(False, True), rays=flat_rays(.15))
    torch.testing.assert_close(s.target_landing, landing)
    torch.testing.assert_close(s.target_apex, apex)


def test_delayed_raw_contact_locks_start_and_apex_from_same_stance(module):
    t = Transition(module)
    t.step()
    t.step(dx=.004, left_dx=.03, left_z=.1543, support=(False, True), contact=(True, True))
    s = t.step(dx=.004, left_dx=.03, left_z=.1743, support=(False, True), contact=(False, True))
    assert s.target_valid[:, 0].all()
    torch.testing.assert_close(s.target_start[:, 0, 2], torch.zeros(2), atol=1e-6, rtol=0.)
    assert (s.target_apex[:, 0] >= s.target_start[:, 0, 2]+.035-1e-6).all()


def test_initial_air_does_not_acquire_or_reward_reference(module):
    t = Transition(module)
    t.support[:, 0] = False
    s = t.step(dx=.004, left_dx=.03, support=(False, True))
    assert not s.target_valid.any()
    assert not s.swing_tracking.any()


def test_descending_reference_locks_low_landing_and_converges_down(module):
    t = Transition(module)
    t.rays[:, t.rays[0, :, 0] < .18, 2] = .16
    t.feet[..., 2] += .16
    t.step()
    s = t.step(dx=.004, left_dx=.02, left_z=.25, support=(False, True))
    assert s.target_valid[:, 0].all()
    torch.testing.assert_close(s.target_landing[:, 0, 2], torch.zeros(2))
    assert (s.target_apex[:, 0] > .16).all()
    target_x = float(s.target_landing[0, 0, 0])
    s = t.step(dx=.004, left_dx=target_x-.02, left_z=.0543, support=(False, True))
    torch.testing.assert_close(s.swing_phase[:, 0], torch.ones(2))
    torch.testing.assert_close(s.swing_tracking, torch.full((2,), .5), atol=1e-5, rtol=0.)


def test_missing_stance_scan_cannot_be_reacquired_in_air(module):
    t = Transition(module)
    t.rays[:] = torch.nan
    t.step()
    s = t.step(dx=.004, left_dx=.02, support=(False, True), rays=flat_rays())
    t.step(dx=.004, left_dx=.02, support=(False, True))
    assert not s.target_valid.any()
    assert not s.swing_tracking.any()


def test_fixed_height_airborne_hold_and_jump_earn_no_swing_shaping(module):
    t = Transition(module)
    t.step()
    t.step(dx=.004, left_dx=.03, support=(False, True), left_z=.09)
    assert not t.step(dx=.004, support=(False, True)).swing_tracking.any()
    s = t.step(dx=.004, left_dx=.03, support=(False, False))
    assert not s.swing_tracking.any()
    assert (s.support_cost >= 1.).all()


def test_signed_round_trip_cancels_in_new_mode(module):
    t = Transition(module)
    a = t.step(dx=.012).signed_progress.clone()
    b = t.step(dx=-.012).signed_progress.clone()
    torch.testing.assert_close(a+b, torch.zeros(2))


def cycle(t, *, rays_at_landing=None, force=0.):
    t.step()
    for _ in range(round(.12/t.dt)):
        t.step(dx=.016*t.dt/.12, left_dx=.08*t.dt/.12, support=(False, True), left_z=.09)
    return t.step(left_z=.0543, rays=rays_at_landing, force=force)


def test_touchdown_requires_observed_sole_coverage(module):
    good = cycle(Transition(module))
    assert (good.alternating_steps > 0.).all()
    rays = flat_rays()
    rays[:, rays[0, :, 0] > .1, 2] = -.15
    bad = cycle(Transition(module), rays_at_landing=rays)
    assert (bad.coverage[:, 0] < .6).all()
    assert not bad.alternating_steps.any()


def test_rejected_landing_does_not_consume_foot_or_progress_frontier(module):
    t = Transition(module)
    rays = flat_rays()
    rays[:, :, 2] = -.2
    assert not cycle(t, rays_at_landing=rays).alternating_steps.any()
    assert (t.state._last_foot == -1).all()
    assert not t.state._credit_frontier.any()
    t.rays = flat_rays()
    assert cycle(t).alternating_steps.any()


def test_partial_reset_and_abort_clear_reference_and_cost_state(module):
    t = Transition(module)
    t.step()
    s = t.step(dx=.004, left_dx=.02, left_z=.09, support=(False, True))
    s.reset([0])
    assert not s.target_valid[0].any()
    assert s.target_valid[1, 0]
    assert not s.signed_progress[0]
    s.abort_step()
    assert not s.target_valid.any()
    assert not s._terrain_candidate_valid.any()


@pytest.mark.parametrize("dt", [.01, .02, .04])
def test_landing_event_integral_is_dt_independent(module, dt):
    s = cycle(Transition(module, dt=dt), force=981.)
    # Two feet are reported loaded, but only the new landing is an event.
    torch.testing.assert_close(s.landing_cost*dt, torch.ones(2))
    torch.testing.assert_close(s.alternating_steps*dt, torch.ones(2))


def test_invalid_scan_never_creates_zero_floor_or_positive_support(module):
    t = Transition(module)
    s = t.step(rays=torch.full_like(t.rays, torch.nan))
    assert torch.isfinite(s.support_cost).all()
    assert (s.support_cost > 0).all()
    assert not s.swing_tracking.any()


def test_production_adapters_settle_terminal_transition_once_before_reset(module):
    t = Transition(module)
    data = SimpleNamespace(root_pos_w=torch.zeros(2, 3), root_quat_w=t.q[:, 0],
                           body_pos_w=t.feet, body_quat_w=t.q)
    force = torch.zeros(2, 3, 2, 3)
    force[..., 2] = 245.
    filtered = torch.zeros(2, 3, 1, 1, 3)
    filtered[..., 2] = 245.
    runtime = SimpleNamespace(_support_state=lambda: (torch.zeros(2, 2), t.support, t.valid),
        net_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w_history=force)), feet_ids=[0, 1],
        sensors={name: SimpleNamespace(data=SimpleNamespace(force_matrix_w_history=filtered))
                 for name in ("left_foot_ground", "right_foot_ground")})
    env = SimpleNamespace(asset=SimpleNamespace(data=data), feet_id_b=[0, 1], step_dt=t.dt,
        command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[.2, 0., 0.]]*2)),
        scene={"height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=t.rays))},
        walking_runtime=runtime, walking_reward_state=t.state, masses=torch.ones(2)*50,
        termination_manager=SimpleNamespace(terminated=torch.tensor([True, False])))
    tree = ast.parse((TASK/"mdp/rewards.py").read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in {"walk_signed_progress", "terrain_walk_swing", "terrain_walk_support",
                            "terrain_walk_landing", "terrain_walk_failure"}]
    scope = {}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[])), "wrappers", "exec"), scope)
    t.state.begin_step(env)
    data.root_pos_w[:, 0] += .004
    torch.testing.assert_close(scope["terrain_walk_support"](env), torch.zeros(2))
    torch.testing.assert_close(scope["walk_signed_progress"](env), torch.ones(2))
    scope["terrain_walk_swing"](env)
    scope["terrain_walk_landing"](env)
    torch.testing.assert_close(scope["terrain_walk_failure"](env)*t.dt, torch.tensor([1., 0.]))
    t.state.reset([0])
    data.root_pos_w[0, 0] = 100.
    torch.testing.assert_close(scope["walk_signed_progress"](env), torch.tensor([0., 1.]))
