"""Budget-v3 gait evidence, diagnostics and terminal success settlement.

No policy or physics is executed here. All rewards share one transition before
reset. Legacy terrain-v2 state and its checkpoint semantics remain separate.
"""
from __future__ import annotations

import torch

from .budget_parameters import BUDGET_REWARD_PARAMETERS
from .budget_progress import BudgetProgressState
from .terrain_walking_rewards import TerrainWalkingRewardState
from .terrain_gait import support_geometry


class BudgetWalkingRewardState(TerrainWalkingRewardState):
    def __init__(self, num_envs, device, *, goal_distance, parameters=None):
        super().__init__(num_envs, device)
        self.parameters = dict(BUDGET_REWARD_PARAMETERS if parameters is None else parameters)
        self.progress = BudgetProgressState(num_envs, device, goal_distance=goal_distance, parameters=self.parameters)
        self.goal_distance = float(goal_distance)
        self.landings_per_foot = torch.zeros(num_envs, 2, device=device, dtype=torch.long)
        for name in ("elapsed", "supported_time", "stable_hold", "yaw_cost", "stall_cost",
                     "success_event", "_window_elapsed", "_window_frontier", "_window_landings", "_event_route_frontier"):
            setattr(self, name, torch.zeros(num_envs, device=device))
        for name in ("success_evidence", "success_paid"):
            setattr(self, name, torch.zeros(num_envs, device=device, dtype=torch.bool))
        self._sums = {}
        self._known_time = {}
        self._rejections = {}

    def _read(self, env):
        self._support_sample = env.walking_runtime.support_components()
        history = env.walking_runtime.net_sensor.data.net_forces_w_history[:, :, env.walking_runtime.feet_ids]
        finite = torch.isfinite(history).all(dim=(1, 3))
        contacts = torch.nan_to_num(history).norm(dim=-1).amax(dim=1) > 1.
        sample = self._support_sample
        return (env.asset.data.body_pos_w[:, env.feet_id_b], contacts, sample["support"],
                sample["valid"] & finite)

    @torch.no_grad()
    def begin_step(self, env):
        super().begin_step(env)
        self.progress.begin_transition(env.asset.data.root_pos_w[:, :2], env.asset.data.root_quat_w,
                                       env.command_manager.get_command("base_velocity"), env.step_dt)

    @torch.no_grad()
    def update(self, env):
        if self._updated:
            return
        if not self._pending:
            raise RuntimeError("Budget reward requires begin_step before reward computation")
        feet, contacts, support, valid = self._read(env)
        data = env.asset.data
        forces = self._support_sample["force_z"]
        self.finish_budget_transition(
            data.root_pos_w[:, :2], feet, contacts, support, valid,
            env.scene["height_scanner"].data.ray_hits_w, data.body_quat_w[:, env.feet_id_b],
            forces, env.masses, data.root_lin_vel_w, data.root_quat_w,
            data.projected_gravity_b, self._support_sample,
        )
        # These trackers normally settle in/reset-after the parent step. v3
        # needs their terminal state for the reward, so consumes them here once.
        runtime = env.walking_runtime
        runtime.settle_reward_transition()
        terminal = env.termination_manager.terminated | env.termination_manager.time_outs
        ids = terminal.nonzero(as_tuple=False).flatten()
        self.success_event.zero_()
        if ids.numel():
            success = runtime.evaluator.episode_success_mask(
                env.command_progress_tracker, runtime.traversal, ids,
                family_ids=runtime.family_ids, terminated=env.termination_manager.terminated,
                time_outs=env.termination_manager.time_outs,
                max_episode_length_s=env.max_episode_length_s,
                extra_success=self.success_evidence,
            )
            credit = success & ~self.success_paid[ids]
            self.success_event[ids] = credit.float()/self._dt
            self.success_paid[ids] |= credit

    @torch.no_grad()
    def finish_budget_transition(self, root_xy, feet, contacts, support, valid, rays, quaternions,
                                 vertical_forces, masses, root_velocity_w, root_quaternion,
                                 projected_gravity, support_sample=None):
        """Pure tensor seam; terminal success itself uses the shared evaluator."""
        previous_foot = self._last_foot.clone()
        previous_frontier = self._credit_frontier.clone()
        super().finish_terrain_transition(root_xy, feet, contacts, support, valid, rays, quaternions,
                                          vertical_forces, masses)
        self.progress.finish_transition(root_xy, root_velocity_w)
        route_valid = self.progress.route_valid
        route_foot_progress = ((feet[..., :2]-self._foot_start)*self.progress.direction[:, None]).sum(dim=-1)
        index = self._last_foot.clamp_min(0)
        foot_forward = route_foot_progress.gather(1, index[:, None]).squeeze(1) >= self.parameters["min_step_distance"]-1e-6
        novel_route = self.progress.raw_route_progress-self._event_route_frontier >= self.parameters["min_root_progress"]-1e-6
        event_allowed = route_valid & novel_route & foot_forward
        keep = event_allowed & (self.alternating_steps > 0)
        rejected_credit = ~event_allowed & (self.alternating_steps > 0)
        self.alternating_steps *= event_allowed
        self._event_route_frontier = torch.where(keep, self.progress.raw_route_progress, self._event_route_frontier)
        self._last_foot = torch.where(rejected_credit, previous_foot, self._last_foot)
        self._credit_frontier = torch.where(rejected_credit, previous_frontier, self._credit_frontier)
        for foot in (0, 1):
            self.landings_per_foot[:, foot] += keep & (self._last_foot == foot)
        self.swing_tracking *= route_valid
        self.elapsed += self._dt
        self.supported_time += support.any(dim=-1)*self._dt

        q_finite = torch.isfinite(root_quaternion).all(dim=-1)
        q = torch.nan_to_num(root_quaternion)
        norm = q.norm(dim=-1)
        q = q/norm.clamp_min(1e-12)[:, None]
        w, x, y, z = q.unbind(-1)
        heading = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        direction = self.progress.direction
        route_heading = torch.atan2(direction[:, 1], direction[:, 0])
        error = torch.atan2((heading-route_heading).sin(), (heading-route_heading).cos())
        heading_valid = (q_finite & torch.isfinite(norm) & (norm > 1e-6)
                         & torch.isfinite(direction).all(dim=-1) & torch.isfinite(error))
        self.yaw_cost.copy_(torch.where(heading_valid, error.square(), 0.))

        p = self.parameters
        _, observed = support_geometry(rays, feet, quaternions)
        quality_support = support & observed & (self.coverage >= p["min_landing_coverage"])
        total_landings = self.landings_per_foot.sum(dim=-1).float()
        self._window_elapsed += self._dt
        due = self._window_elapsed >= p["stall_window_seconds"]-1e-6
        expected = self.progress.command_speed*self._window_elapsed
        stalled = ((self.progress.highwater-self._window_frontier < expected*p["stall_min_progress_fraction"])
                   | (total_landings <= self._window_landings))
        self.stall_cost.copy_(torch.where(due, stalled.float(), self.stall_cost))
        self._window_frontier = torch.where(due, self.progress.highwater, self._window_frontier)
        self._window_landings = torch.where(due, total_landings, self._window_landings)
        self._window_elapsed = torch.where(due, 0., self._window_elapsed)

        fraction = self.supported_time/self.elapsed.clamp_min(self._dt)
        ratio = self.progress.raw_route_progress/self.goal_distance
        speed_ratio = self.progress.forward_velocity/self.progress.command_speed.clamp_min(.1)
        lateral_distance = self.progress.lateral_distance.abs()
        gravity_finite = torch.isfinite(projected_gravity).all(dim=-1)
        upright = gravity_finite & (-projected_gravity[:, 2] >= p["success_min_upright_cosine"])
        stable = (route_valid & heading_valid & upright
                  & (ratio >= p["success_min_goal_ratio"]) & (ratio <= p["success_max_goal_ratio"])
                  & (lateral_distance <= p["success_max_lateral_distance"])
                  & (error.abs() <= p["success_max_yaw_error"]) & quality_support.any(dim=-1)
                  & (speed_ratio >= p["success_min_speed_ratio"])
                  & (speed_ratio <= p["success_max_speed_ratio"])
                  & self.progress.velocity_diagnostic_valid
                  & (self.progress.displacement_velocity_error <= p["success_max_velocity_disagreement"]))
        self.stable_hold = torch.where(stable, self.stable_hold+self._dt, 0.)
        self.success_evidence.copy_(stable & (self.stable_hold >= p["success_hold_seconds"]-1e-6)
            & (fraction >= p["success_min_support_fraction"])
            & (self.landings_per_foot >= p["success_min_landings_per_foot"]).all(dim=-1))

        sample = support_sample or {}
        force_valid = sample.get("force_valid", torch.isfinite(vertical_forces))
        near = sample.get("near_surface", support)
        self._record("support_fraction", support.any(dim=-1).float())
        self._record("support_valid_fraction", valid.float().mean(dim=-1))
        self._record("raw_contact_fraction", contacts.float().mean(dim=-1))
        self._record("coverage_known_fraction", observed.float().mean(dim=-1))
        self._record("coverage_fraction", self.coverage.mean(dim=-1))
        self._record("near_surface_fraction", near.float().mean(dim=-1))
        self._record("force_valid_fraction", force_valid.float().mean(dim=-1))
        force_support = sample.get("force_support", force_valid & (vertical_forces > 1.))
        self._record("upward_force_fraction", force_support.float().mean(dim=-1))
        self._record("geometry_valid_fraction", sample.get("geometry_valid", observed).float().mean(dim=-1))
        mass_valid = torch.isfinite(masses) & (masses > 0)
        load = vertical_forces/torch.where(mass_valid[:, None], masses[:, None]*9.81, torch.nan)
        self._record("vertical_load_ratio", load.sum(dim=-1))
        self._record("target_valid_fraction", self.target_valid.float().mean(dim=-1))
        self._record("moving_swing_fraction", self.moving_swing.float().mean(dim=-1))
        self._record("armed_fraction", self.armed.float().mean(dim=-1))
        self._record("unarmed_air_fraction", (~contacts & ~self.armed & ~self.swing).float().mean(dim=-1))
        self._record("position_physics_disagreement", self.progress.displacement_velocity_error)
        self._record("velocity_diagnostic_valid_fraction", self.progress.velocity_diagnostic_valid.float())
        self._record("route_forward_velocity", self.progress.forward_velocity)
        self._record("lateral_velocity", self.progress.lateral_velocity.abs())
        self._record("yaw_error", torch.where(heading_valid, error.abs(), torch.nan))
        self._record("stall_fraction", self.stall_cost)

        # Landing rejection reasons can overlap. Counts are diagnostic evidence,
        # not disjoint probabilities and never feed back as extra rewards.
        foot_progress = ((feet[..., :2]-self._foot_start)*self._swing_direction).sum(dim=-1)
        root_progress = ((root_xy[:, None]-self._root_start)*self._swing_direction).sum(dim=-1)
        landing = self.landing
        self._reject("invalid_cycle", landing & ~self._cycle_valid)
        self._reject("other_support_lost", landing & ~self._other_supported)
        self._reject("air_time", landing & ((self.air_time < p["min_air_time"]-1e-6)
                                            | (self.air_time > p["max_air_time"]+1e-6)))
        self._reject("foot_progress", landing & (foot_progress < p["min_step_distance"]-1e-6))
        self._reject("root_progress", landing & (root_progress < p["min_root_progress"]-1e-6))
        self._reject("coverage_unknown", landing & ~observed)
        self._reject("coverage_shortfall", landing & observed & (self.coverage < p["min_landing_coverage"]))
        self._reject("repeat_foot", landing & (previous_foot[:, None] == torch.arange(2, device=self.device)))
        self._reject("route_invalid", landing & ~route_valid[:, None])
        self._reject("route_frontier", landing & ~novel_route[:, None])
        self._reject("route_foot_progress", landing & (route_foot_progress < p["min_step_distance"]-1e-6))
        self._reject("new_frontier", landing & ((self._signed_distance-previous_frontier)[:, None]
                                                < p["min_root_progress"]-1e-6))

    def _record(self, name, value):
        if name not in self._sums:
            self._sums[name] = torch.zeros(self.num_envs, device=self.device)
            self._known_time[name] = torch.zeros(self.num_envs, device=self.device)
        self._sums[name] += torch.where(torch.isfinite(value), value, 0.)*self._dt
        self._known_time[name] += torch.isfinite(value)*self._dt

    def _reject(self, name, mask):
        if name not in self._rejections:
            self._rejections[name] = torch.zeros(self.num_envs, device=self.device)
        self._rejections[name] += mask.float().sum(dim=-1)

    @torch.no_grad()
    def episode_logs(self, ids):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        if not ids.numel():
            return {}
        values = {"episode_valid": self.progress.route_valid.float(),
                  "route_progress_ratio": self.progress.raw_route_progress/self.goal_distance,
                  "progress_paid_fraction": self.progress.paid_distance/self.goal_distance,
                  "qualified_steps_left": self.landings_per_foot[:, 0].float(),
                  "qualified_steps_right": self.landings_per_foot[:, 1].float(),
                  "success": self.success_paid.float(), "stable_hold_seconds": self.stable_hold}
        values.update({key: torch.where(self._known_time[key] > 0,
            value/self._known_time[key].clamp_min(1e-12), torch.nan) for key, value in self._sums.items()})
        for key in ("position_physics_disagreement", "route_forward_velocity", "lateral_velocity", "yaw_error", "vertical_load_ratio"):
            if key in self._known_time:
                values[key+"_known_fraction"] = self._known_time[key]/self.elapsed.clamp_min(self._dt)
        values.update({"reject_"+key: value for key, value in self._rejections.items()})
        means = torch.stack([torch.nanmean(v[ids]) for v in values.values()]).detach().cpu().tolist()
        return {"BudgetWalk/"+key: value for key, value in zip(values, means)}

    @torch.no_grad()
    def reset(self, env_ids):
        super().reset(env_ids)
        self.progress.reset(env_ids)
        self._clear_evidence(env_ids)

    def _clear_evidence(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for name in ("landings_per_foot", "elapsed", "supported_time", "stable_hold", "yaw_cost",
                     "stall_cost", "success_event", "success_evidence", "success_paid",
                     "_window_elapsed", "_window_frontier", "_window_landings", "_event_route_frontier"):
            getattr(self, name)[ids] = 0
        for tensor in (*self._sums.values(), *self._known_time.values(), *self._rejections.values()):
            tensor[ids] = 0

    @torch.no_grad()
    def abort_step(self):
        # Do not dispatch through reset(): that would replenish a consumed
        # progress ledger, or reject a still-pending progress transition.
        ids = torch.arange(self.num_envs, device=self.device)
        TerrainWalkingRewardState.reset(self, ids)
        self._clear_evidence(ids)
        self._pending, self._updated = False, False
        self.progress.abort_step()
