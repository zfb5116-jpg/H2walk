"""Tensor-only transition state for the opt-in walking reward experiment.

The adapter captures the executed command before physics and computes once in
RewardManager, before terminal rows teleport. This is reward bookkeeping, not
an observation, footstep planner or certificate of physical support coverage.
"""
from __future__ import annotations

import math

import torch


WALK_REWARD_PARAMETERS = {
    "semantic_version": "alternating_v1",
    "min_air_time": .08,
    "max_air_time": 1.2,
    "min_step_distance": .04,
    "min_root_progress": .01,
    "min_swing_speed": .02,
    "min_command_speed": .1,
    "max_step_speed": 5.,
}


class WalkingRewardState:
    """Shared lifecycle, signed displacement and fixed world swing targets.

    A cycle must start from observed qualified ground support. A contact at a
    riser cannot complete it. Missing channels invalidate that cycle, and both
    feet leaving support invalidate jumping as an alternating walking step.
    """

    def __init__(self, num_envs: int, device: str | torch.device):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs, self.device = num_envs, torch.device(device)
        for name in ("armed", "swing", "liftoff", "landing", "moving_swing", "_cycle_valid",
                     "_other_supported", "_target_valid", "_candidate_valid"):
            setattr(self, name, torch.zeros(num_envs, 2, dtype=torch.bool, device=device))
        for name in ("air_time", "_target_z", "_target_weight", "_candidate_z", "_candidate_weight"):
            setattr(self, name, torch.zeros(num_envs, 2, device=device))
        for name in ("signed_progress", "alternating_steps", "support_quality",
                     "_signed_distance", "_credit_frontier"):
            setattr(self, name, torch.zeros(num_envs, device=device))
        self._foot_start = torch.zeros(num_envs, 2, 2, device=device)
        self._root_start = torch.zeros_like(self._foot_start)
        self._swing_direction = torch.zeros_like(self._foot_start)
        self._last_foot = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._pending = self._updated = False
        self._dt = 0.

    def _read(self, env):
        _, support, valid = env.walking_runtime._support_state()
        history = env.walking_runtime.net_sensor.data.net_forces_w_history[:, :, env.walking_runtime.feet_ids]
        finite_contact = torch.isfinite(history).all(dim=(1, 3))
        contacts = torch.nan_to_num(history).norm(dim=-1).amax(dim=1) > 1.
        return env.asset.data.body_pos_w[:, env.feet_id_b], contacts, support, valid & finite_contact

    @torch.no_grad()
    def begin_step(self, env) -> None:
        feet, contacts, support, valid = self._read(env)
        self.begin_transition(env.asset.data.root_pos_w[:, :2], env.asset.data.root_quat_w,
                              env.command_manager.get_command("base_velocity"), feet,
                              contacts, support, valid, env.step_dt)

    @torch.no_grad()
    def update(self, env) -> None:
        """Settle this transition once, regardless of reward term call order."""
        if self._updated:
            return
        if not self._pending:
            raise RuntimeError("Walking reward requires begin_step before reward computation")
        feet, contacts, support, valid = self._read(env)
        self.finish_transition(env.asset.data.root_pos_w[:, :2], feet, contacts, support, valid)

    @torch.no_grad()
    def begin_transition(self, root_xy, quaternion, command, feet, contacts, support, valid, dt) -> None:
        """Pure CPU entry used by synthetic transition tests and the adapter."""
        if self._pending:
            raise RuntimeError("Previous walking reward transition is still pending")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self._dt = float(dt)
        self._position = root_xy.clone()
        self._feet = feet[..., :2].clone()
        q_finite = torch.isfinite(quaternion).all(dim=-1)
        q = torch.nan_to_num(quaternion)
        q_norm = q.norm(dim=-1)
        q = q / q_norm.clamp_min(1e-12)[:, None]
        w, x, y, z = q.unbind(-1)
        yaw = torch.atan2(2 * (w*z+x*y), 1-2*(y*y+z*z))
        cmd = torch.nan_to_num(command[:, :2])
        self._speed = cmd.norm(dim=-1)
        world = torch.stack((yaw.cos()*cmd[:, 0]-yaw.sin()*cmd[:, 1],
                             yaw.sin()*cmd[:, 0]+yaw.cos()*cmd[:, 1]), dim=-1)
        self._direction = world / self._speed.clamp_min(1e-12)[:, None]
        self._input_valid = (torch.isfinite(root_xy).all(dim=-1) & q_finite & torch.isfinite(q_norm) & (q_norm > 1e-6)
                             & torch.isfinite(command).all(dim=-1) & torch.isfinite(self._speed))
        self._active = self._input_valid & (self._speed >= WALK_REWARD_PARAMETERS["min_command_speed"])
        # Keep an armed stance across the short raw-contact history delay.
        # A fault seen before physics still invalidates this cycle if the
        # channel has recovered by the reward-phase sample of the same step.
        pre_valid = valid & torch.isfinite(feet).all(dim=-1) & self._input_valid[:, None]
        self.armed &= pre_valid
        self._candidate_valid &= pre_valid
        self._cycle_valid &= pre_valid
        self.armed |= support & pre_valid & ~self.swing
        self._pre_support = support & pre_valid
        self._other_supported &= self._pre_support.flip(1)
        self._pending, self._updated = True, False
        self.liftoff.zero_()
        self.landing.zero_()
        self.moving_swing.zero_()
        self.alternating_steps.zero_()
        self.signed_progress.zero_()
        self.support_quality.zero_()

    @torch.no_grad()
    def finish_transition(self, root_xy, feet, contacts, support, valid) -> None:
        if not self._pending:
            raise RuntimeError("Walking reward transition is not pending")
        p = WALK_REWARD_PARAMETERS
        delta = root_xy-self._position
        good = (self._input_valid & torch.isfinite(root_xy).all(dim=-1)
                & (delta.norm(dim=-1) <= p["max_step_speed"]*self._dt+1e-6))
        projection = (delta*self._direction).sum(dim=-1)
        # No per-step clamp or support gate: equal forward/back travel cancels.
        self.signed_progress.copy_(torch.where(good & self._active,
            projection/(self._speed.clamp_min(p["min_command_speed"])*self._dt), 0.))
        self._signed_distance += torch.where(good & self._active, projection, 0.)
        self.support_quality.copy_(-((contacts & ~support) | ~valid).float().mean(dim=-1))
        foot_finite = torch.isfinite(feet).all(dim=-1)
        self.armed &= valid & foot_finite
        self._candidate_valid &= valid & foot_finite
        active = (good & self._active)[:, None] & valid & foot_finite
        aligned = (self._swing_direction*self._direction[:, None]).sum(dim=-1) > .95
        self._cycle_valid &= active & aligned
        # Qualified support can arm a stance, but invalid channels cannot be air.
        self.liftoff.copy_(self.armed & ~contacts & active & ~self.swing)
        self._foot_start = torch.where(self.liftoff[..., None], self._feet, self._foot_start)
        self._root_start = torch.where(self.liftoff[..., None], self._position[:, None], self._root_start)
        self._swing_direction = torch.where(self.liftoff[..., None], self._direction[:, None], self._swing_direction)
        self._cycle_valid |= self.liftoff
        self._other_supported = torch.where(self.liftoff, self._pre_support.flip(1), self._other_supported)
        self._other_supported &= support.flip(1) & valid.flip(1)
        self.swing |= self.liftoff
        self.armed &= ~self.liftoff
        self.air_time = torch.where(self.liftoff, 0., self.air_time)
        self.air_time += self.swing*self._dt
        self.landing.copy_(self.swing & support & valid)
        foot_progress = ((feet[..., :2]-self._foot_start)*self._swing_direction).sum(dim=-1)
        root_progress = ((root_xy[:, None]-self._root_start)*self._swing_direction).sum(dim=-1)
        qualified = (self.landing & self._cycle_valid & self._other_supported
                     & (self.air_time >= p["min_air_time"]-1e-6)
                     & (self.air_time <= p["max_air_time"]+1e-6)
                     & (foot_progress >= p["min_step_distance"]-1e-6)
                     & (root_progress >= p["min_root_progress"]-1e-6)
                     & ((self._signed_distance-self._credit_frontier)[:, None]
                        >= p["min_root_progress"]-1e-6))
        # At most one event. Simultaneous landing never represents alternation.
        unique = qualified.sum(dim=-1) == 1
        index = qualified.long().argmax(dim=-1)
        credit = unique & (index != self._last_foot)
        self.alternating_steps.copy_(credit.float()/self._dt)
        self._last_foot = torch.where(credit, index, self._last_foot)
        # Repeated visits to the same command-progress peak cannot farm steps.
        # A deliberate reversed command integrates motion in its new direction.
        self._credit_frontier = torch.where(credit, self._signed_distance, self._credit_frontier)
        foot_step = ((feet[..., :2]-self._feet)*self._swing_direction).sum(dim=-1)
        self.moving_swing.copy_(self.swing & ~contacts & self._cycle_valid & self._other_supported
                               & (self.air_time <= p["max_air_time"]+1e-6)
                               & (foot_step > p["min_swing_speed"]*self._dt))
        self.swing &= ~self.landing
        self.armed |= support & valid
        self._target_valid &= self.swing
        self._pending, self._updated = False, True

    @torch.no_grad()
    def hold_targets(self, target_z, weight, valid, gate):
        """Lock the last stance candidate at liftoff, never acquire in midair."""
        gate = gate[:, None]
        acquire = self.liftoff & self._candidate_valid & gate
        self._target_z = torch.where(acquire, self._candidate_z, self._target_z)
        self._target_weight = torch.where(acquire, self._candidate_weight, self._target_weight)
        self._target_valid |= acquire
        self._target_valid &= self.swing & gate
        stance = ~self.swing & self.armed
        self._candidate_z = torch.where(stance, torch.nan_to_num(target_z), self._candidate_z)
        self._candidate_weight = torch.where(stance, torch.nan_to_num(weight), self._candidate_weight)
        self._candidate_valid = torch.where(stance, valid & gate & self.armed, self._candidate_valid)
        self._candidate_valid &= gate
        return self._target_z, self._target_weight, self._target_valid

    @torch.no_grad()
    def reset(self, env_ids) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for name in ("armed", "swing", "liftoff", "landing", "moving_swing", "_cycle_valid",
                     "_other_supported", "_target_valid", "_candidate_valid", "air_time",
                     "signed_progress", "alternating_steps", "support_quality",
                     "_signed_distance", "_credit_frontier"):
            getattr(self, name)[ids] = 0
        self._last_foot[ids] = -1

    @torch.no_grad()
    def abort_step(self) -> None:
        self.reset(torch.arange(self.num_envs, device=self.device))
        self._pending, self._updated = False, False
