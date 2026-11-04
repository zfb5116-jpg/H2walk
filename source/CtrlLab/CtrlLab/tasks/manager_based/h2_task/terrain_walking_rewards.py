"""Terrain-aware reward state; no policy, simulator or footstep controller.

Uses the existing pre-physics/reward-before-reset lifecycle. Terrain is queried
in world coordinates. A reference is acquired in stance, locked at liftoff,
and never reacquired while airborne. Geometry is a coarse scan proxy, not a
claim about physical contact area, friction or dynamic balance.
"""
from __future__ import annotations

import torch

from .terrain_gait import (
    TERRAIN_REWARD_PARAMETERS,
    select_swing_targets,
    sole_samples_world,
    support_geometry,
    swing_height_reference,
)
from .walking_rewards import WalkingRewardState


class TerrainWalkingRewardState(WalkingRewardState):
    """Share alternation/progress, add full swing references and support costs."""

    def __init__(self, num_envs: int, device: str | torch.device):
        super().__init__(num_envs, device, parameters=TERRAIN_REWARD_PARAMETERS)
        for name in ("_terrain_candidate_valid", "target_valid"):
            setattr(self, name, torch.zeros(num_envs, 2, dtype=torch.bool, device=device))
        for name in ("_terrain_candidate", "_terrain_candidate_start", "target_landing", "target_start"):
            setattr(self, name, torch.zeros(num_envs, 2, 3, device=device))
        for name in ("_candidate_apex", "target_apex", "coverage", "swing_phase"):
            setattr(self, name, torch.zeros(num_envs, 2, device=device))
        for name in ("swing_tracking", "support_cost", "landing_cost"):
            setattr(self, name, torch.zeros(num_envs, device=device))

    @torch.no_grad()
    def begin_step(self, env) -> None:
        feet, contacts, support, valid = self._read(env)
        data = env.asset.data
        self.begin_transition(data.root_pos_w[:, :2], data.root_quat_w,
                              env.command_manager.get_command("base_velocity"), feet,
                              contacts, support, valid, env.step_dt)
        self.prepare_terrain(env.scene["height_scanner"].data.ray_hits_w, feet,
                             data.body_quat_w[:, env.feet_id_b], support, valid)

    @torch.no_grad()
    def prepare_terrain(self, rays, feet, quaternions, support, valid) -> None:
        """Pre-physics stance candidates; pure tensor seam for CPU tests."""
        points = sole_samples_world(feet, quaternions)
        self._pre_sole = points[..., 2].amin(dim=-1)
        candidate, apex, observed = select_swing_targets(rays, feet, quaternions, self._direction)
        stance = support & valid & ~self.swing & self._active[:, None]
        self._terrain_candidate = torch.where(stance[..., None], candidate, self._terrain_candidate)
        start = torch.cat((feet[..., :2], self._pre_sole[..., None]), dim=-1)
        self._terrain_candidate_start = torch.where(stance[..., None], start, self._terrain_candidate_start)
        self._candidate_apex = torch.where(stance, apex, self._candidate_apex)
        # A stance with a missing query explicitly invalidates its old candidate.
        self._terrain_candidate_valid = torch.where(stance, observed, self._terrain_candidate_valid)
        self._terrain_candidate_valid &= valid & self._active[:, None]
        self.swing_tracking.zero_()
        self.support_cost.zero_()
        self.landing_cost.zero_()

    @torch.no_grad()
    def update(self, env) -> None:
        if self._updated:
            return
        if not self._pending:
            raise RuntimeError("Terrain reward requires begin_step before reward computation")
        feet, contacts, support, valid = self._read(env)
        forces = torch.stack([
            env.walking_runtime.sensors[name].data.force_matrix_w_history[:, 0, 0, :, 2].sum(dim=-1)
            for name in ("left_foot_ground", "right_foot_ground")
        ], dim=-1)
        self.finish_terrain_transition(
            env.asset.data.root_pos_w[:, :2], feet, contacts, support, valid,
            env.scene["height_scanner"].data.ray_hits_w,
            env.asset.data.body_quat_w[:, env.feet_id_b], forces, env.masses,
        )

    @torch.no_grad()
    def finish_terrain_transition(self, root_xy, feet, contacts, support, valid,
                                  rays, quaternions, vertical_forces, masses) -> None:
        """Settle once before reset; costs are nonnegative, shaping bounded."""
        p = TERRAIN_REWARD_PARAMETERS
        coverage, observed = support_geometry(rays, feet, quaternions)
        self.coverage.copy_(coverage)
        quality = observed & (coverage >= p["min_landing_coverage"])
        super().finish_transition(root_xy, feet, contacts, support, valid, landing_quality=quality)

        acquire = self.liftoff & self._terrain_candidate_valid
        self.target_landing = torch.where(acquire[..., None], self._terrain_candidate, self.target_landing)
        # Start, landing and apex are one last-stance snapshot. Raw contact
        # history may clear later, after the foot has already moved upwards.
        self.target_start = torch.where(acquire[..., None], self._terrain_candidate_start, self.target_start)
        self.target_apex = torch.where(acquire, self._candidate_apex, self.target_apex)
        self.target_valid |= acquire
        self.target_valid &= self.swing & self._cycle_valid & valid
        # Candidates are not eligible for the next swing until a new stance sample.
        self._terrain_candidate_valid &= ~self.liftoff

        displacement = self.target_landing[..., :2] - self.target_start[..., :2]
        denominator = displacement.square().sum(dim=-1).clamp_min(1e-8)
        phase = ((feet[..., :2]-self.target_start[..., :2])*displacement).sum(dim=-1)/denominator
        self.swing_phase.copy_(phase.clamp(0., 1.))
        reference = swing_height_reference(self.target_start[..., 2], self.target_landing[..., 2],
                                            self.target_apex, self.swing_phase)
        sole = sole_samples_world(feet, quaternions)[..., 2].amin(dim=-1)
        usable = self.target_valid & self.moving_swing & torch.isfinite(sole) & torch.isfinite(reference)
        score = torch.exp(-((torch.nan_to_num(sole)-torch.nan_to_num(reference))/p["swing_height_std"]).square())
        self.swing_tracking.copy_(torch.where(usable, score, 0.).mean(dim=-1))

        # Toe-off is not forced to retain broad coverage. This is a soft support
        # cost; coverage gates only the touchdown event, not continuous support.
        deficit = torch.where(observed, 1.-coverage, 1.)
        bad_contact = ((contacts & ~support) | ~valid).float().mean(dim=-1)
        self.support_cost.copy_(bad_contact + torch.where(support, deficit, 0.).mean(dim=-1)
                                + (~support.any(dim=-1)).float())
        mass = masses[:, None]
        force_valid = torch.isfinite(vertical_forces) & torch.isfinite(mass) & (mass > 0)
        excess = (torch.nan_to_num(vertical_forces)/(torch.nan_to_num(mass).clamp_min(1e-6)*9.81)
                  - p["landing_force_ratio"]).clamp_min(0.).square().clamp_max(p["landing_cost_cap"])
        self.landing_cost.copy_(torch.where(self.landing & force_valid, excess, 0.).sum(dim=-1)/self._dt)

    @torch.no_grad()
    def reset(self, env_ids) -> None:
        super().reset(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for name in ("_terrain_candidate_valid", "target_valid", "_terrain_candidate", "_terrain_candidate_start", "target_landing",
                     "target_start", "_candidate_apex", "target_apex", "coverage", "swing_phase",
                     "swing_tracking", "support_cost", "landing_cost"):
            getattr(self, name)[ids] = 0
