"""Physical episode state for policy stair training; never supplied to the actor.

Terrain geometry is used here to score observed physical states. It neither
selects joint targets nor augments the policy's observation layout.
"""

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from .foot_geometry import sole_footprint_corners
from .locomotion_progress import FootfallTracker
from .stair_ascent_safety import physical_validity
from .stair_ascent_evidence import StairEpisodeEvidence
from .stair_clearance_rewards import clearance_progress_components, clearance_progress_delta


def quaternion_matrix(q):
    """Active body-to-world rotation; Isaac Lab quaternions are w, x, y, z."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
                        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
                        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def foot_geometry(device, dtype=torch.float32):
    blob = json.loads((Path(__file__).parent / "assets/h2_leg_chain.json").read_text())
    offsets = torch.tensor([blob["legs"][side]["sole_center_local_offset_m"]
                            for side in ("left", "right")], device=device, dtype=dtype)
    return offsets, sole_footprint_corners(device, dtype)


def signed_forward_ledger(previous_x, actual_peak_x, current_x, positive_allowed, physical_valid,
                          goal_reached, dt, maximum_credit_speed=0.3):
    """Pay new world-X distance once; preserve physical backwards debt.

    The peak follows every observed real position, even when credit is capped
    or gated off. Excess speed/airborne motion cannot be banked for later.
    Including the pre-step position also prevents paying an unmeasured reset
    or externally relocated state as a simulated displacement.
    """
    if dt <= 0 or maximum_credit_speed <= 0:
        raise ValueError("Forward ledger requires a positive physical interval and credit speed")
    peak_before_step = torch.maximum(actual_peak_x, previous_x)
    peak_after_step = torch.maximum(peak_before_step, current_x)
    new_distance = (current_x-peak_before_step).clamp_min(0)
    paid = torch.where(positive_allowed & physical_valid & ~goal_reached,
                       new_distance.clamp_max(float(maximum_credit_speed)*dt), 0.0)
    negative = torch.where(physical_valid, (current_x-previous_x).clamp_max(0), 0.0)
    return peak_after_step, paid, negative


def stair_foot_state(soles, rotations, origins, heights, corners, *, depth, steps, width, approach,
                     tile_end_x=9.75, margin=0.008, side_margin=None):
    """Identify a whole sole's true horizontal tread, including every physical edge.

    X identifies the tread, rather than rounding Z; this matters for 3 cm risers.
    The four corners bound the convex collision sole on a straight rectangular
    tread. A corner extending across a riser invalidates the complete footprint.
    """
    local = soles - origins[:, None]
    world_corners = torch.einsum("nfij,fkj->nfki", rotations, corners) + local[:, :, None]
    min_x, max_x = world_corners[..., 0].amin(-1), world_corners[..., 0].amax(-1)
    max_y = world_corners[..., 1].abs().amax(-1)
    level = torch.floor((local[..., 0] - approach) / depth).long() + 1
    level = level.clamp(0, steps)
    start_x = approach + (level - 1) * depth
    end_x = approach + level * depth
    start_x = torch.where(level == 0, torch.full_like(start_x, -1.75), start_x)
    end_x = torch.where(level == steps, torch.full_like(end_x, tile_end_x), end_x)
    plane_z = level * heights[:, None]
    side_margin = margin if side_margin is None else side_margin
    footprint = ((min_x >= start_x + margin) & (max_x <= end_x - margin)
                 & (max_y <= width * 0.5 - side_margin))
    near_height = (local[..., 2] - plane_z).abs() <= 0.03
    flat = rotations[..., 2, 2] > 0.975
    return {"local": local, "corners": world_corners, "level": level, "plane_z": plane_z,
            "footprint": footprint, "geometric": footprint & near_height & flat,
            "min_x": min_x, "height_error": local[..., 2] - plane_z}


class StairAscentProgress:
    """One post-physics update per control epoch, with terminal state captured before reset."""

    SUMMARY_FIELDS = ("duration_s", "survived", "survived20s", "completed_levels", "verified_feet",
                      "root_rise_m", "target_rise_m", "route_progress_ratio", "top_dwell_s", "success",
                      "terrain_level", "left_valid_landings", "right_valid_landings", "stance_slip_m_s",
                      "transferred_levels", "initial_partial", "terminated", "timed_out",
                      "physical_duration_s", "eligible_duration_s", "ever_invalid", "ever_out_of_tile",
                      "ever_below_floor", "minimum_world_z_m", "minimum_local_z_m",
                      "forward_paid_m", "backward_debt_m", "signed_forward_credit_m",
                      "actual_backward_distance_m", "world_x_peak_m", "goal_reached", "platform_hold_s",
                      "force_credit_eligible", "contact_evidence_valid", "first_failure_charged")

    def __init__(self, n, device, masses, *, steps=6, depth=0.35, width=2.4, approach=1.4,
                 episode_length_s=90.0, gamma=0.99, window_size=5000, domain_bounds=None,
                 floor_tolerance=0.05, record_directory=None, contact_body_names=(), strict_episode_force=False,
                 clearance_progress_potential=False):
        self.n, self.device = n, device
        self.strict_episode_force = bool(strict_episode_force)
        if type(clearance_progress_potential) is not bool:
            raise ValueError("Clearance progress activation must be an explicit boolean")
        self.clearance_progress_enabled = clearance_progress_potential
        if self.clearance_progress_enabled and (not self.strict_episode_force or gamma != 0.99):
            raise ValueError("Clearance progress requires the strict transfer force contract and gamma.99")
        self.steps, self.depth, self.width, self.approach = steps, depth, width, approach
        self.episode_length_s, self.gamma = episode_length_s, gamma
        if domain_bounds is None:
            raise ValueError("Stair progress requires its actual authored tile-domain bounds")
        self.domain_bounds, self.floor_tolerance = tuple(domain_bounds), float(floor_tolerance)
        self.contact_body_names = tuple(contact_body_names)
        self.episode_evidence = StairEpisodeEvidence(record_directory, {
            "domain_bounds_local_m": list(self.domain_bounds), "below_floor_tolerance_m": self.floor_tolerance,
            "contact_body_names": list(self.contact_body_names),
            "force_semantics": "ContactSensor world normal-force vectors; not a frictional wrench",
            "invalid_episodes_kept_as_failures_in_metrics": True,
            "min_xyz_is_componentwise_minimum": True,
            "minimum_samples": "post-physics control endpoints; initial pre-physics root is separately stored",
            "root_at_min_world_z_is_one_actual_control_frame": True,
            "world_x_credit_ledger": {"maximum_positive_speed_m_s": 0.3,
                                      "positive_tilt_limit_rad": 0.5, "minimum_supported_weight_fraction": 0.7,
                                      "support_requires_geometric_loaded_sole_and_normal_force_positive_control": True,
                                      "new_actual_peak_only": True, "unpaid_peak_motion_is_never_banked": True,
                                      "negative_physical_dx_charged_even_without_support_or_after_goal": True,
                                      "invalid_episode_credit_stops": True},
        })
        self.recorded_episode_count = 0
        self.completed_episode_count = 0
        self.masses = masses
        self.weight = masses * 9.81
        self.minimum_load = torch.maximum(masses * 0.981, torch.full_like(masses, 20.0))
        self.corners = sole_footprint_corners(device)
        self.footfalls = FootfallTracker(n, device)
        self.recent = deque(maxlen=window_size)
        self.last_update_counter = -1
        self.last_dt = 0.02
        self.pending = False
        self.needs_start = torch.ones(n, dtype=torch.bool, device=device)
        self.partial = torch.zeros_like(self.needs_start)
        self.origins = torch.zeros(n, 3, device=device)
        self.heights = torch.zeros(n, device=device)
        self.terrain_level = torch.zeros(n, dtype=torch.long, device=device)
        self.root_start = torch.zeros(n, 3, device=device)
        self.pre_soles = torch.zeros(n, 2, 3, device=device)
        self.pre_root_x = torch.zeros(n, device=device)
        self.forward_actual_peak_x = torch.zeros(n, device=device)
        self.forward_paid_m = torch.zeros(n, device=device)
        self.backward_debt_m = torch.zeros(n, device=device)
        self.actual_backward_distance_m = torch.zeros(n, device=device)
        self.signed_forward_step_m = torch.zeros(n, device=device)
        self.forward_paid_step_m = torch.zeros(n, device=device)
        self.backward_step_m = torch.zeros(n, device=device)
        self.forward_credit_eligible = torch.zeros(n, dtype=torch.bool, device=device)
        self.duration = torch.zeros(n, device=device)
        self.physical_duration = torch.zeros(n, device=device)
        self.ever_invalid = torch.zeros(n, dtype=torch.bool, device=device)
        self.ever_out_of_tile = torch.zeros_like(self.ever_invalid)
        self.ever_below_floor = torch.zeros_like(self.ever_invalid)
        self.current_out_of_tile = torch.zeros_like(self.ever_invalid)
        self.current_below_floor = torch.zeros_like(self.ever_invalid)
        self.start_counter = torch.zeros(n, dtype=torch.long, device=device)
        self.end_counter = torch.zeros_like(self.start_counter)
        self.first_invalid_counter = torch.full_like(self.start_counter, -1)
        self.min_z_counter = torch.zeros_like(self.start_counter)
        self.root_end = torch.zeros(n, 3, device=device)
        self.minimum_world_xyz = torch.zeros_like(self.root_end)
        self.minimum_local_xyz = torch.zeros_like(self.root_end)
        self.root_at_min_world_z = torch.zeros_like(self.root_end)
        self.first_invalid_root = torch.zeros_like(self.root_end)
        self.ground_force_end = torch.zeros(n, 2, 3, device=device)
        self.raw_force_end = torch.zeros_like(self.ground_force_end)
        self.ground_force_at_min_z = torch.zeros_like(self.ground_force_end)
        self.raw_force_at_min_z = torch.zeros_like(self.ground_force_end)
        self.first_invalid_ground_force = torch.zeros_like(self.ground_force_end)
        self.first_invalid_raw_force = torch.zeros_like(self.ground_force_end)
        body_shape = (n, len(self.contact_body_names))
        self.maximum_body_force_norm = torch.zeros(body_shape, device=device)
        self.maximum_body_contact_time = torch.zeros_like(self.maximum_body_force_norm)
        self.body_force_at_min_z = torch.zeros_like(self.maximum_body_force_norm)
        self.body_contact_time_at_min_z = torch.zeros_like(self.maximum_body_force_norm)
        self.body_contact_time_end = torch.zeros_like(self.maximum_body_force_norm)
        self.body_force_end = torch.zeros_like(self.maximum_body_force_norm)
        self.completed = torch.zeros(n, dtype=torch.long, device=device)
        self.verified = torch.zeros(n, steps, 2, dtype=torch.bool, device=device)
        self.transferred = torch.zeros(n, steps, dtype=torch.bool, device=device)
        self.pending_level = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.loaded_time = torch.zeros(n, 2, device=device)
        self.platform_time = torch.zeros(n, device=device)
        self.platform_max_time = torch.zeros(n, device=device)
        self.top_paid = torch.zeros(n, dtype=torch.bool, device=device)
        self.goal_reached = torch.zeros(n, dtype=torch.bool, device=device)
        self.goal_first_counter = torch.full_like(self.start_counter, -1)
        self.platform_stable = torch.zeros_like(self.goal_reached)
        self.platform_hold_s = torch.zeros(n, device=device)
        self.potential = torch.zeros(n, device=device)
        self.root_rise = torch.zeros(n, device=device)
        self.route_progress = torch.zeros(n, device=device)
        self.slip_integral = torch.zeros(n, device=device)
        self.slip_time = torch.zeros(n, device=device)
        self.current_slip = torch.zeros(n, device=device)
        self.current_support_height = torch.zeros(n, device=device)
        self.current_force_valid = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.positive_control_seen = torch.zeros_like(self.current_force_valid)
        self.episode_positive_control_seen = torch.zeros_like(self.current_force_valid)
        self.force_mismatch_steps = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.failure_charged = torch.zeros(n, dtype=torch.bool, device=device)
        self.first_failure_this_step = torch.zeros_like(self.failure_charged)
        self.events = {name: torch.zeros(n, device=device) for name in
                       ("potential", "foot", "pair", "transfer", "top")}
        self.last_episode = {key: torch.zeros(n, device=device) for key in self.SUMMARY_FIELDS}
        self.last_episode["valid"] = torch.zeros(n, dtype=torch.bool, device=device)
        self.snapshot_before_reset = None
        self._clearance_allocate()

    def _clearance_allocate(self):
        self.clearance_positive_tilt_limit_rad = 0.5
        self.clearance_previous = torch.zeros(self.n, 2, device=self.device)
        self.clearance_level = torch.zeros(self.n, 2, dtype=torch.long, device=self.device)
        self.clearance_near = torch.zeros(self.n, 2, dtype=torch.bool, device=self.device)
        self.clearance_initialized = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.clearance_endpoint_closed = torch.zeros_like(self.clearance_initialized)
        self.clearance_raw_delta = torch.zeros_like(self.clearance_previous)
        self.clearance_applied_delta = torch.zeros_like(self.clearance_previous)
        self.clearance_update_count = 0
        self.clearance_initialization_count = 0

    def _clearance_initialize(self, ids, soles, rotations):
        if not self.clearance_progress_enabled or not ids.numel():
            return
        if rotations is None or not torch.isfinite(rotations[ids]).all():
            raise RuntimeError("Clearance initial state requires actual precontrol foot rotations")
        value, level, near = clearance_progress_components(
            soles[ids], rotations[ids], self.origins[ids], self.heights[ids], self.corners,
            approach=self.approach, depth=self.depth, steps=self.steps)
        if not torch.isfinite(value).all():
            raise RuntimeError("Non-finite actual initial clearance state")
        self.clearance_previous[ids] = value
        self.clearance_level[ids] = level
        self.clearance_near[ids] = near
        self.clearance_initialized[ids] = True
        self.clearance_endpoint_closed[ids] = False
        self.clearance_initialization_count += int(ids.numel())

    def _clearance_update(self, soles, rotations, fz, terminated, timed_out, positive_weight, root_tilt):
        if not self.clearance_progress_enabled:
            return
        if (timed_out is None or timed_out.dtype != torch.bool or timed_out.shape != terminated.shape
                or not self.clearance_initialized.all()):
            raise RuntimeError("Clearance progress requires real timeout masks and precontrol initialization")
        value, level, near = clearance_progress_components(
            soles, rotations, self.origins, self.heights, self.corners,
            approach=self.approach, depth=self.depth, steps=self.steps)
        endpoint = terminated | timed_out | self.ever_invalid
        value = torch.where(endpoint[:, None], 0.0, value)
        eligible = (~endpoint[:, None] & self.current_force_valid.all(-1)[:, None]
                    & (root_tilt < self.clearance_positive_tilt_limit_rad)[:, None]
                    & (fz < self.minimum_load[:, None]) & (fz.flip(-1) >= 0.7*self.weight[:, None])
                    & near & self.clearance_near & (level == self.clearance_level))
        raw, applied = clearance_progress_delta(self.clearance_previous, value, positive_weight,
                                                eligible, self.gamma)
        if not torch.isfinite(raw).all() or not torch.isfinite(applied).all():
            raise RuntimeError("Non-finite clearance progress difference")
        self.clearance_raw_delta.copy_(raw)
        self.clearance_applied_delta.copy_(applied)
        # The old event has already received its own original sign gate. Each
        # new foot's negative debt survives even if the other foot is unpaid.
        self.events["potential"] += applied.sum(-1)
        self.clearance_previous.copy_(value)
        self.clearance_level.copy_(level)
        self.clearance_near.copy_(near)
        self.clearance_endpoint_closed.copy_(endpoint)
        self.clearance_update_count += 1

    def assert_clearance_reset(self, env_ids):
        if not self.clearance_progress_enabled:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        active = self.clearance_initialized[ids]
        if (self.pending or (active & ~self.clearance_endpoint_closed[ids]).any()
                or (self.clearance_previous[ids][active] != 0).any()):
            raise RuntimeError("Cannot reset active clearance history without its real rewarded endpoint")

    def _clearance_reset(self, ids):
        self.assert_clearance_reset(ids)
        for value in (self.clearance_previous, self.clearance_level, self.clearance_near,
                      self.clearance_initialized, self.clearance_endpoint_closed,
                      self.clearance_raw_delta, self.clearance_applied_delta):
            value[ids] = 0

    def clearance_progress_snapshot(self):
        """Live reward-state evidence; this is never supplied to the actor."""
        if not self.clearance_progress_enabled:
            return None
        return dict(enabled=self.clearance_progress_enabled, gamma=self.gamma,
            approach_m=self.approach, depth_m=self.depth, real_riser_count=self.steps,
            strict_episode_force=self.strict_episode_force, foot_components=int(self.clearance_previous.shape[1]),
            positive_tilt_limit_rad=self.clearance_positive_tilt_limit_rad,
            update_count=self.clearance_update_count, initialization_count=self.clearance_initialization_count,
            initialized_envs=int(self.clearance_initialized.sum()), last_control_counter=self.last_update_counter,
            score_min=float(self.clearance_previous.min()), score_max=float(self.clearance_previous.max()),
            last_added_event_min=float(self.clearance_applied_delta.sum(-1).min()),
            last_added_event_max=float(self.clearance_applied_delta.sum(-1).max()),
            last_raw_delta_min=float(self.clearance_raw_delta.min()),
            last_raw_delta_max=float(self.clearance_raw_delta.max()))

    @property
    def force_credit_eligible(self):
        return self.force_mismatch_steps.sum(-1) == 0

    def begin_step(self, root_position, soles, origins, heights, terrain_level, episode_length_buf, control_counter=0,
                   rotations=None):
        ids = self.needs_start.nonzero().flatten()
        if ids.numel():
            self.root_start[ids] = root_position[ids]
            self.forward_actual_peak_x[ids] = root_position[ids, 0]
            self.origins[ids] = origins[ids]
            self.heights[ids] = heights[ids]
            self.terrain_level[ids] = terrain_level[ids]
            self.start_counter[ids] = int(control_counter)
            self.min_z_counter[ids] = int(control_counter)
            self.root_end[ids] = root_position[ids]
            self.minimum_world_xyz[ids] = root_position[ids]
            self.minimum_local_xyz[ids] = root_position[ids]-origins[ids]
            self.root_at_min_world_z[ids] = root_position[ids]
            route_length = self.approach + self.steps*self.depth + 0.25
            self.potential[ids] = 4.0 * ((root_position[ids, 0]-origins[ids, 0])
                                        / route_length).clamp(-0.25, 1.0)
            # RSL-RL can randomize the first episode-length buffer. Such partial
            # episodes train normally, but never enter curriculum/monitor evidence.
            self.partial[ids] = episode_length_buf[ids] != 0
            self._clearance_initialize(ids, soles, rotations)
            self.needs_start[ids] = False
        self.pre_soles.copy_(soles)
        self.pre_root_x.copy_(root_position[:, 0])
        self.forward_actual_peak_x[:] = torch.maximum(self.forward_actual_peak_x, self.pre_root_x)
        for value in (self.signed_forward_step_m, self.forward_paid_step_m, self.backward_step_m):
            value.zero_()
        for value in self.events.values():
            value.zero_()
        self.first_failure_this_step.zero_()
        self.pending = True

    def update(self, counter, root_position, root_velocity, root_tilt, soles, rotations,
               ground_forces, unfiltered_forces, terminated, dt, body_forces=None, body_contact_time=None,
               timed_out=None):
        if not self.pending or self.last_update_counter == int(counter):
            return
        for value in (root_position, root_velocity, root_tilt, soles, rotations,
                      ground_forces, unfiltered_forces):
            if not torch.isfinite(value).all():
                raise RuntimeError("Non-finite physical state in H2 stair progress")
        if not (self.heights > 0).all():
            raise RuntimeError("Stair progress was not initialized with the actual curriculum height")
        self.last_update_counter = int(counter)
        self.last_dt = float(dt)
        self.pending = False
        first_sample = self.duration == 0
        self.duration += dt
        outside, below, valid_physical = physical_validity(
            root_position, self.origins, self.domain_bounds, self.floor_tolerance)
        invalid = ~valid_physical
        first_invalid = invalid & ~self.ever_invalid
        self.first_invalid_counter[first_invalid] = int(counter)
        self.first_invalid_root[first_invalid] = root_position[first_invalid]
        self.first_invalid_ground_force[first_invalid] = ground_forces[first_invalid]
        self.first_invalid_raw_force[first_invalid] = unfiltered_forces[first_invalid]
        self.ever_invalid |= invalid
        self.ever_out_of_tile |= outside
        self.ever_below_floor |= below
        self.current_out_of_tile.copy_(outside)
        self.current_below_floor.copy_(below)
        self.physical_duration += valid_physical*dt
        self.end_counter.fill_(int(counter))
        new_min_z = first_sample | (root_position[:, 2] < self.minimum_world_xyz[:, 2])
        self.root_at_min_world_z[new_min_z] = root_position[new_min_z]
        self.min_z_counter[new_min_z] = int(counter)
        self.ground_force_at_min_z[new_min_z] = ground_forces[new_min_z]
        self.raw_force_at_min_z[new_min_z] = unfiltered_forces[new_min_z]
        self.minimum_world_xyz[:] = torch.where(first_sample[:, None], root_position,
                                                torch.minimum(self.minimum_world_xyz, root_position))
        self.minimum_local_xyz[:] = torch.where(first_sample[:, None], root_position-self.origins,
                                                torch.minimum(self.minimum_local_xyz, root_position-self.origins))
        self.root_end.copy_(root_position)
        self.ground_force_end.copy_(ground_forces)
        self.raw_force_end.copy_(unfiltered_forces)
        if self.contact_body_names:
            if (body_forces is None or body_contact_time is None
                    or body_forces.shape != (*self.maximum_body_force_norm.shape, 3)
                    or body_contact_time.shape != self.maximum_body_force_norm.shape):
                raise RuntimeError("Missing matched all-body contact evidence")
            if not torch.isfinite(body_forces).all() or not torch.isfinite(body_contact_time).all():
                raise RuntimeError("Non-finite all-body stair contact evidence")
            norm = body_forces.norm(dim=-1)
            self.maximum_body_force_norm[:] = torch.maximum(self.maximum_body_force_norm, norm)
            self.maximum_body_contact_time[:] = torch.maximum(self.maximum_body_contact_time, body_contact_time)
            self.body_force_at_min_z[new_min_z] = norm[new_min_z]
            self.body_contact_time_at_min_z[new_min_z] = body_contact_time[new_min_z]
            self.body_force_end.copy_(norm)
            self.body_contact_time_end.copy_(body_contact_time)
        fz = ground_forces[..., 2]
        raw_fz = unfiltered_forces[..., 2]
        if self.strict_episode_force:
            raw_norm = unfiltered_forces.norm(dim=-1)
            ground_norm = ground_forces.norm(dim=-1)
            agreement = ((ground_forces-unfiltered_forces).norm(dim=-1) <= 1.0+0.02*raw_norm)
            relevant = (raw_norm > 5.0) | (ground_norm > 5.0)
            self.force_mismatch_steps += ~agreement & relevant
        else:
            agreement = (fz - raw_fz).abs() <= 1.0 + 0.02 * raw_fz.abs()
            self.force_mismatch_steps += ~agreement
        self.current_force_valid[:] = agreement & (fz >= 0)
        positive = agreement & (fz > 5.0) if self.strict_episode_force else agreement & (fz >= self.minimum_load[:, None])
        self.positive_control_seen |= positive
        self.episode_positive_control_seen |= positive
        # Force qualification gates credit, never the physical contact clock.
        contact = ground_forces.norm(dim=-1) > 5.0
        if not self.strict_episode_force:
            contact &= self.current_force_valid
        failed = terminated | ~self.force_credit_eligible
        self.first_failure_this_step[:] = failed & ~self.failure_charged
        self.failure_charged |= failed
        speed_u = ((root_velocity[:, 0]-0.3)/0.3).clamp(0, 1)
        positive_weight = ((1-speed_u).square()*(1+2*speed_u) * self.force_credit_eligible
                           if self.strict_episode_force else torch.ones_like(speed_u))
        previous_landings = self.footfalls.landings.clone()
        self.footfalls.update(contact, dt)
        new_landing = self.footfalls.landings > previous_landings
        geo = stair_foot_state(soles, rotations, self.origins, self.heights, self.corners,
                               depth=self.depth, steps=self.steps, width=self.width, approach=self.approach,
                               side_margin=.02 if self.strict_episode_force else None,
                               tile_end_x=9.0 if self.strict_episode_force else 9.75)
        level = geo["level"]
        loaded = (fz >= self.minimum_load[:, None]) & self.current_force_valid
        on_tread = loaded & geo["geometric"] & valid_physical[:, None]
        self.pending_level[:] = torch.where(contact, self.pending_level, 0)
        candidate = (geo["footprint"] & (geo["height_error"].abs() <= 0.06)
                     & (level >= 1) & (level == self.completed[:, None] + 1) & valid_physical[:, None])
        self.pending_level[:] = torch.where(new_landing & candidate, level, self.pending_level)
        matching = on_tread & (self.pending_level == level) & (level >= 1)
        self.loaded_time[:] = torch.where(matching, self.loaded_time + dt, 0)
        for foot in (0, 1):
            eligible = (matching[:, foot] & (self.loaded_time[:, foot] >= 0.08 - 1.0e-6)
                        & (level[:, foot] == self.completed + 1))
            ids = eligible.nonzero().flatten()
            if ids.numel():
                lev = level[ids, foot] - 1
                unseen = ~self.verified[ids, lev, foot]
                ids, lev = ids[unseen], lev[unseen]
                self.verified[ids, lev, foot] = True
                self.events["foot"][ids] += 1.0
        next_level = self.completed.clamp_max(self.steps - 1)
        advance = ((self.completed < self.steps)
                   & self.verified[torch.arange(self.n, device=self.device), next_level].all(-1)
                   & on_tread.all(-1) & (level == self.completed[:, None]+1).all(-1))
        self.completed += advance
        self.events["pair"][:] = advance
        # A transfer is a physical load change onto an already verified higher
        # foot, paid once per level. Neither a clock nor an airborne foot earns it.
        valid_fz = torch.where(on_tread, fz.clamp_min(0), 0)
        total = valid_fz.sum(-1)
        share = valid_fz / total[:, None].clamp_min(1)
        for foot in (0, 1):
            lev = (level[:, foot] - 1).clamp(0, self.steps - 1)
            envs = torch.arange(self.n, device=self.device)
            higher = (on_tread[:, foot] & (level[:, foot] >= 1)
                      & (level[:, foot] > level[:, 1-foot])
                      & self.verified[envs, lev, foot] & (share[:, foot] >= 0.65)
                      & (total >= 0.7 * self.weight) & ~self.transferred[envs, lev])
            ids = higher.nonzero().flatten()
            self.transferred[ids, lev[ids]] = True
            self.events["transfer"][ids] += 1.0
        platform_start = self.approach + self.steps * self.depth
        platform_geometry = ((self.completed == self.steps) & on_tread.all(-1) & (level == self.steps).all(-1)
                             & (geo["min_x"] >= platform_start + 0.008).all(-1)
                             & valid_physical & ~self.ever_invalid & ~terminated)
        new_goal = platform_geometry & ~self.goal_reached
        self.goal_first_counter[new_goal] = int(counter)
        self.goal_reached |= platform_geometry
        top = platform_geometry & (root_tilt < 0.25) & (root_velocity.norm(dim=-1) <= 0.15)
        self.platform_stable.copy_(top)
        self.platform_hold_s += top*dt
        supported = on_tread.any(-1) & (total >= 0.7*self.weight)
        self.forward_credit_eligible[:] = (supported & (root_tilt < 0.5) & valid_physical
                                           & ~self.ever_invalid & ~terminated & ~self.goal_reached)
        if self.strict_episode_force:
            self.forward_credit_eligible &= self.force_credit_eligible
        peak, paid, negative = signed_forward_ledger(
            self.pre_root_x, self.forward_actual_peak_x, root_position[:, 0],
            self.forward_credit_eligible, valid_physical & ~self.ever_invalid,
            self.goal_reached, dt)
        self.forward_actual_peak_x.copy_(peak)
        paid = paid * positive_weight
        self.forward_paid_step_m.copy_(paid)
        self.backward_step_m.copy_(negative)
        self.signed_forward_step_m[:] = paid+negative
        self.forward_paid_m += paid
        self.backward_debt_m -= negative
        self.actual_backward_distance_m += (self.pre_root_x-root_position[:, 0]).clamp_min(0)
        self.platform_time[:] = torch.where(top, self.platform_time + dt, 0)
        self.platform_max_time[:] = torch.maximum(self.platform_max_time, self.platform_time)
        top_event = (self.platform_time >= 2.0 - 1.0e-6) & ~self.top_paid
        self.events["top"][:] = top_event
        self.top_paid |= top_event
        for name in ("foot", "pair", "transfer", "top"):
            self.events[name] *= ~terminated & valid_physical
            self.events[name] *= positive_weight
        self.root_rise[:] = root_position[:, 2] - self.root_start[:, 2]
        route_length = platform_start + 0.25
        self.route_progress[:] = ((root_position[:, 0] - self.origins[:, 0]) / route_length).clamp(-0.25, 1.0)
        support_height = (valid_fz * geo["local"][..., 2]).sum(-1) / total.clamp_min(1)
        self.current_support_height[:] = torch.where(total >= 0.7*self.weight, support_height, 0)
        expected_rise = self.steps * self.heights
        # Discount-consistent potential differences telescope; repeated rocking,
        # reweighting and descending cannot repeatedly farm positive progress.
        # This is shaping, not a claim that any milestone is a full ascent.
        phi = (4.0 * self.route_progress
               + 4.0 * (self.current_support_height / expected_rise).clamp(0, 1)
               + 2.0 * (self.root_rise / expected_rise).clamp(-0.25, 1.0))
        phi = torch.where(terminated | invalid, 0.0, phi)
        self.events["potential"][:] = self.gamma * phi - self.potential
        if self.strict_episode_force:
            delta = self.events["potential"]
            delta.copy_(delta.clamp_max(0)+delta.clamp_min(0)*positive_weight)
        self.potential.copy_(phi)
        self._clearance_update(soles, rotations, fz, terminated, timed_out, positive_weight, root_tilt)
        sole_speed = ((soles - self.pre_soles)[..., :2] / dt).norm(dim=-1)
        slip_count = loaded.sum(-1)
        self.current_slip[:] = (sole_speed * loaded).sum(-1) / slip_count.clamp_min(1)
        self.slip_integral += self.current_slip * (slip_count > 0) * dt
        self.slip_time += (slip_count > 0) * dt

    def current_signed_forward_m_s(self):
        """Per-second reward rate; RewardManager integrates this once with dt."""
        return self.signed_forward_step_m/self.last_dt

    def record_episodes(self, env_ids, terminated, timed_out, termination_bits=None):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        valid = (~self.needs_start[ids] & ~self.partial[ids] & (self.duration[ids] > 0)
                 & (terminated[ids] | timed_out[ids]))
        eligible_duration = torch.where(self.ever_invalid[ids], 0.0, self.physical_duration[ids])
        survived = ((eligible_duration >= self.episode_length_s-max(1e-4, 0.5*self.last_dt))
                    & timed_out[ids] & ~terminated[ids] & ~self.ever_invalid[ids])
        expected = self.steps*self.heights[ids]
        success = (survived & (self.completed[ids] == self.steps) & self.verified[ids].all(dim=(1, 2))
                   & (self.platform_max_time[ids] >= 2.0-1.0e-6)
                   & ((self.root_rise[ids]-expected).abs() <= 0.10))
        contact_valid = self.force_credit_eligible[ids] & self.episode_positive_control_seen[ids].all(-1)
        if self.strict_episode_force:
            success &= contact_valid
        values = {
            "duration_s": self.duration[ids], "survived": survived, "survived20s": (eligible_duration >= 20.0-1e-5),
            "completed_levels": self.completed[ids], "verified_feet": self.verified[ids].sum(dim=(1, 2)),
            "root_rise_m": self.root_rise[ids], "target_rise_m": expected,
            "route_progress_ratio": self.route_progress[ids], "top_dwell_s": self.platform_max_time[ids],
            "success": success, "terrain_level": self.terrain_level[ids],
            "left_valid_landings": self.footfalls.landings[ids, 0],
            "right_valid_landings": self.footfalls.landings[ids, 1],
            "stance_slip_m_s": self.slip_integral[ids] / self.slip_time[ids].clamp_min(1e-6),
            "transferred_levels": self.transferred[ids].sum(-1), "initial_partial": self.partial[ids],
            "terminated": terminated[ids], "timed_out": timed_out[ids], "valid": valid,
            "physical_duration_s": self.physical_duration[ids], "eligible_duration_s": eligible_duration,
            "ever_invalid": self.ever_invalid[ids], "ever_out_of_tile": self.ever_out_of_tile[ids],
            "ever_below_floor": self.ever_below_floor[ids],
            "minimum_world_z_m": self.minimum_world_xyz[ids, 2],
            "minimum_local_z_m": self.minimum_local_xyz[ids, 2],
            "forward_paid_m": self.forward_paid_m[ids], "backward_debt_m": self.backward_debt_m[ids],
            "signed_forward_credit_m": self.forward_paid_m[ids]-self.backward_debt_m[ids],
            "actual_backward_distance_m": self.actual_backward_distance_m[ids],
            "world_x_peak_m": self.forward_actual_peak_x[ids], "goal_reached": self.goal_reached[ids],
            "platform_hold_s": self.platform_hold_s[ids],
            "force_credit_eligible": self.force_credit_eligible[ids],
            "contact_evidence_valid": contact_valid,
            "first_failure_charged": self.failure_charged[ids],
        }
        for key, value in values.items():
            self.last_episode[key][ids] = value.to(self.last_episode[key].dtype)
        details = {"env_id": ids, "start_control_counter": self.start_counter[ids], "end_control_counter": self.end_counter[ids],
                   "root_start_world_xyz": self.root_start[ids], "root_end_world_xyz": self.root_end[ids],
                   "minimum_root_world_xyz": self.minimum_world_xyz[ids], "minimum_root_local_xyz": self.minimum_local_xyz[ids],
                   "root_at_min_world_z": self.root_at_min_world_z[ids], "minimum_z_control_counter": self.min_z_counter[ids],
                   "tile_origin_world_xyz": self.origins[ids], "riser_height_m": self.heights[ids],
                   "domain_bounds_local_m": self.duration.new_tensor(self.domain_bounds).expand(len(ids), -1),
                   "below_floor_tolerance_m": self.duration.new_full((len(ids),), self.floor_tolerance),
                   "true_timeout": timed_out[ids], "out_of_tile_at_end": self.current_out_of_tile[ids],
                   "below_floor_at_end": self.current_below_floor[ids],
                   "first_invalid_control_counter": self.first_invalid_counter[ids],
                   "first_invalid_root_world_xyz": self.first_invalid_root[ids],
                   "ground_normal_force_end_w_N": self.ground_force_end[ids], "raw_normal_force_end_w_N": self.raw_force_end[ids],
                   "ground_normal_force_at_min_z_w_N": self.ground_force_at_min_z[ids],
                   "raw_normal_force_at_min_z_w_N": self.raw_force_at_min_z[ids],
                   "ground_normal_force_first_invalid_w_N": self.first_invalid_ground_force[ids],
                   "raw_normal_force_first_invalid_w_N": self.first_invalid_raw_force[ids],
                   "maximum_body_normal_force_N": self.maximum_body_force_norm[ids],
                   "maximum_sampled_body_contact_time_s": self.maximum_body_contact_time[ids],
                   "body_normal_force_at_min_z_N": self.body_force_at_min_z[ids],
                   "body_contact_time_at_min_z_s": self.body_contact_time_at_min_z[ids],
                   "body_normal_force_end_N": self.body_force_end[ids], "body_contact_time_end_s": self.body_contact_time_end[ids],
                   "verified": self.verified[ids], "filtered_force_positive_control": self.episode_positive_control_seen[ids],
                   "sensor_positive_control_ever": self.positive_control_seen[ids],
                   "foot_force_mismatch_control_steps": self.force_mismatch_steps[ids],
                   "goal_first_control_counter": self.goal_first_counter[ids],
                   "platform_stable_at_end": self.platform_stable[ids],
                   "forward_credit_eligible_at_end": self.forward_credit_eligible[ids],
                   "forward_paid_last_step_m": self.forward_paid_step_m[ids],
                   "backward_last_step_m": self.backward_step_m[ids],
                   "signed_forward_last_step_m": self.signed_forward_step_m[ids]}
        for name, bit in (termination_bits or {"terminated": terminated, "time_out": timed_out}).items():
            details["termination_"+name] = bit[ids]
        self.snapshot_before_reset = {"env_ids": ids.clone(), **{key: value.clone() for key, value in {**values, **details}.items()}}
        has_episode = self.duration[ids] > 0
        if has_episode.any():
            self.episode_evidence.append({key: value[has_episode] for key, value in {**values, **details}.items()})
            self.recorded_episode_count += int(has_episode.sum())
        if valid.any():
            rows = torch.stack([values[key].float() for key in self.SUMMARY_FIELDS], -1)
            self.recent.extend(rows[valid].detach().cpu().tolist())
            self.completed_episode_count += int(valid.sum())

    def reset(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._clearance_reset(ids)
        self.needs_start[ids] = True
        self.partial[ids] = False
        for value in (self.duration, self.completed, self.verified, self.transferred, self.pending_level,
                      self.loaded_time, self.platform_time, self.platform_max_time, self.top_paid,
                      self.potential, self.root_rise, self.route_progress, self.slip_integral,
                      self.slip_time, self.current_slip, self.current_support_height, self.physical_duration,
                      self.ever_invalid, self.ever_out_of_tile, self.ever_below_floor, self.current_out_of_tile,
                      self.current_below_floor, self.ground_force_end, self.raw_force_end, self.ground_force_at_min_z,
                      self.raw_force_at_min_z, self.first_invalid_root, self.first_invalid_ground_force,
                      self.first_invalid_raw_force, self.maximum_body_force_norm, self.maximum_body_contact_time,
                      self.body_force_at_min_z, self.body_contact_time_at_min_z, self.body_force_end, self.body_contact_time_end,
                      self.episode_positive_control_seen, self.force_mismatch_steps,
                      self.forward_paid_m, self.backward_debt_m, self.actual_backward_distance_m,
                      self.signed_forward_step_m, self.forward_paid_step_m, self.backward_step_m,
                      self.forward_credit_eligible, self.forward_actual_peak_x, self.pre_root_x,
                      self.goal_reached, self.platform_stable, self.platform_hold_s,
                      self.failure_charged, self.first_failure_this_step):
            value[ids] = 0
        self.first_invalid_counter[ids] = -1
        self.goal_first_counter[ids] = -1
        self.footfalls.reset(ids)

    def summary(self):
        if not self.recent:
            return {"episode_count": 0, "window_episode_count": 0, "completed_episode_count": 0,
                    "survival20s_fraction": 0.0, "survival_horizon_fraction": 0.0,
                    "median_episode_s": 0.0, "mean_episode_s": 0.0, "command_progress_ratio": 0.0,
                    "survival_and_progress_fraction": 0.0, "locomotion_success_fraction": 0.0,
                    "stair_success_fraction": 0.0, "mean_completed_levels": 0.0,
                    "first_level_bilateral_fraction": 0.0, "all_levels_bilateral_fraction": 0.0,
                    "invalid_physical_episode_fraction": 0.0, "physical_valid_episode_count": 0}
        rows = np.asarray(self.recent)
        c = {key: rows[:, i] for i, key in enumerate(self.SUMMARY_FIELDS)}
        physically_valid = c["ever_invalid"] == 0
        eligible = physically_valid.copy()
        if self.strict_episode_force:
            eligible &= c["contact_evidence_valid"] > 0
        forward = np.where(eligible, np.clip(c["route_progress_ratio"], 0.0, 1.0), 0.0)
        levels = np.where(eligible, c["completed_levels"], 0.0)
        success = eligible & (c["success"] > 0)
        survived = eligible & (c["survived"] > 0)
        # These canonical aliases overwrite MotionProbe's raw lifetime/progress
        # before SupervisedRunner assesses patience. Invalid rows remain in the
        # denominator as zero-valued failures, rather than being filtered away.
        return {"episode_count": len(rows), "window_episode_count": len(rows),
                "completed_episode_count": self.completed_episode_count,
                "survival20s_fraction": float(c["survived20s"].mean()),
                "survival_horizon_fraction": float(survived.mean()),
                "mean_episode_s": float(c["eligible_duration_s"].mean()),
                "median_episode_s": float(np.median(c["eligible_duration_s"])),
                "command_progress_ratio": float(forward.mean()),
                "survival_and_progress_fraction": float((survived & (levels >= 1)).mean()),
                "locomotion_success_fraction": float(success.mean()),
                "bilateral_landings_fraction": float((eligible & (c["left_valid_landings"] >= 2)
                                                        & (c["right_valid_landings"] >= 2)).mean()),
                "early_termination_fraction": float(((c["terminated"] > 0) | ~eligible).mean()),
                "survival_fraction": float(survived.mean()),
                "survived20s_fraction": float(c["survived20s"].mean()),
                "stair_success_fraction": float(success.mean()),
                "mean_completed_levels": float(levels.mean()),
                "first_level_bilateral_fraction": float((levels >= 1).mean()),
                "all_levels_bilateral_fraction": float((levels == self.steps).mean()),
                "mean_verified_feet": float(np.where(eligible, c["verified_feet"], 0).mean()),
                "mean_transferred_levels": float(np.where(eligible, c["transferred_levels"], 0).mean()),
                "median_duration_s": float(np.median(c["eligible_duration_s"])),
                "mean_root_rise_m": float(c["root_rise_m"].mean()),
                "mean_valid_upward_root_rise_m": float(np.where(eligible, c["root_rise_m"].clip(0), 0).mean()),
                "mean_route_progress_ratio": float(forward.mean()),
                "mean_platform_dwell_s": float(np.where(eligible, c["top_dwell_s"], 0).mean()),
                "mean_stance_slip_m_s": float(c["stance_slip_m_s"].mean()),
                "mean_terrain_level": float(c["terrain_level"].mean()),
                "physical_valid_episode_count": int(physically_valid.sum()),
                "invalid_physical_episode_count": int((~physically_valid).sum()),
                "invalid_physical_episode_fraction": float((~physically_valid).mean()),
                "force_clean_episode_fraction": float((c["force_credit_eligible"] > 0).mean()),
                "contact_evidence_valid_fraction": float((c["contact_evidence_valid"] > 0).mean()),
                "out_of_tile_episode_fraction": float(c["ever_out_of_tile"].mean()),
                "below_floor_episode_fraction": float(c["ever_below_floor"].mean()),
                "true_timeout_fraction": float(c["timed_out"].mean()),
                "raw_mean_episode_s": float(c["duration_s"].mean()),
                "raw_median_episode_s": float(np.median(c["duration_s"])),
                "mean_physical_duration_s": float(c["physical_duration_s"].mean()),
                "raw_duration20s_fraction": float((c["duration_s"] >= 20.0-1e-5).mean()),
                "raw_mean_route_progress_ratio": float(c["route_progress_ratio"].mean()),
                "raw_mean_completed_levels": float(c["completed_levels"].mean()),
                "mean_forward_paid_m": float(np.where(eligible, c["forward_paid_m"], 0).mean()),
                "mean_backward_debt_m": float(np.where(eligible, c["backward_debt_m"], 0).mean()),
                "mean_signed_forward_credit_m": float(np.where(eligible, c["signed_forward_credit_m"], 0).mean()),
                "raw_mean_forward_paid_m": float(c["forward_paid_m"].mean()),
                "raw_mean_backward_debt_m": float(c["backward_debt_m"].mean()),
                "raw_mean_actual_backward_distance_m": float(c["actual_backward_distance_m"].mean()),
                "goal_reached_fraction": float((eligible & (c["goal_reached"] > 0)).mean()),
                "raw_goal_reached_fraction": float((c["goal_reached"] > 0).mean()),
                "mean_platform_hold_s": float(np.where(eligible, c["platform_hold_s"], 0).mean()),
                "minimum_recorded_world_z_m": float(c["minimum_world_z_m"].min()),
                "minimum_recorded_local_z_m": float(c["minimum_local_z_m"].min()),
                "episode_evidence_rows": self.recorded_episode_count,
                "episode_evidence_written_rows": self.episode_evidence.written,
                "episode_evidence_pending_rows": self.episode_evidence.pending,
                "minimum_valid_air_s": 0.08, "maximum_valid_air_s": 0.6,
                "minimum_loaded_time_s": 0.08, "footprint_margin_m": 0.008,
                "initial_partial_episodes_excluded": True,
                "metrics_are_training_evidence_not_formal_acceptance": True}
