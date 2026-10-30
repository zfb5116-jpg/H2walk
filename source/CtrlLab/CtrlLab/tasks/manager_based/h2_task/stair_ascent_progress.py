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


def stair_foot_state(soles, rotations, origins, heights, corners, *, depth, steps, width, approach,
                     tile_end_x=9.75, margin=0.008):
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
    footprint = ((min_x >= start_x + margin) & (max_x <= end_x - margin)
                 & (max_y <= width * 0.5 - margin))
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
                      "transferred_levels", "initial_partial", "terminated", "timed_out")

    def __init__(self, n, device, masses, *, steps=6, depth=0.35, width=2.4, approach=1.4,
                 episode_length_s=90.0, gamma=0.99, window_size=5000):
        self.n, self.device = n, device
        self.steps, self.depth, self.width, self.approach = steps, depth, width, approach
        self.episode_length_s, self.gamma = episode_length_s, gamma
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
        self.duration = torch.zeros(n, device=device)
        self.completed = torch.zeros(n, dtype=torch.long, device=device)
        self.verified = torch.zeros(n, steps, 2, dtype=torch.bool, device=device)
        self.transferred = torch.zeros(n, steps, dtype=torch.bool, device=device)
        self.pending_level = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.loaded_time = torch.zeros(n, 2, device=device)
        self.platform_time = torch.zeros(n, device=device)
        self.platform_max_time = torch.zeros(n, device=device)
        self.top_paid = torch.zeros(n, dtype=torch.bool, device=device)
        self.potential = torch.zeros(n, device=device)
        self.root_rise = torch.zeros(n, device=device)
        self.route_progress = torch.zeros(n, device=device)
        self.slip_integral = torch.zeros(n, device=device)
        self.slip_time = torch.zeros(n, device=device)
        self.current_slip = torch.zeros(n, device=device)
        self.current_support_height = torch.zeros(n, device=device)
        self.current_force_valid = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.positive_control_seen = torch.zeros_like(self.current_force_valid)
        self.force_mismatch_steps = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.events = {name: torch.zeros(n, device=device) for name in
                       ("potential", "foot", "pair", "transfer", "top")}
        self.last_episode = {key: torch.zeros(n, device=device) for key in self.SUMMARY_FIELDS}
        self.last_episode["valid"] = torch.zeros(n, dtype=torch.bool, device=device)
        self.snapshot_before_reset = None

    def begin_step(self, root_position, soles, origins, heights, terrain_level, episode_length_buf):
        ids = self.needs_start.nonzero().flatten()
        if ids.numel():
            self.root_start[ids] = root_position[ids]
            self.origins[ids] = origins[ids]
            self.heights[ids] = heights[ids]
            self.terrain_level[ids] = terrain_level[ids]
            route_length = self.approach + self.steps*self.depth + 0.25
            self.potential[ids] = 4.0 * ((root_position[ids, 0]-origins[ids, 0])
                                        / route_length).clamp(-0.25, 1.0)
            # RSL-RL can randomize the first episode-length buffer. Such partial
            # episodes train normally, but never enter curriculum/monitor evidence.
            self.partial[ids] = episode_length_buf[ids] != 0
            self.needs_start[ids] = False
        self.pre_soles.copy_(soles)
        for value in self.events.values():
            value.zero_()
        self.pending = True

    def update(self, counter, root_position, root_velocity, root_tilt, soles, rotations,
               ground_forces, unfiltered_forces, terminated, dt):
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
        self.duration += dt
        fz = ground_forces[..., 2]
        raw_fz = unfiltered_forces[..., 2]
        agreement = (fz - raw_fz).abs() <= 1.0 + 0.02 * raw_fz.abs()
        self.current_force_valid[:] = agreement & (fz >= 0)
        self.positive_control_seen |= agreement & (fz >= self.minimum_load[:, None])
        self.force_mismatch_steps += ~agreement
        contact = (ground_forces.norm(dim=-1) > 5.0) & self.current_force_valid
        previous_landings = self.footfalls.landings.clone()
        self.footfalls.update(contact, dt)
        new_landing = self.footfalls.landings > previous_landings
        geo = stair_foot_state(soles, rotations, self.origins, self.heights, self.corners,
                               depth=self.depth, steps=self.steps, width=self.width, approach=self.approach)
        level = geo["level"]
        loaded = (fz >= self.minimum_load[:, None]) & self.current_force_valid
        on_tread = loaded & geo["geometric"]
        self.pending_level[:] = torch.where(contact, self.pending_level, 0)
        candidate = (geo["footprint"] & (geo["height_error"].abs() <= 0.06)
                     & (level >= 1) & (level == self.completed[:, None] + 1))
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
        top = ((self.completed == self.steps) & on_tread.all(-1) & (level == self.steps).all(-1)
               & (geo["min_x"] >= platform_start + 0.008).all(-1) & (root_tilt < 0.25)
               & (root_velocity.norm(dim=-1) <= 0.15))
        self.platform_time[:] = torch.where(top, self.platform_time + dt, 0)
        self.platform_max_time[:] = torch.maximum(self.platform_max_time, self.platform_time)
        top_event = (self.platform_time >= 2.0 - 1.0e-6) & ~self.top_paid
        self.events["top"][:] = top_event
        self.top_paid |= top_event
        for name in ("foot", "pair", "transfer", "top"):
            self.events[name] *= ~terminated
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
        phi = torch.where(terminated, 0.0, phi)
        self.events["potential"][:] = self.gamma * phi - self.potential
        self.potential.copy_(phi)
        sole_speed = ((soles - self.pre_soles)[..., :2] / dt).norm(dim=-1)
        slip_count = loaded.sum(-1)
        self.current_slip[:] = (sole_speed * loaded).sum(-1) / slip_count.clamp_min(1)
        self.slip_integral += self.current_slip * (slip_count > 0) * dt
        self.slip_time += (slip_count > 0) * dt

    def record_episodes(self, env_ids, terminated, timed_out):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        valid = (~self.needs_start[ids] & ~self.partial[ids] & (self.duration[ids] > 0)
                 & (terminated[ids] | timed_out[ids]))
        survived = ((self.duration[ids] >= self.episode_length_s-max(1e-4, 0.5*self.last_dt))
                    & timed_out[ids] & ~terminated[ids])
        expected = self.steps*self.heights[ids]
        success = (survived & (self.completed[ids] == self.steps) & self.verified[ids].all(dim=(1, 2))
                   & (self.platform_max_time[ids] >= 2.0-1.0e-6)
                   & ((self.root_rise[ids]-expected).abs() <= 0.10))
        values = {
            "duration_s": self.duration[ids], "survived": survived, "survived20s": (self.duration[ids] >= 20.0-1e-5),
            "completed_levels": self.completed[ids], "verified_feet": self.verified[ids].sum(dim=(1, 2)),
            "root_rise_m": self.root_rise[ids], "target_rise_m": expected,
            "route_progress_ratio": self.route_progress[ids], "top_dwell_s": self.platform_max_time[ids],
            "success": success, "terrain_level": self.terrain_level[ids],
            "left_valid_landings": self.footfalls.landings[ids, 0],
            "right_valid_landings": self.footfalls.landings[ids, 1],
            "stance_slip_m_s": self.slip_integral[ids] / self.slip_time[ids].clamp_min(1e-6),
            "transferred_levels": self.transferred[ids].sum(-1), "initial_partial": self.partial[ids],
            "terminated": terminated[ids], "timed_out": timed_out[ids], "valid": valid,
        }
        for key, value in values.items():
            self.last_episode[key][ids] = value.to(self.last_episode[key].dtype)
        self.snapshot_before_reset = {"env_ids": ids.clone(), **{key: value.clone() for key, value in values.items()},
                                      "verified": self.verified[ids].clone(),
                                      "filtered_force_positive_control": self.positive_control_seen[ids].clone()}
        if valid.any():
            rows = torch.stack([values[key].float() for key in self.SUMMARY_FIELDS], -1)
            self.recent.extend(rows[valid].detach().cpu().tolist())

    def reset(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.needs_start[ids] = True
        self.partial[ids] = False
        for value in (self.duration, self.completed, self.verified, self.transferred, self.pending_level,
                      self.loaded_time, self.platform_time, self.platform_max_time, self.top_paid,
                      self.potential, self.root_rise, self.route_progress, self.slip_integral,
                      self.slip_time, self.current_slip, self.current_support_height):
            value[ids] = 0
        self.footfalls.reset(ids)

    def summary(self):
        if not self.recent:
            return {"episode_count": 0, "stair_success_fraction": 0.0, "mean_completed_levels": 0.0,
                    "first_level_bilateral_fraction": 0.0, "all_levels_bilateral_fraction": 0.0}
        rows = np.asarray(self.recent)
        c = {key: rows[:, i] for i, key in enumerate(self.SUMMARY_FIELDS)}
        return {"episode_count": len(rows), "survival_fraction": float(c["survived"].mean()),
                "survived20s_fraction": float(c["survived20s"].mean()),
                "stair_success_fraction": float(c["success"].mean()),
                "mean_completed_levels": float(c["completed_levels"].mean()),
                "first_level_bilateral_fraction": float((c["completed_levels"] >= 1).mean()),
                "all_levels_bilateral_fraction": float((c["completed_levels"] == self.steps).mean()),
                "mean_verified_feet": float(c["verified_feet"].mean()),
                "mean_transferred_levels": float(c["transferred_levels"].mean()),
                "median_duration_s": float(np.median(c["duration_s"])),
                "mean_root_rise_m": float(c["root_rise_m"].mean()),
                "mean_route_progress_ratio": float(c["route_progress_ratio"].mean()),
                "mean_platform_dwell_s": float(c["top_dwell_s"].mean()),
                "mean_stance_slip_m_s": float(c["stance_slip_m_s"].mean()),
                "mean_terrain_level": float(c["terrain_level"].mean()),
                "minimum_valid_air_s": 0.08, "maximum_valid_air_s": 0.6,
                "minimum_loaded_time_s": 0.08, "footprint_margin_m": 0.008,
                "initial_partial_episodes_excluded": True,
                "metrics_are_training_evidence_not_formal_acceptance": True}
