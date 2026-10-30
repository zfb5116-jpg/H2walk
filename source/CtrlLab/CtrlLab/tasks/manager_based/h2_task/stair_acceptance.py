"""Independent per-level physical stair acceptance; geometry is evaluator-only."""

import torch
from .foot_geometry import sole_footprint_corners


class StairAcceptance:
    def __init__(self, n, device, origins, masses, *, height, depth, steps, approach):
        self.n, self.device, self.origins = n, device, origins
        self.height, self.depth, self.steps, self.approach = height, depth, steps, approach
        self.threshold = torch.maximum(torch.full((n,), 20.0, device=device), masses * .981)
        self.completed = torch.zeros(n, dtype=torch.long, device=device)
        self.verified = torch.zeros(n, steps, 2, dtype=torch.bool, device=device)
        self.verified_time = torch.full((n, steps, 2), -1.0, device=device)
        self.verified_position = torch.zeros(n, steps, 2, 3, device=device)
        self.previous_landings = torch.zeros(n, 2, device=device)
        self.pending_level = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.stable_time = torch.zeros(n, 2, device=device)
        self.platform_time = torch.zeros(n, device=device)
        self.platform_max_time = torch.zeros(n, device=device)
        self.root_start = None
        self.elapsed = torch.zeros(n, device=device)
        self.corners = sole_footprint_corners(device)

    def update(self, root_position, soles, rotations, forces, landings, dt, active, root_tilt):
        ids = active.nonzero().flatten()
        if not ids.numel():
            return
        if self.root_start is None:
            self.root_start = root_position.clone()
        self.elapsed[ids] += dt
        local = soles[ids] - self.origins[ids, None]
        corners = (rotations[ids, :, None] @ self.corners[None, :, :, :, None]).squeeze(-1)
        corners += local[:, :, None]
        min_x, max_x = corners[..., 0].amin(-1), corners[..., 0].amax(-1)
        max_y = corners[..., 1].abs().amax(-1)
        level = (local[..., 2] / self.height).round().long()
        near_height = (local[..., 2] - level * self.height).abs() <= .03
        level_ok = (level >= 1) & (level <= self.steps)
        start_x = self.approach + (level - 1) * self.depth
        end_x = self.approach + level * self.depth
        end_x = torch.where(level == self.steps, torch.full_like(end_x, 9.0), end_x)
        footprint = (min_x >= start_x + .008) & (max_x <= end_x - .008) & (max_y <= 1.18)
        flat_sole = rotations[ids, :, 2, 2] > .975
        loaded = forces[ids, :, 2] > self.threshold[ids, None]
        geometric_tread = level_ok & near_height & footprint & flat_sole
        on_tread = geometric_tread & loaded
        new_valid = landings[ids] > self.previous_landings[ids]
        self.previous_landings[ids] = landings[ids]
        contact = forces[ids].norm(dim=-1) > 5.0
        self.pending_level[ids] = torch.where(contact, self.pending_level[ids], 0)
        # Arm a valid aerial transition at initial heel/toe contact. Only the
        # subsequent full-foot, near-horizontal80ms loaded pose verifies it.
        landing_candidate = level_ok & footprint & ((local[..., 2] - level * self.height).abs() <= .06)
        self.pending_level[ids] = torch.where(new_valid & landing_candidate, level, self.pending_level[ids])
        matching = on_tread & (self.pending_level[ids] == level)
        self.stable_time[ids] = torch.where(matching, self.stable_time[ids] + dt, 0)
        for foot in (0, 1):
            permitted = (matching[:, foot] & (self.stable_time[ids, foot] >= .08 - 1e-6)
                         & (level[:, foot] == self.completed[ids] + 1))
            sub = ids[permitted]
            lev = level[permitted, foot] - 1
            unseen = ~self.verified[sub, lev, foot]
            sub, lev = sub[unseen], lev[unseen]
            self.verified[sub, lev, foot] = True
            self.verified_time[sub, lev, foot] = self.elapsed[sub]
            self.verified_position[sub, lev, foot] = soles[sub, foot]
        next_level = self.completed[ids].clamp_max(self.steps - 1)
        advance = (self.completed[ids] < self.steps) & self.verified[ids, next_level].all(-1)
        self.completed[ids] += advance
        platform_start = self.approach + self.steps * self.depth
        top = (self.completed[ids] == self.steps) & on_tread.all(-1) & (level == self.steps).all(-1)
        top &= (min_x >= platform_start + .008).all(-1) & (root_tilt[ids] < .25)
        self.platform_time[ids] = torch.where(top, self.platform_time[ids] + dt, 0)
        self.platform_max_time[ids] = torch.maximum(self.platform_max_time[ids], self.platform_time[ids])

    def result(self, end_position, survived):
        rise = end_position[:, 2] - self.root_start[:, 2]
        expected = self.steps * self.height
        joint = (survived & self.verified.all(dim=(1, 2)) & (self.completed == self.steps)
                 & (self.platform_max_time >= 2.0 - 1e-5) & ((rise - expected).abs() <= .10))
        return joint, {"successful_envs": int(joint.sum()), "required_successful_envs": 922,
                       "num_envs": self.n, "success_percent": float(joint.float().mean() * 100),
                       "each_level_both_feet_count": self.verified.all(-1).sum(0).cpu().tolist(),
                       "all_levels_both_feet_envs": int(self.verified.all(dim=(1, 2)).sum()),
                       "platform_dwell2s_envs": int((self.platform_max_time >= 2.0 - 1e-5).sum()),
                       "root_rise_in_range_envs": int(((rise - expected).abs() <= .10).sum()),
                       "expected_net_root_rise_m": expected, "minimum_platform_dwell_s": 2.0,
                       "riser_height_m": self.height, "tread_depth_m": self.depth, "levels": self.steps,
                       "minimum_foot_load_N": ">=max(20N,10%body_weight),80ms",
                       "valid_landing_gate": "original80-600ms flight, alternate foot, other foot supported; initial contacts/hops excluded",
                       "geometry_used_only_by_evaluator": True}
