"""Conservative support domains inferred from measured ground contacts."""

import torch
from .stair_com_x_admittance import (initialize_tactile_cop_latest, stamp_tactile_normal_read, finish_tactile_normal_read, finish_tactile_empty_normal_read, record_tactile_normal_cop)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _extreme_hull(points, local_xy, owner, num_envs):
    """Convex hull of four actual directional extrema, never a nominal box."""
    device = points.device
    directions = points.new_tensor([[-1, 1], [1, 1], [1, -1], [-1, -1]])
    score = local_xy @ directions.T
    maximum = points.new_full((num_envs, 4), -float("inf"))
    maximum.scatter_reduce_(0, owner[:, None].expand(-1, 4), score, reduce="amax", include_self=True)
    candidate = torch.arange(owner.numel(), device=device)[:, None].expand(-1, 4)
    candidate = torch.where(score == maximum[owner], candidate, owner.numel())
    selected = torch.full((num_envs, 4), owner.numel(), dtype=torch.long, device=device)
    selected.scatter_reduce_(0, owner[:, None].expand(-1, 4), candidate, reduce="amin", include_self=True)
    polygon = points[selected.clamp_max(owner.numel() - 1)]
    keep = selected < owner.numel()
    for j in range(1, 4):
        duplicate = (polygon[:, :j, :2] - polygon[:, j, None, :2]).norm(dim=-1) < 1e-5
        keep[:, j] &= ~(duplicate & keep[:, :j]).any(-1)
    four = keep.all(-1)
    remove = torch.zeros_like(keep)
    for j in range(4):
        a, b, c = polygon[:, [i for i in range(4) if i != j], :2].unbind(1)
        p = polygon[:, j, :2]
        signed_area = _cross(b - a, c - a)
        sign = signed_area.sign()
        inside = ((_cross(b - a, p - a) * sign >= -1e-10)
                  & (_cross(c - b, p - b) * sign >= -1e-10)
                  & (_cross(a - c, p - c) * sign >= -1e-10))
        remove[:, j] = four & (signed_area.abs() > 1e-10) & inside
    keep &= ~remove
    count = keep.sum(-1)
    center = (polygon[..., :2] * keep[..., None]).sum(1) / count.clamp_min(1)[:, None]
    relative = polygon[..., :2] - center[:, None]
    angle = torch.atan2(relative[..., 1], relative[..., 0])
    order = torch.where(keep, angle, float("inf")).argsort(-1)
    polygon = polygon.gather(1, order[..., None].expand(-1, -1, 3))
    edge_mask = torch.arange(4, device=device)[None] < count[:, None]
    next_index = (torch.arange(4, device=device)[None] + 1) % count.clamp_min(1)[:, None]
    following = polygon.gather(1, next_index[..., None].expand(-1, -1, 3))
    edge = following[..., :2] - polygon[..., :2]
    # Translate before the shoelace sum to avoid cancellation at large origins.
    area = (_cross(polygon[..., :2] - center[:, None],
                   following[..., :2] - center[:, None]) * edge_mask).sum(-1) * .5
    next_edge = edge.gather(1, next_index[..., None].expand(-1, -1, 2))
    convex = ((_cross(edge, next_edge) >= -1e-8) | ~edge_mask).all(-1)
    edges_distinct = ((edge.norm(dim=-1) >= 1e-5) | ~edge_mask).all(-1)
    valid = (count >= 3) & (area >= .002) & convex & edges_distinct
    return polygon, edge_mask, area, valid


class TactileSupport:
    COUNTERS = (
        "updates", "missing_sensor_frames", "layout_rejected_frames",
        "buffer_rejected_frames", "buffer_at_capacity_frames", "no_contact_frames",
        "nonfinite_contact_frames", "separation_rejected_frames", "force_mismatch_frames", "normal_sign_unconfirmed_frames",
        "foot_positive_control_rejected_frames",
        "insufficient_support_points_frames", "bad_polygon_frames", "height_spread_rejected_frames",
        "foot_tilt_rejected_frames", "load_rejected_frames", "unstable_frames",
        "accepted_support_frames", "cells_observed", "new_cells", "height_conflict_observations",
    )
    MEASUREMENTS = (
        "raw_contact_count", "support_point_count", "hull_vertex_count", "hull_area_m2",
        "filtered_fz_N", "raw_fz_N", "unfiltered_fz_N", "foot_positive_control_passed",
        "support_fz_N", "force_error_N", "min_separation_m", "max_separation_m",
    )

    def __init__(self, env, robot, feet, mass, rotation_from_quat):
        self.env, self.robot, self.feet, self.mass = env, robot, feet, mass
        self.rotation = rotation_from_quat
        self.sensors = [env.scene.sensors.get("stair_left_ground"), env.scene.sensors.get("stair_right_ground")]
        self.unfiltered_sensor = env.scene.sensors.get("contact_forces")
        self.unfiltered_foot_ids = None
        self.unfiltered_foot_names = None
        if self.unfiltered_sensor is not None:
            foot_names = getattr(env.cfg, "feet_body_names", ("left_ankle_pitch_link", "right_ankle_pitch_link"))
            try:
                ids, names = self.unfiltered_sensor.find_bodies(foot_names, preserve_order=True)
            except (AttributeError, ValueError):
                ids, names = [], []
            if len(ids) == 2 and len(set(ids)) == 2 and len(names) == 2:
                self.unfiltered_foot_ids, self.unfiltered_foot_names = list(ids), list(names)
        self.age = torch.zeros(env.num_envs, 2, device=env.device)
        self.stats = {name: torch.zeros_like(self.age, dtype=torch.long) for name in self.COUNTERS}
        self.latest = {name: torch.zeros_like(self.age) for name in self.MEASUREMENTS}
        initialize_tactile_cop_latest(self)
        self.normal_sign = [0, 0]
        self.first_foot_positive_control = [None, None]
        self.last_step = None
        self.duplicate_calls = 0
        depth = (getattr(env.cfg, "contact_gait_reference", None) or {}).get("ground_contact_depth", .006)
        self.min_separation = -(float(depth) + .004)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.age[ids] = 0
        for value in (*self.stats.values(), *self.latest.values()):
            value[ids] = 0

    def _count(self, name, foot, mask):
        self.stats[name][:, foot] += mask.to(torch.long)

    def _reject_all(self, reason, foot):
        self.age[:, foot] = 0
        self.stats[reason][:, foot] += 1

    def update(self, mapping):
        step = int(self.env.common_step_counter)
        if step == self.last_step:
            self.duplicate_calls += 1
            return
        if self.last_step is not None and step != self.last_step + 1:
            self.age.zero_()  # Unobserved time cannot establish stable support.
        self.last_step = step
        device, n = mapping.device, mapping.n
        unfiltered_fz = self.age.new_full((n, 2), float("nan"))
        if self.unfiltered_sensor is not None and self.unfiltered_foot_ids is not None:
            net = self.unfiltered_sensor.data.net_forces_w
            ids = self.unfiltered_foot_ids
            if (net is not None and net.ndim == 3 and net.shape[0] == n and net.shape[-1] == 3
                    and min(ids) >= 0 and max(ids) < net.shape[1]):
                unfiltered_fz = net[:, ids, 2]
        for foot, sensor in enumerate(self.sensors):
            self.stats["updates"][:, foot] += 1
            for value in self.latest.values():
                value[:, foot] = 0
            self.latest["unfiltered_fz_N"][:, foot] = unfiltered_fz[:, foot]
            if sensor is None:
                self._reject_all("missing_sensor_frames", foot)
                continue
            matrix = sensor.data.force_matrix_w
            if (matrix is None or matrix.ndim != 4 or matrix.shape[0] != n or matrix.shape[1] != 1
                    or matrix.shape[-1] != 3 or matrix.shape[2] == 0
                    or sensor.contact_physx_view.sensor_count != n):
                self._reject_all("layout_rejected_frames", foot)
                continue
            filtered = matrix.sum(dim=(1, 2))
            self.latest["filtered_fz_N"][:, foot] = filtered[:, 2]
            forces, points, normals, separation, counts, starts = sensor.contact_physx_view.get_contact_data(
                dt=self.env.cfg.sim.dt)
            if (counts.ndim != 2 or counts.shape[0] != n or counts.shape[1] != matrix.shape[2]
                    or starts.shape != counts.shape or points.ndim != 2 or points.shape[1] != 3
                    or normals.ndim != 2 or normals.shape[1] != 3
                    or forces.numel() != forces.shape[0] or separation.numel() != separation.shape[0]):
                self._reject_all("layout_rejected_frames", foot)
                continue
            counts, starts = counts.long(), starts.long()
            capacity = min(points.shape[0], normals.shape[0], forces.shape[0], separation.shape[0])
            bounds = ((counts >= 0) & (starts >= 0) & (starts <= capacity)
                      & (counts <= capacity) & (starts + counts <= capacity))
            bad_env = ~((counts == 0) | bounds).all(-1)
            self._count("buffer_rejected_frames", foot, bad_env)
            self.latest["raw_contact_count"][:, foot] = counts.clamp_min(0).sum(-1)
            safe_counts = torch.where(bad_env[:, None], 0, counts)
            stamp_tactile_normal_read(self,foot,filtered,bad_env,step)
            count = safe_counts.flatten()
            total = int(count.sum())
            if total and total >= capacity:
                self._reject_all("buffer_at_capacity_frames", foot)
                continue
            self._count("no_contact_frames", foot, safe_counts.sum(-1) == 0)
            if not total:
                finish_tactile_empty_normal_read(self,foot,filtered,unfiltered_fz[:,foot],bad_env,step)
                self.age[:, foot] = 0
                continue
            row = torch.repeat_interleave(torch.arange(count.numel(), device=device), count)
            block = count.cumsum(0) - count
            within = torch.arange(total, device=device) - torch.repeat_interleave(block, count)
            index = starts.flatten()[row] + within
            p, f = points[index], forces[index].flatten()
            normal, distance = normals[index], separation[index].flatten()
            owner = row // counts.shape[1]
            finite = (torch.isfinite(p).all(-1) & torch.isfinite(normal).all(-1)
                      & torch.isfinite(f) & torch.isfinite(distance))
            bad_separation = finite & ((distance > .002) | (distance < self.min_separation))
            self._count("separation_rejected_frames", foot,
                        torch.bincount(owner[bad_separation], minlength=n) > 0)
            invalid_count = torch.bincount(owner[~finite], minlength=n)
            bad_env |= invalid_count > 0
            self._count("nonfinite_contact_frames", foot, invalid_count > 0)
            aggregate = torch.zeros_like(filtered)
            aggregate.index_add_(0, owner, torch.where(finite[:, None], f[:, None] * normal, 0))
            rotation = self.rotation(self.robot.data.body_quat_w[:, self.feet[foot]])
            origin = self.robot.data.body_pos_w[:, self.feet[foot]]
            candidate = (finite & (f > .5) & ((normal.norm(dim=-1) - 1).abs() < .05)
                         & (normal[:, 2].abs() > .98) & (distance <= .002)
                         & (distance >= self.min_separation))
            candidate_count = torch.bincount(owner[candidate], minlength=n)
            low, high = p.new_full((n,), float("inf")), p.new_full((n,), -float("inf"))
            low.scatter_reduce_(0, owner[candidate], p[candidate, 2], reduce="amin", include_self=True)
            high.scatter_reduce_(0, owner[candidate], p[candidate, 2], reduce="amax", include_self=True)
            threshold = torch.maximum(torch.full_like(filtered[:, 2], 20), self.mass * .981)
            loaded = torch.isfinite(filtered).all(-1) & (filtered[:, 2] >= threshold)
            flat = rotation[:, 2, 2] > .995
            tolerance = 1.0 + .02 * filtered.norm(dim=-1)
            unfiltered_ok = (torch.isfinite(unfiltered_fz[:, foot])
                             & ((unfiltered_fz[:, foot] - filtered[:, 2]).abs()
                                <= 1.0 + .02 * filtered[:, 2].abs()))
            control = loaded & flat & (candidate_count >= 3) & (high - low < .004) & ~bad_env
            self._count("foot_positive_control_rejected_frames", foot, control & ~unfiltered_ok)
            if not self.normal_sign[foot]:
                direct = control & unfiltered_ok & ((aggregate - filtered).norm(dim=-1) <= tolerance)
                reverse = control & unfiltered_ok & ((-aggregate - filtered).norm(dim=-1) <= tolerance)
                if bool(direct.any()) and not bool(reverse.any()):
                    self.normal_sign[foot] = 1
                elif bool(reverse.any()) and not bool(direct.any()):
                    self.normal_sign[foot] = -1
            sign = self.normal_sign[foot]
            error = (sign * aggregate - filtered).norm(dim=-1)
            force_ok = (sign != 0) & torch.isfinite(error) & (error <= tolerance) & ~bad_env
            positive_control = control & unfiltered_ok & force_ok
            finish_tactile_normal_read(self,foot,filtered,unfiltered_fz[:,foot],aggregate,bad_env,safe_counts.sum(-1)==0,step)
            self.latest["foot_positive_control_passed"][:, foot] = positive_control
            record_tactile_normal_cop(self,foot,p,f,normal,distance,owner,filtered,unfiltered_fz[:,foot],origin,rotation,positive_control,bad_env,step)
            if self.first_foot_positive_control[foot] is None and bool(positive_control.any()):
                env_id = int(positive_control.nonzero()[0, 0])
                self.first_foot_positive_control[foot] = {
                    "env_id": env_id, "clock": int(mapping.clock), "common_step_counter": step,
                    "simulation_time_s": step * float(self.env.step_dt), "normal_sign": sign,
                    "unfiltered_body_index": self.unfiltered_foot_ids[foot],
                    "unfiltered_body_name": self.unfiltered_foot_names[foot],
                    "filtered_fz_N": float(filtered[env_id, 2]),
                    "raw_fz_N": float(sign * aggregate[env_id, 2]),
                    "unfiltered_fz_N": float(unfiltered_fz[env_id, foot]),
                    "raw_contact_count": int(safe_counts[env_id].sum()),
                    "support_point_count": int(candidate_count[env_id]),
                    "points_w_m": p[(owner == env_id) & candidate].tolist(),
                }
            self.latest["raw_fz_N"][:, foot] = sign * aggregate[:, 2]
            self.latest["force_error_N"][:, foot] = error
            self._count("force_mismatch_frames", foot, (safe_counts.sum(-1) > 0) & ~force_ok)
            if not sign:
                self._count("normal_sign_unconfirmed_frames", foot, safe_counts.sum(-1) > 0)
            valid = candidate & (sign * normal[:, 2] > .98) & force_ok[owner]
            support_z = torch.zeros_like(filtered[:, 2])
            support_z.index_add_(0, owner[valid], (sign * f * normal[:, 2])[valid])
            self.latest["support_fz_N"][:, foot] = support_z
            load_ok = loaded & (support_z >= threshold)
            raw_min, raw_max = p.new_full((n,), float("inf")), p.new_full((n,), -float("inf"))
            raw_min.scatter_reduce_(0, owner[finite], distance[finite], reduce="amin", include_self=True)
            raw_max.scatter_reduce_(0, owner[finite], distance[finite], reduce="amax", include_self=True)
            self.latest["min_separation_m"][:, foot] = torch.where(torch.isfinite(raw_min), raw_min, 0)
            self.latest["max_separation_m"][:, foot] = torch.where(torch.isfinite(raw_max), raw_max, 0)
            owner, p = owner[valid], p[valid]
            point_count = torch.bincount(owner, minlength=n)
            self.latest["support_point_count"][:, foot] = point_count
            self._count("insufficient_support_points_frames", foot, point_count < 3)
            if not owner.numel():
                self.age[:, foot] = 0
                continue
            local = (rotation[owner].transpose(-1, -2) @ (p - origin[owner])[..., None]).squeeze(-1)
            polygon, edge_mask, area, polygon_ok = _extreme_hull(p, local[:, :2], owner, n)
            self.latest["hull_vertex_count"][:, foot] = edge_mask.sum(-1)
            self.latest["hull_area_m2"][:, foot] = area
            low, high = p.new_full((n,), float("inf")), p.new_full((n,), -float("inf"))
            low.scatter_reduce_(0, owner, p[:, 2], reduce="amin", include_self=True)
            high.scatter_reduce_(0, owner, p[:, 2], reduce="amax", include_self=True)
            level = high - low < .004
            self._count("bad_polygon_frames", foot, (point_count >= 3) & ~polygon_ok)
            self._count("height_spread_rejected_frames", foot, (point_count >= 3) & ~level)
            self._count("foot_tilt_rejected_frames", foot, ~flat)
            self._count("load_rejected_frames", foot, ~load_ok)
            eligible = (point_count >= 3) & polygon_ok & level & flat & load_ok & force_ok
            self.age[:, foot] = torch.where(eligible, self.age[:, foot] + self.env.step_dt, 0)
            stable = eligible & (self.age[:, foot] >= .08 - 1e-6)
            self._count("unstable_frames", foot, eligible & ~stable)
            self._count("accepted_support_frames", foot, stable)
            ids = stable.nonzero().flatten()
            for begin in range(0, ids.numel(), 64):
                sub = ids[begin:begin + 64]
                center = origin[sub] - self.env.scene.env_origins[sub]
                ix = torch.floor((center[:, 0] - mapping.x0) / mapping.resolution).long()
                iy = torch.floor((center[:, 1] - mapping.y0) / mapping.resolution).long()
                dy, dx = torch.meshgrid(torch.arange(-6, 7, device=device),
                                        torch.arange(-7, 8, device=device), indexing="ij")
                x, y = ix[:, None] + dx.flatten(), iy[:, None] + dy.flatten()
                within_map = (x >= 0) & (x < mapping.nx) & (y >= 0) & (y < mapping.ny)
                xy = torch.stack((mapping.x0 + (x + .5) * mapping.resolution,
                                  mapping.y0 + (y + .5) * mapping.resolution), -1)
                offsets = p.new_tensor([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]]) * mapping.resolution
                query = xy[:, :, None] + offsets[None, None]
                poly = polygon[sub, :, :2] - self.env.scene.env_origins[sub, None, :2]
                count_vertices = edge_mask[sub].sum(-1)
                next_index = (torch.arange(4, device=device)[None] + 1) % count_vertices[:, None]
                edge = poly.gather(1, next_index[..., None].expand(-1, -1, 2)) - poly
                delta = query[:, :, :, None] - poly[:, None, None]
                cross = _cross(edge[:, None, None], delta)
                length = edge.norm(dim=-1)
                inside = ((cross >= .003 * length[:, None, None] - 1e-7)
                          | ~edge_mask[sub, None, None]).all(-1).all(-1)
                selected_row, selected_cell = (inside & within_map).nonzero(as_tuple=True)
                if not selected_row.numel():
                    continue
                env_ids = sub[selected_row]
                cell = y[selected_row, selected_cell] * mapping.nx + x[selected_row, selected_cell]
                z = ((high[sub] + low[sub]) * .5 - self.env.scene.env_origins[sub, 2])[selected_row]
                known = mapping.tactile_known[env_ids, cell]
                disagreement = known & ((mapping.tactile_height[env_ids, cell] - z).abs() > .008)
                raw_conflict = ((mapping.last_seen[env_ids, cell] >= 0)
                                & (((mapping.maximum[env_ids, cell] - z) > .008)
                                   | ((z - mapping.minimum[env_ids, cell]) > .008)))
                conflict = disagreement | raw_conflict
                new_invalid = conflict & ~mapping.tactile_invalid[env_ids, cell]
                mapping.tactile_invalid[env_ids, cell] |= conflict
                mapping.tactile_height[env_ids, cell] = torch.where(known, mapping.tactile_height[env_ids, cell], z)
                mapping.tactile_known[env_ids, cell] = True
                self.stats["cells_observed"][:, foot].index_add_(0, env_ids, torch.ones_like(env_ids))
                self.stats["new_cells"][:, foot].index_add_(0, env_ids, (~known).long())
                self.stats["height_conflict_observations"][:, foot].index_add_(0, env_ids, conflict.long())
                if bool((~known | new_invalid).any()):
                    mapping.cached_step = -1

    def summary(self):
        return {"normal_sign": list(self.normal_sign), "duplicate_calls_skipped": self.duplicate_calls,
                "first_foot_positive_control": list(self.first_foot_positive_control),
                "unfiltered_foot_body_indices": self.unfiltered_foot_ids,
                "unfiltered_foot_body_names": self.unfiltered_foot_names,
                "foot_positive_control_tolerance": "1N + 2% of abs(filtered Fz); same-environment original foot sensor",
                "foot_positive_control_semantics": "first successful control and normal convention persist across resets; latest flag describes the current frame",
                "minimum_separation_m": self.min_separation, "maximum_separation_m": .002,
                "force_agreement_tolerance": "1N + 2% of filtered normal-force vector magnitude",
                "counts": {name: value.sum(0).tolist() for name, value in self.stats.items()}}
