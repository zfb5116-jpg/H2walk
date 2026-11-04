"""Terrain reward references from the existing GT scan, without Isaac imports.

The sampled sole rectangle is inside H2's audited bottom face. Its support
fraction is a coarse scan geometry proxy, not contact area, CoP, friction or a
kinematic/collision-free footstep certificate. All poses use world coordinates
and wxyz quaternions. Unknown scan regions remain unknown.
"""
from __future__ import annotations

import math

import torch


TERRAIN_REWARD_PARAMETERS = {
    "semantic_version": "terrain_v2",
    "sole_x_min": -.05,
    "sole_x_max": .135,
    "sole_x_count": 6,
    "sole_left_y": (.025, .038, .052),
    "sole_z": -.05429999902844429,
    "query_radius": .04,
    "query_chunk_size": 16,
    "surface_tolerance": .015,
    "candidate_distances": (.16, .20, .24, .28, .32, .36, .40),
    "max_height_change": .24,
    "apex_clearance": .035,
    "min_landing_coverage": .6,
    "swing_height_std": .05,
    "landing_force_ratio": 1.,
    "landing_cost_cap": 25.,
    "min_air_time": .08,
    "max_air_time": 1.2,
    "min_step_distance": .04,
    "min_root_progress": .01,
    "min_swing_speed": .02,
    "min_command_speed": .1,
    "max_step_speed": 5.,
}


def _poses(positions: torch.Tensor, quaternions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.ndim != 3 or positions.shape[1:] != (2, 3):
        raise ValueError("positions must have shape [env, 2, 3]")
    if quaternions.shape != (*positions.shape[:-1], 4):
        raise ValueError("quaternions must have shape [env, 2, 4]")
    if not positions.is_floating_point() or not quaternions.is_floating_point():
        raise ValueError("poses must be floating point")
    if positions.dtype != quaternions.dtype or positions.device != quaternions.device:
        raise ValueError("poses must share dtype and device")
    finite = torch.isfinite(quaternions).all(dim=-1)
    safe = torch.where(finite[..., None], quaternions, 0.)
    norm = torch.linalg.vector_norm(safe, dim=-1)
    valid = finite & torch.isfinite(norm) & (norm > 1e-12) & torch.isfinite(positions).all(dim=-1)
    return safe / torch.where(valid, norm, 1.)[..., None], valid


def _sole_points(reference: torch.Tensor) -> torch.Tensor:
    p = TERRAIN_REWARD_PARAMETERS
    x = torch.linspace(p["sole_x_min"], p["sole_x_max"], p["sole_x_count"],
                       device=reference.device, dtype=reference.dtype)
    y = reference.new_tensor(p["sole_left_y"])
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    left = torch.stack((xx.flatten(), yy.flatten(), torch.full_like(xx.flatten(), p["sole_z"])), dim=-1)
    right = left.clone()
    right[:, 1] *= -1.
    return torch.stack((left, right))


def sole_samples_world(positions: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """Transform 18 interior bottom-face samples per foot using its full pose."""
    quat, valid = _poses(positions, quaternions)
    points = _sole_points(positions)[None]
    vector = quat[..., None, 1:].expand(-1, -1, points.shape[-2], -1)
    points = points.expand(positions.shape[0], -1, -1, -1)
    cross = 2. * torch.linalg.cross(vector, points, dim=-1)
    rotated = points + quat[..., None, :1] * cross + torch.linalg.cross(vector, cross, dim=-1)
    return torch.where(valid[..., None, None], rotated + positions[..., None, :], torch.nan)


def nearest_heights(rays: torch.Tensor, points: torch.Tensor, radius: float = .04) -> torch.Tensor:
    """Nearest finite ray Z within radius for each XY query, otherwise NaN.

    Queries are processed in blocks of at most 16 points, avoiding an
    environment-by-all-sole-points-by-scan-points temporary.
    """
    if isinstance(radius, bool) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if rays.ndim != 3 or rays.shape[-1] != 3 or points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("rays must be [env, ray, 3] and points [env, ..., 3]")
    if rays.shape[0] != points.shape[0]:
        raise ValueError("rays and query points must share the environment count")
    if (not rays.is_floating_point() or not points.is_floating_point()
            or rays.dtype != points.dtype or rays.device != points.device):
        raise ValueError("rays and query points must share floating dtype and device")
    shape = points.shape[:-1]
    flat = points.reshape(points.shape[0], -1, 3)
    if rays.shape[1] == 0 or flat.shape[1] == 0:
        return points.new_full(shape, torch.nan)
    finite_rays = torch.isfinite(rays).all(dim=-1)
    safe_rays = torch.where(finite_rays[..., None], rays, 0.)
    result = []
    chunk_size = TERRAIN_REWARD_PARAMETERS["query_chunk_size"]
    for start in range(0, flat.shape[1], chunk_size):
        query = flat[:, start:start + chunk_size]
        finite_query = torch.isfinite(query).all(dim=-1)
        safe_query = torch.where(finite_query[..., None], query, 0.)
        squared = ((safe_query[:, :, None, 0] - safe_rays[:, None, :, 0]).square()
                   + (safe_query[:, :, None, 1] - safe_rays[:, None, :, 1]).square())
        squared = torch.where(finite_rays[:, None] & finite_query[..., None], squared, torch.inf)
        distance, index = squared.min(dim=-1)
        height = safe_rays[..., 2].gather(1, index)
        result.append(torch.where(distance <= radius * radius + 1e-12, height, torch.nan))
    return torch.cat(result, dim=1).reshape(shape)


def support_geometry(rays: torch.Tensor, positions: torch.Tensor,
                     quaternions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sampled near-surface fraction and complete-query validity.

    Unknown samples count as unsupported in the fraction; any such sample
    makes validity false. Callers must retain that distinction. This function
    does not inspect forces and cannot establish actual physical support.
    """
    points = sole_samples_world(positions, quaternions)
    ground = nearest_heights(rays, points, TERRAIN_REWARD_PARAMETERS["query_radius"])
    known = torch.isfinite(ground) & torch.isfinite(points).all(dim=-1)
    supported = known & ((points[..., 2] - ground).abs() <= TERRAIN_REWARD_PARAMETERS["surface_tolerance"] + 1e-6)
    return supported.to(positions.dtype).mean(dim=-1), known.all(dim=-1)


def select_swing_targets(rays: torch.Tensor, feet: torch.Tensor, quaternions: torch.Tensor,
                         direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the nearest observed flat landing patch in each foot's lane.

    Landing XY is the ankle link reference; landing Z is the sole surface Z,
    not ankle link Z. Candidate sole samples retain only the foot's yaw for a
    flat landing. The apex includes finite rays in the swept sample footprint.
    A path above the maximum allowed rise invalidates the candidate, so a
    farther low patch cannot skip an intervening high obstacle. Reachability,
    unobserved holes and sub-grid terrain geometry are not certified.
    """
    quat, pose_valid = _poses(feet, quaternions)
    if direction.shape != (feet.shape[0], 2):
        raise ValueError("direction must have shape [env, 2]")
    if direction.device != feet.device or direction.dtype != feet.dtype:
        raise ValueError("direction and feet must share dtype and device")
    if rays.ndim != 3 or rays.shape[0] != feet.shape[0] or rays.shape[-1] != 3:
        raise ValueError("rays must have shape [env, ray, 3]")
    if rays.dtype != feet.dtype or rays.device != feet.device:
        raise ValueError("rays and feet must share dtype and device")
    p = TERRAIN_REWARD_PARAMETERS
    finite_direction = torch.isfinite(direction).all(dim=-1)
    safe_direction = torch.where(finite_direction[:, None], direction, 0.)
    length = safe_direction.norm(dim=-1)
    heading_valid = finite_direction & torch.isfinite(length) & (length > 1e-12)
    heading = safe_direction / length.clamp_min(1e-12)[:, None]
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2. * (w*z + x*y), 1. - 2. * (y*y + z*z))
    local = _sole_points(feet)
    offsets = torch.stack((yaw.cos()[..., None] * local[None, :, :, 0] - yaw.sin()[..., None] * local[None, :, :, 1],
                           yaw.sin()[..., None] * local[None, :, :, 0] + yaw.cos()[..., None] * local[None, :, :, 1]), dim=-1)
    distances = feet.new_tensor(p["candidate_distances"])
    candidate_xy = feet[:, :, None, :2] + heading[:, None, None, :] * distances[None, None, :, None]
    sample_xy = candidate_xy[..., None, :] + offsets[:, :, None]
    query = torch.cat((sample_xy, torch.zeros_like(sample_xy[..., :1])), dim=-1)
    heights = nearest_heights(rays, query, p["query_radius"])
    known = torch.isfinite(heights).all(dim=-1)
    floor = heights.nanmedian(dim=-1).values
    spread = torch.where(torch.isfinite(heights), heights, torch.inf).amax(dim=-1) - torch.where(
        torch.isfinite(heights), heights, -torch.inf).amin(dim=-1)
    start_points = sole_samples_world(feet, quaternions)
    start_z = start_points[..., 2].amin(dim=-1)
    start_ground = nearest_heights(rays, start_points, p["query_radius"])
    start_known = torch.isfinite(start_ground).all(dim=-1)
    usable = (known & (spread <= p["surface_tolerance"] + 1e-6)
              & ((floor - start_z[..., None]).abs() <= p["max_height_change"] + 1e-6)
              & pose_valid[..., None] & heading_valid[:, None, None] & start_known[..., None])
    if rays.shape[1] == 0:
        return feet.new_full(feet.shape, torch.nan), feet.new_full(feet.shape[:2], torch.nan), torch.zeros_like(pose_valid)
    finite_rays = torch.isfinite(rays).all(dim=-1)
    safe_rays = torch.where(finite_rays[..., None], rays, 0.)
    relative = safe_rays[:, None, :, :2] - feet[:, :, None, :2]
    forward = (relative * heading[:, None, None]).sum(dim=-1)
    perpendicular = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
    side = (relative * perpendicular[:, None, None]).sum(dim=-1)
    offset_forward = (offsets * heading[:, None, None]).sum(dim=-1)
    offset_side = (offsets * perpendicular[:, None, None]).sum(dim=-1)
    lower = offset_forward.amin(dim=-1)
    upper = offset_forward.amax(dim=-1)
    side_min, side_max = offset_side.amin(dim=-1), offset_side.amax(dim=-1)
    path_peaks = []
    for distance in distances:
        inside = (finite_rays[:, None] & (forward >= lower[..., None] - p["query_radius"])
                  & (forward <= distance + upper[..., None] + p["query_radius"])
                  & (side >= side_min[..., None] - p["query_radius"])
                  & (side <= side_max[..., None] + p["query_radius"]))
        peak = torch.where(inside, safe_rays[:, None, :, 2], -torch.inf).amax(dim=-1)
        path_peaks.append(torch.where(inside.any(dim=-1), peak, torch.nan))
    peaks = torch.stack(path_peaks, dim=-1)
    usable &= torch.isfinite(peaks) & (peaks - start_z[..., None] <= p["max_height_change"] + 1e-6)
    index = usable.to(torch.long).argmax(dim=-1)
    valid = usable.any(dim=-1)
    selected_xy = candidate_xy.gather(2, index[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    selected_z = floor.gather(2, index[..., None]).squeeze(2)
    selected_peak = peaks.gather(2, index[..., None]).squeeze(2)
    landing = torch.cat((selected_xy, selected_z[..., None]), dim=-1)
    apex = torch.maximum(torch.maximum(start_z, selected_z), selected_peak) + p["apex_clearance"]
    return (torch.where(valid[..., None], landing, torch.nan),
            torch.where(valid, apex, torch.nan), valid)


def swing_height_reference(start_z: torch.Tensor, landing_z: torch.Tensor,
                           apex: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """C1 piecewise smoothstep reference, including the lower final tread."""
    phase = u.clamp(0., 1.)
    first = (2. * phase).clamp(0., 1.)
    second = (2. * phase - 1.).clamp(0., 1.)
    first = first.square() * (3. - 2. * first)
    second = second.square() * (3. - 2. * second)
    return torch.where(phase <= .5, start_z + (apex - start_z) * first,
                       apex + (landing_z - apex) * second)
