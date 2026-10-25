"""Local terrain-height queries on existing GT ray hits, without Isaac imports."""

from __future__ import annotations

import math

import torch


def local_ground_height(
    ray_hits_w: torch.Tensor,
    foot_positions_w: torch.Tensor,
    radius: float = 0.10,
) -> torch.Tensor:
    """Median ground Z near each foot's world XY, or NaN when not observed.

    Inputs are ``[env, ray, xyz]`` and ``[env, foot, xyz]``. Only finite rays
    within the horizontal radius are used; an absent scan never creates a
    world-Z-zero floor. The query is local to the foot and does not inspect a
    lower landing surface farther ahead on descending stairs. It estimates a
    local surface height, not foothold coverage or a terrain classification.
    """
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if ray_hits_w.ndim != 3 or ray_hits_w.shape[-1] != 3:
        raise ValueError("ray_hits_w must have shape [env, ray, 3]")
    if foot_positions_w.ndim != 3 or foot_positions_w.shape[-1] != 3:
        raise ValueError("foot_positions_w must have shape [env, foot, 3]")
    if ray_hits_w.shape[0] != foot_positions_w.shape[0]:
        raise ValueError("ray hits and foot positions must have the same environment count")
    if not ray_hits_w.is_floating_point() or not foot_positions_w.is_floating_point():
        raise ValueError("ray hits and foot positions must be floating point")
    if ray_hits_w.device != foot_positions_w.device or ray_hits_w.dtype != foot_positions_w.dtype:
        raise ValueError("ray hits and foot positions must share dtype and device")
    if ray_hits_w.shape[1] == 0:
        return foot_positions_w.new_full(foot_positions_w.shape[:2], torch.nan)

    valid_hits = torch.isfinite(ray_hits_w).all(dim=-1)
    valid_feet = torch.isfinite(foot_positions_w).all(dim=-1)
    safe_hits = torch.where(valid_hits.unsqueeze(-1), ray_hits_w, 0.0)
    safe_feet = torch.where(valid_feet.unsqueeze(-1), foot_positions_w, 0.0)
    delta_xy = safe_hits[:, None, :, :2] - safe_feet[:, :, None, :2]
    nearby = valid_hits[:, None] & valid_feet[:, :, None] & (delta_xy.square().sum(dim=-1) <= radius ** 2)
    heights = safe_hits[:, None, :, 2].expand_as(nearby)
    return torch.where(nearby, heights, torch.nan).nanmedian(dim=-1).values
