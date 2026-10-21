"""Mid-360 FOV, 16-ch × 1.6°, clipped to the paper H^b volume.

Same spherical grid as a full scan, but Warp only casts rays that sample
the reconstruction box within ``max_range``. Observation code scatters those
hits back onto the 16×n_az range image (uncast cells = 0), keeps a strided
azimuth subset of each frame, and stacks those frames across the delay
window (default 1.5 s, 4 frames). Delayed frames can be ego-motion warped
into the current lidar pose before that subsample.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass


def _spherical_dirs(cfg: "YuHbLidarPatternCfg") -> torch.Tensor:
    vertical_angles = torch.linspace(cfg.vertical_fov_range[0], cfg.vertical_fov_range[1], cfg.channels)
    n_az = math.ceil((cfg.horizontal_fov_range[1] - cfg.horizontal_fov_range[0]) / cfg.horizontal_res)
    horizontal_angles = torch.linspace(cfg.horizontal_fov_range[0], cfg.horizontal_fov_range[1], n_az)
    if abs(abs(cfg.horizontal_fov_range[0] - cfg.horizontal_fov_range[1]) - 360.0) < 1e-6:
        horizontal_angles = horizontal_angles[:-1]
    va, ha = torch.meshgrid(
        torch.deg2rad(vertical_angles),
        torch.deg2rad(horizontal_angles),
        indexing="ij",
    )
    return torch.stack(
        [
            torch.cos(va) * torch.cos(ha),
            torch.cos(va) * torch.sin(ha),
            torch.sin(va),
        ],
        dim=-1,
    ).reshape(-1, 3)


def yu_hb_lidar_keep_mask(cfg: "YuHbLidarPatternCfg") -> torch.Tensor:
    """Bool mask over the full spherical grid (True = cast this ray)."""
    dirs = _spherical_dirs(cfg)
    origin = torch.tensor(cfg.origin, dtype=dirs.dtype)
    bmin = torch.tensor(cfg.box_min, dtype=dirs.dtype)
    bmax = torch.tensor(cfg.box_max, dtype=dirs.dtype)
    tmin = float(cfg.min_range)
    tmax = float(cfg.max_range)
    n_slice = max(int(cfg.z_slices), 2)
    zs = torch.linspace(float(bmin[2]), float(bmax[2]), n_slice)
    dz = dirs[:, 2]
    inf = torch.full((dirs.shape[0],), float("inf"), dtype=dirs.dtype)
    keep = torch.zeros(dirs.shape[0], dtype=torch.bool)
    for zg in zs:
        t = torch.where(dz.abs() > 1e-5, (float(zg) - origin[2]) / dz, inf)
        px = origin[0] + t * dirs[:, 0]
        py = origin[1] + t * dirs[:, 1]
        keep |= (
            (t >= tmin)
            & (t <= tmax)
            & (px >= bmin[0])
            & (px <= bmax[0])
            & (py >= bmin[1])
            & (py <= bmax[1])
        )
    if not bool(keep.any()):
        keep[:] = True
    return keep


def yu_hb_lidar_pattern(cfg: "YuHbLidarPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Spherical grid → keep rays that hit H^b height layers within max_range."""
    dirs = _spherical_dirs(cfg)[yu_hb_lidar_keep_mask(cfg)].to(device)
    starts = torch.zeros_like(dirs)
    return starts, dirs


@configclass
class YuHbLidarPatternCfg(PatternBaseCfg):
    """16-ch × 1.6° Mid-360 grid, rays clipped to the paper H^b box."""

    func: Callable = yu_hb_lidar_pattern
    channels: int = 16
    vertical_fov_range: tuple[float, float] = (-45.0, 14.0)
    horizontal_fov_range: tuple[float, float] = (-180.0, 180.0)
    horizontal_res: float = 1.6
    origin: tuple[float, float, float] = (0.0, 0.0, 0.16)
    box_min: tuple[float, float, float] = (-0.55, -0.45, -0.65)
    box_max: tuple[float, float, float] = (1.15, 0.45, 0.55)
    min_range: float = 0.1
    max_range: float = 2.0
    z_slices: int = 9
