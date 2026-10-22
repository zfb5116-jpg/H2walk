"""Tensor-only boundaries for the H2 LiDAR observation adapter.

No simulator or checkpoint imports. Heights and ranges are in metres;
quaternion transforms remain the caller's responsibility.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

import torch


def mesh_hit_ranges(
    hits_w: torch.Tensor,
    starts_w: torch.Tensor,
    *,
    min_range: float = 0.1,
    max_range: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite measured ranges and a mask, never inventing a floor for misses."""
    if not math.isfinite(min_range) or not math.isfinite(max_range) or not 0 <= min_range < max_range:
        raise ValueError("range limits must be finite and 0 <= min_range < max_range")
    if hits_w.ndim != 3 or hits_w.shape[-1] != 3 or starts_w.shape != hits_w.shape:
        raise ValueError("hits_w and starts_w must have matching (B,N,3) shapes")
    if hits_w.device != starts_w.device or not hits_w.is_floating_point() or not starts_w.is_floating_point():
        raise ValueError("hit/start coordinates must be floating point on the same device")
    finite = torch.isfinite(hits_w).all(-1) & torch.isfinite(starts_w).all(-1)
    delta = torch.where(finite[..., None], hits_w - starts_w, torch.zeros_like(hits_w))
    ranges = delta.norm(dim=-1)
    valid = finite & (ranges >= min_range) & (ranges <= max_range)
    return torch.where(valid, ranges, torch.zeros_like(ranges)), valid.to(ranges.dtype)


def expand_ray_geometry(
    starts: torch.Tensor, directions: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize ray geometry to (B,N,3), preserving each sensor's local origin."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    expanded = []
    for name, values in (("starts", starts), ("directions", directions)):
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.ndim != 3 or values.shape[-1] != 3 or values.shape[0] not in (1, batch_size):
            raise ValueError(f"ray {name} must have shape (N,3), (1,N,3) or (B,N,3)")
        if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
            raise ValueError(f"ray {name} must be finite floating point")
        expanded.append(values.expand(batch_size, -1, -1))
    if expanded[0].shape != expanded[1].shape or expanded[0].device != expanded[1].device:
        raise ValueError("ray starts and directions must have equal shapes/devices")
    if expanded[0].shape[1] == 0:
        raise ValueError("the sensor must contain at least one ray")
    return expanded[0], expanded[1]


def validate_and_scatter_scan(
    ranges: torch.Tensor,
    mask: torch.Tensor,
    ray_starts_local: torch.Tensor,
    ray_directions_local: torch.Tensor,
    expected_directions: torch.Tensor,
    keep: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate checkpoint ray order/FOV, then scatter clipped rays to the full image.

    ``expected_directions`` is (full_N,3), taken from the loaded model's actual
    ray buffer. The reconstructor assumes all rays share one local origin.
    Return full range/mask images and the (B,3) local origin; incompatible
    grids fail explicitly rather than silently reinterpreting a checkpoint.
    """
    if ranges.ndim != 2 or mask.shape != ranges.shape:
        raise ValueError("ranges and mask must have equal (B,N) shapes")
    starts, directions = expand_ray_geometry(ray_starts_local, ray_directions_local, ranges.shape[0])
    if starts.shape[:2] != ranges.shape or starts.device != ranges.device:
        raise ValueError("ray geometry must match the measured (B,N) scan/device")
    if expected_directions.ndim != 2 or expected_directions.shape[-1] != 3:
        raise ValueError("model directions must have shape (full_N,3)")
    expected = expected_directions.to(device=ranges.device, dtype=directions.dtype)
    if keep is None:
        keep = torch.ones(expected.shape[0], dtype=torch.bool, device=ranges.device)
    if keep.ndim != 1 or keep.numel() != expected.shape[0] or keep.dtype != torch.bool:
        raise ValueError("keep mask must be boolean and match the checkpoint's full ray count")
    keep = keep.to(ranges.device)
    if int(keep.sum()) != ranges.shape[1]:
        raise ValueError("cast ray count differs from the pattern keep mask")
    if not torch.allclose(directions, expected[keep].unsqueeze(0).expand_as(directions), rtol=1e-5, atol=1e-5):
        raise ValueError("sensor ray order/FOV differs from the checkpoint ray directions")
    origin = starts[:, 0, :]
    if not torch.allclose(starts, origin[:, None, :].expand_as(starts), rtol=0.0, atol=1e-6):
        raise ValueError("height splatting requires a shared ray origin per sensor")
    full_range = ranges.new_zeros(ranges.shape[0], expected.shape[0])
    full_mask = mask.new_zeros(mask.shape[0], expected.shape[0])
    full_range[:, keep], full_mask[:, keep] = ranges, mask
    return full_range, full_mask, origin


def body_height_to_observation(
    body_height: torch.Tensor,
    root_z: torch.Tensor,
    *,
    reference_z: torch.Tensor | None = None,
    convention: str = "height_scan",
    offset: float = 0.5,
) -> torch.Tensor:
    """Convert terrain_z-root_z into Isaac Lab sensor_z-terrain_z-offset.

    Input is already row-major (ny,nx) / flattened in the same order as the
    default GridPattern ordering='xy'; no transpose, clip or scaling occurs.
    ``body_height`` explicitly retains the previous raw output convention.
    The reference sensor contributes its pose only, never its GT ray hits.
    """
    if body_height.ndim != 2 or root_z.shape != body_height.shape[:1]:
        raise ValueError("body_height must be (B,N) and root_z must be (B,)")
    if convention not in ("height_scan", "body_height"):
        raise ValueError("convention must be 'height_scan' or 'body_height'")
    if not math.isfinite(offset):
        raise ValueError("height scan offset must be finite")
    if reference_z is None:
        reference_z = root_z
    if reference_z.shape != root_z.shape or root_z.device != body_height.device or reference_z.device != root_z.device:
        raise ValueError("reference/root heights must match batch shape and device")
    if not bool(torch.isfinite(body_height).all() & torch.isfinite(root_z).all() & torch.isfinite(reference_z).all()):
        raise ValueError("reconstructed map and reference heights must be finite")
    if convention == "body_height":
        return body_height
    return (reference_z - root_z).unsqueeze(-1) - body_height - offset


def velocity_prior(
    like: torch.Tensor, source: str = "zero", estimate_b: torch.Tensor | None = None
) -> torch.Tensor:
    """Provide an explicit zero prior or a caller-supplied body-velocity estimate.

    A zero prior is an untrained cross-embodiment fallback, not a velocity
    estimator. This helper never reads simulator root linear velocity.
    """
    if source == "zero":
        return like.new_zeros((like.shape[0], 3))
    if source != "estimate":
        raise ValueError("velocity_source must be 'zero' or 'estimate'")
    if estimate_b is None:
        raise ValueError("velocity_source='estimate' requires an explicit body-velocity buffer")
    if estimate_b.shape != (like.shape[0], 3) or estimate_b.device != like.device:
        raise ValueError("estimated body velocity must have matching (B,3) shape/device")
    if not estimate_b.is_floating_point() or not bool(torch.isfinite(estimate_b).all()):
        raise ValueError("estimated body velocity must be finite floating point")
    return estimate_b.to(dtype=like.dtype)


def invalidate_lidar_observation(raw: Any, env_ids: Any = None) -> None:
    """Invalidate caches and reset only selected recurrent environment histories.

    Called by RobotEnv reset; safe before a reconstructor was first created.
    Other environments' hidden states remain untouched on a partial reset.
    """
    if env_ids is not None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=getattr(raw, "device", "cpu"))
        if env_ids.numel() == 0:
            return
    model = getattr(raw, "_h2_lidar_reconstructor", None)
    if model is not None:
        model.reset_temporal_state(env_ids)
    hidden = getattr(raw, "_h2_lidar_recon_hidden", None)
    if hidden is not None:
        if env_ids is None:
            raw._h2_lidar_recon_hidden = None
        else:
            hidden = hidden.clone()
            hidden[:, env_ids] = 0.0
            raw._h2_lidar_recon_hidden = hidden
    cache = getattr(raw, "_h2_lidar_recon_cache", None)
    if cache is not None:
        if env_ids is None or len(cache) != 5:
            delattr(raw, "_h2_lidar_recon_cache")
        else:
            cache[4][env_ids] = False
    if hasattr(raw, "_h2_lidar_ray_cache"):
        delattr(raw, "_h2_lidar_ray_cache")


def cache_state_matches(cache: Any, step: int, settings: tuple, tensors: tuple[torch.Tensor, ...]) -> bool:
    """Exact state matching prevents same-step reset/teleport from reusing an old map."""
    return (
        cache is not None and cache[0] == step and cache[1] == settings
        and len(cache) == 5 and bool(cache[4].all())
        and len(cache[2]) == len(tensors)
        and all(before.shape == now.shape and bool(exact_equal_rows(before, now).all()) for before, now in zip(cache[2], tensors))
    )


def exact_equal_rows(before: torch.Tensor, now: torch.Tensor) -> torch.Tensor:
    """Exact per-environment equality, treating unchanged NaN misses as equal."""
    if before.shape != now.shape or before.ndim < 1 or before.device != now.device:
        raise ValueError("cached/current state must have equal batched shapes and devices")
    same = (before == now) | (torch.isnan(before) & torch.isnan(now))
    return same.reshape(before.shape[0], -1).all(dim=1)


@contextmanager
def reconstruction_rows(model: Any, env_ids: torch.Tensor, batch_size: int):
    """Temporarily bind recurrent state to selected rows, then scatter it back.

    This is an adapter for LidarTerrainReconstructor's documented transient
    tensor attributes, never its parameters/buffers. Selecting only input
    tensors without selecting these rings would silently reinitialize them on
    a partial reset. Unselected rings and diagnostics remain unchanged.
    """
    names = (
        "_tok_buf", "_tok_ptr", "_tok_fill", "_pose_rng", "_pose_mask",
        "_pose_pos", "_pose_quat", "_pose_ptr", "_pose_fill", "_last_hit",
        "_last_confidence", "_last_frame_count", "_last_height_span",
    )
    if env_ids.ndim != 1 or env_ids.dtype != torch.long or env_ids.numel() < 1:
        raise ValueError("env_ids must be a nonempty integer vector")
    if bool(((env_ids < 0) | (env_ids >= batch_size)).any()) or env_ids.unique().numel() != env_ids.numel():
        raise ValueError("env_ids must be unique and within the original batch")
    if env_ids.numel() == batch_size and torch.equal(env_ids, torch.arange(batch_size, device=env_ids.device)):
        yield
        return
    original = {name: getattr(model, name, None) for name in names}
    for name, value in original.items():
        if value is not None:
            if value.shape[0] != batch_size:
                raise ValueError(f"transient state {name} does not match the environment batch")
    try:
        for name, value in original.items():
            setattr(model, name, value.index_select(0, env_ids) if value is not None else None)
        yield
    except BaseException:
        for name, value in original.items():
            setattr(model, name, value)
        raise
    else:
        updated = {name: getattr(model, name, None) for name in names}
        try:
            for name, value in updated.items():
                if value is None:
                    continue
                if value.shape[0] != env_ids.numel():
                    raise ValueError(f"updated transient state {name} has an unexpected batch")
                target = original[name]
                if target is None:
                    target = value.new_zeros((batch_size, *value.shape[1:]))
                    if name == "_pose_quat":
                        target[..., 0] = 1.0
                if target.shape[1:] != value.shape[1:]:
                    raise ValueError(f"transient state {name} changed shape during subset inference")
                target.index_copy_(0, env_ids, value)
                original[name] = target
        finally:
            for name, value in original.items():
                setattr(model, name, value)
