"""Torch-only experimental height fusion; no learned parameters or simulator imports."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _masked_median(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Lower median over frame dimension 1, with zero for an empty cell."""
    ordered = torch.where(valid, values, torch.full_like(values, float("inf"))).sort(dim=1).values
    count = valid.sum(dim=1, keepdim=True)
    pick = (count - 1).clamp_min(0) // 2
    median = ordered.gather(1, pick)
    return torch.where(count > 0, median, torch.zeros_like(median))


def history_frame_indices(
    next_pointer: torch.Tensor, fill_count: torch.Tensor, capacity: int, ages_steps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select strictly older ring frames after the current frame was pushed.

    Pointers/fill counts have shape ``(B,)``; requested integer ages are ``(T,)``
    in control steps and must be strictly between zero and ``capacity``.
    Return ``(B,T)`` ring indices and validity. Excluding age zero prevents
    counting the current observation twice. A partially reset environment with
    ``fill_count == 1`` contributes only its separately supplied current frame.
    """
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    if next_pointer.ndim != 1 or fill_count.shape != next_pointer.shape or fill_count.device != next_pointer.device:
        raise ValueError("next_pointer and fill_count must have equal (B,) shapes and devices")
    ages = torch.as_tensor(ages_steps, device=next_pointer.device)
    if next_pointer.is_floating_point() or fill_count.is_floating_point() or ages.is_floating_point() or ages.ndim != 1:
        raise ValueError("ring pointers, fill counts and ages must be integer tensors")
    if bool(((ages <= 0) | (ages >= capacity)).any()):
        raise ValueError("historical ages must be strictly between zero and capacity")
    if ages.unique().numel() != ages.numel():
        raise ValueError("historical ages must be unique")
    if bool(((next_pointer < 0) | (next_pointer >= capacity) | (fill_count < 0) | (fill_count > capacity)).any()):
        raise ValueError("ring pointers or fill counts are out of bounds")
    indices = (next_pointer[:, None] - 1 - ages[None, :]) % capacity
    return indices.long(), fill_count[:, None] > ages[None, :]


def fuse_height_frames(
    heights: torch.Tensor,
    hits: torch.Tensor,
    ages_steps: torch.Tensor,
    *,
    height_spans: torch.Tensor | None = None,
    dispersion_scale_m: float = 0.01,
    layer_scale_m: float = 0.015,
    age_decay_steps: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse already reprojected ``(B,T,H,W)`` heights in metres.

    ``hits > 0.5`` marks a measured cell. ``ages_steps`` is ``(T,)`` or
    ``(B,T)`` in control steps, with zero for the current frame. Each physical
    frame must occur once; this helper does not infer duplicate timestamps.
    Optional ``height_spans`` is the per-frame cell's max-minus-min return
    height in metres, measured *before* upper-band selection.

    Return median height, hit, heuristic confidence and valid-frame count,
    each ``(B,1,H,W)``. Confidence combines age-weighted support, temporal MAD
    plus mean absolute dispersion, and within-cell span. It is not a calibrated
    probability. A single fresh, internally consistent sample has confidence
    0.5; absent/nonfinite samples contribute neither support nor height.
    The scales are positive e-folding distances/times, not layer classifiers.
    """
    dispersion_scale_m = _positive(dispersion_scale_m, "dispersion_scale_m")
    layer_scale_m = _positive(layer_scale_m, "layer_scale_m")
    age_decay_steps = _positive(age_decay_steps, "age_decay_steps")
    if heights.ndim != 4 or heights.shape[1] < 1 or hits.shape != heights.shape:
        raise ValueError("heights and hits must have equal (B,T,H,W) shapes with T >= 1")
    if not heights.is_floating_point() or hits.device != heights.device:
        raise ValueError("heights must be floating point and hits must share their device")
    ages = torch.as_tensor(ages_steps, device=heights.device, dtype=heights.dtype)
    if ages.ndim == 1 and ages.shape[0] == heights.shape[1]:
        ages = ages.unsqueeze(0).expand(heights.shape[0], -1)
    if ages.shape != heights.shape[:2] or not bool(torch.isfinite(ages).all()) or bool((ages < 0).any()):
        raise ValueError("ages_steps must be finite, nonnegative and shaped (T,) or (B,T)")
    valid = (hits > 0.5) & torch.isfinite(hits) & torch.isfinite(heights)
    if height_spans is None:
        spans = torch.zeros_like(heights)
    else:
        if height_spans.shape != heights.shape or height_spans.device != heights.device:
            raise ValueError("height_spans must have the same shape and device as heights")
        valid = valid & torch.isfinite(height_spans) & (height_spans >= 0)
        spans = torch.where(valid, height_spans, torch.zeros_like(height_spans))
    safe = torch.where(valid, heights, torch.zeros_like(heights))
    median = _masked_median(safe, valid)
    deviations = (safe - median).abs()
    count = valid.sum(dim=1, keepdim=True)
    denominator = count.clamp_min(1).to(heights.dtype)
    mad = _masked_median(deviations, valid)
    # MAD alone is zero for two contradictory samples or a repeated majority.
    mean_deviation = torch.where(valid, deviations, torch.zeros_like(deviations)).sum(1, keepdim=True) / denominator
    mean_span = spans.sum(dim=1, keepdim=True) / denominator
    age_weights = torch.exp(-ages / age_decay_steps).unsqueeze(-1).unsqueeze(-1)
    support = (valid.to(heights.dtype) * age_weights).sum(dim=1, keepdim=True)
    confidence = support / (support + 1.0)
    confidence = confidence * torch.exp(-(mad + mean_deviation) / dispersion_scale_m)
    confidence = confidence * torch.exp(-mean_span / layer_scale_m)
    seen = count > 0
    confidence = torch.where(seen, confidence, torch.zeros_like(confidence)).clamp(0.0, 1.0)
    return median, seen.to(heights.dtype), confidence, count


def edge_aware_median(
    height: torch.Tensor,
    *,
    threshold_m: float = 0.01,
    kernel_size: int = 3,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Median of neighbours within ``threshold_m`` metres of each cell's height.

    Input/output shape is ``(B,1,H,W)`` in metres. The positive threshold is an
    inclusive same-surface gate: use a value below 0.02 to preserve a clean
    2 cm step and a one-cell-wide 2 cm feature. Without confidence this also
    preserves isolated returns above the gate. Optional heuristic confidence
    has the same shape and lies in [0,1]. A centre below 0.25 may be replaced
    only when at least 5/8 of its neighbours have confidence >= 0.5 and agree
    within the height gate. A reliable narrow feature remains unchanged.
    Low-confidence real features can be removed by this rule; replay must
    validate the tradeoff. It cannot restore already lost layers. The caller
    must reapply its total residual bound after filtering.
    Nonfinite neighbours are excluded; a nonfinite centre becomes zero.
    """
    threshold_m = _positive(threshold_m, "threshold_m")
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    if height.ndim != 4 or height.shape[1] != 1 or min(height.shape[-2:]) < 1 or not height.is_floating_point():
        raise ValueError("height must be floating point with shape (B,1,H,W)")
    pad = kernel_size // 2
    patches = F.pad(height, (pad, pad, pad, pad), mode="replicate")
    patches = patches.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    patches = patches.reshape(*height.shape, kernel_size * kernel_size)
    centre = height.unsqueeze(-1)
    keep = torch.isfinite(patches) & torch.isfinite(centre) & ((patches - centre).abs() <= threshold_m)
    ordered = torch.where(keep, patches, torch.full_like(patches, float("inf"))).sort(dim=-1).values
    count = keep.sum(dim=-1, keepdim=True)
    filtered = ordered.gather(-1, (count - 1).clamp_min(0) // 2).squeeze(-1)
    filtered = torch.where(count.squeeze(-1) > 0, filtered, torch.zeros_like(filtered))
    if confidence is None:
        return filtered
    if confidence.shape != height.shape or confidence.device != height.device:
        raise ValueError("confidence must have the same shape and device as height")
    if not bool(torch.isfinite(confidence).all()) or bool(((confidence < 0) | (confidence > 1)).any()):
        raise ValueError("confidence must be finite and lie in [0,1]")
    neighbour_count = kernel_size * kernel_size - 1
    if neighbour_count == 0:
        return filtered
    score_patches = F.pad(confidence, (pad, pad, pad, pad), mode="replicate")
    score_patches = score_patches.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    score_patches = score_patches.reshape(*height.shape, kernel_size * kernel_size)
    trusted = torch.isfinite(patches) & (score_patches >= 0.5)
    trusted[..., kernel_size * kernel_size // 2] = False
    trusted_count = trusted.sum(dim=-1, keepdim=True)
    ordered_neighbours = torch.where(trusted, patches, torch.full_like(patches, float("inf"))).sort(dim=-1).values
    neighbour_median = ordered_neighbours.gather(-1, (trusted_count - 1).clamp_min(0) // 2)
    # Use only a stable majority, not a single trusted return near an edge.
    agreeing = trusted & ((patches - neighbour_median).abs() <= threshold_m)
    support = agreeing.sum(dim=-1)
    replace = (confidence < 0.25) & (support >= math.ceil(0.625 * neighbour_count))
    return torch.where(replace, neighbour_median.squeeze(-1), filtered)


def project_confidence_bound(
    base: torch.Tensor, candidate: torch.Tensor, confidence: torch.Tensor, *, residual_limit_m: float = 0.08
) -> torch.Tensor:
    """Project a map onto ``base +/- (1-confidence)*residual_limit_m`` metres.

    All inputs have the same shape; base and candidate must be finite and
    confidence must be finite in ``[0,1]``. Apply after every learned/filtering
    stage to guarantee a total correction bound, rather than summing several
    individually bounded residuals. This adds no learned state.
    """
    residual_limit_m = _positive(residual_limit_m, "residual_limit_m")
    if base.shape != candidate.shape or base.shape != confidence.shape:
        raise ValueError("base, candidate and confidence must have equal shapes")
    if not bool(torch.isfinite(base).all() & torch.isfinite(candidate).all() & torch.isfinite(confidence).all()):
        raise ValueError("base, candidate and confidence must be finite")
    if bool(((confidence < 0) | (confidence > 1)).any()):
        raise ValueError("confidence must lie in [0,1]")
    bound = (1.0 - confidence) * residual_limit_m
    return base + torch.maximum(torch.minimum(candidate - base, bound), -bound)
