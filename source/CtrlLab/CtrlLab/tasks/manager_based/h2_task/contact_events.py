"""Contact-event reductions on the existing force history, without Isaac imports."""

from __future__ import annotations

import math

import torch


def stumble_from_force_history(
    force_history: torch.Tensor,
    force_ratio: float = 5.0,
    temporal_mode: str = "per_sample",
) -> torch.Tensor:
    """Return whether any selected foot had a lateral-dominant contact.

    Input is the sensor's world-frame force history ``[env, time, foot, xyz]``.
    Compare horizontal and vertical force from the *same* physics sample before
    reducing over time and feet. Earlier support must not mask a later riser hit.
    This is a force-ratio heuristic, not a foothold-support-area measurement.

    ``independent_max`` preserves the historical reduction for ablations. The
    caller selects bodies before invoking this stateless helper. Empty body
    selections return False; a sensor history needs at least one time sample.
    """
    if not math.isfinite(force_ratio) or force_ratio <= 0.0:
        raise ValueError("force_ratio must be finite and positive")
    if temporal_mode not in {"per_sample", "independent_max"}:
        raise ValueError("temporal_mode must be 'per_sample' or 'independent_max'")
    if force_history.ndim != 4 or force_history.shape[-1] != 3 or force_history.shape[1] < 1:
        raise ValueError("force_history must have shape [env, time >= 1, foot, 3]")

    horizontal = torch.linalg.vector_norm(force_history[..., :2], dim=-1)
    vertical = force_history[..., 2].abs()
    if temporal_mode == "independent_max":
        return (horizontal.amax(dim=1) > force_ratio * vertical.amax(dim=1)).any(dim=-1)
    return (horizontal > force_ratio * vertical).any(dim=1).any(dim=-1)
