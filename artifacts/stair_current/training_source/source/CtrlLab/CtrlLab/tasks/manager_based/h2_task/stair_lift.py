"""A conservative pre-riser lift point derived only from current raw returns."""

import math

import torch


def pre_riser_lift(mapping, ids, start_world, end_world, foot_corners, max_clearance=.03, margin=.02):
    """Return ``(lift_world, raw_boundary_evidence)`` for forward ascending steps.

    ``foot_corners`` are sole-local ``[4,3]`` or per-step ``[N,4,3]`` points.
    Their XY envelope circle protects all yaw orientations. Only finite raw
    min/max cells with ``last_seen>=0`` can identify a barrier; model, tactile,
    safe-patch and configured stair geometry are deliberately not consulted.

    The forward swept corridor is the start/end segment expanded by the foot
    radius, safety margin and cell half-diagonal. Its nearest raw cell above
    start Z+40mm supplies a boundary at the cell's *back* X edge. Advance stops
    before that boundary minus the foot radius and margin, and before end X.
    Without such a raw witness the centre does not advance. If the original
    standing foot already exceeds this conservative safety plane, it is never
    moved backwards; no further advance is issued and this is explicitly
    reported. This is not a free-space certificate. The caller still requires
    actual whole-foot clear, completed lift, collision, IK and native limits.
    """
    if not math.isclose(float(mapping.resolution), .025, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Pre-riser lift requires the configured 25mm map.")
    if not math.isfinite(float(max_clearance)) or max_clearance <= .012:
        raise ValueError("Lift clearance must remain above the12mm actual-clear threshold.")
    if not math.isfinite(float(margin)) or margin < 0:
        raise ValueError("Pre-riser margin must be finite and nonnegative.")
    device = mapping.device
    ids = torch.as_tensor(ids, dtype=torch.long, device=device).reshape(-1)
    start = torch.as_tensor(start_world, dtype=torch.float32, device=device)
    end = torch.as_tensor(end_world, dtype=torch.float32, device=device)
    corners = torch.as_tensor(foot_corners, dtype=torch.float32, device=device)
    n = ids.numel()
    if corners.shape == (4, 3):
        corners = corners[None].expand(n, -1, -1)
    if start.shape != (n, 3) or end.shape != (n, 3) or corners.shape != (n, 4, 3):
        raise ValueError("Expected N IDs, Nx3 endpoints and4x3 or Nx4x3 sole-local corners.")
    if (not bool(torch.isfinite(start).all() & torch.isfinite(end).all() & torch.isfinite(corners).all())
            or bool(((ids < 0) | (ids >= mapping.n)).any())):
        raise ValueError("Pre-riser inputs must be finite and use valid environment IDs.")
    if bool((end[:, 0] < start[:, 0] - 1e-6).any()):
        raise ValueError("Pre-riser lift is defined for forward steps; use the existing profile otherwise.")
    radius = corners[..., :2].norm(dim=-1).amax(-1)
    if bool((radius <= 1e-6).any()):
        raise ValueError("The sole-local foot envelope must have a positive XY radius.")
    lift = start.clone()
    lift[:, 2] = torch.maximum(start[:, 2], end[:, 2]) + float(max_clearance)
    evidence = {
        "raw_boundary_world_x": start.new_full((n,), float("inf")),
        "raw_boundary_valid": torch.zeros(n, dtype=torch.bool, device=device),
        "raw_cell_index": torch.full((n,), -1, dtype=torch.long, device=device),
        "raw_minimum": start.new_full((n,), float("nan")),
        "raw_maximum": start.new_full((n,), float("nan")),
        "raw_last_seen": torch.full((n,), -1, dtype=mapping.last_seen.dtype, device=device),
        "radius_m": radius,
        "advance_m": start.new_zeros(n),
        "boundary_limited_at_start": torch.zeros(n, dtype=torch.bool, device=device),
        "corridor_radius_m": radius + float(margin) + float(mapping.resolution) * math.sqrt(.5),
    }
    # No map methods are invoked: these are the actual raw cell buffers now.
    for begin in range(0, n, 32):
        finish = min(begin + 32, n)
        selected = ids[begin:finish]
        origin = mapping.env.scene.env_origins[selected]
        a, b = start[begin:finish] - origin, end[begin:finish] - origin
        dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
        x, y = mapping.x[None], mapping.y[None]
        ox, oy = x - a[:, 0, None], y - a[:, 1, None]
        parameter = ((ox * dx[:, None] + oy * dy[:, None])
                     / (dx.square() + dy.square()).clamp_min(1e-12)[:, None]).clamp(0, 1)
        distance2 = (ox - parameter * dx[:, None]).square() + (oy - parameter * dy[:, None]).square()
        raw_minimum = mapping.minimum[selected]
        raw_maximum = mapping.maximum[selected]
        raw_seen = mapping.last_seen[selected]
        corridor = distance2 <= evidence["corridor_radius_m"][begin:finish, None].square()
        real = ((raw_seen >= 0) & torch.isfinite(raw_minimum) & torch.isfinite(raw_maximum)
                & (raw_minimum <= raw_maximum))
        candidates = (real & corridor & (x > a[:, 0, None])
                      & (raw_maximum > a[:, 2, None] + .04))
        first_x, cell = torch.where(candidates, x, float("inf")).min(-1)
        valid = torch.isfinite(first_x)
        boundary = origin[:, 0] + first_x - float(mapping.resolution) * .5
        safe_x = boundary - radius[begin:finish] - float(margin)
        advance = (torch.minimum(end[begin:finish, 0], safe_x) - start[begin:finish, 0]).clamp_min(0)
        advance = torch.where(valid, advance, 0)
        lift[begin:finish, 0] += advance
        evidence["raw_boundary_world_x"][begin:finish] = boundary
        evidence["raw_boundary_valid"][begin:finish] = valid
        evidence["raw_cell_index"][begin:finish] = torch.where(valid, cell, -1)
        owner = torch.arange(finish - begin, device=device)
        evidence["raw_minimum"][begin:finish] = torch.where(valid, raw_minimum[owner, cell], float("nan"))
        evidence["raw_maximum"][begin:finish] = torch.where(valid, raw_maximum[owner, cell], float("nan"))
        evidence["raw_last_seen"][begin:finish] = torch.where(valid, raw_seen[owner, cell], -1)
        evidence["advance_m"][begin:finish] = advance
        evidence["boundary_limited_at_start"][begin:finish] = valid & (safe_x < start[begin:finish, 0])
    return lift, evidence
