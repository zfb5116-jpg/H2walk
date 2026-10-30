"""Task-domain validity, independent of a preferred pelvis pose or gait phase."""

import torch


def tile_domain_bounds(size, spawn_origin_x, top_end_margin, inset=0.10):
    """Bounds relative to the authored spawn origin of this robot's own tile.

    The forward edge is inside the actual top platform, rather than its lower
    floor strip or a different robot's neighboring tile. No global mesh bounds
    or particular curriculum row are assumed.
    """
    length, width = map(float, size)
    origin, end, inset = float(spawn_origin_x), float(top_end_margin), float(inset)
    if not (inset > 0 and 0 < origin < length-end and width > 2*inset
            and length-end > 2*inset):
        raise ValueError("Invalid authored stair-tile domain")
    return (-origin+inset, length-origin-end-inset, -width/2+inset, width/2-inset)


def physical_validity(root_position, origins, bounds, below_floor_tolerance=0.05):
    """Return own-tile exit, impossible below-floor position and joint validity.

    Origin Z is the lowest actual floor. Its small negative tolerance is a
    penetration/escape guard, not a minimum standing height above a tread.
    """
    if root_position.shape != origins.shape or root_position.shape[-1] != 3:
        raise ValueError("Require matched root/origin N x 3 physical states")
    if not torch.isfinite(root_position).all() or not torch.isfinite(origins).all():
        raise RuntimeError("Non-finite H2 stair root/domain state")
    if below_floor_tolerance <= 0:
        raise ValueError("Below-floor tolerance must be positive")
    bounds = torch.as_tensor(bounds, device=root_position.device, dtype=root_position.dtype)
    local = root_position-origins
    outside = ((local[:, 0] < bounds[0]) | (local[:, 0] > bounds[1])
               | (local[:, 1] < bounds[2]) | (local[:, 1] > bounds[3]))
    below = local[:, 2] < -float(below_floor_tolerance)
    return outside, below, ~(outside | below)


def _environment_validity(env):
    return physical_validity(env.scene["robot"].data.root_pos_w, env.scene.terrain.env_origins,
                             env._stair_domain_bounds_local, env.cfg.stair_below_floor_tolerance_m)


def stair_out_of_tile(env):
    return _environment_validity(env)[0]


def stair_below_floor(env):
    return _environment_validity(env)[1]

