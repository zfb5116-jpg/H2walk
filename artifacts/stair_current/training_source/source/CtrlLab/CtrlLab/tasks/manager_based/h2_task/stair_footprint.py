"""Conservative full-area support checks on existing measured map evidence."""

import math

import torch


def footprint_supported(mapping, ids, world_corners_N4x3, target_surface_world_z_N, margin=.008):
    """Return whether every intersected 25mm cell certifies the target plane.

    Corners use ``foot_geometry``'s original order; ``[0, 1, 3, 2]`` forms the
    perimeter. The XY projection is expanded by an outward ``margin`` square,
    giving at least that clearance in world X and Y. SAT includes *all* cells
    intersecting this area, rather than just cells under its four vertices.

    Raw LiDAR, existing bounded models and tactile evidence are resolved only
    through ``mapping._fused_cells``. No models are generated, no unknown cell
    is filled, and no cache/update/query method is called. The complete area
    permits at most one model cell; measured height intervals must jointly
    span less than 22mm and lie inside target Z +/-30mm. Out-of-map evidence
    and degenerate tilted projections fail closed. This geometric predicate
    does not replace the caller's actual contact, load, flight or tilt gates.
    """
    if not math.isfinite(float(margin)) or margin < 0:
        raise ValueError("Footprint margin must be finite and nonnegative.")
    if not math.isclose(float(mapping.resolution), .025, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Footprint support requires the configured 25mm map.")
    device = mapping.device
    ids = torch.as_tensor(ids, dtype=torch.long, device=device).reshape(-1)
    corners = torch.as_tensor(world_corners_N4x3, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_surface_world_z_N, dtype=torch.float32, device=device).reshape(-1)
    n = ids.numel()
    if corners.shape != (n, 4, 3) or target.shape != (n,):
        raise ValueError("Expected N environment IDs, Nx4x3 corners and N plane heights.")
    result = torch.zeros(n, dtype=torch.bool, device=device)
    if not n:
        return result
    resolution = float(mapping.resolution)
    map_low = corners.new_tensor([mapping.x0, mapping.y0])
    map_high = map_low + corners.new_tensor([mapping.nx, mapping.ny]) * resolution

    # Bound allocations even for a malformed very large polygon. Each live
    # subset spans at most the actual map and usually fewer than 150 cells.
    for begin in range(0, n, 64):
        end = min(begin + 64, n)
        positions = torch.arange(begin, end, device=device)
        env_ids = ids[begin:end]
        finite = torch.isfinite(corners[begin:end]).all(dim=(1, 2)) & torch.isfinite(target[begin:end])
        finite &= (env_ids >= 0) & (env_ids < mapping.n)
        positions = positions[finite]
        if not positions.numel():
            continue
        env_ids = ids[positions]
        origin = mapping.env.scene.env_origins[env_ids]
        polygon = corners[positions][:, [0, 1, 3, 2]] - origin[:, None]
        xy = polygon[..., :2]
        edge = torch.roll(xy, -1, dims=1) - xy
        next_edge = torch.roll(edge, -1, dims=1)
        turns = edge[..., 0] * next_edge[..., 1] - edge[..., 1] * next_edge[..., 0]
        normal3 = torch.cross(polygon[:, 1] - polygon[:, 0], polygon[:, 3] - polygon[:, 0], dim=-1)
        area3 = normal3.norm(dim=-1)
        area_xy = normal3[:, 2].abs()
        plane_error = ((polygon[:, 2] - polygon[:, 0]) * normal3).sum(-1).abs() / area3.clamp_min(1e-12)
        geometric = ((edge.norm(dim=-1) > 1e-6).all(-1)
                     & ((turns > 1e-10).all(-1) | (turns < -1e-10).all(-1))
                     & (area_xy > 1e-6) & (area_xy > area3 * 1e-3) & (plane_error <= 1e-5))
        lower, upper = xy.amin(1) - margin, xy.amax(1) + margin
        geometric &= (lower >= map_low).all(-1) & (upper <= map_high).all(-1)
        positions = positions[geometric]
        if not positions.numel():
            continue
        env_ids = ids[positions]
        xy, edge = xy[geometric], edge[geometric]
        lower, upper = lower[geometric], upper[geometric]
        low_cell = torch.floor((lower - map_low) / resolution).long()
        high_cell = torch.ceil((upper - map_low) / resolution).long() - 1
        span = high_cell - low_cell + 1
        width, height = int(span[:, 0].amax()), int(span[:, 1].amax())
        dy, dx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        ix = low_cell[:, 0, None] + dx.flatten()[None]
        iy = low_cell[:, 1, None] + dy.flatten()[None]
        active_box = (ix <= high_cell[:, 0, None]) & (iy <= high_cell[:, 1, None])
        inside = (ix >= 0) & (ix < mapping.nx) & (iy >= 0) & (iy < mapping.ny)
        centers = torch.stack((mapping.x0 + (ix + .5) * resolution,
                               mapping.y0 + (iy + .5) * resolution), -1)
        normals = torch.stack((-edge[..., 1], edge[..., 0]), -1)
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        axes = torch.cat((torch.eye(2, device=device, dtype=xy.dtype).expand(len(positions), -1, -1), normals), 1)
        vertex_projection = torch.einsum("npi,nai->npa", xy, axes)
        safety = margin * axes.abs().sum(-1)
        projection_low = vertex_projection.amin(1) - safety
        projection_high = vertex_projection.amax(1) + safety
        projection = torch.einsum("nci,nai->nca", centers, axes)
        radius = .5 * resolution * axes.abs().sum(-1)
        # Zero-area tangencies need not certify an adjacent cell. A 0.1um
        # tolerance absorbs float32 grid arithmetic without bridging a hole.
        intersects = ((projection + radius[:, None] > projection_low[:, None] + 1e-7)
                      & (projection - radius[:, None] < projection_high[:, None] - 1e-7)).all(-1)
        active = active_box & intersects
        index = iy.clamp(0, mapping.ny - 1) * mapping.nx + ix.clamp(0, mapping.nx - 1)
        known, minimum, maximum, model = mapping._fused_cells(env_ids, index)
        target_local = target[positions] - mapping.env.scene.env_origins[env_ids, 2]
        valid = (inside & known & torch.isfinite(minimum) & torch.isfinite(maximum)
                 & (minimum >= target_local[:, None] - .03)
                 & (maximum <= target_local[:, None] + .03))
        all_known = ((~active) | valid).all(-1) & active.any(-1)
        low = torch.where(active, minimum, float("inf")).amin(-1)
        high = torch.where(active, maximum, -float("inf")).amax(-1)
        result[positions] = (all_known & ((high - low) < .022)
                             & ((model & active).sum(-1) <= 1))
    return result
