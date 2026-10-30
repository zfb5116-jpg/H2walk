"""Collision-foot bottom bounds in the measured sole-center frame."""

import json
from pathlib import Path

import torch


def sole_footprint_corners(device, dtype=torch.float32):
    data = json.loads((Path(__file__).parent / "assets/h2_leg_chain.json").read_text())
    feet = []
    for side in ("left", "right"):
        leg = data["legs"][side]
        low, high = leg["collision_mesh_local_bounds_m"]
        center = leg["sole_center_local_offset_m"]
        feet.append([[x - center[0], y - center[1], low[2] - center[2]]
                     for x in (low[0], high[0]) for y in (low[1], high[1])])
    return torch.tensor(feet, device=device, dtype=dtype)
