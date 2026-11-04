"""A straight ascending stairway with a low approach and a top landing."""

import numpy as np
import trimesh

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


def straight_stairs(difficulty, cfg):
    length, width = cfg.size
    origin_x = 2.0
    first = origin_x + cfg.approach_distance
    meshes = []
    floor = trimesh.creation.box((length, width, 0.10))
    floor.apply_translation((length / 2, width / 2, -0.05))
    meshes.append(floor)
    for level in range(1, cfg.num_steps + 1):
        start = first + (level - 1) * cfg.tread_depth
        end = start + cfg.tread_depth if level < cfg.num_steps else length - 0.25
        height = level * cfg.riser_height
        block = trimesh.creation.box((end - start, cfg.stair_width, height))
        block.apply_translation(((start + end) / 2, width / 2, height / 2))
        meshes.append(block)
    return meshes, np.array((origin_x, width / 2, 0.0))


@configclass
class StraightStairTerrainCfg(SubTerrainBaseCfg):
    function = straight_stairs
    riser_height: float = 0.15
    tread_depth: float = 0.35
    num_steps: int = 6
    stair_width: float = 2.4
    approach_distance: float = 1.4


def straight_stair_generator(*, height=0.15, depth=0.35, steps=6, approach=1.4):
    return TerrainGeneratorCfg(
        size=(12.0, 6.0), num_rows=1, num_cols=1, border_width=3.0,
        curriculum=False, use_cache=False,
        sub_terrains={"ascending_stairway": StraightStairTerrainCfg(
            proportion=1.0, riser_height=height, tread_depth=depth,
            num_steps=steps, approach_distance=approach)},
    )
