"""A height curriculum made entirely of six-step straight ascending stairways."""

import numpy as np
import trimesh

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .stair_terrain import StraightStairTerrainCfg


def curriculum_riser(difficulty, minimum, maximum, levels):
    """Quantize Isaac Lab's randomized row difficulty to a known row height."""
    if levels < 2 or not 0.0 < minimum <= maximum:
        raise ValueError("Stair height curriculum needs at least two positive-height rows")
    row = min(levels - 1, max(0, int(float(difficulty) * levels)))
    return minimum + (maximum - minimum) * row / (levels - 1)


def ascent_stairs(difficulty, cfg):
    length, width = cfg.size
    origin_x = cfg.spawn_origin_x
    first = origin_x+cfg.approach_distance
    height = curriculum_riser(difficulty, cfg.minimum_riser_height,
                              cfg.maximum_riser_height, cfg.height_levels)
    if first+(cfg.num_steps-1)*cfg.tread_depth >= length-cfg.top_end_margin:
        raise ValueError("Stair treads do not fit inside their authored tile")
    floor = trimesh.creation.box((length, width, 0.10))
    floor.apply_translation((length/2, width/2, -0.05))
    meshes = [floor]
    for level in range(1, cfg.num_steps+1):
        start = first+(level-1)*cfg.tread_depth
        end = start+cfg.tread_depth if level < cfg.num_steps else length-cfg.top_end_margin
        block = trimesh.creation.box((end-start, cfg.stair_width, level*height))
        block.apply_translation(((start+end)/2, width/2, level*height/2))
        meshes.append(block)
    return meshes, np.array((origin_x, width/2, 0.0))


@configclass
class AscentStairTerrainCfg(StraightStairTerrainCfg):
    function = ascent_stairs
    minimum_riser_height: float = 0.03
    maximum_riser_height: float = 0.15
    height_levels: int = 8
    spawn_origin_x: float = 2.0
    top_end_margin: float = 0.25


def ascent_stair_generator(*, levels=8, columns=4, minimum=0.03, maximum=0.15,
                           depth=0.35, steps=6, width=2.4, approach=1.4):
    return TerrainGeneratorCfg(
        size=(12.0, 6.0), num_rows=levels, num_cols=columns, border_width=3.0,
        curriculum=True, use_cache=False, difficulty_range=(0.0, 1.0),
        sub_terrains={"ascending_stairway": AscentStairTerrainCfg(
            proportion=1.0, minimum_riser_height=minimum, maximum_riser_height=maximum,
            height_levels=levels, tread_depth=depth, num_steps=steps,
            stair_width=width, approach_distance=approach)},
    )
