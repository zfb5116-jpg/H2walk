"""A genuinely flat first curriculum row, with the original stair bands above."""

from copy import deepcopy
from isaaclab.terrains.trimesh import mesh_terrains
from isaaclab.terrains.height_field import hf_terrains


FLAT_ROW_FRACTION = 1.0 / 7.0


def _later_difficulty(difficulty):
    return max(0.0, min(1.0, (difficulty - FLAT_ROW_FRACTION) / (1.0 - FLAT_ROW_FRACTION)))


def flat_start_pyramid_stairs(difficulty, cfg):
    if difficulty < FLAT_ROW_FRACTION:
        return mesh_terrains.flat_terrain(difficulty, cfg)
    return mesh_terrains.pyramid_stairs_terrain(_later_difficulty(difficulty), cfg)


def flat_start_inverted_stairs(difficulty, cfg):
    if difficulty < FLAT_ROW_FRACTION:
        return mesh_terrains.flat_terrain(difficulty, cfg)
    return mesh_terrains.inverted_pyramid_stairs_terrain(_later_difficulty(difficulty), cfg)


def flat_start_rough(difficulty, cfg):
    if difficulty < FLAT_ROW_FRACTION:
        return mesh_terrains.flat_terrain(difficulty, cfg)
    return hf_terrains.random_uniform_terrain(_later_difficulty(difficulty), cfg)


def with_flat_start(generator):
    cfg = deepcopy(generator)
    cfg.num_rows = 7
    # At 0.35 m/s a 20-second first episode must stay within the flat row.
    cfg.size = (16.0, 6.0)
    for name, terrain in cfg.sub_terrains.items():
        terrain.function = (flat_start_rough if name == "rough_steps" else
                            flat_start_inverted_stairs if "down" in name else
                            flat_start_pyramid_stairs)
    return cfg
