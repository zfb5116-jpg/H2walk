from copy import deepcopy

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=12,  # curriculum level
    num_cols=13,  # terrain type number
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            # proportion=0.2,
            step_height_range=(0.02, 0.12),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            # proportion=0.2,
            step_height_range=(0.02, 0.12),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_low": terrain_gen.MeshPyramidStairsTerrainCfg(
            step_height_range=(0.04, 0.08),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_high": terrain_gen.MeshPyramidStairsTerrainCfg(
            step_height_range=(0.16, 0.24),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv_high": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            step_height_range=(0.16, 0.24),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            obstacle_height_mode="choice",
            obstacle_width_range=(0.20, 0.50),
            obstacle_height_range=(0.02, 0.12),
            num_obstacles=30,
            platform_width=2.0,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            amplitude_range=(0.02, 0.08),
            num_waves=4,
            border_width=0.1,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            stone_height_max=0.08,
            stone_width_range=(0.30, 0.60),
            stone_distance_range=(0.05, 0.20),
            holes_depth=-0.20,
            platform_width=2.0,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            # proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.025, 0.1),
            platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            # proportion=0.2,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            # proportion=0.2,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            # proportion=0.2,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25
        ),
        "flat": terrain_gen.HfPyramidSlopedTerrainCfg(
            # proportion=0.2,
            slope_range=(0.0, 0.0),
            platform_width=2.0,
            border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""

STAIR_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(6.0, 6.0),
    border_width=20.0,
    num_rows=6,  # more gradual H2 stair curriculum
    num_cols=6,  # terrain type number
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stairs_low": terrain_gen.MeshPyramidStairsTerrainCfg(
            # Begin around the reconstructed package's shallow L0 stair scale.
            step_height_range=(0.02, 0.05),
            step_width=0.35,
            platform_width=3.0,
            border_width=0.3,
            holes=False,
        ),
        "stairs_mid": terrain_gen.MeshPyramidStairsTerrainCfg(
            step_height_range=(0.04, 0.08),
            step_width=0.35,
            platform_width=3.0,
            border_width=0.3,
            holes=False,
        ),
        "stairs_high": terrain_gen.MeshPyramidStairsTerrainCfg(
            step_height_range=(0.08, 0.14),
            step_width=0.35,
            platform_width=3.0,
            border_width=0.3,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            step_height_range=(0.02, 0.08),
            step_width=0.35,
            platform_width=3.0,
            border_width=0.3,
            holes=False,
        ),
        "stairs_down_high": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            step_height_range=(0.08, 0.14),
            step_width=0.35,
            platform_width=3.0,
            border_width=0.3,
            holes=False,
        ),
        "rough_steps": terrain_gen.HfRandomUniformTerrainCfg(
            noise_range=(0.01, 0.04),
            noise_step=0.01,
            border_width=0.1,
        ),
    }
)

# Keep ordinary terrain represented at every curriculum row while retaining the
# shallow H2 stair ranges. In Isaac Lab v2.2.1 curriculum generation, ten columns
# with these weights allocate 3 flat, 1 slope, 1 rough and 5 stair columns.
# This is an initial training mixture, not a measured optimal distribution.
MIXED_WALK_TERRAINS_CFG = deepcopy(STAIR_ROUGH_TERRAINS_CFG)
MIXED_WALK_TERRAINS_CFG.num_cols = 10
for _sub_terrain in MIXED_WALK_TERRAINS_CFG.sub_terrains.values():
    _sub_terrain.proportion = 1.0
MIXED_WALK_TERRAINS_CFG.sub_terrains = {
    "flat": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=3.0,
        slope_range=(0.0, 0.0),
        platform_width=3.0,
        border_width=0.3,
    ),
    "gentle_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=1.0,
        slope_range=(0.0, 0.10),
        platform_width=3.0,
        border_width=0.3,
    ),
    **MIXED_WALK_TERRAINS_CFG.sub_terrains,
}


SLOPE_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,  # curriculum level
    num_cols=7,  # terrain type number
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            # proportion=0.2,
            slope_range=(0.4, 0.4),
            platform_width=2.0,
            border_width=0.1
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            # proportion=0.2,
            slope_range=(0.4, 0.4),
            platform_width=2.0,
            border_width=0.1
        ),
    }
)

BOXES_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(80.0, 80.0),
    border_width=1.0,
    num_rows=1,  # curriculum level
    num_cols=1,  # terrain type number
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            # proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.06, 0.06),
            platform_width=1.0
        ),
    }
)

RANDOM_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(80.0, 80.0),
    border_width=1.0,
    num_rows=1,  # curriculum level
    num_cols=1,  # terrain type number
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            # proportion=0.2,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.1
        ),
    }
)

FLAT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,  # curriculum level
    num_cols=7,  # terrain type number
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.HfPyramidSlopedTerrainCfg(
            # proportion=0.2,
            slope_range=(0.0, 0.0),
            platform_width=2.0,
            border_width=0.1
        ),
    }
)
