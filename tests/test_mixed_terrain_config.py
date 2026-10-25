"""CPU checks of the real terrain configuration and official column allocation.

The source configuration is executed with plain config containers. The tests do
not generate terrain meshes or import Isaac Lab, and do not claim gait results.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest


TERRAIN_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/config/terrain_cfg.py"
)


class _Config(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(proportion=1.0, **{key: value for key, value in kwargs.items() if key != "proportion"})
        self.proportion = kwargs.get("proportion", 1.0)


class _TerrainFactories:
    def __getattr__(self, name):
        return _Config


def _load_terrain_configs():
    tree = ast.parse(TERRAIN_SOURCE.read_text(encoding="utf-8"))
    # Only the two Isaac imports are replaced by lightweight config factories.
    body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(alias.name.startswith("isaaclab") for alias in node.names)
        or isinstance(node, ast.ImportFrom) and (node.module or "").startswith("isaaclab")
    )]
    namespace = {"terrain_gen": _TerrainFactories(), "TerrainGeneratorCfg": _Config}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(TERRAIN_SOURCE), "exec"), namespace)
    return namespace


def _column_names(cfg):
    """Isaac Lab v2.2.1 _generate_curriculum_terrains column choice."""
    names = list(cfg.sub_terrains)
    weights = [cfg.sub_terrains[name].proportion for name in names]
    total = sum(weights)
    cumulative = []
    value = 0.0
    for weight in weights:
        value += weight / total
        cumulative.append(value)
    return [next(name for name, bound in zip(names, cumulative) if col / cfg.num_cols + 0.001 < bound)
            for col in range(cfg.num_cols)]


def test_mixed_config_retains_parameters_and_deep_copies_old_stair_configuration():
    namespace = _load_terrain_configs()
    old = namespace["STAIR_ROUGH_TERRAINS_CFG"]
    assert "MIXED_WALK_TERRAINS_CFG" in namespace
    mixed = namespace["MIXED_WALK_TERRAINS_CFG"]
    assert old.num_cols == 6 and mixed.num_cols == 10
    assert old.num_rows == mixed.num_rows == 6
    assert old.size == mixed.size == (6.0, 6.0)
    assert old.curriculum is mixed.curriculum is True
    assert set(mixed.sub_terrains) == set(old.sub_terrains) | {"flat", "gentle_slope"}
    for name, original in old.sub_terrains.items():
        assert mixed.sub_terrains[name] == original
        assert mixed.sub_terrains[name] is not original
        assert original.proportion == mixed.sub_terrains[name].proportion == 1.0
    assert mixed.sub_terrains["flat"].proportion == 3.0
    assert mixed.sub_terrains["flat"].slope_range == (0.0, 0.0)
    assert mixed.sub_terrains["gentle_slope"].proportion == 1.0
    assert mixed.sub_terrains["gentle_slope"].slope_range == (0.0, 0.10)
    mixed.sub_terrains["stairs_low"].step_height_range = (0.5, 0.8)
    assert old.sub_terrains["stairs_low"].step_height_range == (0.02, 0.05)


def test_official_curriculum_column_allocation_has_persistent_ordinary_terrain():
    namespace = _load_terrain_configs()
    assert "MIXED_WALK_TERRAINS_CFG" in namespace
    cfg = namespace["MIXED_WALK_TERRAINS_CFG"]
    columns = _column_names(cfg)
    counts = Counter(columns)
    assert counts == Counter({"flat": 3, "gentle_slope": 1, "rough_steps": 1,
                              "stairs_low": 1, "stairs_mid": 1, "stairs_high": 1,
                              "stairs_down": 1, "stairs_down_high": 1})
    # The official generator's family selection depends on column, not row.
    # Row changes therefore cannot replace these fixed ordinary-terrain columns.


@pytest.mark.parametrize("num_envs", [64, 2048, 4096])
def test_official_importer_environment_column_allocation_exposes_all_families(num_envs):
    namespace = _load_terrain_configs()
    assert "MIXED_WALK_TERRAINS_CFG" in namespace
    cfg = namespace["MIXED_WALK_TERRAINS_CFG"]
    columns = _column_names(cfg)
    # v2.2.1 Importer: floor(env_id / (num_envs / num_cols)).
    assigned = [columns[min(int(env_id / (num_envs / cfg.num_cols)), cfg.num_cols - 1)]
                for env_id in range(num_envs)]
    counts = Counter(assigned)
    assert set(counts) == set(cfg.sub_terrains)
    for name, column_count in Counter(columns).items():
        assert abs(counts[name] - num_envs * column_count / cfg.num_cols) <= column_count
