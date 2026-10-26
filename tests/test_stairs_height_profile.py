"""Real Stairs class methods applied to real terrain cfg data on CPU only.

Isaac imports/configclass facilities are isolated with existing test containers;
no terrain generator, scene, policy, simulator or optimizer is constructed.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from test_mixed_terrain_config import _column_names, _load_terrain_configs
from test_stairs_config import _load_configs


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
@pytest.mark.parametrize("mixed", [True, False])
@pytest.mark.parametrize("profile", ["progressive_20cm", "legacy_14cm"])
def test_height_profile_changes_only_high_families_in_the_instance(name, mixed, profile):
    templates = _load_terrain_configs()
    original = deepcopy({key: templates[key] for key in ("STAIR_ROUGH_TERRAINS_CFG", "MIXED_WALK_TERRAINS_CFG")})
    namespace, _ = _load_configs(templates)
    cfg = namespace[name]()
    assert cfg.stairs_height_profile == "progressive_20cm"
    cfg.stairs_mixed_terrain = mixed
    cfg.stairs_height_profile = profile
    cfg.__post_init__()
    selected_name = "MIXED_WALK_TERRAINS_CFG" if mixed else "STAIR_ROUGH_TERRAINS_CFG"
    template = templates[selected_name]
    generated = cfg.scene.terrain.terrain_generator
    assert generated is not template
    assert generated.num_rows == template.num_rows == 6
    assert generated.num_cols == template.num_cols
    assert _column_names(generated) == _column_names(template)
    assert cfg.scene.terrain.max_init_terrain_level == (0 if name == "RobotStairsEnvCfg" else 3)
    for terrain_name, source in template.sub_terrains.items():
        actual = generated.sub_terrains[terrain_name]
        assert actual is not source
        expected = deepcopy(source)
        if profile == "progressive_20cm" and terrain_name in {"stairs_high", "stairs_down_high"}:
            expected.step_height_range = (0.08, 0.20)
        assert actual == expected
    params = cfg.rewards.feet_stair_clearance.params
    assert params["max_rise"] == (0.24 if profile == "progressive_20cm" else 0.16)
    assert params["rise_weight_scale"] == (0.16 if profile == "progressive_20cm" else None)
    assert params["min_rise"] == 0.02
    assert params["clearance"] == 0.035
    assert params["height_std"] == 0.08
    assert cfg.rewards.feet_stair_clearance.weight == 2.5
    for template_name, saved in original.items():
        assert templates[template_name] == saved


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
def test_invalid_height_profile_is_explicit(name):
    namespace, _ = _load_configs(_load_terrain_configs())
    cfg = namespace[name]()
    cfg.stairs_height_profile = "20cm"
    with pytest.raises(ValueError, match="stairs_height_profile"):
        cfg.__post_init__()


def test_progressive_profile_and_signal_knobs_do_not_modify_module_templates_across_instances():
    templates = _load_terrain_configs()
    saved = deepcopy({key: templates[key] for key in ("STAIR_ROUGH_TERRAINS_CFG", "MIXED_WALK_TERRAINS_CFG")})
    namespace, _ = _load_configs(templates)
    for name, profile, mixed, signal in (
        ("RobotStairsPlayEnvCfg", "progressive_20cm", True, "aligned"),
        ("RobotStairsEnvCfg", "legacy_14cm", False, "legacy"),
        ("RobotStairsEnvCfg", "progressive_20cm", False, "legacy"),
        ("RobotStairsPlayEnvCfg", "legacy_14cm", True, "aligned"),
    ):
        cfg = namespace[name]()
        cfg.stairs_height_profile = profile
        cfg.stairs_mixed_terrain = mixed
        cfg.walking_foot_signal_mode = signal
        cfg.__post_init__()
    assert {key: templates[key] for key in saved} == saved
