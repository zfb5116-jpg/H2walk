"""Execute production config methods on plain containers, without Isaac Lab.

AST extraction omits registration, imports and the configclass decorator only.
These checks cover the GT/reconstruction switch and missing-sensor contract;
they do not construct an Isaac Lab scene or establish simulator integration.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"


class _ContainerBase:
    def __init__(self):
        self.scene = SimpleNamespace(
            terrain=SimpleNamespace(terrain_generator=None),
            height_scanner=object(),
            lidar=None,
        )
        self.observations = SimpleNamespace(
            policy=SimpleNamespace(height_scanner="gt_policy_scan"),
            critic=SimpleNamespace(height_scanner="gt_critic_scan"),
        )
        self.rewards = SimpleNamespace(
            feet_stair_clearance=SimpleNamespace(weight=2.0, params={}),
            feet_clearance=SimpleNamespace(weight=-4.0, params={}),
            feet_swing_height=SimpleNamespace(weight=0.75, params={}),
            track_lin_vel_xy=SimpleNamespace(weight=8.0, params={"std": 0.5}),
            track_forward_progress=SimpleNamespace(weight=2.0, params={"std": 0.5}),
        )
        self.curriculum = SimpleNamespace(terrain_levels=object())
        self.commands = SimpleNamespace(base_velocity=SimpleNamespace())

    def __post_init__(self):
        pass


@pytest.fixture
def configs():
    """Load actual class bodies while isolating the simulator's base classes."""
    names = {"RobotStairsEnvCfg", "RobotStairsPlayEnvCfg", "RobotFlatEnvCfg"}
    source = TASK / "robot_env_cfg.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_configure_walking_height_rewards", "_configure_walking_tracking_rewards"}]
    assert {node.name for node in classes} == names
    for node in classes:
        node.decorator_list = []
    calls = []

    def enable_reconstruction(cfg):
        calls.append(cfg)
        cfg.scene.lidar = "explicit_reconstruction_sensor"
        cfg.observations.policy.height_scanner = "explicit_reconstruction_scan"

    namespace = {
        "RobotRoughEnvCfg": _ContainerBase,
        "RobotRoughPlayEnvCfg": _ContainerBase,
        "deepcopy": deepcopy,
        "STAIR_ROUGH_TERRAINS_CFG": SimpleNamespace(
            curriculum=None, num_cols=6,
            sub_terrains={"stairs_low": SimpleNamespace(proportion=1.0)},
        ),
        "MIXED_WALK_TERRAINS_CFG": SimpleNamespace(
            curriculum=None, num_cols=10,
            sub_terrains={"flat": SimpleNamespace(proportion=3.0)},
        ),
        "_enable_stairs_lidar_reconstruction": enable_reconstruction,
    }
    module = ast.fix_missing_locations(ast.Module(body=[*helpers, *classes], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace, calls


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
def test_stairs_default_uses_gt_without_enabling_reconstruction(configs, name):
    namespace, calls = configs
    cfg = namespace[name]()
    assert cfg.use_lidar_reconstruction is False
    assert cfg.lidar_reliable_fusion is False
    cfg.__post_init__()
    assert calls == []
    assert cfg.scene.lidar is None
    assert cfg.scene.height_scanner is not None
    assert cfg.observations.policy.height_scanner == "gt_policy_scan"
    assert cfg.observations.critic.height_scanner == "gt_critic_scan"
    assert cfg.rewards.feet_stair_clearance.params["target_mode"] == "nearest"
    assert cfg.stairs_clearance_height_reference == "sole"
    assert cfg.rewards.feet_stair_clearance.params["height_reference"] == "sole"
    assert cfg.rewards.feet_stair_clearance.weight == 2.5
    assert cfg.walking_tracking_mode == "command_relative"
    for reward_name, weight in (("track_lin_vel_xy", 8.0), ("track_forward_progress", 2.0)):
        term = getattr(cfg.rewards, reward_name)
        assert term.params == {"std": 0.5, "tracking_mode": "command_relative", "min_command_speed": 0.1}
        assert term.weight == weight
    assert cfg.scene.terrain.max_init_terrain_level == (0 if name == "RobotStairsEnvCfg" else 3)
    assert cfg.scene.terrain.terrain_generator.curriculum is True
    if name == "RobotStairsPlayEnvCfg":
        assert cfg.curriculum.terrain_levels is None


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
def test_reconstruction_requires_explicit_opt_in(configs, name):
    namespace, calls = configs
    cfg = namespace[name]()
    cfg.use_lidar_reconstruction = True
    cfg.__post_init__()
    assert calls == [cfg]
    assert cfg.observations.policy.height_scanner == "explicit_reconstruction_scan"
    assert cfg.observations.critic.height_scanner == "gt_critic_scan"


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
def test_highest_baseline_reaches_the_reward_parameters(configs, name):
    namespace, calls = configs
    cfg = namespace[name]()
    cfg.stairs_clearance_target_mode = "highest"
    cfg.__post_init__()
    assert cfg.rewards.feet_stair_clearance.params["target_mode"] == "highest"
    assert calls == []


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
def test_link_height_baseline_reaches_the_reward_parameters(configs, name):
    namespace, calls = configs
    cfg = namespace[name]()
    cfg.stairs_clearance_height_reference = "link"
    cfg.__post_init__()
    assert cfg.rewards.feet_stair_clearance.params["height_reference"] == "link"
    assert cfg.rewards.feet_stair_clearance.params["target_mode"] == "nearest"
    assert calls == []


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
def test_absolute_tracking_baseline_reaches_both_reward_parameters(configs, name):
    namespace, _ = configs
    cfg = namespace[name]()
    cfg.walking_tracking_mode = "absolute"
    cfg.__post_init__()
    for reward_name in ("track_lin_vel_xy", "track_forward_progress"):
        assert getattr(cfg.rewards, reward_name).params == {
            "std": 0.5, "tracking_mode": "absolute", "min_command_speed": 0.1,
        }


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
def test_unknown_config_tracking_mode_is_explicit(configs, name):
    namespace, _ = configs
    cfg = namespace[name]()
    cfg.walking_tracking_mode = "relative"
    with pytest.raises(ValueError, match="walking_tracking_mode"):
        cfg.__post_init__()


def test_flat_removes_terrain_reward_when_removing_its_sensor(configs):
    namespace, _ = configs
    cfg = namespace["RobotFlatEnvCfg"]()
    cfg.__post_init__()
    assert cfg.scene.height_scanner is None
    assert cfg.observations.policy.height_scanner is None
    assert cfg.observations.critic.height_scanner is None
    assert cfg.rewards.feet_stair_clearance is None
    assert cfg.curriculum.terrain_levels is None
    assert "height_mode" not in cfg.rewards.feet_clearance.params
    assert "tracking_mode" not in cfg.rewards.track_lin_vel_xy.params
    assert "tracking_mode" not in cfg.rewards.track_forward_progress.params
    assert "height_mode" not in cfg.rewards.feet_swing_height.params


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
def test_mixed_walking_terrain_and_height_mode_reach_production_config(configs, name):
    namespace, _ = configs
    cfg = namespace[name]()
    assert cfg.stairs_mixed_terrain is True
    assert cfg.walking_height_mode == "terrain"
    cfg.__post_init__()
    assert cfg.scene.terrain.terrain_generator.num_cols == 10
    for term in (cfg.rewards.feet_clearance, cfg.rewards.feet_swing_height):
        assert term.params["height_mode"] == "terrain"
        assert term.params["min_clearance"] == 0.03
        assert term.params["terrain_sensor_name"] == "height_scanner"
        assert term.params["ground_query_radius"] == 0.10
    assert cfg.rewards.feet_clearance.weight == -4.0
    assert cfg.rewards.feet_swing_height.weight == 0.75


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
@pytest.mark.parametrize("mixed", [True, False])
@pytest.mark.parametrize("height_mode", ["terrain", "body_legacy"])
def test_terrain_and_height_mode_switches_are_independent(configs, name, mixed, height_mode):
    namespace, _ = configs
    cfg = namespace[name]()
    cfg.stairs_mixed_terrain = mixed
    cfg.walking_height_mode = height_mode
    cfg.__post_init__()
    assert cfg.scene.terrain.terrain_generator.num_cols == (10 if mixed else 6)
    assert cfg.scene.terrain.terrain_generator.curriculum is (mixed or name == "RobotStairsEnvCfg")
    if name == "RobotStairsPlayEnvCfg":
        assert (cfg.curriculum.terrain_levels is None) is mixed
    for term in (cfg.rewards.feet_clearance, cfg.rewards.feet_swing_height):
        assert term.params["height_mode"] == height_mode
    assert cfg.rewards.feet_stair_clearance.params["height_reference"] == "sole"
    assert cfg.rewards.feet_stair_clearance.params["target_mode"] == "nearest"


def test_train_play_configuration_does_not_mutate_shared_terrain_templates(configs):
    namespace, _ = configs
    mixed = namespace["MIXED_WALK_TERRAINS_CFG"]
    legacy = namespace["STAIR_ROUGH_TERRAINS_CFG"]
    originals = deepcopy((mixed, legacy))
    instances = []
    for name, use_mixed in (("RobotStairsEnvCfg", True), ("RobotStairsPlayEnvCfg", True),
                            ("RobotStairsEnvCfg", False), ("RobotStairsPlayEnvCfg", False)):
        cfg = namespace[name]()
        cfg.stairs_mixed_terrain = use_mixed
        cfg.__post_init__()
        instances.append(cfg)
        source = mixed if use_mixed else legacy
        generated = cfg.scene.terrain.terrain_generator
        assert generated is not source
        assert generated.sub_terrains is not source.sub_terrains
        key = next(iter(source.sub_terrains))
        assert generated.sub_terrains[key] is not source.sub_terrains[key]
        generated.sub_terrains[key].proportion = 99.0
    assert (mixed, legacy) == originals
    assert len({id(cfg.scene.terrain.terrain_generator) for cfg in instances}) == 4


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
def test_walking_height_mode_validation_is_early(configs, name):
    namespace, _ = configs
    cfg = namespace[name]()
    cfg.walking_height_mode = "unknown"
    with pytest.raises(ValueError, match="walking_height_mode"):
        cfg.__post_init__()
