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
        self.curriculum = SimpleNamespace(
            terrain_levels=object(),
            lin_vel_cmd_levels=SimpleNamespace(func="progress_callback", params={"min_episodes": 32}),
        )
        self.commands = SimpleNamespace(base_velocity=SimpleNamespace())

    def __post_init__(self):
        pass


def _load_configs(terrain_templates=None):
    """Load actual class bodies while isolating the simulator's base classes."""
    names = {"RobotStairsEnvCfg", "RobotStairsPlayEnvCfg", "RobotFlatEnvCfg"}
    source = TASK / "robot_env_cfg.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_configure_walking_height_rewards", "_configure_walking_tracking_rewards",
                                 "_configure_stairs_height_profile", "_configure_walking_command_curriculum"}]
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
            sub_terrains={
                "stairs_low": SimpleNamespace(proportion=1.0, step_height_range=(0.02, 0.05)),
                "stairs_high": SimpleNamespace(proportion=1.0, step_height_range=(0.08, 0.14)),
                "stairs_down_high": SimpleNamespace(proportion=1.0, step_height_range=(0.08, 0.14)),
            },
        ),
        "MIXED_WALK_TERRAINS_CFG": SimpleNamespace(
            curriculum=None, num_cols=10,
            sub_terrains={
                "flat": SimpleNamespace(proportion=3.0),
                "stairs_low": SimpleNamespace(proportion=1.0, step_height_range=(0.02, 0.05)),
                "stairs_high": SimpleNamespace(proportion=1.0, step_height_range=(0.08, 0.14)),
                "stairs_down_high": SimpleNamespace(proportion=1.0, step_height_range=(0.08, 0.14)),
            },
        ),
        "_enable_stairs_lidar_reconstruction": enable_reconstruction,
    }
    if terrain_templates is not None:
        namespace.update({name: terrain_templates[name]
                          for name in ("STAIR_ROUGH_TERRAINS_CFG", "MIXED_WALK_TERRAINS_CFG")})
    module = ast.fix_missing_locations(ast.Module(body=[*helpers, *classes], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace, calls


@pytest.fixture
def configs():
    return _load_configs()


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
    assert cfg.walking_tracking_mode == "command_relative_mixed"
    assert cfg.walking_tracking_relative_weight == 0.75
    for reward_name, weight in (("track_lin_vel_xy", 8.0), ("track_forward_progress", 2.0)):
        term = getattr(cfg.rewards, reward_name)
        assert term.params == {"std": 0.5, "tracking_mode": "command_relative_mixed",
                               "min_command_speed": 0.1, "relative_weight": 0.75}
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
            "relative_weight": 0.75,
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
    assert cfg.curriculum.lin_vel_cmd_levels.params == {"min_episodes": 32}


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


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
@pytest.mark.parametrize("mode", ["absolute", "command_relative", "command_relative_mixed"])
def test_tracking_modes_and_relative_weight_reach_both_terms(configs, name, mode):
    namespace, _ = configs
    cfg = namespace[name]()
    cfg.walking_tracking_mode = mode
    cfg.walking_tracking_relative_weight = 0.6
    cfg.__post_init__()
    for reward_name in ("track_lin_vel_xy", "track_forward_progress"):
        params = getattr(cfg.rewards, reward_name).params
        assert params["tracking_mode"] == mode
        assert params["relative_weight"] == 0.6
        assert params["min_command_speed"] == 0.1


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_relative_tracking_weight_accepts_closed_interval_endpoints(configs, weight):
    namespace, _ = configs
    cfg = namespace["RobotStairsEnvCfg"]()
    cfg.walking_tracking_relative_weight = weight
    cfg.__post_init__()
    assert cfg.rewards.track_lin_vel_xy.params["relative_weight"] == weight
    assert cfg.rewards.track_forward_progress.params["relative_weight"] == weight


@pytest.mark.parametrize("weight", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_relative_tracking_weight_rejects_nonfinite_and_out_of_range(configs, weight):
    namespace, _ = configs
    cfg = namespace["RobotStairsEnvCfg"]()
    cfg.walking_tracking_relative_weight = weight
    with pytest.raises(ValueError, match="walking_tracking_relative_weight"):
        cfg.__post_init__()


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
@pytest.mark.parametrize("height_mode", ["terrain", "body_legacy"])
@pytest.mark.parametrize("signal_mode", ["aligned", "legacy"])
def test_foot_signal_mode_and_height_baseline_are_independent(configs, name, height_mode, signal_mode):
    namespace, _ = configs
    cfg = namespace[name]()
    cfg.walking_height_mode = height_mode
    cfg.walking_foot_signal_mode = signal_mode
    cfg.__post_init__()
    swing = cfg.rewards.feet_swing_height.params
    aligned_terrain = height_mode == "terrain" and signal_mode == "aligned"
    assert swing["height_std"] == (0.03 if aligned_terrain else 0.10)
    assert swing["movement_threshold"] == (0.1 if aligned_terrain else 0.15)
    assert swing["movement_gate_inclusive"] is aligned_terrain
    stair = cfg.rewards.feet_stair_clearance.params
    assert stair["command_gate_mode"] == ("progress" if signal_mode == "aligned" else "legacy")
    assert stair["min_forward_command"] == 0.1
    assert stair["max_lateral_command"] == 0.15
    assert stair["max_yaw_command"] == 0.3


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
def test_unknown_foot_signal_mode_is_explicit(configs, name):
    namespace, _ = configs
    cfg = namespace[name]()
    cfg.walking_foot_signal_mode = "auto"
    with pytest.raises(ValueError, match="walking_foot_signal_mode"):
        cfg.__post_init__()


@pytest.mark.parametrize("name,expected", [("RobotStairsEnvCfg", "terrain_columns"),
                                           ("RobotStairsPlayEnvCfg", "global")])
def test_command_curriculum_defaults_reach_existing_callback(configs, name, expected):
    namespace, _ = configs
    cfg = namespace[name]()
    assert cfg.walking_command_curriculum_mode == expected
    cfg.__post_init__()
    term = cfg.curriculum.lin_vel_cmd_levels
    assert term.func == "progress_callback"
    assert term.params == {"min_episodes": 32, "grouping": expected, "min_group_episodes": 32}
    if name == "RobotStairsEnvCfg":
        assert cfg.scene.terrain.terrain_generator.curriculum is True


def test_training_can_explicitly_restore_global_command_curriculum(configs):
    namespace, _ = configs
    cfg = namespace["RobotStairsEnvCfg"]()
    cfg.walking_command_curriculum_mode = "global"
    cfg.__post_init__()
    assert cfg.curriculum.lin_vel_cmd_levels.params["grouping"] == "global"
    assert cfg.curriculum.lin_vel_cmd_levels.params["min_group_episodes"] == 32


@pytest.mark.parametrize("name", ["RobotStairsEnvCfg", "RobotStairsPlayEnvCfg"])
def test_invalid_command_curriculum_mode_is_explicit(configs, name):
    namespace, _ = configs
    cfg = namespace[name]()
    cfg.walking_command_curriculum_mode = "terrain_family"
    with pytest.raises(ValueError, match="walking_command_curriculum_mode"):
        cfg.__post_init__()
