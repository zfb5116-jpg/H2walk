"""Execute production config methods on plain containers, without Isaac Lab.

AST extraction omits registration, imports and the configclass decorator only.
These checks cover the GT/reconstruction switch and missing-sensor contract;
they do not construct an Isaac Lab scene or establish simulator integration.
"""

from __future__ import annotations

import ast
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
        self.rewards = SimpleNamespace(feet_stair_clearance=SimpleNamespace(weight=2.0, params={}))
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
        "STAIR_ROUGH_TERRAINS_CFG": SimpleNamespace(curriculum=None),
        "_enable_stairs_lidar_reconstruction": enable_reconstruction,
    }
    module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
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
    assert cfg.rewards.feet_stair_clearance.weight == 2.5
    assert cfg.scene.terrain.max_init_terrain_level == (0 if name == "RobotStairsEnvCfg" else 3)
    assert cfg.scene.terrain.terrain_generator.curriculum is (name == "RobotStairsEnvCfg")


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


def test_flat_removes_terrain_reward_when_removing_its_sensor(configs):
    namespace, _ = configs
    cfg = namespace["RobotFlatEnvCfg"]()
    cfg.__post_init__()
    assert cfg.scene.height_scanner is None
    assert cfg.observations.policy.height_scanner is None
    assert cfg.observations.critic.height_scanner is None
    assert cfg.rewards.feet_stair_clearance is None
    assert cfg.curriculum.terrain_levels is None
