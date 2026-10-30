"""CPU configuration contracts for scan-preserving staged locomotion tasks.

Real terrain configuration data is copied into plain containers. No terrain
mesh, Isaac Lab environment, policy, or optimizer is constructed.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_mixed_terrain_config import _column_names, _load_terrain_configs


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"


def _load_profile_module():
    package_name = "_h2_profile_test_package"
    package = ModuleType(package_name)
    package.__path__ = [str(TASK)]
    sys.modules[package_name] = package
    name = package_name + ".walking_profiles"
    spec = importlib.util.spec_from_file_location(name, TASK / "walking_profiles.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def profile_module():
    return _load_profile_module()


def _config():
    return SimpleNamespace(
        use_lidar_reconstruction=False,
        walking_reward_parameters=_load_profile_module().WALK_REWARD_PARAMETERS.copy(),
        episode_length_s=20.0,
        scene=SimpleNamespace(
            height_scanner=SimpleNamespace(pattern_cfg=SimpleNamespace(resolution=.05, size=[1.6, .8]), mesh_prim_paths=["/World/ground"]),
            lidar=None,
            terrain=SimpleNamespace(terrain_generator=None, terrain_type="generator", max_init_terrain_level=3, prim_path="/World/ground"),
            robot=SimpleNamespace(actuators={
                "leg": SimpleNamespace(min_delay=0, max_delay=6, stiffness=200.0, effort_limit=360.0),
                "arm": SimpleNamespace(min_delay=0, max_delay=6, stiffness=40.0, effort_limit=54.0),
            }),
        ),
        observations=SimpleNamespace(
            policy=SimpleNamespace(height_scanner=object(), enable_corruption=True),
            critic=SimpleNamespace(height_scanner=object()),
        ),
        commands=SimpleNamespace(base_velocity=SimpleNamespace(
            ranges=SimpleNamespace(lin_vel_x=(-.1, .3), lin_vel_y=(-.08, .08), ang_vel_z=(-.3, .3), heading=(-3.14, 3.14)),
            limit_ranges=SimpleNamespace(lin_vel_x=(-1., 1.2), lin_vel_y=(-.25, .25), ang_vel_z=(-.8, .8), heading=(-3.14, 3.14)),
            resampling_time_range=(8, 12), rel_standing_envs=.05, rel_heading_envs=1., heading_command=True,
        )),
        events=SimpleNamespace(
            push_robot=SimpleNamespace(func="push"), physics_material=SimpleNamespace(func="material_randomization"),
            add_motor_pd_and_motor_strength=SimpleNamespace(func="gain_randomization"),
            reset_base=SimpleNamespace(params={
                "pose_range": {"x": (-.25, .25), "y": (-.25, .25), "yaw": (-3.14, 3.14)},
                "velocity_range": {"x": (0., 0.), "y": (0., 0.), "z": (0., 0.)},
            }),
        ),
        curriculum=SimpleNamespace(terrain_levels=object(), lin_vel_cmd_levels=object()),
        terminations=object(), actions=object(),
        rewards=SimpleNamespace(
            feet_stair_clearance=SimpleNamespace(weight=2.5, params={}),
            feet_air_time_variance=SimpleNamespace(weight=-1., params={"command_name": "base_velocity"}),
            feet_swing_height=SimpleNamespace(weight=.75, params={}),
            feet_air_time=SimpleNamespace(weight=.45, params={}),
            feet_contact_time=SimpleNamespace(weight=.2, params={}),
            walk_signed_progress=SimpleNamespace(weight=0., params={}),
            walk_alternating_steps=SimpleNamespace(weight=0., params={}),
            walk_support_quality=SimpleNamespace(weight=0., params={}),
        ),
    )


def _apply(module, cfg, stage, **kwargs):
    terrain = _load_terrain_configs()
    module.apply_walking_profile(
        cfg, stage, terrain_template=terrain["MIXED_WALK_TERRAINS_CFG"],
        flat_template=terrain["MIXED_WALK_TERRAINS_CFG"].sub_terrains["flat"],
        # Default spawn is the center: inverted pyramid moves uphill outward.
        stairs_up_template=terrain["STAIR_ROUGH_TERRAINS_CFG"].sub_terrains["stairs_down"],
        stairs_down_template=terrain["STAIR_ROUGH_TERRAINS_CFG"].sub_terrains["stairs_low"],
        **kwargs,
    )
    return terrain


def test_bootstrap_keeps_scan_actions_and_terminations(profile_module):
    cfg = _config()
    sensor, actor_scan, critic_scan = cfg.scene.height_scanner, cfg.observations.policy.height_scanner, cfg.observations.critic.height_scanner
    actions, terminations = cfg.actions, cfg.terminations
    _apply(profile_module, cfg, "bootstrap")
    assert cfg.scene.height_scanner is sensor
    assert cfg.observations.policy.height_scanner is actor_scan
    assert cfg.observations.critic.height_scanner is critic_scan
    assert cfg.actions is actions and cfg.terminations is terminations
    assert cfg.scene.lidar is None
    assert cfg.scene.terrain.terrain_type == "generator"
    generated = cfg.scene.terrain.terrain_generator
    assert generated.num_rows == generated.num_cols == 1
    assert generated.size == (10.0, 10.0)
    assert _column_names(generated) == ["flat"]
    assert generated.sub_terrains["flat"].slope_range == (0., 0.)
    assert cfg.curriculum.terrain_levels is None
    assert cfg.curriculum.lin_vel_cmd_levels is None


@pytest.mark.parametrize("stage", ["shallow", "stairs_08", "stairs_12", "stairs_16", "stairs_20"])
def test_stage_has_persistent_flat_up_and_down_columns_without_mutating_templates(profile_module, stage):
    cfg = _config()
    terrain = _load_terrain_configs()
    saved = deepcopy(terrain["MIXED_WALK_TERRAINS_CFG"])
    profile_module.apply_walking_profile(
        cfg, stage, terrain_template=terrain["MIXED_WALK_TERRAINS_CFG"],
        flat_template=terrain["MIXED_WALK_TERRAINS_CFG"].sub_terrains["flat"],
        stairs_up_template=terrain["STAIR_ROUGH_TERRAINS_CFG"].sub_terrains["stairs_down"],
        stairs_down_template=terrain["STAIR_ROUGH_TERRAINS_CFG"].sub_terrains["stairs_low"],
    )
    generated = cfg.scene.terrain.terrain_generator
    assert generated.num_rows == 1 and generated.num_cols == 10
    assert generated.curriculum is True
    assert cfg.scene.terrain.max_init_terrain_level == 0
    assert _column_names(generated) == ["flat"] * 6 + ["up"] * 2 + ["down"] * 2
    expected = profile_module.STAGE_HEIGHT_RANGES[stage]
    for family in ("up", "down"):
        assert generated.sub_terrains[family].step_height_range == expected
        assert generated.sub_terrains[family].platform_width == 3.0
        assert generated.sub_terrains[family].step_width == .35
        assert generated.sub_terrains[family].border_width == .3
    assert terrain["MIXED_WALK_TERRAINS_CFG"] == saved
    generated.sub_terrains["up"].step_height_range = (.9, 1.)
    assert terrain["STAIR_ROUGH_TERRAINS_CFG"].sub_terrains["stairs_down"].step_height_range == (.02, .08)


def test_forward_command_and_reset_frame_are_fixed_together(profile_module):
    cfg = _config()
    _apply(profile_module, cfg, "bootstrap")
    command = cfg.commands.base_velocity
    assert command.ranges.lin_vel_x == command.limit_ranges.lin_vel_x == (.2, .2)
    assert command.ranges.lin_vel_y == command.limit_ranges.lin_vel_y == (0., 0.)
    assert command.ranges.ang_vel_z == command.limit_ranges.ang_vel_z == (0., 0.)
    assert command.ranges.heading == command.limit_ranges.heading == (0., 0.)
    assert command.ranges is not command.limit_ranges
    assert command.rel_standing_envs == command.rel_heading_envs == 0.
    assert command.heading_command is False
    assert command.resampling_time_range == (20., 20.)
    assert cfg.events.reset_base.params["pose_range"] == {"x": (-.1, .1), "y": (-.1, .1), "yaw": (0., 0.)}


def test_default_profile_simplifies_independent_noise_and_randomization_knobs(profile_module):
    cfg = _config()
    _apply(profile_module, cfg, "shallow")
    assert cfg.events.push_robot is None
    assert cfg.events.physics_material is None
    assert cfg.events.add_motor_pd_and_motor_strength is None
    assert cfg.observations.policy.enable_corruption is False
    assert all(a.min_delay == a.max_delay == 0 for a in cfg.scene.robot.actuators.values())
    assert cfg.scene.robot.actuators["leg"].stiffness == 200.
    assert cfg.scene.robot.actuators["arm"].effort_limit == 54.


@pytest.mark.parametrize("flag", ["enable_push", "enable_domain_randomization", "enable_observation_noise", "enable_actuator_delay"])
def test_each_ablation_knob_restores_only_its_original_configuration(profile_module, flag):
    cfg = _config()
    _apply(profile_module, cfg, "shallow", **{flag: True})
    assert (cfg.events.push_robot is not None) is (flag == "enable_push")
    assert (cfg.events.physics_material is not None) is (flag == "enable_domain_randomization")
    assert (cfg.events.add_motor_pd_and_motor_strength is not None) is (flag == "enable_domain_randomization")
    assert cfg.observations.policy.enable_corruption is (flag == "enable_observation_noise")
    assert (cfg.scene.robot.actuators["leg"].max_delay == 6) is (flag == "enable_actuator_delay")


@pytest.mark.parametrize("stage", [None, "flat", "stairs20", 20])
def test_unknown_stage_fails_before_mutation(profile_module, stage):
    cfg = _config()
    generator = cfg.scene.terrain.terrain_generator
    with pytest.raises(ValueError, match="stage"):
        _apply(profile_module, cfg, stage)
    assert cfg.scene.terrain.terrain_generator is generator
    assert cfg.curriculum.terrain_levels is not None


@pytest.mark.parametrize("speed", [0., .09, -.2, float("nan"), float("inf"), True, .4])
def test_invalid_or_tile_crossing_command_is_rejected(profile_module, speed):
    with pytest.raises(ValueError, match="speed|tile"):
        _apply(profile_module, _config(), "shallow", forward_speed=speed)


@pytest.mark.parametrize("missing", ["sensor", "actor", "critic", "reconstruction", "pattern"])
def test_missing_or_different_scan_contract_is_explicit(profile_module, missing):
    cfg = _config()
    if missing == "sensor":
        cfg.scene.height_scanner = None
    elif missing == "actor":
        cfg.observations.policy.height_scanner = None
    elif missing == "critic":
        cfg.observations.critic.height_scanner = None
    elif missing == "reconstruction":
        cfg.use_lidar_reconstruction = True
    else:
        cfg.scene.height_scanner.pattern_cfg.size = [1., 1.]
    with pytest.raises(ValueError, match="scan|GT|reconstruction"):
        _apply(profile_module, cfg, "bootstrap")


def test_profile_does_not_touch_checkpoint_or_normalizer_configuration(profile_module):
    cfg = _config()
    cfg.normalizer = object()
    cfg.checkpoint = "original_checkpoint"
    normalizer = cfg.normalizer
    _apply(profile_module, cfg, "shallow")
    assert cfg.normalizer is normalizer
    assert cfg.checkpoint == "original_checkpoint"


@pytest.mark.parametrize("flag", ["enable_push", "enable_domain_randomization", "enable_observation_noise", "enable_actuator_delay"])
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_ablation_flags_reject_ambiguous_values(profile_module, flag, value):
    with pytest.raises(ValueError, match="booleans"):
        _apply(profile_module, _config(), "bootstrap", **{flag: value})


@pytest.mark.parametrize("duration", [0., -20., float("nan"), float("inf"), True, "20"])
def test_invalid_episode_duration_is_explicit(profile_module, duration):
    cfg = _config()
    cfg.episode_length_s = duration
    with pytest.raises(ValueError, match="duration"):
        _apply(profile_module, cfg, "bootstrap")


def test_disabled_delay_requires_an_explicit_actuator_delay_contract(profile_module):
    cfg = _config()
    del cfg.scene.robot.actuators["leg"].max_delay
    with pytest.raises(ValueError, match="delay configuration"):
        _apply(profile_module, cfg, "bootstrap")
    assert cfg.scene.terrain.terrain_generator is None


def test_missing_reset_rejects_profile_before_mutation(profile_module):
    cfg = _config()
    cfg.events.reset_base = None
    with pytest.raises(ValueError, match="reset_base"):
        _apply(profile_module, cfg, "bootstrap")
    assert cfg.scene.terrain.terrain_generator is None


def _resolved_profile(module, stage="shallow", speed=.2, **flags):
    cfg = _config()
    cfg.walking_forward_speed = speed
    for flag in ("enable_push", "enable_domain_randomization", "enable_observation_noise", "enable_actuator_delay"):
        setattr(cfg, "walking_" + flag, flags.get(flag, False))
    _apply(module, cfg, stage, forward_speed=speed, **flags)
    return cfg


@pytest.mark.parametrize("stage", ["bootstrap", "shallow", "stairs_08", "stairs_12", "stairs_16", "stairs_20"])
def test_read_only_validator_accepts_fresh_profiles_without_reapplying_them(profile_module, stage):
    cfg = _resolved_profile(profile_module, stage)
    generator = cfg.scene.terrain.terrain_generator
    command = cfg.commands.base_velocity
    old_ranges = deepcopy(command.ranges.__dict__)
    reset_params = deepcopy(cfg.events.reset_base.params)
    assert profile_module.validate_walking_profile(cfg) is None
    assert cfg.scene.terrain.terrain_generator is generator
    assert cfg.commands.base_velocity is command
    assert command.ranges.__dict__ == old_ranges
    assert cfg.events.reset_base.params == reset_params


@pytest.mark.parametrize("flag", ["enable_push", "enable_domain_randomization", "enable_observation_noise", "enable_actuator_delay"])
def test_validator_accepts_a_fresh_constructor_ablation(profile_module, flag):
    cfg = _resolved_profile(profile_module, speed=.15, **{flag: True})
    profile_module.validate_walking_profile(cfg)


@pytest.mark.parametrize("mutate", [
    lambda cfg: setattr(cfg, "walking_stage", "stairs_20"),
    lambda cfg: setattr(cfg, "walking_enable_push", True),
    lambda cfg: setattr(cfg, "walking_enable_domain_randomization", True),
    lambda cfg: setattr(cfg, "walking_enable_observation_noise", True),
    lambda cfg: setattr(cfg, "walking_enable_actuator_delay", True),
    lambda cfg: setattr(cfg, "walking_forward_speed", .15),
    lambda cfg: setattr(cfg, "stage_support_height_tolerance", .04),
    lambda cfg: setattr(cfg.scene.height_scanner, "mesh_prim_paths", ["/World/another_ground"]),
    lambda cfg: setattr(cfg, "walking_stage_height_range", (.2, .2)),
    lambda cfg: setattr(cfg, "walking_stage_column_families", ("flat",)),
    lambda cfg: setattr(cfg.scene.terrain.terrain_generator, "size", (6., 6.)),
    lambda cfg: setattr(cfg.scene.terrain.terrain_generator, "num_rows", 2),
    lambda cfg: setattr(cfg.scene.terrain.terrain_generator, "num_cols", 1),
    lambda cfg: setattr(cfg.scene.terrain.terrain_generator, "curriculum", False),
    lambda cfg: setattr(cfg.scene.terrain.terrain_generator.sub_terrains["up"], "step_height_range", (.16, .2)),
    lambda cfg: setattr(cfg.scene.terrain.terrain_generator.sub_terrains["up"], "proportion", 1.),
    lambda cfg: setattr(cfg.commands.base_velocity.ranges, "lin_vel_y", (-.1, .1)),
    lambda cfg: setattr(cfg.commands.base_velocity.ranges, "ang_vel_z", (-.3, .3)),
    lambda cfg: setattr(cfg.commands.base_velocity, "heading_command", True),
    lambda cfg: cfg.events.reset_base.params["pose_range"].update(yaw=(-3.14, 3.14)),
    lambda cfg: setattr(cfg.observations.policy, "enable_corruption", True),
    lambda cfg: setattr(cfg.events, "push_robot", object()),
    lambda cfg: setattr(cfg.scene.robot.actuators["leg"], "max_delay", 6),
    lambda cfg: setattr(cfg.observations.critic, "height_scanner", None),
    lambda cfg: setattr(cfg.curriculum, "terrain_levels", object()),
])
def test_post_init_override_cannot_silently_relabel_a_resolved_profile(profile_module, mutate):
    cfg = _resolved_profile(profile_module)
    mutate(cfg)
    with pytest.raises(ValueError, match="construct a fresh profile with constructor arguments"):
        profile_module.validate_walking_profile(cfg)


@pytest.mark.parametrize("flag,event", [
    ("enable_push", "push_robot"),
    ("enable_domain_randomization", "physics_material"),
    ("enable_domain_randomization", "add_motor_pd_and_motor_strength"),
])
def test_enabled_event_flag_requires_the_event_to_survive_configuration(profile_module, flag, event):
    cfg = _resolved_profile(profile_module, **{flag: True})
    setattr(cfg.events, event, None)
    with pytest.raises(ValueError, match="construct a fresh profile with constructor arguments"):
        profile_module.validate_walking_profile(cfg)


@pytest.mark.parametrize("stem", [
    "RobotWalkBootstrap", "RobotWalkShallow", "RobotWalkStairs08",
    "RobotWalkStairs12", "RobotWalkStairs16", "RobotWalkStairs20",
])
@pytest.mark.parametrize("play", [False, True])
def test_validator_accepts_the_resolved_production_config_class_bodies(profile_module, stem, play):
    from test_walking_env_config import _load_configs

    namespace = _load_configs()
    cfg = namespace[stem + ("PlayEnvCfg" if play else "EnvCfg")]()
    cfg.__post_init__()
    profile_module.validate_walking_profile(cfg)


@pytest.mark.parametrize("stage", ["bootstrap", "shallow", "stairs_08", "stairs_12", "stairs_16", "stairs_20"])
@pytest.mark.parametrize("mode", ["legacy", "bounded"])
def test_stage_timing_mode_is_explicit_and_keeps_existing_reward_parameters(profile_module, stage, mode):
    cfg = _config()
    original_params = cfg.rewards.feet_air_time_variance.params
    original_params["sensor_cfg"] = "existing_sensor"
    _apply(profile_module, cfg, stage, cross_time_mode=mode, max_time_difference=.3)
    assert cfg.walking_cross_time_mode == mode
    assert cfg.walking_max_time_difference == .3
    assert cfg.rewards.feet_air_time_variance.params == {
        "command_name": "base_velocity", "sensor_cfg": "existing_sensor",
        "cross_time_mode": mode, "max_time_difference": .3,
    }
    assert cfg.rewards.feet_air_time_variance.weight == -1.
    assert original_params == {"command_name": "base_velocity", "sensor_cfg": "existing_sensor"}


@pytest.mark.parametrize("kwargs", [
    {"cross_time_mode": "clip"}, {"cross_time_mode": None}, {"cross_time_mode": []},
    {"max_time_difference": 0.}, {"max_time_difference": -.5}, {"max_time_difference": True},
    {"max_time_difference": float("nan")}, {"max_time_difference": float("inf")},
    {"max_time_difference": "0.5"},
])
def test_invalid_timing_configuration_fails_before_profile_mutation(profile_module, kwargs):
    cfg = _config()
    with pytest.raises(ValueError, match="cross_time_mode|max_time_difference"):
        _apply(profile_module, cfg, "shallow", **kwargs)
    assert cfg.scene.terrain.terrain_generator is None
    assert cfg.curriculum.terrain_levels is not None


@pytest.mark.parametrize("mutate", [
    lambda cfg: setattr(cfg, "walking_cross_time_mode", "legacy"),
    lambda cfg: setattr(cfg, "walking_max_time_difference", .3),
    lambda cfg: cfg.rewards.feet_air_time_variance.params.update(cross_time_mode="legacy"),
    lambda cfg: cfg.rewards.feet_air_time_variance.params.update(max_time_difference=.3),
    lambda cfg: cfg.rewards.feet_air_time_variance.params.pop("cross_time_mode"),
    lambda cfg: setattr(cfg.rewards, "feet_air_time_variance", None),
])
def test_post_init_timing_relabel_is_rejected(profile_module, mutate):
    cfg = _resolved_profile(profile_module)
    mutate(cfg)
    with pytest.raises(ValueError, match="Invalid resolved walking profile"):
        profile_module.validate_walking_profile(cfg)


def test_resolved_timing_limit_cannot_use_boolean_equality_to_bypass_validation(profile_module):
    cfg = _resolved_profile(profile_module, max_time_difference=1.)
    cfg.rewards.feet_air_time_variance.params["max_time_difference"] = True
    with pytest.raises(ValueError, match="walking_max_time_difference"):
        profile_module.validate_walking_profile(cfg)

