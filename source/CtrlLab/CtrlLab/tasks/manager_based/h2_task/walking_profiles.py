"""Configuration-only profiles for a scanned flat-to-stairs training protocol.

This module does not import Isaac Lab or create terrain. Apply a profile once to
a fresh configuration after the existing Stairs post-init. A different stage
requires a new environment: no live mesh replacement or automatic promotion is
performed here. The original Stairs task remains the baseline.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from .staged_curriculum import STAGE_HEIGHT_RANGES, STAGE_ORDER, stage_column_families


def _validate_scan(cfg: Any) -> None:
    if getattr(cfg, "use_lidar_reconstruction", False) or getattr(cfg.scene, "lidar", None) is not None:
        raise ValueError("Walking profiles require the GT scan, not lidar reconstruction")
    scanner = getattr(cfg.scene, "height_scanner", None)
    if scanner is None:
        raise ValueError("Walking profiles require the height scanner")
    if tuple(getattr(scanner, "mesh_prim_paths", ())) != (cfg.scene.terrain.prim_path,):
        raise ValueError("GT scan and filtered support must reference the same terrain root")
    for group in (cfg.observations.policy, cfg.observations.critic):
        if getattr(group, "height_scanner", None) is None:
            raise ValueError("Both policy and critic must retain the GT scan observation")
    pattern = getattr(scanner, "pattern_cfg", None)
    if (pattern is None or not math.isclose(getattr(pattern, "resolution", 0.0), .05)
            or tuple(getattr(pattern, "size", ())) != (1.6, .8)):
        raise ValueError("Walking profiles require the existing 561-point GT scan pattern")


def validate_walking_profile(cfg: Any) -> None:
    """Check resolved staged configuration before any environment is built.

    Hydra-style overrides can change a stage or ablation flag after post-init
    has already materialized terrain, events and commands. This read-only
    guard rejects that inconsistent state rather than silently reapplying a
    profile or changing evaluation conditions. Constructor/subclass arguments
    which were present when post-init ran remain supported.
    """
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    try:
        stage = cfg.walking_stage
        require(isinstance(stage, str) and stage in STAGE_ORDER, "Unknown resolved walking stage")
        heights = STAGE_HEIGHT_RANGES[stage]
        tolerance = getattr(cfg, "stage_support_height_tolerance", .015)
        require(not isinstance(tolerance, bool) and isinstance(tolerance, (int, float))
                and math.isfinite(tolerance) and tolerance > 0, "Invalid stage support height tolerance")
        if stage != "bootstrap":
            require(tolerance < heights[0], "Support tolerance must remain smaller than the stage's minimum riser")
        families = stage_column_families(stage)
        require(tuple(cfg.walking_stage_height_range) == heights, "Stage height metadata disagrees with stage")
        require(tuple(cfg.walking_stage_column_families) == families, "Stage column metadata disagrees with stage")
        _validate_scan(cfg)
        terrain = cfg.scene.terrain
        require(terrain.terrain_type == "generator", "Stage terrain must use the generator protocol")
        generator = terrain.terrain_generator
        require(generator is not None, "Stage terrain generator is missing")
        require(tuple(generator.size) == (10.0, 10.0), "Resolved terrain tile size disagrees with profile")
        require(type(generator.num_rows) is int and generator.num_rows == 1, "Stage must retain one terrain row")
        require(type(generator.num_cols) is int and generator.num_cols == len(families), "Resolved column count disagrees with stage")
        require(generator.curriculum is True and generator.use_cache is False, "Fixed-column generator settings changed")
        require(terrain.max_init_terrain_level == 0, "Stage must start on its only terrain row")
        expected_names = ("flat",) if stage == "bootstrap" else ("flat", "up", "down")
        require(tuple(generator.sub_terrains) == expected_names, "Resolved terrain family order disagrees with stage")
        flat = generator.sub_terrains["flat"]
        require(tuple(flat.slope_range) == (0.0, 0.0), "Ordinary terrain is no longer flat")
        require(flat.proportion == (1.0 if stage == "bootstrap" else 6.0), "Flat column proportion changed")
        if stage != "bootstrap":
            for family in ("up", "down"):
                sub_terrain = generator.sub_terrains[family]
                require(tuple(sub_terrain.step_height_range) == heights, f"Resolved {family} height range disagrees with stage")
                require(sub_terrain.proportion == 2.0, f"Resolved {family} column proportion changed")
                require(sub_terrain.step_width == .35 and sub_terrain.platform_width == 3.0
                        and sub_terrain.border_width == .3 and sub_terrain.holes is False,
                        f"Resolved {family} stair geometry changed")
        require(cfg.curriculum.terrain_levels is None and cfg.curriculum.lin_vel_cmd_levels is None,
                "Legacy terrain or speed progression is enabled inside a fixed stage")

        speed = cfg.walking_forward_speed
        duration = cfg.episode_length_s
        require(not isinstance(speed, bool) and isinstance(speed, (int, float))
                and math.isfinite(speed) and speed >= .1, "Invalid resolved forward speed")
        require(not isinstance(duration, bool) and isinstance(duration, (int, float))
                and math.isfinite(duration) and duration > 0, "Invalid resolved episode duration")
        require(speed * duration + .1 < 10.0 / 2 - .3, "Resolved command path crosses a terrain tile")
        command = cfg.commands.base_velocity
        for ranges in (command.ranges, command.limit_ranges):
            require(tuple(ranges.lin_vel_x) == (speed, speed), "Resolved forward command disagrees with walking_forward_speed")
            for name in ("lin_vel_y", "ang_vel_z", "heading"):
                require(tuple(getattr(ranges, name)) == (0.0, 0.0), f"Resolved {name} command must remain zero")
        require(command.heading_command is False and command.rel_heading_envs == 0.0
                and command.rel_standing_envs == 0.0, "Resolved command mode disagrees with forward-only profile")
        require(tuple(command.resampling_time_range) == (duration, duration), "Resolved command duration disagrees with episode")
        pose = cfg.events.reset_base.params["pose_range"]
        require(tuple(pose["yaw"]) == (0.0, 0.0), "Reset yaw no longer aligns with the terrain frame")
        require(tuple(pose["x"]) == (-.1, .1) and tuple(pose["y"]) == (-.1, .1), "Reset XY bounds changed after profile resolution")

        flags = {name: getattr(cfg, "walking_" + name) for name in (
            "enable_push", "enable_domain_randomization", "enable_observation_noise", "enable_actuator_delay")}
        require(all(type(value) is bool for value in flags.values()), "Resolved ablation flags must be booleans")
        for event_name, flag_name in (
            ("push_robot", "enable_push"), ("physics_material", "enable_domain_randomization"),
            ("add_motor_pd_and_motor_strength", "enable_domain_randomization"),
        ):
            require((getattr(cfg.events, event_name) is not None) == flags[flag_name],
                    f"Resolved event {event_name} disagrees with its ablation flag")
        require(cfg.observations.policy.enable_corruption is flags["enable_observation_noise"],
                "Resolved observation corruption disagrees with its ablation flag")
        actuators = cfg.scene.robot.actuators
        require(bool(actuators), "Stage actuator configuration is empty")
        for name, actuator in actuators.items():
            require(type(actuator.min_delay) is int and type(actuator.max_delay) is int
                    and 0 <= actuator.min_delay <= actuator.max_delay,
                    f"Invalid resolved actuator delay for {name}")
            if not flags["enable_actuator_delay"]:
                require(actuator.min_delay == actuator.max_delay == 0,
                        f"Resolved actuator delay for {name} disagrees with disabled delay flag")
        if flags["enable_actuator_delay"]:
            require(any(actuator.max_delay > 0 for actuator in actuators.values()),
                    "Delay is labeled enabled but no actuator has nonzero delay")
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid resolved walking profile: {error}; construct a fresh profile with constructor arguments") from error


def apply_walking_profile(
    cfg: Any,
    stage: str,
    *,
    terrain_template: Any,
    flat_template: Any,
    stairs_up_template: Any,
    stairs_down_template: Any,
    forward_speed: float = .2,
    enable_push: bool = False,
    enable_domain_randomization: bool = False,
    enable_observation_noise: bool = False,
    enable_actuator_delay: bool = False,
) -> None:
    """Apply one explicit stage while preserving the scan/action architecture.

    ``stairs_up_template`` must be an *inverted* pyramid and
    ``stairs_down_template`` a regular pyramid: the existing spawn is at their
    center, and a positive world-X command moves outward from that platform.
    Family names describe that physical direction, not the source class name.

    Bootstrap has one flat column. Every stair stage has ten fixed columns:
    six flat, two uphill, and two downhill. The generator's curriculum flag is
    retained solely for deterministic column selection; all runtime terrain
    and command expansion terms are disabled. One generated row samples the
    stage's height range, so the actual tile step height is not a fixed value.

    The ablation flags retain the corresponding fresh baseline configuration
    when true, and disable it when false. They do not restore an event already
    removed by a different profile or Play parent. Train and Play should both
    derive from the same staged Train configuration and apply this helper.
    Checkpoints, normalizers, rewards, terminations and actions are untouched.
    """
    if not isinstance(stage, str) or stage not in STAGE_ORDER:
        raise ValueError(f"Unknown walking stage: {stage!r}; expected one of {STAGE_ORDER}")
    flags = (enable_push, enable_domain_randomization, enable_observation_noise, enable_actuator_delay)
    if any(type(value) is not bool for value in flags):
        raise ValueError("Walking profile ablation flags must be booleans")
    if isinstance(forward_speed, bool) or not isinstance(forward_speed, (int, float)):
        raise ValueError("forward_speed must be a finite speed in m/s")
    if not math.isfinite(forward_speed) or forward_speed < .1:
        raise ValueError("forward_speed must be finite and at least .1 m/s")
    duration = cfg.episode_length_s
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Walking profile episode duration must be finite and positive")
    # With zero yaw and a fixed command, the intended path including reset-X
    # variation must stay within this tile. This is a configuration bound, not
    # proof that an actual policy will obey the command or stay inside it.
    if forward_speed * duration + .1 >= 10.0 / 2 - .3:
        raise ValueError("Configured forward speed and episode duration would cross the terrain tile")
    _validate_scan(cfg)
    if cfg.events.reset_base is None:
        raise ValueError("Walking profile requires reset_base to align the command and terrain frames")
    if not enable_actuator_delay:
        for name, actuator in cfg.scene.robot.actuators.items():
            if not hasattr(actuator, "min_delay") or not hasattr(actuator, "max_delay"):
                raise ValueError(f"Actuator {name!r} has no explicit delay configuration")

    generator = deepcopy(terrain_template)
    generator.size = (10.0, 10.0)
    generator.num_rows = 1
    generator.num_cols = len(stage_column_families(stage))
    generator.curriculum = True
    generator.use_cache = False
    flat = deepcopy(flat_template)
    flat.slope_range = (0.0, 0.0)
    flat.proportion = 1.0 if stage == "bootstrap" else 6.0
    generator.sub_terrains = {"flat": flat}
    if stage != "bootstrap":
        for family, template in (("up", stairs_up_template), ("down", stairs_down_template)):
            terrain = deepcopy(template)
            terrain.proportion = 2.0
            terrain.step_height_range = STAGE_HEIGHT_RANGES[stage]
            terrain.step_width = .35
            terrain.platform_width = 3.0
            terrain.border_width = .3
            terrain.holes = False
            generator.sub_terrains[family] = terrain
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = generator
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None

    command = cfg.commands.base_velocity
    ranges = deepcopy(command.ranges)
    ranges.lin_vel_x = (float(forward_speed), float(forward_speed))
    ranges.lin_vel_y = ranges.ang_vel_z = ranges.heading = (0.0, 0.0)
    command.ranges = ranges
    command.limit_ranges = deepcopy(ranges)
    command.heading_command = False
    command.rel_heading_envs = 0.0
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (float(duration), float(duration))
    reset_params = deepcopy(cfg.events.reset_base.params)
    reset_params["pose_range"].update({"x": (-.1, .1), "y": (-.1, .1), "yaw": (0.0, 0.0)})
    cfg.events.reset_base.params = reset_params

    if not enable_push:
        cfg.events.push_robot = None
    if not enable_domain_randomization:
        cfg.events.physics_material = None
        cfg.events.add_motor_pd_and_motor_strength = None
    cfg.observations.policy.enable_corruption = enable_observation_noise
    if not enable_actuator_delay:
        for actuator in cfg.scene.robot.actuators.values():
            actuator.min_delay = actuator.max_delay = 0
    cfg.walking_stage = stage
    cfg.walking_stage_height_range = STAGE_HEIGHT_RANGES[stage]
    cfg.walking_stage_column_families = stage_column_families(stage)
