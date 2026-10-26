from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .assets import (
    H2_ANKLE_PITCH_JOINT_NAMES,
    H2_ANKLE_ROLL_JOINT_NAMES,
    H2_ARM_JOINT_NAMES,
    H2_FOOT_BODY_NAMES,
    H2_HEIGHT_SCANNER_PRIM_PATH,
    H2_LIDAR_RECON_CHECKPOINT,
    H2_HIP_KNEE_JOINT_NAMES,
    H2_LEG_JOINT_NAMES,
    H2_SHOULDER_ELBOW_WRIST_ROLL_JOINT_NAMES,
    H2_STANDING_JOINT_POS,
    H2_WAIST_JOINT_NAMES,
    H2_WRIST_PITCH_YAW_JOINT_NAMES,
    UNITREE_H2_CFG,
)
from .config import *
from .lidar_hb_pattern import YuHbLidarPatternCfg
from .mdp.observations import reconstructed_lidar_height_scan


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a Unitree H2 humanoid."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot: ArticulationCfg = UNITREE_H2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.03),
            joint_pos=H2_STANDING_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hip_knee": DelayedPDActuatorCfg(
                joint_names_expr=H2_HIP_KNEE_JOINT_NAMES,
                effort_limit=360.0,
                effort_limit_sim=360.0,
                velocity_limit=30.0,
                velocity_limit_sim=30.0,
                stiffness=200.0,
                damping=4.0,
                friction=0.01,
                armature=0.01,
                min_delay=0,
                max_delay=6,
            ),
            "ankle_roll": DelayedPDActuatorCfg(
                joint_names_expr=H2_ANKLE_ROLL_JOINT_NAMES,
                effort_limit=19.0,
                effort_limit_sim=19.0,
                velocity_limit=30.0,
                velocity_limit_sim=30.0,
                stiffness=40.0,
                damping=2.0,
                friction=0.01,
                armature=0.01,
                min_delay=0,
                max_delay=6,
            ),
            "ankle_pitch": DelayedPDActuatorCfg(
                joint_names_expr=H2_ANKLE_PITCH_JOINT_NAMES,
                effort_limit=66.0,
                effort_limit_sim=66.0,
                velocity_limit=30.0,
                velocity_limit_sim=30.0,
                stiffness=40.0,
                damping=2.0,
                friction=0.01,
                armature=0.01,
                min_delay=0,
                max_delay=6,
            ),
            "waist": DelayedPDActuatorCfg(
                joint_names_expr=H2_WAIST_JOINT_NAMES,
                effort_limit=120.0,
                effort_limit_sim=120.0,
                velocity_limit=20.0,
                velocity_limit_sim=20.0,
                stiffness=150.0,
                damping=3.0,
                friction=0.01,
                armature=0.01,
                min_delay=0,
                max_delay=6,
            ),
            "shoulder_elbow_wrist_roll": DelayedPDActuatorCfg(
                joint_names_expr=H2_SHOULDER_ELBOW_WRIST_ROLL_JOINT_NAMES,
                effort_limit=54.0,
                effort_limit_sim=54.0,
                velocity_limit=20.0,
                velocity_limit_sim=20.0,
                stiffness=40.0,
                damping=2.0,
                friction=0.01,
                armature=0.005,
                min_delay=0,
                max_delay=6,
            ),
            "wrist_pitch_yaw": DelayedPDActuatorCfg(
                joint_names_expr=H2_WRIST_PITCH_YAW_JOINT_NAMES,
                effort_limit=25.0,
                effort_limit_sim=25.0,
                velocity_limit=20.0,
                velocity_limit_sim=20.0,
                stiffness=20.0,
                damping=1.0,
                friction=0.01,
                armature=0.005,
                min_delay=0,
                max_delay=6,
            ),
        },
    )

    height_scanner = RayCasterCfg(
        prim_path=H2_HEIGHT_SCANNER_PRIM_PATH,
        # Center the map 0.3 m ahead of the pelvis: x in [-0.5, 1.1] m.
        offset=RayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 20.0)),
        ray_alignment="yaw",
        # TerProg H^b: 33 x 17 samples, 5 cm resolution, 1.6 x 0.8 m window.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    lidar: RayCasterCfg | None = None
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sm60_camera_left: CameraCfg | None = None
    sm60_camera_right: CameraCfg | None = None
    sm60_camera_housing: AssetBaseCfg | None = None
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class RobotRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for H2 locomotion velocity tracking."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=2048, env_spacing=3.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    feet_body_names = H2_FOOT_BODY_NAMES
    command_progress_min_speed: float = 0.1
    command_progress_max_step_speed: float = 5.0

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.resampling_time_range = (8, 12)
        self.rewards.track_lin_vel_xy.weight = 8.0
        self.rewards.track_ang_vel_z.weight = 2.0

        self.rewards.base_linear_velocity.weight = -2.0
        self.rewards.base_angular_velocity.weight = -2.0
        self.rewards.base_acc.weight = -3e-4
        self.rewards.base_height.weight = -5.0
        self.rewards.flat_orientation.weight = -12.0
        self.rewards.base_lateral_velocity.weight = -1.2
        self.rewards.base_lateral_acc.weight = -1e-3

        self.rewards.action_rate.weight = -0.015
        self.rewards.action_rate2.weight = -0.02
        self.rewards.joint_vel.weight = -8e-4
        self.rewards.joint_acc.weight = -2e-7
        self.rewards.joint_torques.weight = -1e-5
        self.rewards.joint_power.weight = -1e-4
        self.rewards.joint_power_var.weight = -8e-5
        self.rewards.joint_leg_deviation.weight = -0.2
        self.rewards.joint_waist_deviation.weight = -1.0
        self.rewards.joint_arm_deviation.weight = -0.2
        self.rewards.dof_pos_limits.weight = -10.0
        self.rewards.dof_vel_limits.weight = -10.0
        self.rewards.dof_tor_limits.weight = -10.0

        self.rewards.feet_air_time.weight = 0.45
        self.rewards.feet_contact_time.weight = 0.2
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_vel.weight = -0.04
        self.rewards.feet_slide.weight = -0.8
        self.rewards.feet_stumble.weight = -1.5
        self.rewards.feet_vel.weight = -2e-3
        self.rewards.feet_acc.weight = -8e-4
        self.rewards.feet_clearance.weight = -4.0
        self.rewards.feet_swing_height.weight = 0.75
        self.rewards.track_forward_progress.weight = 2.0
        self.rewards.feet_stair_clearance.weight = 2.0
        self.rewards.support_force_z_axis.weight = -2e-4

        self.rewards.stand_still.weight = -2.0
        self.rewards.undesired_contacts.weight = -2.0

    def _enable_sm60_cameras(self):
        camera_spawn = sim_utils.PinholeCameraCfg(
            focal_length=1.0,
            horizontal_aperture=6.155,
            vertical_aperture=3.03,
            clipping_range=(0.05, 30.0),
        )
        camera_rotation = (0.7071068, 0.0, 0.7071068, 0.0)
        camera_common = {
            "width": 320,
            "height": 200,
            "data_types": ["rgb"],
            "update_period": 1.0 / 30.0,
            "spawn": camera_spawn,
        }
        self.scene.sm60_camera_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head/sm60_left",
            offset=CameraCfg.OffsetCfg(
                pos=(0.015, 0.0, 0.03), rot=camera_rotation, convention="ros"
            ),
            **camera_common,
        )
        self.scene.sm60_camera_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head/sm60_right",
            offset=CameraCfg.OffsetCfg(
                pos=(0.015, 0.0, -0.03), rot=camera_rotation, convention="ros"
            ),
            **camera_common,
        )
        self.scene.sm60_camera_housing = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head/sm60_housing",
            spawn=sim_utils.CuboidCfg(
                size=(0.0287, 0.0382, 0.1026),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08)),
            ),
        )


@configclass
class RobotRoughPlayEnvCfg(RobotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.max_init_terrain_level = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self._enable_sm60_cameras()


def _enable_stairs_lidar_reconstruction(cfg):
    """Enable frozen cross-embodiment perception with explicit map semantics.

    Reliability fusion is opt-in; both paths return the height_scan convention
    before ObservationManager applies the existing clipping and 0.2 scaling.
    The zero velocity prior is explicit until a deployable estimate is supplied.
    """
    lidar_pattern = YuHbLidarPatternCfg(
        channels=16,
        vertical_fov_range=(-45.0, 14.0),
        horizontal_fov_range=(-180.0, 180.0),
        horizontal_res=1.6,
        origin=(0.0, 0.0, 0.16),
        box_min=(-0.55, -0.45, -0.65),
        box_max=(1.15, 0.45, 0.55),
        min_range=0.1,
        max_range=2.0,
        z_slices=9,
    )
    cfg.scene.lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        mesh_prim_paths=["/World/ground"],
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.16)),
        ray_alignment="base",
        pattern_cfg=lidar_pattern,
        max_distance=2.0,
        debug_vis=False,
    )
    cfg.scene.lidar.update_period = cfg.decimation * cfg.sim.dt
    cfg.observations.policy.height_scanner = ObsTerm(
        func=reconstructed_lidar_height_scan,
        clip=(-1.0, 1.0),
        scale=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "checkpoint": str(H2_LIDAR_RECON_CHECKPOINT),
            "use_reliable_fusion": bool(cfg.lidar_reliable_fusion),
            "output_convention": "height_scan",
            "height_scan_offset": 0.5,
            "reference_sensor_cfg": SceneEntityCfg("height_scanner"),
            "velocity_source": "zero",
        },
    )


def _configure_walking_height_rewards(cfg):
    """Choose shared foot-height shaping without changing observations or weights."""
    if cfg.walking_height_mode not in {"terrain", "body_legacy"}:
        raise ValueError("walking_height_mode must be 'terrain' or 'body_legacy'")
    if cfg.walking_foot_signal_mode not in {"aligned", "legacy"}:
        raise ValueError("walking_foot_signal_mode must be 'aligned' or 'legacy'")
    for term in (cfg.rewards.feet_clearance, cfg.rewards.feet_swing_height):
        term.params.update({
            "height_mode": cfg.walking_height_mode,
            "min_clearance": 0.03,
            "terrain_sensor_name": "height_scanner",
            "ground_query_radius": 0.10,
        })
    aligned_terrain = cfg.walking_height_mode == "terrain" and cfg.walking_foot_signal_mode == "aligned"
    cfg.rewards.feet_swing_height.params.update({
        "height_std": 0.03 if aligned_terrain else 0.10,
        "movement_threshold": 0.1 if aligned_terrain else 0.15,
        "movement_gate_inclusive": aligned_terrain,
    })
    cfg.rewards.feet_stair_clearance.params.update({
        "command_gate_mode": "progress" if cfg.walking_foot_signal_mode == "aligned" else "legacy",
        "min_forward_command": 0.1,
        "max_lateral_command": 0.15,
        "max_yaw_command": 0.3,
    })


def _configure_walking_tracking_rewards(cfg):
    """Select velocity-tracking bandwidth without changing weights or commands."""
    if cfg.walking_tracking_mode not in ("absolute", "command_relative", "command_relative_mixed"):
        raise ValueError("walking_tracking_mode must be 'absolute', 'command_relative' or 'command_relative_mixed'")
    if not 0.0 <= cfg.walking_tracking_relative_weight <= 1.0:
        raise ValueError("walking_tracking_relative_weight must be finite and in [0, 1]")
    for term in (cfg.rewards.track_lin_vel_xy, cfg.rewards.track_forward_progress):
        term.params.update({
            "tracking_mode": cfg.walking_tracking_mode,
            "min_command_speed": 0.1,
            "relative_weight": cfg.walking_tracking_relative_weight,
        })


def _configure_walking_command_curriculum(cfg):
    """Require per-column progress for training while retaining a global baseline."""
    if cfg.walking_command_curriculum_mode not in {"global", "terrain_columns"}:
        raise ValueError("walking_command_curriculum_mode must be 'global' or 'terrain_columns'")
    cfg.curriculum.lin_vel_cmd_levels.params.update({
        "grouping": cfg.walking_command_curriculum_mode,
        "min_group_episodes": 32,
    })


def _configure_stairs_height_profile(cfg):
    """Apply a height profile only to an already copied per-instance generator."""
    if cfg.stairs_height_profile not in {"progressive_20cm", "legacy_14cm"}:
        raise ValueError("stairs_height_profile must be 'progressive_20cm' or 'legacy_14cm'")
    progressive = cfg.stairs_height_profile == "progressive_20cm"
    if progressive:
        for terrain_name in ("stairs_high", "stairs_down_high"):
            cfg.scene.terrain.terrain_generator.sub_terrains[terrain_name].step_height_range = (0.08, 0.20)
    cfg.rewards.feet_stair_clearance.params.update({
        "max_rise": 0.24 if progressive else 0.16,
        # Keep shallow-step signal magnitude while widening the admissible rise.
        "rise_weight_scale": 0.16 if progressive else None,
    })


@configclass
class RobotStairsEnvCfg(RobotRoughEnvCfg):
    """Train one scanned-terrain policy on ordinary ground and stair terrains."""

    lidar_reliable_fusion: bool = False
    use_lidar_reconstruction: bool = False
    stairs_clearance_target_mode: str = "nearest"
    stairs_clearance_height_reference: str = "sole"
    stairs_mixed_terrain: bool = True
    walking_height_mode: str = "terrain"
    walking_tracking_mode: str = "command_relative_mixed"
    walking_tracking_relative_weight: float = 0.75
    walking_foot_signal_mode: str = "aligned"
    stairs_height_profile: str = "progressive_20cm"
    walking_command_curriculum_mode: str = "terrain_columns"

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = deepcopy(
            MIXED_WALK_TERRAINS_CFG if self.stairs_mixed_terrain else STAIR_ROUGH_TERRAINS_CFG
        )
        # Preserve upstream's shallow start and fall detection for H2.
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.curriculum = True
        self.rewards.feet_stair_clearance.weight = 2.5
        self.rewards.feet_stair_clearance.params.update(
            {"clearance": 0.035, "min_rise": 0.02, "max_rise": 0.16,
             "height_std": 0.08, "target_mode": self.stairs_clearance_target_mode,
             "height_reference": self.stairs_clearance_height_reference}
        )
        _configure_stairs_height_profile(self)
        _configure_walking_height_rewards(self)
        _configure_walking_tracking_rewards(self)
        _configure_walking_command_curriculum(self)
        if self.use_lidar_reconstruction:
            _enable_stairs_lidar_reconstruction(self)


@configclass
class RobotStairsPlayEnvCfg(RobotRoughPlayEnvCfg):
    """Play the same scanned-terrain policy, retaining ordinary terrain coverage."""

    lidar_reliable_fusion: bool = False
    use_lidar_reconstruction: bool = False
    stairs_clearance_target_mode: str = "nearest"
    stairs_clearance_height_reference: str = "sole"
    stairs_mixed_terrain: bool = True
    walking_height_mode: str = "terrain"
    walking_tracking_mode: str = "command_relative_mixed"
    walking_tracking_relative_weight: float = 0.75
    walking_foot_signal_mode: str = "aligned"
    stairs_height_profile: str = "progressive_20cm"
    walking_command_curriculum_mode: str = "global"

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = deepcopy(
            MIXED_WALK_TERRAINS_CFG if self.stairs_mixed_terrain else STAIR_ROUGH_TERRAINS_CFG
        )
        self.scene.terrain.max_init_terrain_level = 3
        # Generator curriculum fixes terrain families by column; runtime terrain
        # progression is a separate manager term. Disable only that term for
        # mixed play, so columns cannot randomly lose a family at evaluation.
        self.scene.terrain.terrain_generator.curriculum = bool(self.stairs_mixed_terrain)
        if self.stairs_mixed_terrain:
            self.curriculum.terrain_levels = None
        self.rewards.feet_stair_clearance.weight = 2.5
        self.rewards.feet_stair_clearance.params.update(
            {"clearance": 0.035, "min_rise": 0.02, "max_rise": 0.16,
             "height_std": 0.08, "target_mode": self.stairs_clearance_target_mode,
             "height_reference": self.stairs_clearance_height_reference}
        )
        _configure_stairs_height_profile(self)
        _configure_walking_height_rewards(self)
        _configure_walking_tracking_rewards(self)
        _configure_walking_command_curriculum(self)
        if self.use_lidar_reconstruction:
            _enable_stairs_lidar_reconstruction(self)


@configclass
class RobotFlatEnvCfg(RobotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # The flat task has no ray caster, so remove the terrain observation from both groups.
        self.observations.policy.height_scanner = None
        self.observations.critic.height_scanner = None
        self.rewards.feet_stair_clearance = None
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.resampling_time_range = (10, 10)


@configclass
class RobotFlatPlayEnvCfg(RobotFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self._enable_sm60_cameras()
