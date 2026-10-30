"""Independent policy-owned H2 ascent task; existing flat/reference tasks are unchanged."""

from copy import deepcopy

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .assets import H2_LEG_JOINT_NAMES
from .direct_leg_action import DirectLegPositionAction
from .robot_env_cfg import RobotStairsEnvCfg
from .config.terminations_cfg import TerminationsCfg
from .stair_ascent_safety import stair_below_floor, stair_out_of_tile
from .stair_ascent_terrain import ascent_stair_generator
from . import stair_ascent_rewards as rewards


@configclass
class StairAscentRewardsCfg:
    world_forward = RewTerm(func=rewards.stair_signed_forward, weight=12.0)
    heading_error = RewTerm(func=rewards.stair_heading_error, weight=-1.0)
    platform_hold = RewTerm(func=rewards.stair_platform_hold, weight=1.5)
    potential = RewTerm(func=rewards.stair_event, weight=1.0, params={"name": "potential"})
    first_valid_foot = RewTerm(func=rewards.stair_event, weight=3.0, params={"name": "foot"})
    bilateral_level = RewTerm(func=rewards.stair_event, weight=8.0, params={"name": "pair"})
    high_foot_transfer = RewTerm(func=rewards.stair_event, weight=3.0, params={"name": "transfer"})
    stable_top = RewTerm(func=rewards.stair_event, weight=30.0, params={"name": "top"})
    failure = RewTerm(func=rewards.stair_failure, weight=-50.0)
    stance_slip = RewTerm(func=rewards.stair_stance_slip, weight=-0.3)
    tilt_margin = RewTerm(func=rewards.stair_tilt_margin, weight=-0.5)
    drive_request = RewTerm(func=rewards.stair_drive_request, weight=-0.001)
    target_change = RewTerm(func=rewards.stair_target_change, weight=-0.002)
    native_overspeed = RewTerm(func=rewards.stair_native_overspeed, weight=-0.1)
    soft_limit_excess = RewTerm(func=rewards.stair_soft_limit_excess, weight=-5.0)


@configclass
class StairAscentCurriculumCfg:
    terrain_levels = CurrTerm(func=rewards.stair_height_curriculum)
    lin_vel_cmd_levels = None


@configclass
class StairAscentTerminationsCfg(TerminationsCfg):
    # The original five terms remain intact. These are immediate failures,
    # never successful timeouts or a standing-height/posture prescription.
    stair_below_floor = DoneTerm(func=stair_below_floor)
    stair_out_of_tile = DoneTerm(func=stair_out_of_tile)


@configclass
class StairAscentEnvCfg(RobotStairsEnvCfg):
    # These class defaults are visible during RobotStairsEnvCfg.__post_init__,
    # preventing its reference-only flat terrain replacement.
    contact_gait_reference = None
    stair_reference = None
    stance_preload_torques = {}
    stance_orientation_feedback = None
    lateral_com_feedback = None
    progressive_walking_curriculum = False
    stair_minimum_height: float = 0.03
    stair_maximum_height: float = 0.15
    stair_height_levels: int = 8
    stair_shaping_gamma: float = 0.99
    stair_tread_depth: float = 0.35
    stair_steps: int = 6
    stair_width: float = 2.4
    stair_approach_distance: float = 1.4
    stair_below_floor_tolerance_m: float = 0.05
    stair_domain_safety_inset_m: float = 0.10

    def __post_init__(self):
        # Reuse the measured native gains/limits/delay and existing 29-joint
        # proprioception. Replace objectives only after the parent finishes.
        super().__post_init__()
        self.scene.num_envs = 1024
        self.episode_length_s = 90.0
        self.scene.terrain.terrain_generator = ascent_stair_generator(
            levels=self.stair_height_levels, minimum=self.stair_minimum_height,
            maximum=self.stair_maximum_height, depth=self.stair_tread_depth,
            steps=self.stair_steps, width=self.stair_width, approach=self.stair_approach_distance)
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum = StairAscentCurriculumCfg()
        self.rewards = StairAscentRewardsCfg()
        self.terminations = StairAscentTerminationsCfg()
        action = self.actions.JointPositionAction
        action.class_type = DirectLegPositionAction
        action.joint_names = H2_LEG_JOINT_NAMES
        action.preserve_order = True
        action.scale = 1.0
        action.use_default_offset = False
        action.clip = None
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_standing_envs = 0.0
        command.debug_vis = False
        command.resampling_time_range = (90.0, 90.0)
        for ranges in (command.ranges, command.limit_ranges):
            ranges.lin_vel_x = (0.3, 0.3)
            ranges.lin_vel_y = (0.0, 0.0)
            ranges.ang_vel_z = (0.0, 0.0)
            ranges.heading = (0.0, 0.0)
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        # Both network terrain slots receive the same real LiDAR-derived map.
        # Simulator stair geometry is restricted to rewards/curriculum/evaluation.
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scanner.params["side_noise_probability"] = 0.0
        self.observations.critic.height_scanner = deepcopy(self.observations.policy.height_scanner)
        for label, body in (("stair_ascent_left_ground", "left_ankle_pitch_link"),
                            ("stair_ascent_right_ground", "right_ankle_pitch_link")):
            setattr(self.scene, label, ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/"+body, history_length=3,
                update_period=self.sim.dt, filter_prim_paths_expr=["/World/ground/terrain/mesh"]))


@configclass
class StairAscentPlayEnvCfg(StairAscentEnvCfg):
    stair_minimum_height: float = 0.15
    stair_maximum_height: float = 0.15
    stair_height_levels: int = 2

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.curriculum.terrain_levels = None
        # Keep deterministic row construction; every row now has 15 cm risers.
        self.observations.policy.enable_corruption = False
