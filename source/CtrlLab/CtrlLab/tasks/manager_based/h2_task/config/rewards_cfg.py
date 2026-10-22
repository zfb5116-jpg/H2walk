import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .. import mdp as h2_mdp
from ..assets import (
    H2_ARM_JOINT_NAMES,
    H2_BASE_BODY_NAMES,
    H2_FALL_CONTACT_BODY_NAMES,
    H2_FOOT_BODY_NAMES,
    H2_LEG_JOINT_NAMES,
    H2_UNDESIRED_CONTACT_BODY_NAMES,
    H2_WAIST_JOINT_NAMES,
)


@configclass
class RewardsCfg:
    """Reward terms for the H2 locomotion MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=8.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_forward_progress = RewTerm(
        func=h2_mdp.track_forward_velocity,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-2.0)
    base_lateral_velocity = RewTerm(
        func=h2_mdp.base_lateral_velocity_l2,
        weight=-1.2,
        params={"command_name": "base_velocity", "command_threshold": 0.15},
    )
    base_lateral_acc = RewTerm(
        func=h2_mdp.base_lateral_acc_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=H2_BASE_BODY_NAMES)},
    )
    base_acc = RewTerm(
        func=h2_mdp.base_acc_l2,
        weight=-2.5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=H2_BASE_BODY_NAMES)},
    )
    base_height = RewTerm(func=h2_mdp.base_height_l1, weight=-5.0, params={"target_height": 0.92})
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-7.0)

    # -- joint and action regularization
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-8e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    joint_power = RewTerm(func=h2_mdp.joint_power_norm, weight=-1e-4)
    joint_power_var = RewTerm(func=h2_mdp.joint_power_var, weight=-5e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.015)
    action_rate2 = RewTerm(func=h2_mdp.action_smooth_l2, weight=-0.02)

    # -- posture
    joint_leg_deviation = RewTerm(
        func=h2_mdp.joint_deviation_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=H2_LEG_JOINT_NAMES)},
    )
    joint_waist_deviation = RewTerm(
        func=h2_mdp.joint_deviation_l2,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=H2_WAIST_JOINT_NAMES)},
    )
    joint_arm_deviation = RewTerm(
        func=h2_mdp.joint_deviation_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=H2_ARM_JOINT_NAMES)},
    )
    # -- joint limits
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-10.0, params={"soft_ratio": 0.75})
    dof_tor_limits = RewTerm(func=h2_mdp.joint_tor_limits, weight=-10.0, params={"soft_ratio": 0.75})

    # -- feet
    feet_air_time = RewTerm(
        func=h2_mdp.feet_air_time,
        weight=0.45,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_FOOT_BODY_NAMES, preserve_order=True),
            "command_name": "base_velocity",
            "threshold": 0.45,
        },
    )
    feet_contact_time = RewTerm(
        func=h2_mdp.feet_contact_time,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_FOOT_BODY_NAMES, preserve_order=True),
            "command_name": "base_velocity",
            "threshold": 0.45,
        },
    )
    feet_air_time_variance = RewTerm(
        func=h2_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_FOOT_BODY_NAMES, preserve_order=True),
        },
    )
    feet_air_vel = RewTerm(
        func=h2_mdp.feet_air_vel_norm,
        weight=-0.04,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_FOOT_BODY_NAMES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=H2_FOOT_BODY_NAMES, preserve_order=True),
        },
    )
    feet_slide = RewTerm(
        func=h2_mdp.feet_slide,
        weight=-0.8,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_FOOT_BODY_NAMES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=H2_FOOT_BODY_NAMES, preserve_order=True),
        },
    )
    feet_vel = RewTerm(
        func=h2_mdp.feet_vel_norm,
        weight=-2e-3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=H2_FOOT_BODY_NAMES, preserve_order=True)},
    )
    feet_acc = RewTerm(
        func=h2_mdp.feet_acc_norm,
        weight=-8e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=H2_FOOT_BODY_NAMES, preserve_order=True)},
    )
    feet_stumble = RewTerm(
        func=h2_mdp.feet_stumble_norm,
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_FOOT_BODY_NAMES, preserve_order=True)},
    )
    feet_clearance = RewTerm(
        func=h2_mdp.feet_clearance_l2,
        weight=-4.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_FOOT_BODY_NAMES, preserve_order=True),
            "target_feet_height": -0.70,
        },
    )
    feet_swing_height = RewTerm(
        func=h2_mdp.feet_swing_height,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_FOOT_BODY_NAMES, preserve_order=True),
            "target_feet_height": -0.67,
            "height_std": 0.10,
        },
    )
    feet_stair_clearance = RewTerm(
        func=h2_mdp.feet_stair_clearance,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_FOOT_BODY_NAMES, preserve_order=True),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "clearance": 0.04,
            "min_rise": 0.04,
            "max_rise": 0.24,
            "height_std": 0.12,
        },
    )
    support_force_z_axis = RewTerm(
        func=h2_mdp.support_force_z_axis_norm,
        weight=-1e-4,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_FOOT_BODY_NAMES, preserve_order=True)},
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_UNDESIRED_CONTACT_BODY_NAMES),
        },
    )
    fall_body_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-12.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_FALL_CONTACT_BODY_NAMES),
        },
    )
    stand_still = RewTerm(
        func=h2_mdp.stand_still_joint_deviation_l1,
        weight=-2.0,
        params={"command_name": "base_velocity", "command_threshold": 0.01},
    )

    # Four-foot Go2-only terms are intentionally disabled for H2.
    feet_mirror_diff = None
    floating_base = None
    wbc_cmd_ddq = None
