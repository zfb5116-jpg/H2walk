import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg


CTRLLAB_ROOT = Path(__file__).resolve().parents[7]
DEFAULT_H2_USD_PATH = CTRLLAB_ROOT / "assets" / "isaaclab" / "unitree_robots" / "h2" / "h2.usd"
H2_USD_PATH = os.environ.get("CTRLLAB_H2_USD_PATH", str(DEFAULT_H2_USD_PATH))
H2_HEIGHT_SCANNER_PRIM_PATH = os.environ.get("CTRLLAB_H2_HEIGHT_SCANNER_PRIM_PATH", "{ENV_REGEX_NS}/Robot/pelvis")
H2_LIDAR_RECON_CHECKPOINT = os.environ.get(
    "CTRLLAB_H2_LIDAR_RECON_CHECKPOINT",
    "logs/rsl_rl/stairs_l0b2_20261020/lidar_reconstructor_phase1.pt",
)

# Joint order follows the official Unitree MuJoCo H2 model in unitree_rl_mjlab.
H2_LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
]

H2_HIP_KNEE_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
]

H2_ANKLE_ROLL_JOINT_NAMES = [
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

H2_ANKLE_PITCH_JOINT_NAMES = [
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
]

H2_WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

H2_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

H2_SHOULDER_ELBOW_WRIST_ROLL_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

H2_WRIST_PITCH_YAW_JOINT_NAMES = [
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

H2_JOINT_NAMES = H2_LEG_JOINT_NAMES + H2_WAIST_JOINT_NAMES + H2_ARM_JOINT_NAMES
H2_NUM_ACTIONS = len(H2_JOINT_NAMES)

H2_FOOT_BODY_NAMES = ["left_ankle_pitch_link", "right_ankle_pitch_link"]
H2_BASE_BODY_NAMES = ["pelvis"]
H2_FALL_CONTACT_BODY_NAMES = [
    ".*pelvis.*",
    ".*torso.*",
    ".*waist.*",
    ".*head.*",
]
H2_UNDESIRED_CONTACT_BODY_NAMES = [
    ".*pelvis.*",
    ".*torso.*",
    ".*waist.*",
    ".*hip.*",
    ".*knee.*",
    ".*shoulder.*",
    ".*elbow.*",
    ".*head.*",
]

H2_STANDING_JOINT_POS = {
    "left_hip_pitch_joint": -0.25,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.5,
    "left_ankle_roll_joint": 0.0,
    "left_ankle_pitch_joint": -0.25,
    "right_hip_pitch_joint": -0.25,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.5,
    "right_ankle_roll_joint": 0.0,
    "right_ankle_pitch_joint": -0.25,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.87,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.87,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


# Match the H2 home keyframe in Unitree's official RL config:
# https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/assets/robots/unitree_h2/h2_constants.py
UNITREE_H2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=H2_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Match the Unitree H2 home keyframe used by the official RL config.
        pos=(0.0, 0.0, 1.03),
        joint_pos=H2_STANDING_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_knee": DCMotorCfg(
            joint_names_expr=H2_HIP_KNEE_JOINT_NAMES,
            effort_limit=360.0,
            saturation_effort=360.0,
            velocity_limit=30.0,
            stiffness=200.0,
            damping=4.0,
            friction=0.01,
        ),
        "ankle_roll": DCMotorCfg(
            joint_names_expr=H2_ANKLE_ROLL_JOINT_NAMES,
            effort_limit=19.0,
            saturation_effort=19.0,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.01,
        ),
        "ankle_pitch": DCMotorCfg(
            joint_names_expr=H2_ANKLE_PITCH_JOINT_NAMES,
            effort_limit=66.0,
            saturation_effort=66.0,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.01,
        ),
        "waist": DCMotorCfg(
            joint_names_expr=H2_WAIST_JOINT_NAMES,
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=20.0,
            stiffness=150.0,
            damping=3.0,
            friction=0.01,
        ),
        "shoulder_elbow_wrist_roll": DCMotorCfg(
            joint_names_expr=H2_SHOULDER_ELBOW_WRIST_ROLL_JOINT_NAMES,
            effort_limit=54.0,
            saturation_effort=54.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.01,
        ),
        "wrist_pitch_yaw": DCMotorCfg(
            joint_names_expr=H2_WRIST_PITCH_YAW_JOINT_NAMES,
            effort_limit=25.0,
            saturation_effort=25.0,
            velocity_limit=20.0,
            stiffness=20.0,
            damping=1.0,
            friction=0.01,
        ),
    },
)
