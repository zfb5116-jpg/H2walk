"""Opt-in contact diagnosis and scanned flat-to-stairs configurations.

Original Stairs and Flat task configurations remain unchanged. These classes
only configure sensors and the explicit staged protocol; importing them does
not create a scene or start training.
"""

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .assets import H2_FOOT_BODY_NAMES, H2_JOINT_NAMES
from .config.terrain_cfg import MIXED_WALK_TERRAINS_CFG, STAIR_ROUGH_TERRAINS_CFG
from .robot_env_cfg import RobotSceneCfg, RobotStairsEnvCfg
from .walking_profiles import apply_walking_profile


# Collision-bearing rigid links from the audited H2 USD, not all 29 joint
# names. Keep one filter per link: one Robot/.* expression would match N*B
# bodies rather than N clones of one named body in a batched PhysX view.
H2_DIAGNOSTIC_COLLISION_BODY_NAMES = (
    "head_yaw_link",
    "left_ankle_pitch_link",
    "left_elbow_link",
    "left_hand_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_wrist_yaw_link",
    "pelvis",
    "right_ankle_pitch_link",
    "right_elbow_link",
    "right_hand_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_wrist_yaw_link",
    "torso_link",
)


@configclass
class WalkingSceneCfg(RobotSceneCfg):
    """Reserve separate one-body filtered channels without changing observations."""

    pelvis_ground: ContactSensorCfg | None = None
    pelvis_robot: ContactSensorCfg | None = None
    left_foot_ground: ContactSensorCfg | None = None
    right_foot_ground: ContactSensorCfg | None = None


def _generated_terrain_collision_path(terrain):
    """Resolve the collider path of Isaac Lab's generator import protocol.

    TerrainImporter imports generated mesh as name='terrain'; the public
    create_prim_from_mesh creates its colliding Mesh child at '/mesh'. Neither
    parent Xform is a collision shape. Plane/USD import protocols need a
    separately verified path and are deliberately unsupported here.
    """
    if terrain.terrain_type != "generator":
        raise ValueError("Walking filtered contacts support only the generator terrain import protocol")
    path = getattr(terrain, "prim_path", None)
    if (not isinstance(path, str) or not path.startswith("/") or path.endswith("/")
            or not all(part.isidentifier() for part in path.split("/")[1:])):
        raise ValueError("Terrain prim_path must be an absolute, explicit USD prim path")
    return path + "/terrain/mesh"


def _configure_contact_diagnostic_sensors(cfg):
    if type(cfg.contact_source_diagnostics) is not bool or type(cfg.staged_evidence) is not bool:
        raise ValueError("Contact diagnostics and staged evidence flags must be booleans")
    active = cfg.contact_source_diagnostics or cfg.staged_evidence
    ground_filters = [_generated_terrain_collision_path(cfg.scene.terrain)] if active else []
    robot_filters = ["{ENV_REGEX_NS}/Robot/" + body
                     for body in H2_DIAGNOSTIC_COLLISION_BODY_NAMES if body != "pelvis"]
    names_and_bodies = (
        ("pelvis_ground", "pelvis", ground_filters),
        ("pelvis_robot", "pelvis", robot_filters),
        ("left_foot_ground", H2_FOOT_BODY_NAMES[0], ground_filters),
        ("right_foot_ground", H2_FOOT_BODY_NAMES[1], ground_filters),
    )
    for name, body, filters in names_and_bodies:
        sensor = None
        if active:
            # Filtered contact reporting is one-to-many in Isaac Lab v2.2.1.
            # Each primary selects exactly one rigid body in each environment.
            sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + body,
                filter_prim_paths_expr=list(filters),
                history_length=3,
                update_period=cfg.sim.dt,
                max_contact_data_count_per_prim=64,
                debug_vis=False,
            )
        setattr(cfg.scene, name, sensor)


@configclass
class RobotContactDiagnosticEnvCfg(RobotStairsEnvCfg):
    """Original scanned Stairs MDP with opt-in pre-reset contact diagnostics."""

    scene: WalkingSceneCfg = WalkingSceneCfg(num_envs=2048, env_spacing=3.5)
    contact_source_diagnostics: bool = True
    staged_evidence: bool = False
    walking_diagnostic_joint_names: tuple[str, ...] = tuple(H2_JOINT_NAMES)
    contact_diagnostic_threshold: float = 1.0
    contact_diagnostic_max_samples: int = 8
    stage_support_height_tolerance: float = .04

    def __post_init__(self):
        super().__post_init__()
        _configure_contact_diagnostic_sensors(self)


@configclass
class RobotContactDiagnosticPlayEnvCfg(RobotContactDiagnosticEnvCfg):
    """Use the same MDP and diagnostic channels for evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64


@configclass
class RobotWalkBootstrapEnvCfg(RobotContactDiagnosticEnvCfg):
    """Fixed low-speed flat profile retaining GT scan and all 29 actions."""

    staged_evidence: bool = True
    walking_stage: str = "bootstrap"
    walking_forward_speed: float = .2
    # Keep this smaller than the shallow stage's minimum 2 cm riser, so a
    # neighboring tread in the ray query cannot stand in for a landed sole.
    stage_support_height_tolerance: float = .015
    walking_enable_push: bool = False
    walking_enable_domain_randomization: bool = False
    walking_enable_observation_noise: bool = False
    walking_enable_actuator_delay: bool = False

    def __post_init__(self):
        super().__post_init__()
        apply_walking_profile(
            self,
            self.walking_stage,
            terrain_template=MIXED_WALK_TERRAINS_CFG,
            flat_template=MIXED_WALK_TERRAINS_CFG.sub_terrains["flat"],
            # The spawn is at the center platform. Moving outward climbs the
            # inverted pyramid and descends the ordinary pyramid.
            stairs_up_template=STAIR_ROUGH_TERRAINS_CFG.sub_terrains["stairs_down"],
            stairs_down_template=STAIR_ROUGH_TERRAINS_CFG.sub_terrains["stairs_low"],
            forward_speed=self.walking_forward_speed,
            enable_push=self.walking_enable_push,
            enable_domain_randomization=self.walking_enable_domain_randomization,
            enable_observation_noise=self.walking_enable_observation_noise,
            enable_actuator_delay=self.walking_enable_actuator_delay,
        )


@configclass
class RobotWalkBootstrapPlayEnvCfg(RobotWalkBootstrapEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64


@configclass
class RobotWalkShallowEnvCfg(RobotWalkBootstrapEnvCfg):
    walking_stage: str = "shallow"


@configclass
class RobotWalkShallowPlayEnvCfg(RobotWalkShallowEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64


@configclass
class RobotWalkStairs08EnvCfg(RobotWalkBootstrapEnvCfg):
    walking_stage: str = "stairs_08"


@configclass
class RobotWalkStairs08PlayEnvCfg(RobotWalkStairs08EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64


@configclass
class RobotWalkStairs12EnvCfg(RobotWalkBootstrapEnvCfg):
    walking_stage: str = "stairs_12"


@configclass
class RobotWalkStairs12PlayEnvCfg(RobotWalkStairs12EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64


@configclass
class RobotWalkStairs16EnvCfg(RobotWalkBootstrapEnvCfg):
    walking_stage: str = "stairs_16"


@configclass
class RobotWalkStairs16PlayEnvCfg(RobotWalkStairs16EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64


@configclass
class RobotWalkStairs20EnvCfg(RobotWalkBootstrapEnvCfg):
    walking_stage: str = "stairs_20"


@configclass
class RobotWalkStairs20PlayEnvCfg(RobotWalkStairs20EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
