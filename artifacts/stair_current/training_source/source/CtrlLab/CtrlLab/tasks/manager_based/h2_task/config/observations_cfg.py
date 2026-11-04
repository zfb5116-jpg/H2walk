import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup, SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from .. import mdp as h2_mdp
from ..assets import H2_BASE_BODY_NAMES, H2_FOOT_BODY_NAMES, H2_JOINT_NAMES


DOF_IDX = H2_JOINT_NAMES
POLICY_HISTORY_LENGTH = 5


@configclass
class ObservationsCfg:
    """Observation specifications for the H2 MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the exported H2 policy group."""

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=POLICY_HISTORY_LENGTH,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-1, 1),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=POLICY_HISTORY_LENGTH,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True, joint_names=DOF_IDX)},
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=POLICY_HISTORY_LENGTH,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True, joint_names=DOF_IDX)},
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=POLICY_HISTORY_LENGTH,
        )
        last_action = ObsTerm(func=h2_mdp.last_action_in_joint_order, clip=(-100, 100), history_length=POLICY_HISTORY_LENGTH)
        # Exteroceptive terrain perception from the existing pelvis-mounted ray caster.
        # This makes the actor terrain-aware; the critic already receives this scan below.
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            clip=(-1.0, 1.0),
            scale=0.2,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )

        def __post_init__(self):
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for critic training."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100, 100), scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-1, 1))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True, joint_names=DOF_IDX)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            clip=(-100, 100),
            scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True, joint_names=DOF_IDX)},
        )
        last_action = ObsTerm(func=h2_mdp.last_action_in_joint_order, clip=(-100, 100))
        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            scale=0.01,
            clip=(-100, 100),
            params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True, joint_names=DOF_IDX)},
        )
        ddq = ObsTerm(
            func=h2_mdp.whole_body_ddq,
            clip=(-100, 100),
            scale=0.001,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=H2_BASE_BODY_NAMES,
                    joint_names=DOF_IDX,
                    preserve_order=True,
                )
            },
        )
        feet_pos_b = ObsTerm(
            func=h2_mdp.feet_pos_b,
            clip=(-100, 100),
            scale=0.2,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=H2_FOOT_BODY_NAMES, preserve_order=True)},
        )
        contact_state = ObsTerm(
            func=h2_mdp.body_contact_state,
            clip=(0.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
        )
        contact_force = ObsTerm(
            func=h2_mdp.body_contact_force,
            scale=0.001,
            clip=(-100, 100),
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
        )
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            clip=(-1.0, 1.0),
            scale=0.2,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )

    critic: CriticCfg = CriticCfg()

    @configclass
    class RndStateCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100, 100), scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-1, 1))
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True, joint_names=DOF_IDX)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            clip=(-100, 100),
            scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True, joint_names=DOF_IDX)},
        )
        feet_pos_b = ObsTerm(
            func=h2_mdp.feet_pos_b,
            clip=(-100, 100),
            scale=0.2,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=H2_FOOT_BODY_NAMES, preserve_order=True)},
        )

    rnd_state: RndStateCfg = RndStateCfg()
