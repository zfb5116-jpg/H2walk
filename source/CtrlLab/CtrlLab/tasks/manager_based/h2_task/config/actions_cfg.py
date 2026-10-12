import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from ..assets import H2_JOINT_NAMES


DOF_IDX = H2_JOINT_NAMES


@configclass
class ActionsCfg:
    """Action specifications for the H2 MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=DOF_IDX,
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
