import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp as h2_mdp
from ..assets import H2_BASE_BODY_NAMES, H2_UNDESIRED_CONTACT_BODY_NAMES


@configclass
class TerminationsCfg:
    """Termination terms for the H2 MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_BASE_BODY_NAMES),
            "threshold": 1.0,
        },
    )

    contact_out_of_time = DoneTerm(
        func=h2_mdp.illegal_contact_out_of_time,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_UNDESIRED_CONTACT_BODY_NAMES),
            "threshold": 0.25,
        },
    )
