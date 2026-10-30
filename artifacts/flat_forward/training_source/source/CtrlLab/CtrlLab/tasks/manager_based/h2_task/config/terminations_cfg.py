import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp as h2_mdp
from ..assets import (
    H2_BASE_BODY_NAMES,
    H2_FALL_CONTACT_BODY_NAMES,
    H2_UNDESIRED_CONTACT_BODY_NAMES,
)


@configclass
class TerminationsCfg:
    """Termination terms for the H2 MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.75})

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_BASE_BODY_NAMES),
            "threshold": 1.0,
        },
    )

    fall_body_contact = DoneTerm(
        func=h2_mdp.illegal_contact_out_of_time,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_FALL_CONTACT_BODY_NAMES),
            # End a fall quickly while leaving brief non-fall contact transients to the reward.
            "threshold": 0.08,
        },
    )

    contact_out_of_time = DoneTerm(
        func=h2_mdp.illegal_contact_out_of_time,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=H2_UNDESIRED_CONTACT_BODY_NAMES),
            "threshold": 0.25,
        },
    )
