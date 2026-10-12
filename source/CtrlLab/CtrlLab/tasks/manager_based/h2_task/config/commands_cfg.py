import math

from ..mdp import UniformLevelVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        debug_vis=True,
        heading_command=True,
        heading_control_stiffness=0.5,
        ranges=UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.6), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)
        ),
        limit_ranges=UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 1.0), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-0.8, 0.8), heading=(-math.pi, math.pi)
        ),
    )
