from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from ..mdp import lin_vel_cmd_levels, ang_vel_cmd_levels, terrain_levels_lifetime


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(terrain_levels_lifetime)
    lin_vel_cmd_levels = CurrTerm(lin_vel_cmd_levels)
    # ang_vel_cmd_levels = CurrTerm(ang_vel_cmd_levels)
