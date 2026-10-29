"""MuJoCo H2 stair-climbing branch.

The policy controls the 12 leg joints only.  Waist and arm joints are kept at
the H2 standing pose by :class:`H2StairsEnv`.
"""

from .h2_stairs_env import H2StairsEnv

__all__ = ["H2StairsEnv"]
