"""Versioned budget-v3 constants shared by configuration and reward state."""
from .budget_progress import BUDGET_PROGRESS_PARAMETERS
from .terrain_gait import TERRAIN_REWARD_PARAMETERS


BUDGET_REWARD_PARAMETERS = {
    **TERRAIN_REWARD_PARAMETERS,
    **BUDGET_PROGRESS_PARAMETERS,
    "semantic_version": "budget_v3",
    "success_min_landings_per_foot": 2,
    "success_min_support_fraction": .8,
    "success_hold_seconds": .2,
    "success_max_lateral_distance": .15,
    "success_max_yaw_error": .2,
    "success_min_upright_cosine": .9,
    "success_max_velocity_disagreement": .25,
    "success_min_speed_ratio": .8,
    "success_max_speed_ratio": 1.2,
    "success_min_goal_ratio": .8,
    "success_max_goal_ratio": 1.2,
    "stall_window_seconds": 4.,
    "stall_min_progress_fraction": .1,
}
