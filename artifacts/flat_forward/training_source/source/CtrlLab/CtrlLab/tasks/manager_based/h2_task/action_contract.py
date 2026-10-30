"""Name-based mapping between an action term and proprioceptive joint order."""

from collections.abc import Sequence

import torch


def actions_in_joint_order(
    actions: torch.Tensor,
    action_joint_names: Sequence[str],
    requested_joint_names: Sequence[str],
    *,
    strict: bool = True,
) -> torch.Tensor:
    """Select by joint name; uncontrolled observed joints are zero only when allowed."""
    if actions.ndim != 2 or actions.shape[1] != len(action_joint_names):
        raise ValueError("Action tensor width must match the resolved action joint names")
    if len(set(action_joint_names)) != len(action_joint_names):
        raise ValueError("Action joint names must be unique")
    source = {name: index for index, name in enumerate(action_joint_names)}
    missing = [name for name in requested_joint_names if name not in source]
    if strict and missing:
        raise ValueError(f"No action channel exists for joints: {missing}")
    output = actions.new_zeros((actions.shape[0], len(requested_joint_names)))
    target_ids = [i for i, name in enumerate(requested_joint_names) if name in source]
    source_ids = [source[requested_joint_names[i]] for i in target_ids]
    if target_ids:
        output[:, target_ids] = actions[:, source_ids]
    return output
