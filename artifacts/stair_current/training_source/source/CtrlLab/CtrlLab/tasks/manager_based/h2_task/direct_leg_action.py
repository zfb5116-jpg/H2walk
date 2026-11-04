"""Twelve policy coordinates mapped to the articulation's complete soft range."""

import torch
from isaaclab.envs.mdp.actions import JointPositionAction


def soft_range_targets(raw, limits):
    """Map unbounded policy coordinates to finite, per-environment joint limits."""
    midpoint = limits.mean(-1)
    half_range = (limits[..., 1] - limits[..., 0]) * 0.5
    return midpoint + half_range * torch.tanh(raw)


def targets_to_raw(targets, limits, epsilon=1.0e-5):
    """Inverse used by supervised initialization; clipping is explicit at the edge."""
    midpoint = limits.mean(-1)
    half_range = (limits[..., 1] - limits[..., 0]) * 0.5
    normalized = ((targets - midpoint) / half_range).clamp(-1.0 + epsilon, 1.0 - epsilon)
    return torch.atanh(normalized)


class DirectLegPositionAction(JointPositionAction):
    """Native delayed position PD, with no reference or additional effort port.

    The policy owns all twelve leg targets. Native gain/strength randomization,
    solver effort and velocity limits, friction, armature and transport delay
    continue to be supplied by the articulation's existing actuators.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self.action_dim != 12:
            raise ValueError(f"Direct H2 stair control requires 12 leg actions, got {self.action_dim}")
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        if not torch.isfinite(limits).all() or not (limits[..., 1] > limits[..., 0]).all():
            raise ValueError("Direct leg control needs finite, nonempty runtime soft joint ranges")
        self._zero_target = torch.zeros_like(self._processed_actions)
        # No stair receiver/impedance/effort-priority attachment is installed.
        self.gait_reference = None

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise RuntimeError("Non-finite direct leg policy action")
        self._raw_actions.copy_(actions)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._processed_actions.copy_(soft_range_targets(actions, limits))

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target(self._zero_target, joint_ids=self._joint_ids)
        self._asset.set_joint_effort_target(self._zero_target, joint_ids=self._joint_ids)

