from typing import Sequence

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionManager

from ..assets import H2_ARM_JOINT_NAMES, H2_WAIST_JOINT_NAMES
from ..action_contract import actions_in_joint_order


class CtrlLabActionManager(ActionManager):
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._prev_prev_action = torch.zeros_like(self._prev_action)
        self._robot = env.scene["robot"]
        self._action_joint_names = tuple(self.get_term("JointPositionAction")._joint_names)
        self._locked_action_ids = []
        self._held_joint_ids = []
        if bool(getattr(env.cfg, "lock_upper_body_during_training", False)):
            locked_names = set(H2_WAIST_JOINT_NAMES + H2_ARM_JOINT_NAMES)
            self._locked_action_ids = [index for index, name in enumerate(self._action_joint_names) if name in locked_names]
            self._held_joint_ids = self._robot.find_joints(
                H2_WAIST_JOINT_NAMES + H2_ARM_JOINT_NAMES, preserve_order=True)[0]
            self._hold_upper_body()

    @property
    def held_joint_ids(self) -> tuple[int, ...]:
        return tuple(self._held_joint_ids)

    def joint_actions(self, joint_names, *, strict=True) -> torch.Tensor:
        return actions_in_joint_order(self.action, self._action_joint_names, joint_names, strict=strict)

    def _hold_upper_body(self, env_ids=None):
        if self._held_joint_ids:
            targets = self._robot.data.default_joint_pos
            if env_ids is not None:
                targets = targets[env_ids]
            self._robot.set_joint_position_target(
                targets[:, self._held_joint_ids], joint_ids=self._held_joint_ids, env_ids=env_ids)

    @property
    def prev_prev_action(self) -> torch.Tensor:
        return self._prev_prev_action

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            self._prev_prev_action[:] = 0.0
        else:
            self._prev_prev_action[env_ids] = 0.0
        super().reset(env_ids)
        self._hold_upper_body(env_ids)
        return {}

    def apply_action(self):
        super().apply_action()
        self._hold_upper_body()

    def process_action(self, action: torch.Tensor):
        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        if self._locked_action_ids:
            action = action.clone()
            action[:, self._locked_action_ids] = 0.0
        # store the input action
        self._prev_prev_action[:] = self._prev_action
        super().process_action(action)
