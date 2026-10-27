from typing import Sequence

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionManager

from ..assets import H2_ARM_JOINT_NAMES, H2_WAIST_JOINT_NAMES


class CtrlLabActionManager(ActionManager):
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._prev_prev_action = torch.zeros_like(self._prev_action)
        self._locked_action_ids = []
        if bool(getattr(env.cfg, "lock_upper_body_during_training", False)):
            action_cfg = env.cfg.actions.JointPositionAction
            locked_names = set(H2_WAIST_JOINT_NAMES + H2_ARM_JOINT_NAMES)
            self._locked_action_ids = [index for index, name in enumerate(action_cfg.joint_names) if name in locked_names]

    @property
    def prev_prev_action(self) -> torch.Tensor:
        return self._prev_prev_action

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            self._prev_prev_action[:] = 0.0
        else:
            self._prev_prev_action[env_ids] = 0.0
        super().reset(env_ids)
        return {}

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
