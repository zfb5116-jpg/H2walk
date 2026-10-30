"""Solver-integrated PD drives with the same sampled command transport delay."""

import torch

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer


class DelayedImplicitActuator(ImplicitActuator):
    """Delay all setpoints, then let PhysX integrate the bounded PD drive."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def reset(self, env_ids=None):
        count = self._num_envs if env_ids is None or env_ids == slice(None) else len(env_ids)
        delay = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (count,),
                              dtype=torch.int, device=self._device)
        for buffer in (self.positions_delay_buffer, self.velocities_delay_buffer, self.efforts_delay_buffer):
            buffer.set_time_lag(delay, env_ids)
            buffer.reset(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    class_type: type = DelayedImplicitActuator
    min_delay: int = 0
    max_delay: int = 6
