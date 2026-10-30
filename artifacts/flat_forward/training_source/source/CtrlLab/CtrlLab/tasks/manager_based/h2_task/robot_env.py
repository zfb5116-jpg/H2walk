from typing import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn
from isaaclab.utils.math import quat_apply_inverse

from .manager import CtrlLabActionManager
from .locomotion_progress import LocomotionProgress


class RobotEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.asset: Articulation = self.scene["robot"]
        self.masses = torch.sum(self.asset.root_physx_view.get_masses(), dim=-1).to(self.device)

        # feet buffer
        feet_body_names = getattr(cfg, "feet_body_names", ".*foot")
        self.feet_id_b, feet_names = self.asset.find_bodies(feet_body_names)
        self.num_feet = len(self.feet_id_b)
        if self.num_feet == 0:
            raise ValueError(f"No foot bodies matched feet_body_names={feet_body_names!r}; got {feet_names}.")
        self.feet_pos_b = torch.zeros(self.num_envs, self.num_feet, 3, dtype=torch.float32, device=self.device)
        self.prev_feet_pos_b = torch.zeros_like(self.feet_pos_b)
        self.feet_vel_b = torch.zeros_like(self.feet_pos_b)
        self._progress_sensor = self.scene["contact_forces"]
        self._progress_foot_ids, _ = self._progress_sensor.find_bodies(feet_body_names, preserve_order=True)
        if len(self._progress_foot_ids) != 2:
            raise ValueError("H2 locomotion curriculum requires two ordered foot contact bodies")
        self.locomotion_progress = LocomotionProgress(
            self.num_envs, self.device, episode_length_s=cfg.episode_length_s)

    def _finish_progress_step(self, env_ids=None):
        contact = self._progress_sensor.data.net_forces_w[:, self._progress_foot_ids, :].norm(dim=-1) > 5.0
        self.locomotion_progress.finish_step(self.asset.data.root_pos_w[:, :2], contact, env_ids)

    def _reset_idx(self, env_ids: Sequence[int]):
        if hasattr(self, "locomotion_progress"):
            # Capture the terminal displacement/contact before terrain origins or
            # root state change. The curriculum then sees complete episodes.
            self._finish_progress_step(env_ids)
            self.locomotion_progress.record_episodes(env_ids, self.termination_manager.terminated)
        super()._reset_idx(env_ids)
        if hasattr(self, "locomotion_progress"):
            self.locomotion_progress.reset(env_ids)
        self.update_feet_state()
        self.prev_feet_pos_b[env_ids] = self.feet_pos_b[env_ids]
        self.feet_vel_b[env_ids] = 0.0

    def load_managers(self):
        super().load_managers()
        self.action_manager = CtrlLabActionManager(self.cfg.actions, self)

    def update_feet_state(self):
        base_quat = self.asset.data.root_quat_w.unsqueeze(1).expand(-1, self.num_feet, -1)
        base_pos_w = self.asset.data.root_pos_w.unsqueeze(1)
        feet_pos_w = self.asset.data.body_pos_w[:, self.feet_id_b, :]
        self.feet_pos_b[:] = quat_apply_inverse(base_quat, feet_pos_w - base_pos_w)
        self.feet_vel_b[:] = (self.feet_pos_b - self.prev_feet_pos_b) / self.step_dt

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.prev_feet_pos_b[:] = self.feet_pos_b
        self.update_feet_state()
        self.locomotion_progress.begin_step(
            self.asset.data.root_pos_w[:, :2], self.asset.data.root_quat_w,
            self.command_manager.get_command("base_velocity"), self.step_dt)
        result = super().step(action)
        # Reset environments were finalized by _reset_idx; only finish the rest.
        self._finish_progress_step()
        return result
