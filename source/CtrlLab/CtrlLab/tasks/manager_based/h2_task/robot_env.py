from typing import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn
from isaaclab.utils.math import quat_apply_inverse

from .command_progress import CommandProgressTracker
from .manager import CtrlLabActionManager
from .networks.lidar_observation import invalidate_lidar_observation
from .walking_runtime import WalkingRuntime
from .walking_profiles import validate_walking_profile


class RobotEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        if getattr(cfg, "staged_evidence", False):
            # Reject stale post-init overrides before creating any scene.
            validate_walking_profile(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        self.asset: Articulation = self.scene["robot"]
        self.masses = torch.sum(self.asset.root_physx_view.get_masses(), dim=-1).to(self.device)

        # feet buffer
        feet_body_names = getattr(cfg, "feet_body_names", ".*foot")
        # Match ContactSensor SceneEntityCfg(preserve_order=True) foot ordering.
        self.feet_id_b, feet_names = self.asset.find_bodies(feet_body_names, preserve_order=True)
        self.num_feet = len(self.feet_id_b)
        if self.num_feet == 0:
            raise ValueError(f"No foot bodies matched feet_body_names={feet_body_names!r}; got {feet_names}.")
        self.feet_pos_b = torch.zeros(self.num_envs, self.num_feet, 3, dtype=torch.float32, device=self.device)
        self.prev_feet_pos_b = torch.zeros_like(self.feet_pos_b)
        self.feet_vel_b = torch.zeros_like(self.feet_pos_b)
        self.command_progress_tracker = CommandProgressTracker(
            self.num_envs,
            self.device,
            min_command_speed=getattr(cfg, "command_progress_min_speed", 0.1),
            max_step_speed=getattr(cfg, "command_progress_max_step_speed", 5.0),
        )
        self.walking_runtime = None
        if getattr(cfg, "contact_source_diagnostics", False) or getattr(cfg, "staged_evidence", False):
            self.walking_runtime = WalkingRuntime(self)

    def _reset_idx(self, env_ids: Sequence[int]):
        tracker = getattr(self, "command_progress_tracker", None)
        if tracker is not None:
            # Curriculum runs inside the parent reset before the scene teleport.
            # Include the terminal transition while its position is still valid.
            tracker.finish_step(env_ids, self.asset.data.root_pos_w[:, :2], for_reset=True)
        walking = getattr(self, "walking_runtime", None)
        walking_log, walking_sample = walking.before_reset(env_ids) if walking is not None else ({}, None)
        super()._reset_idx(env_ids)
        # Parent reset replaces extras["log"] after curriculum computation.
        # Restore its diagnostics alongside, rather than over, manager logs.
        progress_log = self.__dict__.pop("_command_progress_pending_log", None)
        if progress_log is not None:
            self.extras.setdefault("log", {}).update(progress_log)
        self.extras.setdefault("log", {}).update(walking_log)
        if walking_sample is not None:
            self.extras["walking_diagnostics"] = walking_sample
        if tracker is not None:
            tracker.reset(env_ids, generation=getattr(self, "_command_progress_generation", 0))
        if walking is not None:
            walking.reset(env_ids)
        invalidate_lidar_observation(self, env_ids)
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
        tracker = self.command_progress_tracker
        # The parent resamples commands after physics and resets. Capture the
        # command/yaw applied to this transition, rather than the next command.
        tracker.begin_step(
            self.asset.data.root_pos_w[:, :2],
            self.asset.data.root_quat_w,
            self.command_manager.get_command("base_velocity"),
            self.step_dt,
        )
        walking = getattr(self, "walking_runtime", None)
        try:
            if walking is not None:
                walking.begin_step()
            self.prev_feet_pos_b[:] = self.feet_pos_b
            self.update_feet_state()
            result = super().step(action)
            # Reset environments were consumed before their scene reset; only
            # the remaining environments still have a pending transition.
            env_ids = tracker.pending.nonzero(as_tuple=False).flatten()
            tracker.finish_step(env_ids, self.asset.data.root_pos_w[:, :2])
            if walking is not None:
                walking.finish_step()
        except Exception:
            tracker.abort_step()
            if walking is not None:
                walking.abort_step()
            raise
        return result
