"""Same-width physical context environment for the independent stair task."""
from .stair_ascent_env import StairAscentEnv
from .stair_task_context import stair_task_context_runtime_contract


class StairTransferEnv(StairAscentEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self._stair_task_context_schema_bootstrap = True
        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally:
            self._stair_task_context_schema_bootstrap = False
        if not hasattr(self, "initialization_reset_generation"):
            import torch
            self.initialization_reset_generation = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if cfg.stair_task_context_dim != 5 or not cfg.stair_strict_episode_force:
            raise RuntimeError("Stair transfer requires its exact five-channel and force-qualification contract")
        self.stair_task_context_contract = stair_task_context_runtime_contract(self)

    def _reset_idx(self, env_ids):
        if not hasattr(self, "initialization_reset_generation"):
            import torch
            self.initialization_reset_generation = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.initialization_reset_generation[env_ids] += 1
        super()._reset_idx(env_ids)
