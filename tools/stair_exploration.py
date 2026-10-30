"""Physical exploration budget for the unchanged twelve-coordinate H2 actor.

Only the existing Gaussian std parameter is projected after Adam steps. Policy
means and the full-soft-range action map retain their original authority.
"""
import torch


class StairExplorationBudget:
    def __init__(self, policy, optimizer, soft_limits, maximum_angle_std,
                 *, initial_fraction=0.5, minimum_fraction=0.05, initialize=True,
                 stiffness=None, effort_limits=None):
        if policy.std.shape != (12,) or soft_limits.shape[-2:] != (12,2):
            raise ValueError("Require the existing12-action std and N x12x2 soft limits")
        if not 0 < minimum_fraction <= initial_fraction <= 1:
            raise ValueError("Invalid exploration fraction ordering")
        self.policy = policy
        self.maximum_angle = torch.as_tensor(maximum_angle_std,device=policy.std.device,dtype=policy.std.dtype)
        if self.maximum_angle.shape != (12,) or not bool((self.maximum_angle>0).all()):
            raise ValueError("Specify one positive physical angular std ceiling per leg joint")
        half_range = ((soft_limits[...,1]-soft_limits[...,0])*0.5).detach()
        if not torch.isfinite(half_range).all() or not bool((half_range>0).all()):
            raise ValueError("Exploration requires finite nonempty joint ranges")
        self.half_range_max = half_range.amax(0)
        # tanh is1-Lipschitz. RMS displacement from the deterministic target
        # is at most half_range*raw_std, for every possible policy mean.
        self.maximum = self.maximum_angle/self.half_range_max
        self.minimum = self.maximum*minimum_fraction
        self.initial = self.maximum*initial_fraction
        self.optimizer_steps = 0
        self.projected_entries = torch.zeros((),device=policy.std.device,dtype=torch.long)
        self.negative_proposals = torch.zeros_like(self.projected_entries)
        self.stiffness = stiffness.detach().clone() if stiffness is not None else None
        self.effort_limits = effort_limits.detach().clone() if effort_limits is not None else None
        with torch.no_grad():
            if not bool(torch.isfinite(policy.std).all()):
                raise RuntimeError("Nonfinite input stair exploration parameter before initialization")
            if initialize:
                policy.std.copy_(self.initial)
            self._project()
        self.handle = optimizer.register_step_post_hook(self._after_step)

    @torch.no_grad()
    def _project(self):
        value = self.policy.std
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("Nonfinite stair exploration parameter before projection")
        bounded = torch.maximum(torch.minimum(value,self.maximum),self.minimum)
        self.projected_entries += (bounded!=value).sum()
        self.negative_proposals += (value<=0).sum()
        value.copy_(bounded)

    def _after_step(self, optimizer, args, kwargs):
        self.optimizer_steps += 1
        self._project()

    def assert_valid(self):
        std=self.policy.std.detach()
        if (not bool(torch.isfinite(std).all()) or not bool((std>=self.minimum).all())
                or not bool((std<=self.maximum).all())):
            raise RuntimeError("Stair exploration left its physical variance budget")

    def snapshot(self):
        self.assert_valid()
        std=self.policy.std.detach()
        result=dict(policy_std=std.cpu().tolist(),minimum_raw_std=self.minimum.cpu().tolist(),
            maximum_raw_std=self.maximum.cpu().tolist(),initial_raw_std=self.initial.cpu().tolist(),
            maximum_angle_rms_rad=self.maximum_angle.cpu().tolist(),
            current_angle_rms_upper_bound_rad=(std*self.half_range_max).cpu().tolist(),
            optimizer_steps=self.optimizer_steps,projected_parameter_entries=int(self.projected_entries),
            negative_optimizer_proposals_projected=int(self.negative_proposals),
            policy_mean_and_soft_range_unrestricted=True,network_structure_changed=False,
            enforcement="Training-entry optimizer hook; reconstructed on resume, not serialized into the neural network",
            interpretation="Target-coordinate RMS bound, not a limit on deterministic joint targets or measured motor torque")
        if self.stiffness is not None and self.effort_limits is not None:
            request = self.stiffness*self.maximum_angle
            result["PD_target_noise_sigma_bound_Nm"] = request.amax(0).cpu().tolist()
            result["PD_target_noise_sigma_bound_fraction_of_cap"] = (
                request/self.effort_limits.clamp_min(1e-6)).amax(0).cpu().tolist()
        return result
