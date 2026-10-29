"""Read-only tensor diagnostics for terminal contact and body state.

These reductions do not alter rewards, observations, actions or terminations.
The caller must capture aligned sensor histories and terminal state before reset.
"""

from __future__ import annotations

import math
from types import FunctionType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .robot_env import RobotEnv


@torch.no_grad()
def capture_terminal_predicates(env: RobotEnv, env_ids: torch.Tensor) -> dict[str, object]:
    """Re-evaluate only audited pure predicates, without changing the manager.

    Isaac Lab v2.2.1 get_term retains the last matching term for a done row;
    it cannot establish co-occurrence. Keep those reported values separately.
    Unknown/stateful terms are never called again. Recomputed values describe
    the current pre-reset state, not an interception of the earlier compute.
    """
    reported, recomputed, skipped = {}, {}, []
    manager = env.termination_manager
    for name in manager.active_terms:
        reported[name] = manager.get_term(name)[env_ids].detach().clone()
        cfg = manager.get_term_cfg(name)
        function = cfg.func
        module = getattr(function, "__module__", "")
        symbol = getattr(function, "__name__", "")
        official = module == "isaaclab.envs.mdp.terminations" and symbol in {"time_out", "bad_orientation", "illegal_contact"}
        h2 = module.endswith(".h2_task.mdp.terminations") and symbol == "illegal_contact_out_of_time"
        if not isinstance(function, FunctionType) or not (official or h2):
            skipped.append(name)
            continue
        value = function(env, **cfg.params)
        if not isinstance(value, torch.Tensor) or value.shape != (env.num_envs,) or value.dtype != torch.bool:
            raise RuntimeError(f"Diagnostic termination predicate {name!r} returned an invalid tensor")
        recomputed[name] = value[env_ids].detach().clone()
    return {"reported_term_flags": reported, "recomputed_predicate_flags": recomputed,
            "predicates_not_recomputed": tuple(skipped)}


INVALID = -1
NO_NET_TRIGGER = 0
GROUND = 1
SELF = 2
BOTH = 3
UNATTRIBUTED = 4


def _positive_threshold(threshold: float):
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")


def _floating_tensor(value: torch.Tensor, name: str, device=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on the same device as the other diagnostic inputs")


def _filtered_channel(value, reference: torch.Tensor, name: str, threshold: float):
    n, t = reference.shape[:2]
    empty = torch.zeros(n, t, dtype=torch.bool, device=reference.device)
    if value is None:
        return empty, empty, torch.full((n,), torch.nan, device=reference.device)
    _floating_tensor(value, name, reference.device)
    if value.ndim != 5 or value.shape[:3] != (n, t, 1) or value.shape[-1] != 3:
        raise ValueError(f"{name} must have aligned shape [env, time, 1, filter, 3]")
    if value.shape[3] == 0:
        return empty, empty, torch.full((n,), torch.nan, device=reference.device)
    # Check each filtered body before reducing. Summing opposite contact normals
    # first would erase contacts; these values are not tangential friction forces.
    pair_norm = torch.linalg.vector_norm(value[:, :, 0], dim=-1)
    valid = torch.isfinite(value[:, :, 0]).all(dim=-1).all(dim=-1) & torch.isfinite(pair_norm).all(dim=-1)
    detected = (pair_norm > threshold).any(dim=-1)
    peak = pair_norm.amax(dim=(1, 2))
    peak = torch.where(valid.all(dim=1), peak, torch.nan)
    return detected, valid, peak


@torch.no_grad()
def classify_pelvis_contact(
    net_history: torch.Tensor,
    ground_history: torch.Tensor | None,
    self_history: torch.Tensor | None,
    threshold: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Classify sources present at the same samples as a pelvis net-force trigger.

    ``net_history`` is ``[env,time,1,3]``; each filtered history is
    ``[env,time,1,filter,3]`` from a *single-body* sensor. Histories must have the
    same timestamps, update period and length; matching tensor shapes alone do
    not establish timestamp alignment. A missing channel (None or zero filters)
    is invalid, never evidence of no contact. Each force threshold is strict.

    Codes: -1 invalid, 0 no net trigger, 1 ground, 2 self, 3 both sources across
    trigger samples, 4 at least one unattributed trigger sample. Code 3 does not
    imply simultaneous contacts: ``both_same_sample`` reports that separately.
    An unattributed sample takes precedence over known sources in the window.
    Source flags are diagnostic observations, not a causal reconstruction from
    net-force subtraction. Opposite source forces may cancel the net vector;
    ``ground_detected/self_detected`` retain those observations even for code 0.

    Nonfinite force data makes the corresponding sample/window invalid. Peaks
    are NaN for invalid channels and must be omitted from scalar means by callers.
    The caller should restrict attribution counts to ``valid`` rows.
    """
    _positive_threshold(threshold)
    _floating_tensor(net_history, "net_history")
    if net_history.ndim != 4 or net_history.shape[1] < 1 or net_history.shape[2:] != (1, 3):
        raise ValueError("net_history must have shape [env, time >= 1, 1, 3]")
    net_norm = torch.linalg.vector_norm(net_history[:, :, 0], dim=-1)
    net_valid = torch.isfinite(net_history[:, :, 0]).all(dim=-1) & torch.isfinite(net_norm)
    net_samples = net_norm > threshold
    ground, ground_valid, ground_peak = _filtered_channel(ground_history, net_history, "ground_history", threshold)
    robot, robot_valid, robot_peak = _filtered_channel(self_history, net_history, "self_history", threshold)
    sample_valid = net_valid & ground_valid & robot_valid
    ground_samples = net_samples & ground & ground_valid & net_valid
    self_samples = net_samples & robot & robot_valid & net_valid
    unknown_samples = net_samples & ~ground & ~robot & sample_valid

    sample_codes = torch.zeros_like(net_samples, dtype=torch.long)
    sample_codes[ground_samples] = GROUND
    sample_codes[self_samples] = SELF
    sample_codes[ground_samples & self_samples] = BOTH
    sample_codes[unknown_samples] = UNATTRIBUTED
    sample_codes[~sample_valid] = INVALID
    ground_any, self_any = ground_samples.any(dim=1), self_samples.any(dim=1)
    unknown_any = unknown_samples.any(dim=1)
    codes = torch.zeros_like(ground_any, dtype=torch.long)
    codes[ground_any] = GROUND
    codes[self_any] = SELF
    codes[ground_any & self_any] = BOTH
    codes[unknown_any] = UNATTRIBUTED
    valid = sample_valid.all(dim=1)
    codes[~valid] = INVALID
    return {
        "source_code": codes,
        "source_code_samples": sample_codes,
        "valid": valid,
        "sample_valid": sample_valid,
        "net_trigger": net_samples.any(dim=1),
        "net_trigger_samples": net_samples,
        "ground_at_trigger": ground_any,
        "self_at_trigger": self_any,
        "unattributed_at_trigger": unknown_any,
        "both_in_history": ground_any & self_any,
        "both_same_sample": (ground_samples & self_samples & sample_valid).any(dim=1),
        "ground_detected": (ground & ground_valid).any(dim=1),
        "self_detected": (robot & robot_valid).any(dim=1),
        "net_peak_n": torch.where(net_valid.all(dim=1), net_norm.amax(dim=1), torch.nan),
        "ground_peak_pair_n": ground_peak,
        "self_peak_pair_n": robot_peak,
    }


@torch.no_grad()
def terminal_state_metrics(
    *,
    pelvis_position_w: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    joint_position_error: torch.Tensor,
    joint_velocity: torch.Tensor,
    applied_torque: torch.Tensor,
    feet_force_w: torch.Tensor,
    feet_velocity_w: torch.Tensor,
    raw_action: torch.Tensor,
    processed_action: torch.Tensor,
    ground_height: torch.Tensor | None = None,
    threshold: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return detached per-environment scalars from pre-reset body/action tensors.

    Ground height must be a finite, measured local terrain estimate; absent data
    yields NaN clearance and a false flag, never a world-zero fallback. Foot force
    fields are explicitly *net-force proxies*; they do not measure ground-only
    support, support polygon, or area. Action targets keep their supplied units.
    Invalid fields produce NaN and ``state_valid=False``; callers must use finite
    masks for scalar aggregation. No input tensor is changed or retained by view.
    """
    _positive_threshold(threshold)
    _floating_tensor(pelvis_position_w, "pelvis_position_w")
    if pelvis_position_w.ndim != 2 or pelvis_position_w.shape[1] != 3:
        raise ValueError("pelvis_position_w must have shape [env, 3]")
    n, device = pelvis_position_w.shape[0], pelvis_position_w.device
    vectors = {"projected_gravity_b": projected_gravity_b,
               "joint_position_error": joint_position_error, "joint_velocity": joint_velocity,
               "applied_torque": applied_torque, "feet_force_w": feet_force_w,
               "feet_velocity_w": feet_velocity_w, "raw_action": raw_action,
               "processed_action": processed_action}
    for name, value in vectors.items():
        _floating_tensor(value, name, device)
        ndim = 3 if name.startswith("feet_") else 2
        if value.ndim != ndim or value.shape[0] != n or value.shape[1] < 1:
            raise ValueError(f"{name} has an incompatible diagnostic shape")
    if projected_gravity_b.shape != (n, 3):
        raise ValueError("projected_gravity_b must have shape [env, 3]")
    if joint_position_error.shape != joint_velocity.shape or joint_velocity.shape != applied_torque.shape:
        raise ValueError("joint state tensor shapes must match")
    if feet_force_w.shape != feet_velocity_w.shape or feet_force_w.shape[-1] != 3:
        raise ValueError("foot force/velocity tensors must have matching shape [env, foot, 3]")
    if raw_action.shape != processed_action.shape:
        raise ValueError("raw_action and processed_action must have matching shape [env, action]")

    def row_valid(value):
        return torch.isfinite(value.flatten(start_dim=1)).all(dim=1)

    def metric(value, source):
        return torch.where(row_valid(source) & torch.isfinite(value), value, torch.nan).clone()

    gravity_norm = torch.linalg.vector_norm(projected_gravity_b, dim=-1)
    gravity_valid = row_valid(projected_gravity_b) & torch.isfinite(gravity_norm) & (gravity_norm > 1e-6)
    cosine = (-projected_gravity_b[:, 2] / gravity_norm.clamp_min(1e-6)).clamp(-1., 1.)
    tilt = torch.where(gravity_valid, torch.acos(cosine), torch.nan)
    contact_proxy = torch.linalg.vector_norm(feet_force_w, dim=-1) > threshold
    xy_speed = torch.linalg.vector_norm(feet_velocity_w[..., :2], dim=-1)
    contact_speed = torch.where(contact_proxy, xy_speed, 0.).amax(dim=1)
    result = {
        "pelvis_height_w": metric(pelvis_position_w[:, 2], pelvis_position_w),
        "pelvis_tilt_rad": tilt,
        "joint_position_error_max": metric(joint_position_error.abs().amax(dim=1), joint_position_error),
        "joint_velocity_abs_max": metric(joint_velocity.abs().amax(dim=1), joint_velocity),
        "joint_torque_abs_max": metric(applied_torque.abs().amax(dim=1), applied_torque),
        "raw_action_abs_max": metric(raw_action.abs().amax(dim=1), raw_action),
        "processed_action_abs_max": metric(processed_action.abs().amax(dim=1), processed_action),
        "feet_positive_net_z_count": metric((feet_force_w[..., 2] > threshold).sum(dim=1).to(feet_force_w.dtype), feet_force_w),
        "feet_net_z_sum_n": metric(feet_force_w[..., 2].sum(dim=1), feet_force_w),
        "feet_net_force_peak_n": metric(torch.linalg.vector_norm(feet_force_w, dim=-1).amax(dim=1), feet_force_w),
        "feet_contact_proxy_xy_speed_max": torch.where(row_valid(feet_force_w) & row_valid(feet_velocity_w)
                                                       & torch.isfinite(contact_speed), contact_speed, torch.nan),
    }
    state_valid = row_valid(pelvis_position_w) & gravity_valid
    for value in vectors.values():
        state_valid &= row_valid(value)
    for value in result.values():
        state_valid &= torch.isfinite(value)
    result["state_valid"] = state_valid
    if ground_height is None:
        ground_valid = torch.zeros(n, dtype=torch.bool, device=device)
        clearance = torch.full((n,), torch.nan, device=device, dtype=pelvis_position_w.dtype)
    else:
        _floating_tensor(ground_height, "ground_height", device)
        if ground_height.shape != (n,):
            raise ValueError("ground_height must have shape [env]")
        ground_valid = torch.isfinite(ground_height)
        clearance = torch.where(ground_valid & row_valid(pelvis_position_w),
                                pelvis_position_w[:, 2] - ground_height, torch.nan)
    result["ground_height_valid"] = ground_valid
    result["pelvis_ground_clearance"] = clearance
    return result
