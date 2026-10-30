"""Teacher-to-direct-position bootstrap math and observation-history contract.

Pure Torch and Python: importing this module does not import Isaac Sim. These
helpers transform command requests, not measured motor torques. They never
write to the teacher ActionManager or change network/observation dimensions.
"""
from __future__ import annotations

from math import prod
from collections.abc import Sequence
import torch

H2_LEG_JOINT_NAMES = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_roll", "ankle_pitch")
)
H2_JOINT_NAMES = H2_LEG_JOINT_NAMES + (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
) + tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")
)

POLICY_DIM, CRITIC_DIM = 1029, 866
POLICY_ACTION_SLICE = slice(323, 468)
CRITIC_ACTION_SLICE = slice(70, 99)
POLICY_SCAN_SLICE = slice(468, 1029)
CRITIC_SCAN_SLICE = slice(305, 866)


def _limits_like(value: torch.Tensor, soft_limits: torch.Tensor):
    limits = torch.as_tensor(soft_limits, device=value.device, dtype=value.dtype)
    if limits.shape[-2:] != (12, 2):
        raise ValueError("Expected soft limits [...,12,2] in resolved action-joint order")
    lower, upper = torch.broadcast_tensors(limits[..., 0], limits[..., 1], value)[:2]
    if not bool(torch.isfinite(lower).all() & torch.isfinite(upper).all() & (upper > lower).all()):
        raise ValueError("Require finite original soft intervals of positive width")
    return lower, upper


def decode_full_soft_range(raw_action: torch.Tensor, soft_limits: torch.Tensor) -> torch.Tensor:
    """Student12 raw outputs -> absolute original-soft-range joint positions."""
    if raw_action.ndim < 1 or raw_action.shape[-1] != 12 or raw_action.dtype not in (torch.float32, torch.float64):
        raise ValueError("Require float32/64 raw actions [...,12]")
    if not bool(torch.isfinite(raw_action).all()):
        raise ValueError("Nonfinite raw action")
    lower, upper = _limits_like(raw_action, soft_limits)
    return .5*(lower+upper)+.5*(upper-lower)*torch.tanh(raw_action)


@torch.no_grad()
def pos_only_equivalent_label(
    q_teacher: torch.Tensor,
    v_teacher: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    soft_limits: torch.Tensor,
    *,
    external_effort: torch.Tensor,
    inverse_epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Encode teacher published q/v as a student pos-only raw12 label.

    q_eq = q_teacher + D/K*v_teacher equates the *PD request* for the same
    current q/qd when student's vtarget is0 and leg external effort is0.
    For delayed execution D/K must remain the same between publication and
    receipt; the original q/v buffers must share lag and compute-call clock.
    This function does not certify those runtime preconditions or tracking.

    Soft clipping or finite-atanh clipping breaks exact request equivalence.
    Both masks and the resulting algebraic request error are returned.
    """
    if q_teacher.ndim != 2 or q_teacher.shape[-1] != 12 or q_teacher.dtype not in (torch.float32, torch.float64):
        raise ValueError("Expected published float32/64 q_teacher[N,12]")
    if not 0 < inverse_epsilon < .1:
        raise ValueError("Require an explicit finite inverse guard in(0,.1)")
    values = [torch.as_tensor(x, dtype=q_teacher.dtype, device=q_teacher.device)
              for x in (q_teacher, v_teacher, stiffness, damping, external_effort)]
    q, v, k, d, effort = torch.broadcast_tensors(*values)
    if q.shape != q_teacher.shape:
        raise ValueError("Published q/v/K/D/effort must broadcast to[N,12]")
    if not all(bool(torch.isfinite(x).all()) for x in (q, v, k, d, effort)):
        raise ValueError("Nonfinite published teacher command or gain")
    if not bool((k > 0).all() & (d >= 0).all()):
        raise ValueError("Position-only equivalence requires K>0 and D>=0")
    if bool((effort != 0).any()):
        raise ValueError("This bootstrap assumes exactly zero published leg external effort")
    lower, upper = _limits_like(q, soft_limits)
    midpoint, half_range = .5*(lower+upper), .5*(upper-lower)
    q_equiv_unclipped = q+(d/k)*v
    q_equiv_soft = torch.maximum(torch.minimum(q_equiv_unclipped, upper), lower)
    soft_clipped = (q_equiv_unclipped < lower) | (q_equiv_unclipped > upper)
    normalized = (q_equiv_soft-midpoint)/half_range
    normalized_safe = normalized.clamp(-1+inverse_epsilon, 1-inverse_epsilon)
    atanh_clipped = normalized != normalized_safe
    raw = torch.atanh(normalized_safe)
    q_decoded = decode_full_soft_range(raw, soft_limits)
    # q_actual and qd_actual cancel from the difference. This is a command
    # expression only; a native implicit drive need not output this exact torque.
    request_error = k*(q_decoded-q)-d*v
    return dict(q_equiv_unclipped=q_equiv_unclipped, q_equiv_soft=q_equiv_soft,
        raw_action=raw, q_decoded=q_decoded, zero_velocity_target=torch.zeros_like(q),
        soft_clipped=soft_clipped, atanh_clipped=atanh_clipped,
        exact_equivalence_preconditions_mask=~(soft_clipped | atanh_clipped),
        request_error_Nm=request_error, soft_clip_request_delta_Nm=k*(q_equiv_soft-q_equiv_unclipped),
        finite_inverse_request_delta_Nm=k*(q_decoded-q_equiv_soft))


def assert_observation_layout(observation_manager, action_term) -> dict[str, object]:
    """Fail closed on runtime term/order/layout drift before collecting data."""
    policy_names = ["velocity_commands", "base_ang_vel", "projected_gravity", "joint_pos_rel",
                    "joint_vel_rel", "last_action", "height_scanner"]
    critic_names = ["base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
                    "joint_pos_rel", "joint_vel_rel", "last_action", "joint_effort", "ddq",
                    "feet_pos_b", "contact_state", "contact_force", "height_scanner"]
    expected = {
        "policy": (policy_names, [3, 15, 15, 145, 145, 145, 561]),
        "critic": (critic_names, [3, 3, 3, 3, 29, 29, 29, 29, 35, 6, 34, 102, 561]),
    }
    record = {}
    for group, (names, widths) in expected.items():
        actual_names = list(observation_manager.active_terms[group])
        actual_widths = [prod(shape) for shape in observation_manager.group_obs_term_dim[group]]
        if actual_names != names or actual_widths != widths:
            raise ValueError(f"{group} bootstrap layout mismatch: {actual_names}/{actual_widths}")
        record[group] = dict(names=names, widths=widths, total=sum(widths))
    if tuple(action_term._joint_names) != H2_LEG_JOINT_NAMES:
        raise ValueError("Teacher/student action12 order must be the declared H2 leg order")
    return record


@torch.no_grad()
def teacher_position_label(q_target: torch.Tensor, v_target: torch.Tensor,
                           K: torch.Tensor, D: torch.Tensor,
                           lower: torch.Tensor, upper: torch.Tensor,
                           eps: float = 1e-4) -> dict[str, torch.Tensor]:
    """Collector API for an asserted-zero-external flat teacher.

    The collector MUST separately assert published leg external effort is0;
    this compact API has no force-port argument with which to verify it.
    ``position`` is decode(raw), including the finite inverse guard, so the
    returned affine error describes the student's actual submitted position.
    """
    lower = torch.as_tensor(lower, dtype=q_target.dtype, device=q_target.device)
    upper = torch.as_tensor(upper, dtype=q_target.dtype, device=q_target.device)
    lower, upper, _ = torch.broadcast_tensors(lower, upper, q_target)
    result = pos_only_equivalent_label(q_target, v_target, K, D,
        torch.stack((lower, upper), dim=-1), external_effort=torch.zeros_like(q_target),
        inverse_epsilon=eps)
    return dict(raw=result["raw_action"], equivalent=result["q_equiv_unclipped"],
        position=result["q_decoded"], clipped_mask=result["soft_clipped"] | result["atanh_clipped"],
        affine_request_error=torch.as_tensor(K, dtype=q_target.dtype, device=q_target.device)
            * (result["q_decoded"]-result["q_equiv_unclipped"]),
        soft_position=result["q_equiv_soft"], soft_clipped_mask=result["soft_clipped"],
        inverse_clipped_mask=result["atanh_clipped"],
        request_expression_error=result["request_error_Nm"])


class ActionHistoryAdapter:
    """Independent student raw-action history; never mutates teacher state.

    Buffer shape[N,5,29], oldest->newest. Initialize at an actual episode
    reset before teacher warmup; advance labels even during discarded warmup.
    rewrite() is read-only/idempotent. advance() moves to the NEXT observation,
    then clears done rows, matching ActionManager/ObservationManager resets.
    """

    def __init__(self, num_envs: int, history_length: int = 5, num_observed_joints: int = 29,
                 *, action_joint_names: Sequence[str] = H2_LEG_JOINT_NAMES,
                 observation_joint_names: Sequence[str] = H2_JOINT_NAMES,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        if num_envs < 1 or history_length != 5 or num_observed_joints != 29:
            raise ValueError("This fixed network contract requires N>0, history5 and observed29")
        if len(action_joint_names) != 12 or len(set(action_joint_names)) != 12:
            raise ValueError("Require twelve unique action-joint names")
        if len(observation_joint_names) != 29 or len(set(observation_joint_names)) != 29:
            raise ValueError("Require29 unique observed-joint names, not native articulation31 order")
        if any(n not in observation_joint_names for n in action_joint_names):
            raise ValueError("Every action joint must occur in the29-joint observation layout")
        self.num_envs = num_envs
        self.action_joint_names = tuple(action_joint_names)
        self.observation_joint_names = tuple(observation_joint_names)
        self._target_ids = torch.tensor([self.observation_joint_names.index(n) for n in self.action_joint_names],
                                        dtype=torch.long, device=device)
        self._history = torch.zeros((num_envs, 5, 29), dtype=dtype, device=device)
        self._last_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    @property
    def history(self) -> torch.Tensor:
        return self._history.clone()

    @property
    def latest_raw12(self) -> torch.Tensor:
        """Previous published student raw12, also usable by a shadow reconstructor."""
        return self._history[:, -1, self._target_ids].clone()

    def reset(self, env_ids=None):
        """Clear observation history, retaining epoch guards for duplicate calls."""
        ids = slice(None) if env_ids is None else env_ids
        self._history[ids] = 0

    @torch.no_grad()
    def advance(self, raw_labels: torch.Tensor, dones: torch.Tensor, *, step: int) -> torch.Tensor:
        """Commit this step's label for obs(t+1), resetting done rows to0.

        Call ONLY after snapshotting student obs(t). The label for t must not
        leak into that same observation. Returns rows advanced; repeated calls
        at the same epoch leave every history/reset value unchanged.
        """
        if raw_labels.shape != (self.num_envs, 12) or raw_labels.device != self._history.device:
            raise ValueError("Expected raw label[N,12] on adapter device")
        if dones.shape != (self.num_envs,) or dones.dtype != torch.bool or dones.device != self._history.device:
            raise ValueError("Expected done bool[N] on adapter device")
        if not bool(torch.isfinite(raw_labels).all()):
            raise ValueError("Nonfinite student history label")
        if step < 0 or bool((self._last_step > step).any()):
            raise ValueError("History epochs must be nonnegative and monotone")
        fresh = self._last_step != step
        ids = fresh.nonzero(as_tuple=False).flatten()
        if ids.numel():
            self._history[ids, :-1] = self._history[ids, 1:].clone()
            self._history[ids, -1] = 0
            self._history[ids[:, None], -1, self._target_ids] = raw_labels[ids].to(self._history.dtype)
            self._history[ids[dones[ids]]] = 0
            self._last_step[ids] = step
        return fresh

    def rewrite_policy(self, policy_obs: torch.Tensor) -> torch.Tensor:
        if policy_obs.shape != (self.num_envs, POLICY_DIM) or policy_obs.device != self._history.device:
            raise ValueError("Expected raw policy observation[N,1029] on adapter device")
        result = policy_obs.clone()
        result[:, POLICY_ACTION_SLICE] = self._history.clamp(-100, 100).reshape(self.num_envs, 145)
        return result

    def rewrite_critic(self, critic_obs: torch.Tensor) -> torch.Tensor:
        if critic_obs.shape != (self.num_envs, CRITIC_DIM) or critic_obs.device != self._history.device:
            raise ValueError("Expected raw critic observation[N,866] on adapter device")
        result = critic_obs.clone()
        result[:, CRITIC_ACTION_SLICE] = self._history[:, -1].clamp(-100, 100)
        return result

    def rewrite(self, policy_obs: torch.Tensor, critic_obs: torch.Tensor | None = None):
        """Copy/rewrite RAW observations, before the student normalizers."""
        return self.rewrite_policy(policy_obs), None if critic_obs is None else self.rewrite_critic(critic_obs)


class DirectActionHistory(ActionHistoryAdapter):
    """Compact collector API; append once per20ms step after snapshot obs(t)."""

    def __init__(self, n: int, device: str | torch.device,
                 action_joint_names: Sequence[str], observed_joint_names: Sequence[str]):
        super().__init__(n, 5, 29, action_joint_names=action_joint_names,
                         observation_joint_names=observed_joint_names, device=device)
        self._append_epoch = 0

    def append(self, raw12: torch.Tensor, reset_mask_bool: torch.Tensor, *, step: int | None = None):
        """Update NEXT obs history; optional source epoch guards duplicate calls.

        Without ``step``, callers must append exactly once per environment
        control step, never once for each of its four actuator writes.
        """
        epoch = self._append_epoch if step is None else int(step)
        result = self.advance(raw12, reset_mask_bool, step=epoch)
        self._append_epoch = max(self._append_epoch, epoch+1)
        return result

    def latest(self) -> torch.Tensor:
        return self.latest_raw12
