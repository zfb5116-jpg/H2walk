"""Explicit walking stages and conservative completed-episode evidence.

No simulator is imported or terrain rebuilt here. A stage is selected when an
environment is constructed. Passing evidence proposes a next stage, rather
than changing a live mesh. Observation and evaluation windows are deliberately
not policy state and are not restored from a policy checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .command_progress import CommandProgressTracker


STAGE_ORDER = ("bootstrap", "shallow", "stairs_08", "stairs_12", "stairs_16", "stairs_20")
STAGE_HEIGHT_RANGES = MappingProxyType({
    "bootstrap": (0.0, 0.0), "shallow": (0.02, 0.05), "stairs_08": (0.05, 0.08),
    "stairs_12": (0.08, 0.12), "stairs_16": (0.12, 0.16), "stairs_20": (0.20, 0.20),
})
FAMILY_NAMES = ("flat", "up", "down")


def stage_column_families(stage: str) -> tuple[str, ...]:
    """Fixed columns shared by training/play profiles and evidence grouping."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown walking stage: {stage!r}")
    return ("flat",) if stage == "bootstrap" else ("flat",) * 6 + ("up",) * 2 + ("down",) * 2


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _generation(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("generation must be a nonnegative integer")


class StairTraversalTracker:
    """Track supported surface levels before terminal poses are reset.

    ``ground_contact`` must identify actual terrain contact, e.g. a separate
    one-body filtered foot sensor. Whole-body net force is not an acceptable
    substitute. ``ground_height`` is finite local GT surface Z at each foot;
    absent rays must remain NaN. These are crossing/support evidence, not a
    full sole-coverage or landing-quality measurement.

    Events count new supported-height records of at least ``min_level_change``
    per foot. Revisiting the same two treads cannot manufacture more records.
    A single larger step still counts once, and terminal net height must also
    agree with the required up/down direction.
    """

    def __init__(self, num_envs: int, device: str | torch.device, *,
                 min_level_change: float = 0.015, max_step_speed: float = 5.0):
        _positive_integer("num_envs", num_envs)
        if not all(math.isfinite(v) and v > 0 for v in (min_level_change, max_step_speed)):
            raise ValueError("Level change and speed limit must be finite and positive")
        self.num_envs = num_envs
        self.min_level_change = min_level_change
        self.max_step_speed = max_step_speed
        self.elapsed_time = torch.zeros(num_envs, device=device)
        self.device = self.elapsed_time.device
        for name in ("initial_supported_height", "last_supported_height", "_highest_record", "_lowest_record"):
            setattr(self, name, torch.zeros(num_envs, 2, device=self.device))
        for name in ("up_events", "down_events"):
            setattr(self, name, torch.zeros(num_envs, 2, dtype=torch.long, device=self.device))
        self.support_seen = torch.zeros(num_envs, 2, dtype=torch.bool, device=self.device)
        for name in ("pending", "reset_ready", "reported", "_has_last_position"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.bool, device=self.device))
        self.valid = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.episode_generation = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._start_position = torch.zeros(num_envs, 3, device=self.device)
        self._last_position = torch.zeros_like(self._start_position)
        self._input_valid = torch.ones_like(self.valid)
        self._dt = 0.0

    def _ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.ndim != 1 or (ids.numel() and ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("env_ids must be a one-dimensional integer sequence")
        ids = ids.to(torch.long)
        if torch.any((ids < 0) | (ids >= self.num_envs)):
            raise ValueError("env_ids out of range")
        return ids.unique()

    def _tensor(self, value: torch.Tensor, shape: tuple[int, ...], name: str, *, boolean=False):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.device != self.device:
            raise ValueError(f"{name} must have shape {shape} on {self.device}")
        if (boolean and value.dtype != torch.bool) or (not boolean and not value.is_floating_point()):
            raise ValueError(f"{name} has an invalid dtype")

    @torch.no_grad()
    def begin_step(self, root_position_w: torch.Tensor, dt: float) -> None:
        self._tensor(root_position_w, (self.num_envs, 3), "root_position_w")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if self.pending.any():
            raise RuntimeError("Previous stage-evidence transition is still pending")
        finite = torch.isfinite(root_position_w).all(dim=1)
        unchanged = torch.isclose(root_position_w, self._last_position, atol=1e-5, rtol=0).all(dim=1)
        self.valid &= ~self._has_last_position | unchanged
        self._input_valid.copy_(finite)
        self._start_position.copy_(root_position_w)
        self._dt = dt
        self.pending.fill_(True)
        self.reset_ready.fill_(False)

    @torch.no_grad()
    def finish_step(self, env_ids: Sequence[int] | torch.Tensor | None,
                    root_position_w: torch.Tensor, ground_height: torch.Tensor,
                    ground_contact: torch.Tensor, *, for_reset: bool = False) -> torch.Tensor:
        self._tensor(root_position_w, (self.num_envs, 3), "root_position_w")
        self._tensor(ground_height, (self.num_envs, 2), "ground_height")
        self._tensor(ground_contact, (self.num_envs, 2), "ground_contact", boolean=True)
        ids = self._ids(env_ids)
        ids = ids[self.pending[ids]]
        if not ids.numel():
            return ids
        finite_pose = torch.isfinite(root_position_w[ids]).all(dim=1)
        delta = root_position_w[ids] - self._start_position[ids]
        good = self._input_valid[ids] & finite_pose & (delta.norm(dim=1) <= self.max_step_speed * self._dt + 1e-6)
        heights, contact = ground_height[ids], ground_contact[ids]
        finite_height = torch.isfinite(heights)
        # A measured terrain contact with unknown surface height is incomplete
        # evidence. No world-Z-zero floor is silently substituted.
        self.valid[ids] &= good & (~contact | finite_height).all(dim=1)
        observed = contact & finite_height & good[:, None]
        first = observed & ~self.support_seen[ids]
        self.initial_supported_height[ids] = torch.where(first, heights, self.initial_supported_height[ids])
        high = torch.where(first, heights, self._highest_record[ids])
        low = torch.where(first, heights, self._lowest_record[ids])
        up = observed & ~first & (heights - high >= self.min_level_change - 1e-6)
        down = observed & ~first & (low - heights >= self.min_level_change - 1e-6)
        self.up_events[ids] += up
        self.down_events[ids] += down
        self._highest_record[ids] = torch.where(up | first, heights, high)
        self._lowest_record[ids] = torch.where(down | first, heights, low)
        self.last_supported_height[ids] = torch.where(observed, heights, self.last_supported_height[ids])
        self.support_seen[ids] |= observed
        self.elapsed_time[ids] += self._dt
        self._last_position[ids] = root_position_w[ids]
        self._has_last_position[ids] = finite_pose
        self.pending[ids] = False
        if for_reset:
            self.reset_ready[ids] = True
        return ids

    @torch.no_grad()
    def reset(self, env_ids: Sequence[int] | torch.Tensor, generation: int = 0) -> None:
        _generation(generation)
        ids = self._ids(env_ids)
        for name in ("elapsed_time", "initial_supported_height", "last_supported_height", "_highest_record",
                     "_lowest_record", "up_events", "down_events"):
            getattr(self, name)[ids] = 0
        for name in ("support_seen", "pending", "reset_ready", "reported", "_has_last_position"):
            getattr(self, name)[ids] = False
        self.valid[ids] = True
        self.episode_generation[ids] = generation

    @torch.no_grad()
    def abort_step(self) -> None:
        self.valid[self.pending] = False
        self.pending.fill_(False)
        self.reset_ready.fill_(False)


class StageEvaluator:
    """Require separate flat/up/down results before proposing a next stage.

    No reward or radial displacement is inspected. Natural early failures and
    incomplete evidence stay in each family's denominator. Initial/manual
    resets, duplicate callbacks and evidence from another generation do not
    count. Evaluation windows are fresh on environment reconstruction.
    """

    def __init__(self, stage: str, *, min_group_episodes: int = 32, interval_steps: int = 1000,
                 success_rate: float = 0.8, min_episode_fraction: float = 0.98,
                 min_progress_ratio: float = 0.8, max_progress_ratio: float = 1.2,
                 min_target_distance: float = 0.5, min_active_time: float = 2.0,
                 min_stair_events: int = 2, generation: int = 0):
        stage_column_families(stage)
        for name, value in (("min_group_episodes", min_group_episodes), ("interval_steps", interval_steps),
                            ("min_stair_events", min_stair_events)):
            _positive_integer(name, value)
        _generation(generation)
        values = (success_rate, min_episode_fraction, min_progress_ratio, max_progress_ratio,
                  min_target_distance, min_active_time)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Stage thresholds must be finite")
        if not (0 < success_rate <= 1 and 0 < min_episode_fraction <= 1
                and 0 < min_progress_ratio <= max_progress_ratio
                and min_target_distance > 0 and min_active_time > 0):
            raise ValueError("Stage thresholds are outside their valid ranges")
        self.stage = stage
        self.generation = generation
        self.required_families = (0,) if stage == "bootstrap" else (0, 1, 2)
        self.min_group_episodes = min_group_episodes
        self.interval_steps = interval_steps
        self.success_rate = success_rate
        self.min_episode_fraction = min_episode_fraction
        self.min_progress_ratio = min_progress_ratio
        self.max_progress_ratio = max_progress_ratio
        self.min_target_distance = min_target_distance
        self.min_active_time = min_active_time
        self.min_stair_events = min_stair_events
        self.group_episode_counts = dict.fromkeys(self.required_families, 0)
        self.group_success_counts = dict.fromkeys(self.required_families, 0)
        self.evaluation_count = 0
        self.last_evaluation_step = 0
        self._last_step = 0
        self._family_snapshot = None

    @torch.no_grad()
    def observe(self, command_tracker: CommandProgressTracker, traversal_tracker: StairTraversalTracker,
                env_ids: Sequence[int] | torch.Tensor, *, family_ids: torch.Tensor,
                terminated: torch.Tensor, time_outs: torch.Tensor,
                max_episode_length_s: float, current_step: int) -> dict[str, object] | None:
        n, device = traversal_tracker.num_envs, traversal_tracker.device
        if not math.isfinite(max_episode_length_s) or max_episode_length_s <= 0:
            raise ValueError("Maximum episode duration must be finite and positive")
        if isinstance(current_step, bool) or not isinstance(current_step, int) or current_step < self._last_step:
            raise ValueError("current_step must be a nondecreasing nonnegative integer")
        for name, value in (("family_ids", family_ids), ("terminated", terminated), ("time_outs", time_outs)):
            expected = torch.long if name == "family_ids" else torch.bool
            if not isinstance(value, torch.Tensor) or value.shape != (n,) or value.device != device or value.dtype != expected:
                raise ValueError(f"{name} must have shape {(n,)} and dtype {expected} on {device}")
        declared = torch.zeros_like(family_ids, dtype=torch.bool)
        for family in self.required_families:
            declared |= family_ids == family
        if not declared.all():
            raise ValueError("family_ids include an undeclared stage family")
        if self._family_snapshot is not None and not torch.equal(family_ids, self._family_snapshot):
            raise ValueError("Stage family assignments must remain fixed")
        if self._family_snapshot is None:
            self._family_snapshot = family_ids.clone()
        ids = traversal_tracker._ids(env_ids)
        ready = traversal_tracker.reset_ready[ids] & command_tracker.reset_ready[ids] & ~traversal_tracker.reported[ids]
        ready_ids = ids[ready]
        candidate = ((terminated[ready_ids] | time_outs[ready_ids])
                     & (traversal_tracker.episode_generation[ready_ids] == self.generation)
                     & (command_tracker.episode_generation[ready_ids] == self.generation))
        sample = ready_ids[candidate]
        if sample.numel():
            elapsed = command_tracker.elapsed_time[sample]
            active = command_tracker.forward_active_time[sample]
            target = command_tracker.forward_target_distance[sample]
            progress = command_tracker.forward_signed_progress[sample]
            inactive = command_tracker.inactive_forward_error[sample]
            finite = (torch.isfinite(elapsed) & torch.isfinite(active) & torch.isfinite(target)
                      & torch.isfinite(progress) & torch.isfinite(inactive))
            ratio = progress / target.clamp_min(1e-12)
            effective = (progress - inactive) / target.clamp_min(1e-12)
            valid = (command_tracker.valid[sample] & traversal_tracker.valid[sample] & finite
                     & (inactive >= 0) & traversal_tracker.support_seen[sample].all(dim=1)
                     & torch.isclose(elapsed, traversal_tracker.elapsed_time[sample], atol=1e-3, rtol=1e-4))
            success = (valid & time_outs[sample] & ~terminated[sample]
                       & (elapsed >= self.min_episode_fraction * max_episode_length_s)
                       & (active >= self.min_active_time) & (target >= self.min_target_distance)
                       & (ratio >= self.min_progress_ratio) & (ratio <= self.max_progress_ratio)
                       & (effective >= self.min_progress_ratio))
            height_change = (traversal_tracker.last_supported_height[sample]
                             - traversal_tracker.initial_supported_height[sample])
            minimum_height = self.min_stair_events * STAGE_HEIGHT_RANGES[self.stage][0] * 0.75
            up = ((traversal_tracker.up_events[sample] >= self.min_stair_events).all(dim=1)
                  & (height_change >= minimum_height - 1e-6).all(dim=1))
            down = ((traversal_tracker.down_events[sample] >= self.min_stair_events).all(dim=1)
                    & (height_change <= -minimum_height + 1e-6).all(dim=1))
            family = family_ids[sample]
            success &= ((family == 0) | ((family == 1) & up) | ((family == 2) & down))
            counts = torch.bincount(family, minlength=3)
            successes = torch.bincount(family, weights=success.double(), minlength=3)
            values = torch.cat((counts.double(), successes)).cpu().tolist()
            for group in self.required_families:
                self.group_episode_counts[group] += int(values[group])
                self.group_success_counts[group] += int(values[3 + group])
        # Independent consumption bit: the command curriculum may already have
        # consumed its own tracker.reported without affecting stage evidence.
        traversal_tracker.reported[ready_ids] = True
        self._last_step = current_step
        if (current_step - self.last_evaluation_step < self.interval_steps
                or any(count < self.min_group_episodes for count in self.group_episode_counts.values())):
            return None
        rates = {group: self.group_success_counts[group] / count for group, count in self.group_episode_counts.items()}
        passed = all(rate >= self.success_rate for rate in rates.values())
        stage_index = STAGE_ORDER.index(self.stage)
        self.evaluation_count += 1
        total = sum(self.group_episode_counts.values())
        result = {"stage": self.stage, "stage_index": stage_index, "evaluation_id": self.evaluation_count,
                  "episodes": total, "successes": sum(self.group_success_counts.values()),
                  "success_rate": sum(self.group_success_counts.values()) / total,
                  "group_episodes": dict(self.group_episode_counts), "group_success_rates": rates,
                  "passed": passed, "window_checkpointed": False,
                  "next_stage": STAGE_ORDER[stage_index + 1] if passed and stage_index + 1 < len(STAGE_ORDER) else None}
        self.last_evaluation_step = current_step
        for group in self.required_families:
            self.group_episode_counts[group] = self.group_success_counts[group] = 0
        return result
