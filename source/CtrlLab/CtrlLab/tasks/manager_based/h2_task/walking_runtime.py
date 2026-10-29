"""Opt-in environment adapter for pre-reset diagnostics and stage evidence.

Only tensor reads and bookkeeping live here. No policy, optimizer, sensor
construction or physics step is performed by this adapter.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .contact_diagnostics import capture_terminal_predicates, classify_pelvis_contact, terminal_state_metrics
from .foot_geometry import lowest_sole_height_world
from .staged_curriculum import FAMILY_NAMES, STAGE_ORDER, StageEvaluator, StairTraversalTracker
from .terrain_clearance import local_ground_height

if TYPE_CHECKING:
    from .robot_env import RobotEnv


class WalkingRuntime:
    def __init__(self, env: RobotEnv) -> None:
        self.env = env
        self.diagnostics = bool(getattr(env.cfg, "contact_source_diagnostics", False))
        self.threshold = getattr(env.cfg, "contact_diagnostic_threshold", 1.0)
        self.max_samples = getattr(env.cfg, "contact_diagnostic_max_samples", 8)
        self.support_tolerance = getattr(env.cfg, "stage_support_height_tolerance", .015)
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("Contact diagnostic threshold must be finite and positive")
        if isinstance(self.max_samples, bool) or not isinstance(self.max_samples, int) or self.max_samples < 0:
            raise ValueError("contact_diagnostic_max_samples must be a nonnegative integer")
        if not math.isfinite(self.support_tolerance) or self.support_tolerance <= 0:
            raise ValueError("Stage support height tolerance must be finite and positive")
        names = ("pelvis_ground", "pelvis_robot", "left_foot_ground", "right_foot_ground")
        self.sensors = {name: env.scene[name] for name in names}
        expected_bodies = ("pelvis", "pelvis", *env.cfg.feet_body_names)
        if len(expected_bodies) != 4:
            raise ValueError("Walking diagnostics require exactly two ordered foot bodies")
        for name, body in zip(names, expected_bodies):
            sensor = self.sensors[name]
            if sensor.body_names != [body]:
                raise ValueError(f"{name} must select the single body {body!r}")
            if sensor.cfg.history_length != 3 or not math.isclose(sensor.cfg.update_period, env.physics_dt):
                raise ValueError("All diagnostic histories must use three samples at the physics period")
            history = getattr(sensor.data, "force_matrix_w_history", None)
            if history is None or history.ndim != 5 or history.shape[:3] != (env.num_envs, 3, 1) or history.shape[3] < 1:
                raise RuntimeError(f"{name} requires a single-body filtered force history; check Isaac Lab support")
            if history.shape[3] != len(sensor.cfg.filter_prim_paths_expr):
                raise RuntimeError(f"{name} filtered history does not match the configured collider filters")
        self.net_sensor = env.scene["contact_forces"]
        if self.net_sensor.cfg.history_length != 3 or not math.isclose(self.net_sensor.cfg.update_period, env.physics_dt):
            raise ValueError("The original contact history must use the same physics period")
        self.pelvis_id, pelvis_names = self.net_sensor.find_bodies(["pelvis"], preserve_order=True)
        self.feet_ids, feet_names = self.net_sensor.find_bodies(env.cfg.feet_body_names, preserve_order=True)
        if pelvis_names != ["pelvis"] or feet_names != list(env.cfg.feet_body_names):
            raise ValueError("The original contact sensor body mapping does not match H2")
        joint_names = list(env.cfg.walking_diagnostic_joint_names)
        self.joint_ids, actual_names = env.asset.find_joints(joint_names, preserve_order=True)
        if actual_names != joint_names:
            raise ValueError("Walking diagnostic joint names must retain action order")
        self.counts = dict.fromkeys(("natural_episodes_total", "valid_sources_total", "net_trigger_total",
                                    "ground_at_trigger_total", "self_at_trigger_total", "both_same_sample_total",
                                    "unattributed_total", "invalid_sources_total"), 0)
        self.traversal = self.evaluator = self.family_ids = None
        self.latest_stage = None
        if getattr(env.cfg, "staged_evidence", False):
            if "height_scanner" not in env.scene.sensors:
                raise ValueError("Stage evidence requires the GT height scanner")
            columns = tuple(env.cfg.walking_stage_column_families)
            terrain = env.scene.terrain
            types = terrain.terrain_types
            if types.dtype != torch.long or types.shape != (env.num_envs,) or bool(((types < 0) | (types >= len(columns))).any()):
                raise ValueError("Stage terrain column assignments are invalid")
            self.column_ids = types.clone()
            lookup = torch.tensor([FAMILY_NAMES.index(family) for family in columns], device=env.device)
            self.family_ids = lookup[types]
            self.traversal = StairTraversalTracker(env.num_envs, env.device)
            self.tile_origins = env.scene.env_origins[:, :2].detach().clone()
            margin = getattr(env.cfg, "stage_tile_boundary_margin", .3)
            size = env.cfg.scene.terrain.terrain_generator.size
            if not math.isfinite(margin) or margin < 0 or len(size) != 2:
                raise ValueError("Stage tile boundary margin must be finite and nonnegative")
            self.tile_limits = torch.as_tensor(size, device=env.device, dtype=self.tile_origins.dtype) / 2 - margin
            if not bool(torch.isfinite(self.tile_limits).all() & (self.tile_limits > 0).all()):
                raise ValueError("Stage terrain dimensions must leave a positive bounded evaluation area")
            self.evaluator = StageEvaluator(
                env.cfg.walking_stage,
                min_group_episodes=getattr(env.cfg, "stage_min_group_episodes", 32),
                interval_steps=env.max_episode_length,
            )

    def _ground_heights(self, positions: torch.Tensor) -> torch.Tensor:
        scanner = self.env.scene.sensors.get("height_scanner")
        if scanner is None:
            return positions.new_full(positions.shape[:2], torch.nan)
        return local_ground_height(scanner.data.ray_hits_w, positions)

    @torch.no_grad()
    def _support_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Certify upward terrain force near the sole, retaining an explicit validity mask."""
        feet = self.env.asset.data.body_pos_w[:, self.env.feet_id_b]
        heights = self._ground_heights(feet)
        quaternions = self.env.asset.data.body_quat_w[:, self.env.feet_id_b]
        sole = lowest_sole_height_world(feet, quaternions)
        geometric_valid = torch.isfinite(sole) & torch.isfinite(heights)
        near_surface = (sole - heights).abs() <= self.support_tolerance
        contacts = []
        valid_channels = []
        for name in ("left_foot_ground", "right_foot_ground"):
            pairs = self.sensors[name].data.force_matrix_w_history[:, 0, 0]
            valid = torch.isfinite(pairs).all(dim=(1, 2))
            # A horizontal riser collision is not upward support. Sum the
            # measured vertical forces, not the vector norm of any collision.
            contacts.append(valid & (pairs[..., 2].sum(dim=1) > self.threshold))
            valid_channels.append(valid)
        valid = torch.stack(valid_channels, dim=1) & geometric_valid
        return heights, torch.stack(contacts, dim=1) & valid & near_surface, valid

    def ground_support(self) -> tuple[torch.Tensor, torch.Tensor]:
        heights, support, _ = self._support_state()
        return heights, support

    def begin_step(self) -> None:
        self.env.extras.pop("walking_diagnostics", None)
        if self.traversal is not None:
            self.traversal.begin_step(self.env.asset.data.root_pos_w, self.env.step_dt)
            self._check_tile_bounds(torch.arange(self.env.num_envs, device=self.env.device))

    def _check_tile_bounds(self, ids: Sequence[int] | torch.Tensor) -> None:
        if not torch.equal(self.env.scene.env_origins[:, :2], self.tile_origins):
            raise ValueError("Stage terrain origins must stay fixed")
        position = self.env.asset.data.root_pos_w[ids, :2]
        inside = torch.isfinite(position).all(dim=1) & ((position - self.tile_origins[ids]).abs() <= self.tile_limits).all(dim=1)
        self.traversal.valid[ids] &= inside

    def _finish_traversal(self, ids: Sequence[int] | torch.Tensor, *, for_reset: bool = False) -> None:
        if self.traversal is not None:
            self._check_tile_bounds(ids)
            heights, contacts, valid = self._support_state()
            # Invalid sensor or scan data cannot be relabeled as ordinary air
            # time and later earn a stage pass.
            self.traversal.valid[ids] &= valid[ids].all(dim=1)
            self.traversal.finish_step(ids, self.env.asset.data.root_pos_w, heights, contacts, for_reset=for_reset)

    def finish_step(self) -> None:
        if self.traversal is not None:
            ids = self.traversal.pending.nonzero(as_tuple=False).flatten()
            if ids.numel():
                self._finish_traversal(ids)

    def abort_step(self) -> None:
        if self.traversal is not None:
            self.traversal.abort_step()
            # A parent exception after a terminal callback must not leave a
            # promotion recommendation from an incomplete environment step.
            self.latest_stage = None

    def reset(self, ids: Sequence[int] | torch.Tensor) -> None:
        if self.traversal is not None:
            self.traversal.reset(ids)

    def _stage_logs(self) -> dict[str, int | float]:
        latest = self.latest_stage or {}
        logs = {"Stage/index": STAGE_ORDER.index(self.evaluator.stage),
                "Stage/evaluation_id": self.evaluator.evaluation_count,
                "Stage/has_evaluation": float(self.latest_stage is not None),
                "Stage/passed": float(latest.get("passed", False)),
                "Stage/next_index": STAGE_ORDER.index(latest["next_stage"]) if latest.get("next_stage") else -1,
                "Stage/window_checkpointed": 0.0}
        logs["Stage/recommendation_only"] = 1.0
        for group, name in enumerate(FAMILY_NAMES):
            logs[f"Stage/{name}_required"] = float(group in self.evaluator.required_families)
            logs[f"Stage/{name}_pending"] = self.evaluator.group_episode_counts.get(group, 0)
            logs[f"Stage/{name}_episodes"] = latest.get("group_episodes", {}).get(group, 0)
            logs[f"Stage/{name}_success_rate"] = latest.get("group_success_rates", {}).get(group, 0.0)
        return logs

    @torch.no_grad()
    def before_reset(self, env_ids: Sequence[int] | torch.Tensor) -> tuple[dict[str, int | float], dict[str, object] | None]:
        """Return logs and a bounded detached snapshot before parent reset mutates state."""
        env = self.env
        ids = torch.as_tensor(env_ids, device=env.device)
        if ids.ndim != 1 or (ids.numel() and ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Terminal environment IDs must be a one-dimensional integer sequence")
        ids = ids.to(torch.long).unique()
        if bool(((ids < 0) | (ids >= env.num_envs)).any()):
            raise ValueError("Terminal environment IDs are out of range")
        self._finish_traversal(ids, for_reset=True)
        logs = {}
        if self.evaluator is not None:
            if not torch.equal(env.scene.terrain.terrain_types, self.column_ids):
                raise ValueError("Stage terrain columns must stay fixed")
            result = self.evaluator.observe(
                env.command_progress_tracker, self.traversal, ids, family_ids=self.family_ids,
                terminated=env.termination_manager.terminated, time_outs=env.termination_manager.time_outs,
                max_episode_length_s=env.max_episode_length_s, current_step=env.common_step_counter,
            )
            if result is not None:
                self.latest_stage = result
            logs.update(self._stage_logs())
        natural = (env.command_progress_tracker.reset_ready[ids]
                   & (env.termination_manager.terminated[ids] | env.termination_manager.time_outs[ids]))
        selected = ids[natural]
        snapshot = None
        if self.diagnostics and selected.numel():
            classified = classify_pelvis_contact(
                self.net_sensor.data.net_forces_w_history[selected][:, :, self.pelvis_id],
                self.sensors["pelvis_ground"].data.force_matrix_w_history[selected],
                self.sensors["pelvis_robot"].data.force_matrix_w_history[selected], self.threshold,
            )
            valid = classified["valid"]
            count_tensors = [torch.tensor(selected.numel(), device=env.device), valid.sum(),
                             classified["net_trigger"].sum(),
                             (classified["ground_at_trigger"] & valid).sum(),
                             (classified["self_at_trigger"] & valid).sum(),
                             (classified["both_same_sample"] & valid).sum(),
                             (classified["unattributed_at_trigger"] & valid).sum(), (~valid).sum()]
            for name, value in zip(self.counts, torch.stack(count_tensors).cpu().tolist()):
                self.counts[name] += int(value)
            sample_ids = selected[:self.max_samples]
            take = min(self.max_samples, selected.numel())
            if take:
                data = env.asset.data
                processed = env.action_manager.get_term("JointPositionAction").processed_actions
                pelvis_ground = self._ground_heights(data.root_pos_w[:, None])[:, 0]
                metrics = terminal_state_metrics(
                    pelvis_position_w=data.root_pos_w[sample_ids], projected_gravity_b=data.projected_gravity_b[sample_ids],
                    joint_position_error=(data.joint_pos - data.default_joint_pos)[sample_ids][:, self.joint_ids],
                    joint_velocity=data.joint_vel[sample_ids][:, self.joint_ids],
                    applied_torque=data.applied_torque[sample_ids][:, self.joint_ids],
                    feet_force_w=self.net_sensor.data.net_forces_w[sample_ids][:, self.feet_ids],
                    feet_velocity_w=data.body_lin_vel_w[sample_ids][:, env.feet_id_b],
                    raw_action=env.action_manager.action[sample_ids], processed_action=processed[sample_ids],
                    ground_height=pelvis_ground[sample_ids], threshold=self.threshold,
                )
                snapshot = {"env_ids": sample_ids.detach().clone(), "step": env.common_step_counter,
                            "episode_seconds": env.episode_length_buf[sample_ids].detach().clone() * env.step_dt,
                            "pelvis_position_w": data.root_pos_w[sample_ids].detach().clone(),
                            "joint_position": data.joint_pos[sample_ids][:, self.joint_ids].detach().clone(),
                            "joint_velocity": data.joint_vel[sample_ids][:, self.joint_ids].detach().clone(),
                            "joint_torque": data.applied_torque[sample_ids][:, self.joint_ids].detach().clone(),
                            "raw_action": env.action_manager.action[sample_ids].detach().clone(),
                            "processed_action": processed[sample_ids].detach().clone(),
                            "terminated": env.termination_manager.terminated[sample_ids].detach().clone(),
                            "time_outs": env.termination_manager.time_outs[sample_ids].detach().clone(),
                            "joint_names": tuple(env.cfg.walking_diagnostic_joint_names),
                            **capture_terminal_predicates(env, sample_ids),
                            **metrics, **{key: value[:take].detach().clone() for key, value in classified.items()}}
                heights, support, support_valid = self._support_state()
                snapshot["filtered_ground_support"] = support[sample_ids].detach().clone()
                snapshot["ground_support_valid"] = support_valid[sample_ids].detach().clone()
                snapshot["foot_ground_height"] = heights[sample_ids].detach().clone()
                snapshot["pelvis_ground_force_history"] = self.sensors["pelvis_ground"].data.force_matrix_w_history[sample_ids].detach().clone()
                snapshot["pelvis_robot_force_history"] = self.sensors["pelvis_robot"].data.force_matrix_w_history[sample_ids].detach().clone()
                snapshot["pelvis_robot_filter_paths"] = tuple(self.sensors["pelvis_robot"].cfg.filter_prim_paths_expr)
                if self.family_ids is not None:
                    snapshot["family_ids"] = self.family_ids[sample_ids].detach().clone()
                    snapshot["terrain_columns"] = self.column_ids[sample_ids].detach().clone()
                terrain = getattr(env.scene, "terrain", None)
                if terrain is not None:
                    snapshot["terrain_columns"] = terrain.terrain_types[sample_ids].detach().clone()
                    snapshot["terrain_rows"] = terrain.terrain_levels[sample_ids].detach().clone()
        if self.diagnostics:
            logs.update({f"ContactDiagnostics/{name}": value for name, value in self.counts.items()})
        return logs, snapshot
