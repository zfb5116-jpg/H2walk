"""Capture contemporaneous stair-entry and first-swing map evidence.

This module imports no simulator. Hooks only copy current buffers; in particular,
they never call map.update(), map.surfaces(), or a controller query a second time.
All retained data is an owned CPU NumPy copy, including under inference_mode().
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch


_LAYERS = {
    "safe": (np.bool_, False),
    "surface": (np.float32, np.nan),
    "minimum": (np.float32, np.inf),
    "maximum": (np.float32, -np.inf),
    "first_seen": (np.int32, 1_000_000),
    "last_seen": (np.int32, -1),
    "modelled": (np.bool_, False),
    "model_height": (np.float32, np.nan),
    "tactile_known": (np.bool_, False),
    "tactile_height": (np.float32, np.nan),
    "tactile_invalid": (np.bool_, False),
    "fusion_conflict": (np.bool_, False),
}


def _cpu(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def _rows(value, ids):
    if torch.is_tensor(value):
        index = torch.as_tensor(ids, dtype=torch.long, device=value.device)
        return _cpu(value[index])
    return _cpu(np.asarray(value)[ids])


def _rotate(quaternion, vector):
    """Rotate CPU copies using the controller's wxyz quaternion convention."""
    imaginary = quaternion[..., 1:]
    return vector + 2 * np.cross(imaginary, np.cross(imaginary, vector) + quaternion[..., :1] * vector)


class StairEntryEvidence:
    """CPU-owned one-time entry and first-swing snapshots, never backfilled."""

    columns = 56
    rows = 28

    def __init__(self, reference, raw, robot):
        self.reference, self.raw, self.robot = reference, raw, robot
        self.mapping = reference.map
        if not math.isclose(float(self.mapping.resolution), .025, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Entry evidence requires the configured 25mm map grid.")
        self.joint_ids = _cpu(reference.joint_ids).astype(np.int64).reshape(-1)
        if self.joint_ids.size != 12:
            raise ValueError("Entry evidence requires the twelve actual lower-body joint IDs.")
        self.seen = set()
        self.query_cache = {}
        self.chunks = []
        self.landing_chunks = []
        self.first_swings = {}
        self.landing_seen = set()
        self.installation_step = int(raw.common_step_counter)
        episode_length = getattr(raw, "episode_length_buf", None)
        self.initial_episode_lengths = _cpu(episode_length) if episode_length is not None else None
        self.errors = []
        self.original_entry_plan = reference._entry_plan
        self.original_hold = reference._hold
        self.original_over_target_patch = getattr(reference, "_over_target_patch", None)

    def _error(self, stage, ids, error):
        self.errors.append({"stage": stage, "env_ids": _cpu(ids).reshape(-1).tolist(),
                            "common_step_counter": int(self.raw.common_step_counter),
                            "error": repr(error)})

    def remember_query(self, ids, soles):
        indices = _cpu(ids).astype(np.int64).reshape(-1)
        positions = _rows(soles, indices)
        step = int(self.raw.common_step_counter)
        for row, env_id in enumerate(indices.tolist()):
            if env_id not in self.seen:
                self.query_cache[env_id] = {
                    "soles": positions[row].copy(), "step": step,
                    "clock": int(self.mapping.clock), "cached_clock": int(self.mapping.cached_step),
                }

    def remember_result(self, ids, result):
        indices = _cpu(ids).astype(np.int64).reshape(-1)
        preparation, first_pair, ready = (_cpu(value) for value in result)
        for row, env_id in enumerate(indices.tolist()):
            cached = self.query_cache.get(env_id)
            if cached is not None and env_id not in self.seen:
                cached.update(preparation=preparation[row].copy(), first_pair=first_pair[row].copy(),
                              ready=bool(ready[row]), cached_clock=int(self.mapping.cached_step))

    def capture(self, ids, com, velocity, soles, code):
        if int(code) != 11:
            return
        indices = _cpu(ids).astype(np.int64).reshape(-1)
        approaching = _rows(self.reference.approach, indices).astype(bool)
        indices = np.array([e for e, approach in zip(indices.tolist(), approaching.tolist())
                            if approach and e not in self.seen], dtype=np.int64)
        for begin in range(0, indices.size, 64):
            selected = indices[begin:begin + 64]
            chunk = self._capture_chunk(selected, com, velocity, soles)
            self.chunks.append(chunk)
            self.seen.update(selected.tolist())
            for env_id in selected.tolist():
                self.query_cache.pop(env_id, None)

    def _capture_chunk(self, ids, com, velocity, soles, requested_bounds=None):
        mapping, count = self.mapping, ids.size
        step = int(self.raw.common_step_counter)
        hold_soles = _rows(soles, ids)
        origins = _rows(self.raw.scene.env_origins, ids)
        queries = [self.query_cache.get(int(e)) for e in ids]
        available = np.array([q is not None and q["step"] == step for q in queries])
        query_soles = np.stack([q["soles"] if valid else np.full((2, 3), np.nan, np.float32)
                                for q, valid in zip(queries, available)])
        # Missing/stale query is explicitly marked. The current hold soles can
        # locate its crop, but are never relabelled as original query soles.
        centers = np.where(available[:, None, None], query_soles, hold_soles) - origins[:, None]
        front_x = centers[..., 0].max(axis=1)
        mean_y = centers[..., 1].mean(axis=1)
        requested = np.stack((front_x - .45, front_x + .95, mean_y - .35, mean_y + .35), axis=-1)
        if requested_bounds is not None:
            requested = requested_bounds
        resolution = float(mapping.resolution)
        # Select cell centres in the requested half-open intervals. Exactly
        # 56x28 cells are retained; out-of-map cells have inside=False.
        ix0 = np.ceil((requested[:, 0] - mapping.x0) / resolution - .5).astype(np.int64)
        iy0 = np.ceil((requested[:, 2] - mapping.y0) / resolution - .5).astype(np.int64)
        ix = np.broadcast_to(ix0[:, None, None] + np.arange(self.columns)[None, None],
                             (count, self.rows, self.columns)).copy()
        iy = np.broadcast_to(iy0[:, None, None] + np.arange(self.rows)[None, :, None],
                             (count, self.rows, self.columns)).copy()
        inside = (ix >= 0) & (ix < mapping.nx) & (iy >= 0) & (iy < mapping.ny)
        index = np.clip(iy, 0, mapping.ny - 1) * mapping.nx + np.clip(ix, 0, mapping.nx - 1)
        chunk = {
            "env_id": ids.astype(np.int32), "inside": inside,
            "cell_ix": ix.astype(np.int32), "cell_iy": iy.astype(np.int32),
            "cell_x": (mapping.x0 + (ix + .5) * resolution).astype(np.float32),
            "cell_y": (mapping.y0 + (iy + .5) * resolution).astype(np.float32),
            "requested_bounds_xy": requested.astype(np.float32),
            "origin_w": origins.astype(np.float32),
            "query_soles_w": query_soles.astype(np.float32),
            "hold_soles_w": hold_soles.astype(np.float32),
            "query_available_same_step": available,
            "query_common_step_counter": np.array([q["step"] if valid else -1 for q, valid in zip(queries, available)], np.int64),
            "query_clock": np.array([q["clock"] if valid else -1 for q, valid in zip(queries, available)], np.int64),
            "query_cached_clock": np.array([q["cached_clock"] if valid else -1 for q, valid in zip(queries, available)], np.int64),
            "clock": np.full(count, int(mapping.clock), np.int64),
            "cached_clock": np.full(count, int(mapping.cached_step), np.int64),
            "common_step_counter": np.full(count, step, np.int64),
            "time_s": np.full(count, step * float(self.raw.step_dt), np.float64),
            "actual_com_w": _rows(com, ids).astype(np.float32),
            "actual_velocity_w": _rows(velocity, ids).astype(np.float32),
            "root_state_w": _rows(self.robot.data.root_state_w, ids).astype(np.float32),
            "entry_diagnostics": _rows(self.reference.entry_plan_diagnostics, ids).astype(np.float32),
            "approach": _rows(self.reference.approach, ids).astype(bool),
            "query_ready": np.array([q.get("ready", False) if valid else False
                                      for q, valid in zip(queries, available)], dtype=bool),
        }
        episode_length = getattr(self.raw, "episode_length_buf", None)
        chunk["episode_length_steps"] = (_rows(episode_length, ids).astype(np.int64)
                                           if episode_length is not None else np.full(count, -1, np.int64))
        for key, query_key in (("query_preparation_world", "preparation"), ("query_first_pair_world", "first_pair")):
            chunk[key] = np.stack([q[query_key] if valid and query_key in q else np.full((2, 3), np.nan)
                                   for q, valid in zip(queries, available)]).astype(np.float32)
        for name, (dtype, fill) in _LAYERS.items():
            value = getattr(mapping, name, None)
            if value is None:
                chunk[name] = np.full(inside.shape, fill, dtype=dtype)
                continue
            env_index = torch.as_tensor(ids[:, None], dtype=torch.long, device=value.device)
            cell_index = torch.as_tensor(index.reshape(count, -1), dtype=torch.long, device=value.device)
            selected = _cpu(value[env_index, cell_index]).reshape(inside.shape)
            chunk[name] = np.where(inside, selected, fill).astype(dtype)
        for key, attribute in (("joint_position12", "joint_pos"), ("joint_velocity12", "joint_vel"),
                               ("joint_position_target12", "joint_pos_target"),
                               ("joint_velocity_target12", "joint_vel_target")):
            value = getattr(self.robot.data, attribute, None)
            chunk[key] = (_rows(value, ids)[:, self.joint_ids].astype(np.float32)
                          if value is not None else np.full((count, 12), np.nan, np.float32))
        for key, attribute, shape in (("approach_block_reason", "approach_block_reason", ()),
                                      ("approach_obstacle_x", "approach_obstacle_x", ()),
                                      ("reference_contact_age", "contact_age", (2,))):
            value = getattr(self.reference, attribute, None)
            chunk[key] = (_rows(value, ids).astype(np.float32) if value is not None
                          else np.full((count,) + shape, np.nan, np.float32))
        flat = getattr(self.reference, "flat", None)
        for key, attribute, shape in (("flat_origin_w", "origin", (3,)), ("heading_w", "heading", (4,))):
            value = getattr(flat, attribute, None)
            chunk[key] = (_rows(value, ids).astype(np.float32) if value is not None
                          else np.full((count,) + shape, np.nan, np.float32))
        return chunk

    def _first_episode(self, ids):
        length = getattr(self.raw, "episode_length_buf", None)
        known = length is not None and self.initial_episode_lengths is not None
        if not known:
            return np.zeros(len(ids), bool), np.zeros(len(ids), bool)
        expected = self.initial_episode_lengths[ids] + int(self.raw.common_step_counter) - self.installation_step
        return _rows(length, ids) == expected, np.ones(len(ids), bool)

    def capture_landing(self, ids, soles, original_over):
        """Record two time bins from the same first observed swing of each kind."""
        indices = _cpu(ids).astype(np.int64).reshape(-1)
        state = _rows(self.reference.state, indices)
        phase = _rows(self.reference.time, indices)
        preparing = _rows(self.reference.preparing, indices).astype(bool)
        side = _rows(self.reference.swing_side, indices).astype(np.int64)
        target = _rows(self.reference.swing_target, indices)
        start = _rows(self.reference.swing_start, indices)
        over = _cpu(original_over).astype(bool).reshape(-1)
        first_episode, episode_known = self._first_episode(indices)
        step, dt = int(self.raw.common_step_counter), float(self.raw.step_dt)
        groups = {}
        for row, env_id in enumerate(indices.tolist()):
            if state[row] != self.reference.SWING or not np.isfinite(phase[row]):
                continue  # LAND also invokes the original patch check.
            kind = 0 if preparing[row] else 1
            if kind == 1 and target[row, 2] - start[row, 2] <= .04:
                continue
            token = (step - int(round(float(phase[row]) / dt)), int(side[row]))
            key = (env_id, kind)
            if key not in self.first_swings:
                self.first_swings[key] = {
                    "token": token, "first_phase_s": float(phase[row]), "last_phase_s": float(phase[row]),
                    "first_episode": bool(first_episode[row]), "first_episode_known": bool(episode_known[row]),
                }
            first = self.first_swings[key]
            if token != first["token"]:
                continue  # A later swing must never fill the first swing's missing bin.
            first["last_phase_s"] = max(first["last_phase_s"], float(phase[row]))
            if episode_known[row] and not first_episode[row]:
                continue
            for stage, threshold in enumerate((.32, .52)):
                event = (env_id, kind, stage)
                if phase[row] >= threshold - 1e-6 and event not in self.landing_seen:
                    groups.setdefault((kind, stage), []).append(row)
        for (kind, stage), positions in groups.items():
            for begin in range(0, len(positions), 64):
                positions_chunk = np.array(positions[begin:begin + 64], dtype=np.int64)
                selected = indices[positions_chunk]
                chunk = self._landing_chunk(selected, soles, kind, stage, phase[positions_chunk],
                                            over[positions_chunk], first_episode[positions_chunk],
                                            episode_known[positions_chunk])
                self.landing_chunks.append(chunk)
                self.landing_seen.update((int(e), kind, stage) for e in selected)

    @staticmethod
    def _remove_entry_fields(chunk):
        for key in list(chunk):
            if key.startswith("query_") or key in ("entry_diagnostics", "approach_block_reason", "approach_obstacle_x"):
                chunk.pop(key)
        chunk["actual_soles_w"] = chunk.pop("hold_soles_w")
        return chunk

    def _landing_chunk(self, ids, soles, kind, stage, phase, original_over, first_episode, episode_known):
        count, reference = len(ids), self.reference
        origin = _rows(self.raw.scene.env_origins, ids)
        heading, flat_origin = _rows(reference.flat.heading, ids), _rows(reference.flat.origin, ids)
        target_local = _rows(reference.swing_target, ids)
        target_world = flat_origin + _rotate(heading, target_local)
        target_map = target_world - origin
        bounds = np.stack((target_map[:, 0] - .70, target_map[:, 0] + .70,
                           target_map[:, 1] - .35, target_map[:, 1] + .35), -1)
        chunk = self._remove_entry_fields(self._capture_chunk(
            ids, reference.actual_com, reference.actual_velocity, soles, requested_bounds=bounds))
        side = _rows(reference.swing_side, ids).astype(np.int64)
        feet = _cpu(reference.flat.feet).astype(np.int64)
        quaternions = _rows(self.robot.data.body_quat_w, ids)[:, feet]
        swing_quat = quaternions[np.arange(count), side]
        swing_sole = chunk["actual_soles_w"][np.arange(count), side]
        corners = _cpu(reference.foot_corners)[side]
        world_corners = swing_sole[:, None] + _rotate(swing_quat[:, None], corners)
        plane = target_world.copy()
        plane[:, 2] += float(reference.cfg["ground_contact_depth"])
        ix = np.floor((target_map[:, 0] - self.mapping.x0) / self.mapping.resolution).astype(np.int64)
        iy = np.floor((target_map[:, 1] - self.mapping.y0) / self.mapping.resolution).astype(np.int64)
        inside = (ix >= 0) & (ix < self.mapping.nx) & (iy >= 0) & (iy < self.mapping.ny)
        cell = np.clip(iy, 0, self.mapping.ny - 1) * self.mapping.nx + np.clip(ix, 0, self.mapping.nx - 1)
        env_index = torch.as_tensor(ids, dtype=torch.long, device=self.mapping.surface.device)
        cell_index = torch.as_tensor(cell, dtype=torch.long, device=self.mapping.surface.device)
        surface = _cpu(self.mapping.surface[env_index, cell_index])
        safe = _cpu(self.mapping.safe[env_index, cell_index])
        chunk.update(
            stage=np.full(count, stage, np.int8), kind=np.full(count, kind, np.int8),
            phase_threshold_s=np.full(count, (.32, .52)[stage], np.float32),
            phase_s=phase.astype(np.float32), original_over=original_over.astype(bool),
            swing_side=side.astype(np.int8),
            swing_start_step=np.array([self.first_swings[(int(e), kind)]["token"][0] for e in ids], np.int64),
            first_episode=first_episode.astype(bool), first_episode_known=episode_known.astype(bool),
            swing_target_w=target_world.astype(np.float32), target_plane_w=plane.astype(np.float32),
            swing_start_w=(flat_origin + _rotate(heading, _rows(reference.swing_start, ids))).astype(np.float32),
            actual_foot_quaternions_w=quaternions.astype(np.float32),
            actual_swing_sole_w=swing_sole.astype(np.float32), actual_swing_quaternion_w=swing_quat.astype(np.float32),
            actual_swing_corners_w=world_corners.astype(np.float32),
            target_cell_inside=inside, target_cached_safe=inside & safe,
            target_cached_surface_z_m=np.where(inside, surface, np.nan).astype(np.float32),
        )
        return chunk

    def _empty_arrays(self):
        arrays = {"env_id": np.empty(0, np.int32),
                  "inside": np.empty((0, self.rows, self.columns), bool)}
        for key, dtype in (("cell_ix", np.int32), ("cell_iy", np.int32),
                           ("cell_x", np.float32), ("cell_y", np.float32)):
            arrays[key] = np.empty((0, self.rows, self.columns), dtype)
        for key, (dtype, _) in _LAYERS.items():
            arrays[key] = np.empty((0, self.rows, self.columns), dtype)
        for key in ("query_clock", "query_cached_clock", "query_common_step_counter",
                    "clock", "cached_clock", "common_step_counter", "episode_length_steps"):
            arrays[key] = np.empty(0, np.int64)
        for key in ("query_available_same_step", "query_ready", "approach"):
            arrays[key] = np.empty(0, bool)
        arrays["time_s"] = np.empty(0, np.float64)
        for key, shape in {
            "requested_bounds_xy": (4,), "origin_w": (3,), "query_soles_w": (2, 3),
            "hold_soles_w": (2, 3), "actual_com_w": (3,), "actual_velocity_w": (3,),
            "root_state_w": (13,), "entry_diagnostics": (6,),
            "query_preparation_world": (2, 3), "query_first_pair_world": (2, 3),
            "joint_position12": (12,), "joint_velocity12": (12,),
            "joint_position_target12": (12,), "joint_velocity_target12": (12,),
            "approach_block_reason": (), "approach_obstacle_x": (),
            "reference_contact_age": (2,), "flat_origin_w": (3,), "heading_w": (4,),
        }.items():
            arrays[key] = np.empty((0,) + shape, np.float32)
        return arrays

    def _empty_landing_arrays(self):
        arrays = self._remove_entry_fields(self._empty_arrays())
        for key, dtype in (("stage", np.int8), ("kind", np.int8), ("swing_side", np.int8),
                           ("phase_threshold_s", np.float32), ("phase_s", np.float32),
                           ("swing_start_step", np.int64), ("original_over", bool),
                           ("first_episode", bool), ("first_episode_known", bool),
                           ("target_cell_inside", bool), ("target_cached_safe", bool),
                           ("target_cached_surface_z_m", np.float32)):
            arrays[key] = np.empty(0, dtype)
        for key, shape in {"swing_target_w": (3,), "target_plane_w": (3,), "swing_start_w": (3,),
                           "actual_foot_quaternions_w": (2, 4), "actual_swing_sole_w": (3,),
                           "actual_swing_quaternion_w": (4,), "actual_swing_corners_w": (4, 3)}.items():
            arrays[key] = np.empty((0,) + shape, np.float32)
        return arrays

    def _save_landing(self, path):
        arrays = ({key: np.concatenate([chunk[key] for chunk in self.landing_chunks], axis=0)
                   for key in self.landing_chunks[0]} if self.landing_chunks else self._empty_landing_arrays())
        first = sorted(self.first_swings.items())
        arrays.update(
            first_swing_env_id=np.array([key[0] for key, _ in first], np.int32),
            first_swing_kind=np.array([key[1] for key, _ in first], np.int8),
            first_swing_start_step=np.array([value["token"][0] for _, value in first], np.int64),
            first_swing_side=np.array([value["token"][1] for _, value in first], np.int8),
            first_swing_first_phase_s=np.array([value["first_phase_s"] for _, value in first], np.float32),
            first_swing_last_phase_s=np.array([value["last_phase_s"] for _, value in first], np.float32),
            first_swing_first_episode=np.array([value["first_episode"] for _, value in first], bool),
            first_swing_first_episode_known=np.array([value["first_episode_known"] for _, value in first], bool),
            first_swing_captured_stage_mask=np.array([
                [(key[0], key[1], stage) in self.landing_seen for stage in range(2)]
                for key, _ in first], bool).reshape(-1, 2),
            joint_ids=self.joint_ids.copy(), resolution_m=np.array(float(self.mapping.resolution)),
            map_shape_yx=np.array([self.mapping.ny, self.mapping.nx], np.int32),
            grid_origin_xy=np.array([self.mapping.x0, self.mapping.y0], np.float64),
            kind_labels=np.array(["first_preparation_swing", "first_high_swing"]),
            stage_labels=np.array(["phase_ge_0p32_seconds", "phase_ge_0p52_seconds"]),
            foot_corners_local_m=(_cpu(self.reference.foot_corners) if hasattr(self.reference, "foot_corners")
                                  else np.empty((0, 4, 3), np.float32)),
            errors_json=np.array(json.dumps(self.errors, ensure_ascii=False)),
            capture_rule=np.array("post original patch check, SWING only, first same-swing time bins; later swings cannot fill missing bins; known subsequent episodes excluded"),
            plane_rule=np.array("target_plane_w is controller sole target plus configured penetration depth; target_cached_surface_z_m is the unchanged cached map plane in env-local Z"),
            first_episode_rule=np.array("episode_length equals installation length plus global-step delta; unknown is explicitly flagged"),
            cache_rule=np.array("all map layers copied as-is after the original query; no extra surfaces/update/extent call"),
        )
        np.savez_compressed(path, **arrays)

    def save(self, path):
        """Write entry and adjacent landing files from the owned CPU chunks."""
        path = Path(path)
        if path.suffix.lower() != ".npz":
            path = path / "entry_windows.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = ({key: np.concatenate([chunk[key] for chunk in self.chunks], axis=0)
                   for key in self.chunks[0]} if self.chunks else self._empty_arrays())
        arrays.update(
            joint_ids=self.joint_ids.copy(), resolution_m=np.array(float(self.mapping.resolution)),
            map_shape_yx=np.array([self.mapping.ny, self.mapping.nx], dtype=np.int32),
            grid_origin_xy=np.array([self.mapping.x0, self.mapping.y0], dtype=np.float64),
            errors_json=np.array(json.dumps(self.errors, ensure_ascii=False)),
            capture_rule=np.array("first code11 while approach=True; before original hold; per-env once across resets"),
            cache_rule=np.array("safe/surface/model layers copied as-is; cached_clock can lag current raw evidence; no map methods called"),
        )
        np.savez_compressed(path, **arrays)
        self._save_landing(path.with_name("landing_windows.npz"))
        return path


def install_entry_evidence(reference, raw, robot):
    """Install transparent hooks and return the recorder (idempotent)."""
    installed = getattr(reference, "_entry_evidence_recorder", None)
    if installed is not None:
        return installed
    recorder = StairEntryEvidence(reference, raw, robot)

    def entry_plan_hook(self, ids, soles):
        try:
            recorder.remember_query(ids, soles)
        except Exception as error:
            recorder._error("query", ids, error)
        result = recorder.original_entry_plan(ids, soles)
        try:
            recorder.remember_result(ids, result)
        except Exception as error:
            recorder._error("query_result", ids, error)
        return result

    def hold_hook(self, ids, com, velocity, soles, code):
        try:
            recorder.capture(ids, com, velocity, soles, code)
        except Exception as error:
            recorder._error("pre_hold_capture", ids, error)
        return recorder.original_hold(ids, com, velocity, soles, code)

    def over_target_patch_hook(self, ids, soles):
        result = recorder.original_over_target_patch(ids, soles)
        try:
            recorder.capture_landing(ids, soles, result)
        except Exception as error:
            recorder._error("landing_capture", ids, error)
        return result

    reference._entry_plan = MethodType(entry_plan_hook, reference)
    reference._hold = MethodType(hold_hook, reference)
    if recorder.original_over_target_patch is not None:
        reference._over_target_patch = MethodType(over_target_patch_hook, reference)
    reference._entry_evidence_recorder = recorder
    return recorder
