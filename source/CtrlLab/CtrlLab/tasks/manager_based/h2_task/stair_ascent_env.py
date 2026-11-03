"""A policy-owned stair environment with pre-reset physical progress capture."""

import torch
from pathlib import Path

from .direct_leg_action import DirectLegPositionAction
from .robot_env import RobotEnv
from .stair_ascent_progress import StairAscentProgress, foot_geometry, quaternion_matrix
from .stair_ascent_safety import tile_domain_bounds


class StairAscentEnv(RobotEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        term = self.action_manager.get_term("JointPositionAction")
        if not isinstance(term, DirectLegPositionAction) or self.action_manager.total_action_dim != 12:
            raise RuntimeError("StairAscentEnv must use twelve direct leg-position actions")
        if cfg.contact_gait_reference is not None or cfg.stair_reference is not None:
            raise RuntimeError("The policy stair task cannot install a scripted walking reference")
        generator = self.scene.terrain.cfg.terrain_generator
        if generator is None or "ascending_stairway" not in generator.sub_terrains:
            raise RuntimeError("The policy stair task requires its actual ascending stair generator")
        geometry = generator.sub_terrains["ascending_stairway"]
        self._stair_minimum_height = float(geometry.minimum_riser_height)
        self._stair_maximum_height = float(geometry.maximum_riser_height)
        self._stair_height_rows = int(geometry.height_levels)
        if generator.num_rows != self._stair_height_rows:
            raise RuntimeError("Terrain row count and stair height metadata disagree")
        self._stair_domain_bounds_local = tile_domain_bounds(
            generator.size, geometry.spawn_origin_x, geometry.top_end_margin,
            cfg.stair_domain_safety_inset_m)
        self._stair_foot_ids, names = self.asset.find_bodies(cfg.feet_body_names, preserve_order=True)
        if len(names) != 2:
            raise RuntimeError("Stair progress requires the two actual ankle-pitch foot bodies")
        self._stair_unfiltered_ids, _ = self.scene["contact_forces"].find_bodies(
            cfg.feet_body_names, preserve_order=True)
        self._stair_sensors = [self.scene["stair_ascent_left_ground"], self.scene["stair_ascent_right_ground"]]
        self._stair_sole_offset, _ = foot_geometry(self.device)
        output = getattr(cfg, "io_descriptors_output_dir", None)
        self.stair_episode_evidence_directory = str(Path(output)/"stair_episode_evidence") if output else None
        self.stair_progress = StairAscentProgress(
            self.num_envs, self.device, self.masses, steps=int(geometry.num_steps),
            depth=float(geometry.tread_depth), width=float(geometry.stair_width),
            approach=float(geometry.approach_distance), episode_length_s=cfg.episode_length_s,
            gamma=float(cfg.stair_shaping_gamma),
            domain_bounds=self._stair_domain_bounds_local,
            floor_tolerance=cfg.stair_below_floor_tolerance_m,
            record_directory=self.stair_episode_evidence_directory,
            contact_body_names=self.scene["contact_forces"].body_names,
            strict_episode_force=getattr(cfg, "stair_strict_episode_force", False),
            clearance_progress_potential=getattr(cfg, "stair_clearance_progress_potential", False),
        )
        context = int(getattr(cfg, "stair_task_context_dim", 0))
        if context not in (0, 5):
            raise RuntimeError("Undeclared stair context width")
        expected = {"policy": (1029+context,), "critic": (866+context,), "rnd_state": (73,)}
        for name, shape in expected.items():
            actual = tuple(self.observation_manager.group_obs_dim[name])
            if actual != shape:
                raise RuntimeError(f"Stair observation contract changed: {name}={actual}, expected {shape}")

    def _stair_foot_pose(self):
        rotation = quaternion_matrix(self.asset.data.body_quat_w[:, self._stair_foot_ids])
        sole = (self.asset.data.body_pos_w[:, self._stair_foot_ids]
                + torch.einsum("nfij,fj->nfi", rotation, self._stair_sole_offset))
        return sole, rotation

    def _stair_heights(self):
        row = self.scene.terrain.terrain_levels
        return (self._stair_minimum_height + (self._stair_maximum_height-self._stair_minimum_height)
                * row / (self._stair_height_rows-1))

    def _update_stair_progress(self):
        if not hasattr(self, "stair_progress") or not self.stair_progress.pending:
            return
        sole, rotation = self._stair_foot_pose()
        forces = []
        for sensor in self._stair_sensors:
            matrix = sensor.data.force_matrix_w
            if matrix is None or matrix.ndim != 4 or tuple(matrix.shape) != (self.num_envs, 1, 1, 3):
                raise RuntimeError("Stair ground-filtered contact sensor has no valid N x 1 x filters x 3 force matrix")
            forces.append(matrix.sum(dim=(1, 2)))
        filtered = torch.stack(forces, dim=1)
        unfiltered = self.scene["contact_forces"].data.net_forces_w[:, self._stair_unfiltered_ids]
        tilt = torch.acos((-self.asset.data.projected_gravity_b[:, 2]).clamp(-1, 1))
        velocity = (self.asset.data.root_link_lin_vel_w if self.stair_progress.strict_episode_force
                    else self.asset.data.root_lin_vel_w)
        self.stair_progress.update(
            self.common_step_counter, self.asset.data.root_pos_w, velocity,
            tilt, sole, rotation, filtered, unfiltered, self.termination_manager.terminated, self.step_dt,
            body_forces=self.scene["contact_forces"].data.net_forces_w,
            body_contact_time=self.scene["contact_forces"].data.current_contact_time,
            timed_out=self.termination_manager.time_outs,
        )

    def _reset_idx(self, env_ids):
        if hasattr(self, "stair_progress"):
            self._update_stair_progress()
            self.stair_progress.assert_clearance_reset(env_ids)
            # Curriculum runs inside super()._reset_idx, so it reads this exact
            # terminal epoch rather than reset positions or a later sensor map.
            self.stair_progress.record_episodes(
                env_ids, self.termination_manager.terminated, self.termination_manager.time_outs,
                termination_bits={name: self.termination_manager.get_term(name)
                                  for name in self.termination_manager.active_terms})
        super()._reset_idx(env_ids)
        if hasattr(self, "stair_progress"):
            self.stair_progress.reset(env_ids)

    def step(self, action):
        sole, rotation = self._stair_foot_pose()
        self.stair_progress.begin_step(
            self.asset.data.root_pos_w, sole, self.scene.terrain.env_origins,
            self._stair_heights(), self.scene.terrain.terrain_levels, self.episode_length_buf,
            control_counter=self.common_step_counter, rotations=rotation)
        result = super().step(action)
        self.stair_progress.episode_evidence.maybe_flush(self.common_step_counter)
        return result

    def close(self):
        try:
            if hasattr(self, "stair_progress"):
                self.stair_progress.episode_evidence.close()
        finally:
            super().close()
