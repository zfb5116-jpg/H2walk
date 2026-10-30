"""Policy-owned generated-flat plant for bounded closed-loop initialization.

The old accepted flat controller is deliberately absent from this action path.
It may be queried externally as a teacher; only the original direct twelve
position coordinates can write the plant's leg targets.
"""
from copy import deepcopy

import torch
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.terrains import TerrainGeneratorCfg, MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from .config.terminations_cfg import TerminationsCfg
from .direct_leg_action import DirectLegPositionAction
from .robot_env import RobotEnv
from .stair_ascent_cfg import StairAscentEnvCfg
from .stair_ascent_progress import foot_geometry, quaternion_matrix
from .stair_ascent_safety import physical_validity


def initialization_flat_generator():
    """A real finite16m x6m mesh, with its spawn origin at the tile centre."""
    return TerrainGeneratorCfg(size=(16.0, 6.0), num_rows=1, num_cols=1,
        border_width=3.0, curriculum=False, use_cache=False,
        sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)})


def _zero_diagnostic_reward(env):
    return torch.zeros(env.num_envs, device=env.device)


def _validity(env):
    return physical_validity(env.scene["robot"].data.root_pos_w,
        env.scene.terrain.env_origins, env._stair_domain_bounds_local,
        env._stair_below_floor_tolerance)


def initialization_out_of_tile(env):
    return _validity(env)[0]


def initialization_below_floor(env):
    return _validity(env)[1]


@configclass
class DirectInitializationTerminationsCfg(TerminationsCfg):
    # Original five native failure/timeout terms remain byte-for-byte inherited.
    stair_below_floor = TerminationTermCfg(func=initialization_below_floor)
    stair_out_of_tile = TerminationTermCfg(func=initialization_out_of_tile)


@configclass
class DirectInitializationRewardsCfg:
    diagnostic_zero = RewardTermCfg(func=_zero_diagnostic_reward, weight=1.0)


@configclass
class DirectInitializationCurriculumCfg:
    terrain_levels = None
    lin_vel_cmd_levels = None


@configclass
class DirectInitializationEnvCfg(StairAscentEnvCfg):
    """Same direct plant/network layout/physics; flat initialization only."""
    initialization_domain_inset_m: float = 0.10
    initialization_floor_tolerance_m: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1024
        self.episode_length_s = 20.0
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = initialization_flat_generator()
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum = DirectInitializationCurriculumCfg()
        self.rewards = DirectInitializationRewardsCfg()
        self.terminations = DirectInitializationTerminationsCfg()
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scanner.params["side_noise_probability"] = 0.0
        self.observations.critic.height_scanner = deepcopy(self.observations.policy.height_scanner)
        # Inherited direct action: full original soft range, native v/E target0.
        # No body height, pelvis attitude, phase or teacher action is prescribed.
        if self.actions.JointPositionAction.class_type is not DirectLegPositionAction:
            raise RuntimeError("Initialization requires the unchanged direct leg action")


class DirectInitializationEnv(RobotEnv):
    """Direct plant with real-reset hooks and independent physical-domain guards."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._initialization_reset_callbacks = {"before": [], "after": []}
        super().__init__(cfg, render_mode, **kwargs)
        if self.num_envs > 1024:
            raise ValueError("This bounded initializer supports at most1024 environments")
        term = self.action_manager.get_term("JointPositionAction")
        if (type(term) is not DirectLegPositionAction or term.gait_reference is not None
                or self.action_manager.total_action_dim != 12):
            raise RuntimeError("Initialization plant action authority is not purely direct12")
        if any(getattr(cfg, name, None) is not None for name in
               ("contact_gait_reference", "stair_reference", "stance_orientation_feedback", "lateral_com_feedback")):
            raise RuntimeError("Teacher/reference/legacy feedback must stay outside the plant")
        if cfg.stance_preload_torques:
            raise RuntimeError("Direct initialization cannot install preload in the plant")
        generator = self.scene.terrain.cfg.terrain_generator
        if (generator is None or set(generator.sub_terrains) != {"flat"}
                or type(generator.sub_terrains["flat"]) is not MeshPlaneTerrainCfg):
            raise RuntimeError("Initialization needs its actual generated flat mesh")
        length, width = map(float, generator.size)
        inset = float(cfg.initialization_domain_inset_m)
        if not 0 < inset < min(length, width)/2:
            raise ValueError("Invalid own-flat-tile safety inset")
        # MeshPlaneTerrainCfg returns the geometric tile centre as env origin.
        self._stair_domain_bounds_local = (-length/2+inset, length/2-inset,
                                           -width/2+inset, width/2-inset)
        self.domain_bounds_local = self._stair_domain_bounds_local
        self._stair_below_floor_tolerance = float(cfg.initialization_floor_tolerance_m)
        self.initialization_reset_generation = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._stair_foot_ids, names = self.asset.find_bodies(cfg.feet_body_names, preserve_order=True)
        if names != ["left_ankle_pitch_link", "right_ankle_pitch_link"]:
            raise RuntimeError("Initialization foot order changed")
        self._stair_unfiltered_ids, _ = self.scene["contact_forces"].find_bodies(cfg.feet_body_names, preserve_order=True)
        self._stair_sensors = [self.scene["stair_ascent_left_ground"], self.scene["stair_ascent_right_ground"]]
        self._stair_sole_offset, _ = foot_geometry(self.device)
        for group, expected in (("policy", (1029,)), ("critic", (866,)), ("rnd_state", (73,))):
            if tuple(self.observation_manager.group_obs_dim[group]) != expected:
                raise RuntimeError(f"Initialization changed the existing{group} observation layout")

    def add_reset_callback(self, callback, *, when="after"):
        """callback(env_ids): before teleport, or after true reset/feet refresh.

        The oracle registers an after callback; collectors register before for
        terminal evidence. No synthetic reset, warmup or rescue is performed.
        """
        if when not in self._initialization_reset_callbacks or not callable(callback):
            raise ValueError("Expected callable and when='before'|'after'")
        callbacks = self._initialization_reset_callbacks[when]
        if callback in callbacks:
            raise ValueError("Reset callback is already registered")
        callbacks.append(callback)

    def remove_reset_callback(self, callback, *, when="after"):
        self._initialization_reset_callbacks[when].remove(callback)

    def _reset_idx(self, env_ids):
        callbacks = getattr(self, "_initialization_reset_callbacks", {"before": [], "after": []})
        for callback in tuple(callbacks["before"]):
            callback(env_ids)
        super()._reset_idx(env_ids)
        if hasattr(self, "initialization_reset_generation"):
            self.initialization_reset_generation[env_ids] += 1
        for callback in tuple(callbacks["after"]):
            callback(env_ids)

    def _stair_foot_pose(self):
        rotation = quaternion_matrix(self.asset.data.body_quat_w[:, self._stair_foot_ids])
        sole = (self.asset.data.body_pos_w[:, self._stair_foot_ids]
                + torch.einsum("nfij,fj->nfi", rotation, self._stair_sole_offset))
        return sole, rotation
