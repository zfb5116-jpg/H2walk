"""Fixed world stair direction expressed in the original base-frame command."""
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand


def world_stair_velocity_in_base(quaternion, speed=0.3):
    """Project world [+speed,0,0] through the full inverse root rotation.

    The existing command observation still contains base-frame vx, vy and yaw
    rate in their original units. This does not alter an observation layout or
    its frozen normalization, nor prescribe a joint or body-height target.
    """
    world = torch.zeros_like(quaternion[:, :3])
    world[:, 0] = speed
    xyz = quaternion[:, 1:]
    cross = 2.0*torch.cross(xyz, world, dim=-1)
    return world-quaternion[:, :1]*cross+torch.cross(xyz, cross, dim=-1)


class WorldStairVelocityCommand(UniformVelocityCommand):
    """A fixed world velocity goal; native actuator control is unchanged."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.heading_command or cfg.rel_standing_envs != 0.0:
            raise ValueError('World stair command requires zero yaw rate and no standing subset')
        self.metrics['world_root_link_xy_error'] = torch.zeros(self.num_envs, device=self.device)
        self.metrics['world_root_COM_xy_error'] = torch.zeros(self.num_envs, device=self.device)

    def _set_world_command(self, env_ids):
        base = world_stair_velocity_in_base(self.robot.data.root_quat_w[env_ids])
        self.vel_command_b[env_ids, :2] = base[:, :2]
        self.vel_command_b[env_ids, 2] = 0.0

    def _resample_command(self, env_ids):
        # Keep the original resampling/RNG path, then express the fixed goal in
        # the current base frame before the first post-reset observation.
        super()._resample_command(env_ids)
        self._set_world_command(env_ids)

    def _update_command(self):
        self._set_world_command(slice(None))

    def _update_metrics(self):
        self._set_world_command(slice(None))
        super()._update_metrics()
        steps = self.cfg.resampling_time_range[1]/self._env.step_dt
        world = torch.zeros_like(self.robot.data.root_link_lin_vel_w[:, :2])
        world[:, 0] = 0.3
        self.metrics['world_root_link_xy_error'] += torch.norm(
            self.robot.data.root_link_lin_vel_w[:, :2]-world, dim=-1)/steps
        self.metrics['world_root_COM_xy_error'] += torch.norm(
            self.robot.data.root_lin_vel_w[:, :2]-world, dim=-1)/steps

    def runtime_contract(self):
        return dict(kind='fixed_world_plusX_base_command_v1', world_linear_velocity_m_s=[0.3, 0.0, 0.0],
            command_frame='full inverse root quaternion; base-frame XY m/s, yaw rate rad/s',
            command_observation_width=3, yaw_rate_rad_s=0.0,
            prefix_command_units_and_base_frame_preserved=True,
            inherited_critic_base_velocity_point='root rigid-body COM; original body frame unchanged',
            context_velocity_point='root-link origin; original yaw-only frame unchanged',
            explicit_world_metrics=['world_root_link_xy_error', 'world_root_COM_xy_error'],
            frozen_normalization_unchanged=True, joint_or_body_height_controller=False)
