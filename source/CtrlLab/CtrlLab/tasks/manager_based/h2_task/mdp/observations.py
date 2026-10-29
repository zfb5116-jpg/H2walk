import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.math import quat_apply
from ..assets import H2_LEG_JOINT_NAMES
from ..networks.lidar_terrain_reconstructor import LidarTerrainReconstructor
from ..lidar_hb_pattern import YuHbLidarPatternCfg, yu_hb_lidar_keep_mask


def reconstructed_lidar_height_scan(
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        checkpoint: str = "logs/rsl_rl/stairs_l0b2_20261020/lidar_reconstructor_phase1.pt",
) -> torch.Tensor:
    """Run the optional frozen reconstructor and return its 17x33 body map."""
    raw = env.unwrapped
    sensor = raw.scene.sensors[sensor_cfg.name]
    cache = getattr(raw, "_h2_lidar_recon_cache", None)
    step = int(getattr(raw, "common_step_counter", -1))
    if cache is not None and cache[0] == step:
        return cache[1]

    model = getattr(raw, "_h2_lidar_reconstructor", None)
    if model is None:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = LidarTerrainReconstructor(
            n_ch=int(blob["n_ch"]),
            n_az=int(blob["n_az"]),
            n_frames=int(blob["n_frames"]),
            proprio_dim=int(blob["proprio_dim"]),
            gru_delay_steps=int(blob["gru_delay_steps"]),
            use_height_splat=bool(blob["use_height_splat"]),
        )
        model.load_state_dict(blob["model_state_dict"], strict=True)
        model.to(raw.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        setattr(raw, "_h2_lidar_reconstructor", model)
        print(f"[h2-lidar-recon] loaded frozen stair reconstructor from {checkpoint}", flush=True)

    ray_hits = sensor.data.ray_hits_w
    n_env, n_rays = ray_hits.shape[:2]
    sensor_pos = sensor.data.pos_w
    sensor_quat = sensor.data.quat_w
    q = sensor_quat.unsqueeze(1).expand(-1, n_rays, -1).reshape(-1, 4)
    ray_start_local = sensor.ray_starts
    ray_dir_local = sensor.ray_directions
    if ray_start_local.ndim == 2:
        ray_start_local = ray_start_local.unsqueeze(0)
    if ray_dir_local.ndim == 2:
        ray_dir_local = ray_dir_local.unsqueeze(0)
    if ray_start_local.shape[0] == 1 and n_env > 1:
        ray_start_local = ray_start_local.expand(n_env, -1, -1)
    if ray_dir_local.shape[0] == 1 and n_env > 1:
        ray_dir_local = ray_dir_local.expand(n_env, -1, -1)
    starts = quat_apply(q, ray_start_local.reshape(-1, 3)).view(n_env, n_rays, 3) + sensor_pos.unsqueeze(1)
    directions = quat_apply(q, ray_dir_local.reshape(-1, 3)).view(n_env, n_rays, 3)
    ray_range = torch.linalg.norm(ray_hits - starts, dim=-1)
    robot = raw.scene["robot"]
    terrain_types = raw.scene.terrain.terrain_types
    cache_key = (step, terrain_types.clone(), robot.data.root_pos_w.clone())
    ray_cache = getattr(raw, "_h2_lidar_ray_cache", None)
    if ray_cache is not None and ray_cache[0] == step and torch.equal(ray_cache[1], terrain_types) and torch.allclose(ray_cache[2], robot.data.root_pos_w):
        rng, mask = ray_cache[3], ray_cache[4]
    else:
        mesh_ok = (
            torch.isfinite(ray_hits).all(dim=-1)
            & (ray_hits[..., 2] > -2.0)
            & (ray_range >= 0.1)
            & (ray_range <= model.max_range)
        )
        dz = directions[..., 2]
        plane_t = torch.where(
            dz < -1.0e-4,
            -starts[..., 2] / dz,
            torch.full_like(dz, float("inf")),
        )
        plane_ok = (plane_t >= model.min_range) & (plane_t <= model.max_range)
        plane_hits = starts + directions * plane_t.unsqueeze(-1)
        hit_pos = torch.where(mesh_ok.unsqueeze(-1), ray_hits, plane_hits)
        valid = mesh_ok | plane_ok
        hit_range = torch.linalg.norm(hit_pos - starts, dim=-1)
        valid &= torch.isfinite(hit_pos[..., 2]) & (hit_range <= model.max_range)
        rng = torch.where(valid, hit_range, torch.zeros_like(hit_range))
        mask = valid.to(dtype=rng.dtype)
        setattr(raw, "_h2_lidar_ray_cache", (*cache_key, rng, mask))

    pattern = sensor.cfg.pattern_cfg
    if isinstance(pattern, YuHbLidarPatternCfg):
        keep = yu_hb_lidar_keep_mask(pattern).to(device=rng.device)
        if int(keep.sum()) == n_rays:
            full_rng = rng.new_zeros(n_env, model.n_ch * model.n_az)
            full_mask = mask.new_zeros(n_env, model.n_ch * model.n_az)
            full_rng[:, keep] = rng
            full_mask[:, keep] = mask
            rng, mask = full_rng, full_mask

    origin_b = ray_start_local[:, 0, :]
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    hidden = getattr(raw, "_h2_lidar_recon_hidden", None)
    episode_lengths = getattr(raw, "episode_length_buf", None)
    if episode_lengths is not None:
        reset_ids = torch.nonzero(episode_lengths == 0, as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            model.reset_temporal_state(reset_ids)
            if hidden is not None:
                hidden = hidden.clone()
                hidden[:, reset_ids] = 0.0

    proprio = rng.new_zeros((n_env, int(model.proprio_dim)))
    if proprio.shape[-1] >= 45:
        # GO2 input layout: command, scaled IMU, 12 leg positions/velocities, actions.
        proprio[:, 0:3] = raw.command_manager.get_command("base_velocity")
        proprio[:, 3:6] = robot.data.root_ang_vel_b * 0.2
        gravity_w = rng.new_zeros((n_env, 3))
        gravity_w[:, 2] = -1.0
        proprio[:, 6:9] = quat_apply_inverse(root_quat, gravity_w)
        leg_joint_ids = robot.find_joints(H2_LEG_JOINT_NAMES, preserve_order=True)[0]
        proprio[:, 9:21] = robot.data.joint_pos[:, leg_joint_ids] - robot.data.default_joint_pos[:, leg_joint_ids]
        proprio[:, 21:33] = robot.data.joint_vel[:, leg_joint_ids] * 0.05
        proprio[:, 33:45] = raw.action_manager.action[:, leg_joint_ids]
    if proprio.shape[-1] > 45:
        n_velocity = min(3, proprio.shape[-1] - 45)
        proprio[:, 45 : 45 + n_velocity] = robot.data.root_lin_vel_b[:, :n_velocity]
    with torch.no_grad():
        _, h_recon, hidden = model.forward_height_splat(
            rng,
            mask,
            sensor_pos,
            sensor_quat,
            root_pos,
            root_quat,
            proprio,
            hidden,
            origin_b,
        )
    setattr(raw, "_h2_lidar_recon_hidden", hidden.detach())
    height_map = h_recon.view(n_env, -1)
    setattr(raw, "_h2_lidar_recon_cache", (step, height_map))
    return height_map


def feet_pos_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    num_bodies = asset.data.body_pos_w.shape[1] if isinstance(asset_cfg.body_ids, slice) else len(asset_cfg.body_ids)
    base_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)
    base_pos_w = asset.data.root_pos_w.unsqueeze(1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    return quat_apply_inverse(base_quat, feet_pos_w - base_pos_w).reshape(env.num_envs, -1)


def body_contact_state(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    return contacts.float()


def body_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    body_forces = torch.mean(net_contact_forces, dim=1)[:, sensor_cfg.body_ids]
    return body_forces.reshape(body_forces.shape[0], -1)


def feet_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_id = sensor_cfg.body_ids
    feet_forces = torch.mean(net_contact_forces, dim=1)[:, feet_id]
    return feet_forces.reshape(feet_forces.shape[0], -1)


def whole_body_ddq(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    base_body_id = asset_cfg.body_ids[0] if isinstance(asset_cfg.body_ids, list) else asset_cfg.body_ids
    base_acc = asset.data.body_acc_w[:, base_body_id]
    ddq = torch.cat([base_acc, asset.data.joint_acc[:, asset_cfg.joint_ids]], dim=-1)
    return ddq
