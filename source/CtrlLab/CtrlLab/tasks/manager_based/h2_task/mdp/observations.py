import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_apply
from ..assets import H2_LEG_JOINT_NAMES
from ..networks.lidar_terrain_reconstructor import LidarTerrainReconstructor
from ..networks.lidar_observation import (
    body_height_to_observation,
    cache_state_matches,
    expand_ray_geometry,
    exact_equal_rows,
    invalidate_lidar_observation,
    mesh_hit_ranges,
    reconstruction_rows,
    validate_and_scatter_scan,
    velocity_prior,
)
from ..lidar_hb_pattern import YuHbLidarPatternCfg, yu_hb_lidar_keep_mask


def reconstructed_lidar_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    checkpoint: str = "logs/rsl_rl/stairs_l0b2_20261020/lidar_reconstructor_phase1.pt",
    use_reliable_fusion: bool = False,
    output_convention: str = "height_scan",
    height_scan_offset: float = 0.5,
    reference_sensor_cfg: SceneEntityCfg | None = None,
    velocity_source: str = "zero",
    velocity_estimate_attr: str = "_h2_base_velocity_estimate_b",
) -> torch.Tensor:
    """Frozen cross-embodiment reconstruction with an explicit H2 map contract.

    The source weights were trained on Go2. The zero velocity prior is an
    explicit experimental fallback, not a learned H2 estimator; choose
    velocity_source='estimate' and provide a finite (B,3) body estimate for
    deployment work. Simulator root linear velocity is never an input.
    The reference scanner contributes only its pose to the height convention.
    ObsTerm applies clipping/scaling once, after this function returns.
    """
    raw = env.unwrapped
    sensor = raw.scene.sensors[sensor_cfg.name]
    robot = raw.scene["robot"]
    step = int(getattr(raw, "common_step_counter", -1))
    root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w
    sensor_pos, sensor_quat = sensor.data.pos_w, sensor.data.quat_w
    n_env = root_pos.shape[0]
    reference_z = (
        raw.scene.sensors[reference_sensor_cfg.name].data.pos_w[:, 2]
        if reference_sensor_cfg is not None else root_pos[:, 2]
    )
    command = raw.command_manager.get_command("base_velocity")
    action = raw.action_manager.action
    estimate = getattr(raw, velocity_estimate_attr, None) if velocity_source == "estimate" else None
    velocity = velocity_prior(root_pos, velocity_source, estimate)
    settings = (
        str(checkpoint), bool(use_reliable_fusion), output_convention,
        float(height_scan_offset), reference_sensor_cfg.name if reference_sensor_cfg is not None else None,
        velocity_source, velocity_estimate_attr,
    )
    episode_lengths = getattr(raw, "episode_length_buf", None)
    state = [
        root_pos, root_quat, sensor_pos, sensor_quat, reference_z, command,
        robot.data.joint_pos, robot.data.joint_vel, robot.data.root_ang_vel_b, action, velocity,
        sensor.data.ray_hits_w,
    ]
    if episode_lengths is not None:
        state.append(episode_lengths)
    terrain_types = getattr(raw.scene.terrain, "terrain_types", None)
    if isinstance(terrain_types, torch.Tensor):
        state.append(terrain_types)
    state = tuple(state)
    cache = getattr(raw, "_h2_lidar_recon_cache", None)
    if cache_state_matches(cache, step, settings, state):
        return cache[3]

    # Recompute only changed/invalid rows at the same step. Merely clearing
    # selected histories and then forwarding the full batch double-counts the
    # current physical frame in every unchanged environment.
    same_step = cache is not None and len(cache) == 5 and cache[0] == step and cache[1] == settings
    if same_step:
        changed = ~cache[4].clone()
        if len(cache[2]) != len(state):
            changed[:] = True
        else:
            for previous, current in zip(cache[2], state):
                if previous.shape != current.shape:
                    changed[:] = True
                    break
                changed |= ~exact_equal_rows(previous, current)
        invalidate_lidar_observation(raw, torch.nonzero(changed, as_tuple=False).flatten())
    else:
        changed = torch.ones(n_env, dtype=torch.bool, device=root_pos.device)
        if cache is not None and cache[0] == step:
            # A different observation contract at the same physical frame
            # deliberately cold-restarts memory instead of double-pushing it.
            invalidate_lidar_observation(raw)
    if episode_lengths is not None:
        reset_ids = torch.nonzero(episode_lengths == 0, as_tuple=False).flatten()
        invalidate_lidar_observation(raw, reset_ids)

    model_key = (str(checkpoint), bool(use_reliable_fusion))
    model = getattr(raw, "_h2_lidar_reconstructor", None)
    if model is None or getattr(raw, "_h2_lidar_model_settings", None) != model_key:
        invalidate_lidar_observation(raw)
        changed[:] = True
        same_step = False
        # This archive contains a tensor state dict and primitive metadata.
        # Do not enable arbitrary pickle objects when reading user paths.
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(blob, dict) or "model_state_dict" not in blob:
            raise ValueError("LiDAR checkpoint must contain model_state_dict and sensor metadata")
        state_dict = blob["model_state_dict"]
        if not bool(blob.get("use_height_splat", False)):
            raise ValueError("H2 stair observation requires a height-splat reconstructor checkpoint")
        model = LidarTerrainReconstructor(
            n_ch=int(blob["n_ch"]),
            n_az=int(blob["n_az"]),
            n_frames=int(blob["n_frames"]),
            proprio_dim=int(blob["proprio_dim"]),
            gru_delay_steps=int(blob["gru_delay_steps"]),
            use_height_splat=True,
            use_refine=any(key.startswith("refine.") for key in state_dict),
            use_reliable_fusion=bool(use_reliable_fusion),
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(raw.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        raw._h2_lidar_reconstructor = model
        raw._h2_lidar_model_settings = model_key

    ray_hits = sensor.data.ray_hits_w
    if ray_hits.ndim != 3 or ray_hits.shape[0] != n_env or ray_hits.shape[-1] != 3:
        raise ValueError("LiDAR ray hits must have shape (num_envs,num_rays,3)")
    ray_start_local, ray_dir_local = expand_ray_geometry(sensor.ray_starts, sensor.ray_directions, n_env)
    n_rays = ray_hits.shape[1]
    if ray_start_local.shape[1] != n_rays:
        raise ValueError("LiDAR ray geometry and hit count differ")
    q = sensor_quat[:, None, :].expand(-1, n_rays, -1).reshape(-1, 4)
    starts = quat_apply(q, ray_start_local.reshape(-1, 3)).view(n_env, n_rays, 3) + sensor_pos[:, None, :]
    pattern = sensor.cfg.pattern_cfg
    # The lower bound belongs to the sensor pattern, not the neural module.
    rng, mask = mesh_hit_ranges(
        ray_hits, starts,
        min_range=float(getattr(pattern, "min_range", 0.1)),
        max_range=min(float(getattr(pattern, "max_range", model.max_range)), float(model.max_range)),
    )
    keep = yu_hb_lidar_keep_mask(pattern) if isinstance(pattern, YuHbLidarPatternCfg) else None
    expected_dirs = model.ray_dirs.reshape(3, -1).transpose(0, 1)
    rng, mask, origin_b = validate_and_scatter_scan(
        rng, mask, ray_start_local, ray_dir_local, expected_dirs, keep,
    )

    # The transferred archive has either 45 measured inputs or those 45 plus
    # a 3-vector velocity estimate. Other embeddings need an explicit adapter.
    if int(model.proprio_dim) not in (45, 48):
        raise ValueError("Go2-to-H2 proprio adapter supports only 45 or 48 checkpoint inputs")
    leg_joint_ids = robot.find_joints(H2_LEG_JOINT_NAMES, preserve_order=True)[0]
    if len(leg_joint_ids) != 12 or action.ndim != 2 or action.shape[0] != n_env or action.shape[1] < 12:
        raise ValueError("H2 proprio adapter requires the 12 configured leg joints/actions")
    proprio = rng.new_zeros((n_env, int(model.proprio_dim)))
    proprio[:, 0:3] = command
    proprio[:, 3:6] = robot.data.root_ang_vel_b * 0.2
    gravity_w = rng.new_zeros((n_env, 3))
    gravity_w[:, 2] = -1.0
    proprio[:, 6:9] = quat_apply_inverse(root_quat, gravity_w)
    proprio[:, 9:21] = robot.data.joint_pos[:, leg_joint_ids] - robot.data.default_joint_pos[:, leg_joint_ids]
    proprio[:, 21:33] = robot.data.joint_vel[:, leg_joint_ids] * 0.05
    # Actions follow H2_JOINT_NAMES (legs first), not articulation joint indices.
    proprio[:, 33:45] = action[:, :12]
    if model.proprio_dim == 48:
        proprio[:, 45:48] = velocity
    active_ids = torch.nonzero(changed, as_tuple=False).flatten()
    hidden_full = getattr(raw, "_h2_lidar_recon_hidden", None)
    hidden_in = hidden_full[:, active_ids] if hidden_full is not None else None
    with torch.no_grad(), reconstruction_rows(model, active_ids, n_env):
        _, h_recon, hidden = model.forward_height_splat(
            rng[active_ids], mask[active_ids], sensor_pos[active_ids], sensor_quat[active_ids],
            root_pos[active_ids], root_quat[active_ids], proprio[active_ids], hidden_in, origin_b[active_ids],
        )
    if hidden_full is None:
        hidden_full = hidden.new_zeros((hidden.shape[0], n_env, hidden.shape[2]))
    else:
        hidden_full = hidden_full.clone()
    hidden_full[:, active_ids] = hidden.detach()
    raw._h2_lidar_recon_hidden = hidden_full
    active_map = body_height_to_observation(
        h_recon.view(active_ids.numel(), -1), root_pos[active_ids, 2], reference_z=reference_z[active_ids],
        convention=output_convention, offset=height_scan_offset,
    )
    height_map = cache[3].clone() if same_step else active_map.new_empty((n_env, active_map.shape[1]))
    height_map[active_ids] = active_map
    raw._h2_lidar_recon_cache = (
        step, settings, tuple(value.detach().clone() for value in state), height_map,
        torch.ones(n_env, dtype=torch.bool, device=root_pos.device),
    )
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
