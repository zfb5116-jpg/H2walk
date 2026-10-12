import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse


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
