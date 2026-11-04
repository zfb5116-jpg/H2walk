import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def illegal_feet_stumble(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 水平方向力 (X, Y)
    horizontal_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :2], dim=-1).max(dim=1).values
    # 垂直方向力 (Z)
    vertical_force = torch.abs(net_contact_forces[:, :, sensor_cfg.body_ids, 2]).max(dim=1).values
    # 如果水平力比垂直力大 5 倍，则视为“绊倒”
    stumble_condition = horizontal_force > 10.0 * vertical_force
    # 判断每个环境中是否有任意一个脚部满足这个条件
    stumble = torch.any(stumble_condition, dim=-1)  # [num_envs], dtype: bool
    return stumble


def illegal_contact_out_of_time(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > threshold, dim=1)
