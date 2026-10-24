import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from ..contact_events import stumble_from_force_history


def illegal_feet_stumble(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        force_ratio: float = 10.0,
        temporal_mode: str = "per_sample",
) -> torch.Tensor:
    """Time-aligned foot contact heuristic; not enabled by TerminationsCfg."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return stumble_from_force_history(
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :],
        force_ratio=force_ratio,
        temporal_mode=temporal_mode,
    )


def illegal_contact_out_of_time(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > threshold, dim=1)
