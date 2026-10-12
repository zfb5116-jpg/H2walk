import gymnasium as gym

from . import agents
# flat
gym.register(
    id="Unitree-H2-Velocity-Flat",
    entry_point=f"{__name__}.robot_env:RobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robot_env_cfg:RobotFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H2-Velocity-Flat-Play",
    entry_point=f"{__name__}.robot_env:RobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robot_env_cfg:RobotFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H2FlatPPORunnerCfg",
    },
)
# rough
gym.register(
    id="Unitree-H2-Velocity-Rough",
    entry_point=f"{__name__}.robot_env:RobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robot_env_cfg:RobotRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H2-Velocity-Rough-Play",
    entry_point=f"{__name__}.robot_env:RobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robot_env_cfg:RobotRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H2RoughPPORunnerCfg",
    },
)
from .robot_env import RobotEnv
from .robot_env_cfg import RobotFlatPlayEnvCfg, RobotRoughPlayEnvCfg
