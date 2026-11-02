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

gym.register(
    id="Unitree-H2-Velocity-Stairs",
    entry_point=f"{__name__}.robot_env:RobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robot_env_cfg:RobotStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H2StairsPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H2-Velocity-Stairs-Play",
    entry_point=f"{__name__}.robot_env:RobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robot_env_cfg:RobotStairsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H2StairsPPORunnerCfg",
    },
)
from .robot_env import RobotEnv
from .robot_env_cfg import RobotFlatPlayEnvCfg, RobotRoughPlayEnvCfg

# Independent, policy-owned stair learning. Existing reference and flat tasks
# retain their original configuration and entry points.
gym.register(
    id="Unitree-H2-Stair-Ascent-Policy",
    entry_point=f"{__name__}.stair_transfer_env:StairTransferEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_transfer_cfg:StairTransferEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.stair_transfer_ppo_cfg:H2StairTransferPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H2-Stair-Ascent-Policy-Play",
    entry_point=f"{__name__}.stair_transfer_env:StairTransferEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_transfer_cfg:StairTransferPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.stair_transfer_ppo_cfg:H2StairTransferPPORunnerCfg",
    },
)

# Independent direct plant used only by the bounded closed-loop initializer.
gym.register(
    id="Unitree-H2-Direct-Flat-Initialization",
    entry_point=f"{__name__}.direct_initialization:DirectInitializationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_initialization:DirectInitializationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.stair_ascent_ppo_cfg:H2StairAscentPPORunnerCfg",
    },
)

# Independent guarded PPO repair; neither published flat task is replaced.
for _suffix in ('', '-Play'):
    gym.register(
        id='Unitree-H2-Flat-Balance-Policy'+_suffix,
        entry_point=f'{__name__}.flat_balance_env:FlatBalanceEnv',
        disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point': f'{__name__}.flat_balance_cfg:FlatBalance'+('Play' if _suffix else '')+'EnvCfg',
            'rsl_rl_cfg_entry_point': f'{agents.__name__}.flat_balance_ppo_cfg:H2FlatBalancePPORunnerCfg',
        },
    )
