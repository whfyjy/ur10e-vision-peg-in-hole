"""UR10e vision peg-in-hole task registration."""

import gymnasium as gym

from .peg_hole_env_cfg import PegHoleEnvCfg

gym.register(
    id="UR10e-VisionPegInHole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": PegHoleEnvCfg},
)
