import gymnasium as gym

from .walking_robot_cfg import WalkingRobotEnvCfg, WalkingRobotEnvPLayCfg
from .agents.rsl_rl_walking_robot_cfg import WalkingRobotPPORunnerCfg

'''
    Register the RoughWalkingRobot-v3  and RoughWalkingRobot-Play-v3
'''
gym.register(
    id="RoughWalkingRobot-v3",
    entry_point="leg_robot.tasks.locomotion_liked_unitree.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WalkingRobotEnvCfg,
        "rsl_rl_cfg_entry_point": WalkingRobotPPORunnerCfg,
    },
)

gym.register(
    id="RoughWalkingRobot-Play-v3",
    entry_point="leg_robot.tasks.locomotion_liked_unitree.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WalkingRobotEnvPLayCfg,
        "rsl_rl_cfg_entry_point": WalkingRobotPPORunnerCfg,
    },
)