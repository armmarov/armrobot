import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-PM01-Walking-Direct-v0",
    entry_point=f"{__name__}.armrobotlegging_env:ArmrobotleggingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.armrobotlegging_env_cfg:ArmrobotleggingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01WalkingPPORunnerCfg",
    },
)
