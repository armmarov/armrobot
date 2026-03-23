from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PM01WalkingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 10000
    save_interval = 200
    experiment_name = "pm01_walking"
    empirical_normalization = True  # Run 23: restore (proven stable in Runs 13-19)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,   # Run 23: restore
        critic_obs_normalization=True,  # Run 23: restore
        # Run 44: wider first layer to handle 960-dim history input (was 512 for 64-dim)
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[768, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=False,  # Run 48: clip_param=0.2 too tight for return scale ~560 — value fn couldn't track
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=2,  # Run 31: match EngineAI (was 5, caused value loss spikes)
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.994,
        lam=0.9,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
