from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    class_name = "ActorCriticRecurrent"
    init_noise_std = 0.8
    activation = "elu"
    actor_hidden_dims = [128, 128, 128]
    critic_hidden_dims = [128, 128, 128]
    
    rnn_type = "gru"
    rnn_hidden_size = 64
    rnn_num_layers = 1

@configclass
class WalkingRobotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 50
    experiment_name = "walking_robot_unitree"   
    empirical_normalization = False

    # resume = True
    # load_checkpoint = "model_8650.pt"
    # load_run = "2025-01-02_15-36-20"

    policy = RslRlPpoActorCriticRecurrentCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )