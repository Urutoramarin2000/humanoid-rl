
from humanoid.envs.cowa.cowa_config import CowaCfg, CowaCfgPPO


class CowaCfg_VAE(CowaCfg):
    class env(CowaCfg.env):
        frame_stack = 3
        num_latent = 16
        num_est_prob = 3


class CowaCfgPPO_VAE(CowaCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunnerVAE'
    

    class policy(CowaCfgPPO.policy):
        init_noise_std = 1.0
        # VAE para
        encoder_hidden_dims=[256, 128]
        decoder_hidden_dims=[64, 128]

    class algorithm(CowaCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2 ##2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

        # VAE para
        vae_learning_rate = 5.e-4
        kl_weight = 1.
        value_loss_coef = 1.0
        max_grad_norm_vae = 7e-1
        num_adaptation_module_substeps = 1

    class runner:
        policy_class_name = 'ActorCritic_VAE'
        algorithm_class_name = 'PPO_VAE'
        num_steps_per_env = 60  # per iteration
        max_iterations = 10001  # number of policy updates        #  xxw

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'cowa_vae'
        run_name = 'vae'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = '/home/zcowa'  # updated from load_run and chkpt