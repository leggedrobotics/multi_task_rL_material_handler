# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

# TODO: Change, these values are for m545 !!!

@configclass
class MaterialHandlerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Policy
    
    # Runner
    num_steps_per_env = 3 # per iteration
    max_iterations = 6000 # number of policy updates
    empirical_normalization = False

    # logging
    logger = "wandb"
    experiment_name = "Isaac-Material-Handler-v0"
    run_name = "train"
    save_interval = 50
    

    # Policy
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.4,
        actor_hidden_dims= (256, 128, 128),
        critic_hidden_dims= (256, 128, 128),
        activation="tanh", # elu, relu, selu, crelu, lrelu, tanh, sigmoid
    )
    # Algorithm
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00,
        num_learning_epochs=5,
        num_mini_batches=4,  # mini batch size = num_envs * nsteps / nminibatches
        learning_rate=5e-4,
        schedule="fixed", # adaptive, fixed
        gamma=0.98,
        lam=0.95,
        desired_kl=0.00,
        max_grad_norm=0.5,
    )
