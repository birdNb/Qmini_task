# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QminiRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # G1 style training configuration
    num_steps_per_env = 48  # Steps per environment per rollout
    max_iterations = 2100
    save_interval = 300
    experiment_name = "qmini_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],  # G1 style: deep network for complex locomotion
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,  # G1 style: standard PPO clip parameter
        entropy_coef=0.01,  # G1 style: moderate exploration
        num_learning_epochs=5,  # G1 style: multiple epochs for sample efficiency
        num_mini_batches=8,  # steps=48 → batch=6
        learning_rate=3.0e-4,  # G1 style: moderate learning rate for stable training
        schedule="adaptive",  # Adaptive learning rate scheduling
        gamma=0.99,  # G1 style: standard discount factor
        lam=0.95,  # G1 style: GAE lambda
        desired_kl=0.015,  # G1 style: target KL divergence for adaptive learning rate
        max_grad_norm=1.0,  # Gradient clipping
    )


# @configclass
# class QminiFlatPPORunnerCfg(QminiRoughPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         self.max_iterations = 15000
#         self.experiment_name = "qmini_flat"
#         self.policy.actor_hidden_dims = [128, 128, 128]
#         self.policy.critic_hidden_dims = [128, 128, 128]


# Backward compatibility alias
PPORunnerCfg = QminiRoughPPORunnerCfg
