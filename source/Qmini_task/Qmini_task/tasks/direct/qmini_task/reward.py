# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for Qmini task."""

import torch


def compute_rewards(env) -> torch.Tensor:
    """Compute total reward for the environment.
    
    Args:
        env: The environment instance
        
    Returns:
        Total reward tensor of shape (num_envs,)
    """
    current_pos = env.joint_pos[:, env._controlled_joint_indices]
    current_vel = env.joint_vel[:, env._controlled_joint_indices]
    target_pos = env._target_pos.unsqueeze(0).expand_as(current_pos)

    root_state = env.robot.data.root_state_w
    base_quat = root_state[:, 3:7]
    base_height = root_state[:, 2]  # Z coordinate (height)
    base_lin_vel = root_state[:, 7:10]
    base_ang_vel = root_state[:, 10:13]
    roll, pitch, _ = env._quat_to_euler(base_quat)
    orientation_error = torch.sqrt(roll * roll + pitch * pitch)
    roll_deg = torch.rad2deg(roll)
    pitch_deg = torch.rad2deg(pitch)

    pos_error = current_pos - target_pos
    joint_error = torch.norm(pos_error, dim=1)
    joint_vel_norm = torch.norm(current_vel, dim=1)
    lin_vel_norm = torch.norm(base_lin_vel, dim=1)
    ang_vel_norm = torch.norm(base_ang_vel, dim=1)
    action_rate = torch.norm(env.actions - env._prev_actions, dim=1)

    # Only reward target_joint_pos when root height >= 0.3m
    height_threshold = 0.3  # [m]
    height_ok = base_height >= height_threshold
    height_ok_mask = height_ok.float()
    
    # For joint velocity: reward when height < 0.35m, penalty when height >= 0.35m
    joint_vel_height_threshold = 0.35  # [m]
    joint_vel_low_height = base_height < joint_vel_height_threshold
    joint_vel_low_height_mask = joint_vel_low_height.float()

    rew_alive = env.cfg.rew_scale_alive * (1.0 - env.reset_terminated.float())
    rew_term = env.cfg.rew_scale_terminated * env.reset_terminated.float()
    # Joint rewards are only active when height >= 0.3m
    rew_joint = -env.cfg.rew_scale_joint * joint_error * height_ok_mask
    # Joint velocity: reward when height < 0.35m (encourage fast movement to stand up)
    #                  penalty when height >= 0.35m (encourage stability)
    rew_joint_vel = (
        env.cfg.rew_scale_joint_vel * joint_vel_norm * joint_vel_low_height_mask  # Reward for fast movement when low
        - env.cfg.rew_scale_joint_vel * joint_vel_norm * (1.0 - joint_vel_low_height_mask)  # Penalty when high
    )
    # Reduce penalties when height is low to allow exploration during stand-up
    # Penalties are reduced when height < 0.3m to encourage standing up
    penalty_reduction = 1.0 - (1.0 - height_ok_mask) * 0.5  # Reduce by 50% when height < 0.3m
    rew_upright = -env.cfg.rew_scale_upright * orientation_error * penalty_reduction
    rew_base_lin = -env.cfg.rew_scale_base_lin_vel * lin_vel_norm * penalty_reduction
    rew_base_ang = -env.cfg.rew_scale_base_ang_vel * ang_vel_norm * penalty_reduction
    rew_action = -env.cfg.rew_scale_action_rate * action_rate

    # Stand-up task specific rewards
    if hasattr(env.cfg, 'enable_standup_task') and env.cfg.enable_standup_task:
        # Height progress reward: encourage increasing height
        # Use squared progress to give more reward for higher positions
        target_height = env.cfg.target_base_height
        height_progress = torch.clamp(base_height / target_height, 0.0, 1.0)
        # Square the progress to give more reward as height increases (non-linear)
        rew_height_progress = env.cfg.rew_scale_height_progress * (height_progress ** 2)
        
        # Target height reward: exponential reward for reaching target height
        # Use a wider tolerance to give reward even when not exactly at target
        height_error = torch.abs(base_height - target_height)
        rew_height_target = env.cfg.rew_scale_height_target * torch.exp(-height_error / 0.15)
    else:
        rew_height_progress = torch.zeros(env.scene.num_envs, device=env.device)
        rew_height_target = torch.zeros(env.scene.num_envs, device=env.device)

    orientation_ok = (torch.abs(roll) < env.cfg.success_pitch_tol) & (
        torch.abs(pitch) < env.cfg.success_pitch_tol
    )
    # Success also requires height >= 0.3m
    success_mask = (
        (torch.max(torch.abs(pos_error), dim=1).values < env._success_joint_tol)
        & orientation_ok
        & height_ok
    )

    # Stand-up success: height reached + orientation ok
    if hasattr(env.cfg, 'enable_standup_task') and env.cfg.enable_standup_task:
        standup_height_ok = base_height >= (env.cfg.target_base_height - 0.05)  # Within 5cm of target
        standup_success_mask = standup_height_ok & orientation_ok
        rew_standup_success = env.cfg.rew_scale_standup_success * standup_success_mask.float()
        rew_success = env.cfg.rew_scale_success * success_mask.float() + rew_standup_success
    else:
        rew_success = env.cfg.rew_scale_success * success_mask.float()
        rew_standup_success = torch.zeros(env.scene.num_envs, device=env.device)

    total_reward = (
        rew_alive
        + rew_term
        + rew_joint
        + rew_joint_vel
        + rew_upright
        + rew_base_lin
        + rew_base_ang
        + rew_action
        + rew_success
        + rew_height_progress
        + rew_height_target
    )

    # Logging
    if env._tb_step % 32 == 0:
        env._tb_writer.add_scalar("pose/roll_deg", roll_deg.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("pose/pitch_deg", pitch_deg.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("pose/base_height", base_height.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("reward/total", total_reward.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("reward/upright_penalty", rew_upright.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("reward/joint_penalty", rew_joint.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("reward/action_rate_penalty", rew_action.mean().item(), env._tb_step)
        if hasattr(env.cfg, 'enable_standup_task') and env.cfg.enable_standup_task:
            env._tb_writer.add_scalar("reward/height_progress", rew_height_progress.mean().item(), env._tb_step)
            env._tb_writer.add_scalar("reward/height_target", rew_height_target.mean().item(), env._tb_step)
            env._tb_writer.add_scalar("reward/standup_success", rew_standup_success.mean().item(), env._tb_step)

    return total_reward
