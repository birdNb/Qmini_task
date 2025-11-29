# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for Qmini task.
Based on HoST (Humanoid Standing-up Control) framework.
Implements multi-stage curriculum learning with multi-critic reward structure.
"""

import torch
import numpy as np


def sigmoid(x, value_at_1):
    """Sigmoid function for tolerance calculation."""
    scale = np.sqrt(-2 * np.log(value_at_1))
    return torch.exp(-0.5 * (x * scale) ** 2)


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, value_at_margin=0.1):
    """Tolerance function from HoST framework.
    
    Args:
        x: Input value tensor
        bounds: (lower, upper) bounds for the value
        margin: Margin width for smooth transition
        value_at_margin: Value at margin boundary
        
    Returns:
        Reward value in [0, 1] range
    """
    lower, upper = bounds
    assert lower < upper
    assert margin >= 0

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0.0)
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        value = torch.where(in_bounds, 1.0, sigmoid(d.double(), value_at_margin))
    
    return value.float()


def compute_rewards(env) -> torch.Tensor:
    """Compute total reward based on HoST framework.
    
    Implements:
    - Multi-stage rewards (Phase 1: h<0.45m, Phase 3: h≥0.65m)
    - Multi-critic structure: Task (multiplied), Style, Regularization, Post-task rewards
    
    Args:
        env: The environment instance
        
    Returns:
        Total reward tensor of shape (num_envs,)
    """
    # ========== Extract State Information ==========
    current_pos = env.joint_pos[:, env._controlled_joint_indices]
    current_vel = env.joint_vel[:, env._controlled_joint_indices]
    target_pos = env._target_pos.unsqueeze(0).expand_as(current_pos)

    root_state = env.robot.data.root_state_w
    base_quat = root_state[:, 3:7]  # [w, x, y, z]
    base_height = root_state[:, 2]  # Z coordinate (height)
    base_lin_vel = root_state[:, 7:10]
    base_ang_vel = root_state[:, 10:13]
    roll, pitch, yaw = env._quat_to_euler(base_quat)
    
    # Compute projected gravity (gravity vector in base frame)
    # Gravity in world frame is [0, 0, -1] (assuming z-up)
    gravity_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).expand(env.scene.num_envs, -1)
    # Rotate gravity to base frame (inverse quaternion rotation)
    # For quaternion [w, x, y, z], inverse is [w, -x, -y, -z]
    quat_inv = base_quat.clone()
    quat_inv[:, 1:4] = -quat_inv[:, 1:4]  # Negate x, y, z components
    projected_gravity = env._quat_apply(quat_inv, gravity_world)
    
    roll_deg = torch.rad2deg(roll)
    pitch_deg = torch.rad2deg(pitch)
    
    # Get joint information
    # Get joint torques if available
    if hasattr(env.robot.data, 'applied_torque'):
        joint_torques = env.robot.data.applied_torque[:, env._controlled_joint_indices]
        # Power = torque * velocity
        power = torch.abs(torch.sum(joint_torques * current_vel, dim=1))
    else:
        joint_torques = torch.zeros((env.scene.num_envs, len(env._controlled_joint_indices)), device=env.device)
        power = torch.zeros(env.scene.num_envs, device=env.device)
    
    # Compute joint acceleration (approximate from velocity change)
    if not hasattr(env, '_prev_joint_vel'):
        env._prev_joint_vel = current_vel.clone()
    joint_accel = (current_vel - env._prev_joint_vel) / env.step_dt
    env._prev_joint_vel = current_vel.clone()
    
    # ========== Phase Detection ==========
    # Phase thresholds (adjusted for Qmini robot scale)
    phase1_threshold = env.cfg.target_base_height_phase1 if hasattr(env.cfg, 'target_base_height_phase1') else 0.25
    phase3_threshold = env.cfg.target_base_height_phase3 if hasattr(env.cfg, 'target_base_height_phase3') else 0.35
    
    phase3_mask = (base_height >= phase3_threshold).float()
    
    # ========== Task Rewards (rtask) - ADDED (not multiplied to avoid gradient vanishing) ==========
    # Use addition instead of multiplication to provide gradient signal even when one component is low
    
    # 1. Orientation reward: tolerance(-projected_gravity[2], [threshold, inf], 1.0, 0.05)
    # When upright, projected_gravity[2] should be close to -1, so -projected_gravity[2] ≈ 1
    orientation_threshold = env.cfg.orientation_threshold if hasattr(env.cfg, 'orientation_threshold') else 0.99
    orientation_reward = tolerance(
        -projected_gravity[:, 2],
        [orientation_threshold, np.inf],
        1.0,
        0.05
    )
    # Always active, but scale by height progress to encourage both orientation and height
    height_progress = torch.clamp(base_height / phase3_threshold, 0.0, 1.0)
    orientation_reward = orientation_reward * (0.3 + 0.7 * height_progress)  # Scale from 0.3 to 1.0
    rew_orientation_task = env.cfg.rew_scale_orientation_task * orientation_reward
    
    # 2. Head/Base height reward: progressive reward for height
    # Use base height as proxy for head height (Qmini doesn't have separate head)
    target_head_height = env.cfg.target_head_height if hasattr(env.cfg, 'target_head_height') else 0.43
    
    # Progressive height reward: reward for any height progress, not just target
    # Use a smooth function that rewards from 0.15m onwards
    min_height = 0.15  # Minimum height to start rewarding
    height_reward_progress = torch.clamp((base_height - min_height) / (target_head_height - min_height), 0.0, 1.0)
    # Also add tolerance-based reward for reaching target
    height_reward_target = tolerance(
        base_height,
        [target_head_height * 0.9, np.inf],  # Start rewarding from 90% of target
        target_head_height,
        0.1
    )
    # Combine progressive and target rewards
    height_reward = 0.5 * height_reward_progress + 0.5 * height_reward_target
    
    # 3. Stand-up phase reward: extra reward for the critical stand-up phase (0.25m to 0.43m)
    # This phase is after flipping but before fully standing
    standup_phase_start = 0.25  # Height where stand-up phase begins
    standup_phase_mask = (base_height >= standup_phase_start).float() * (base_height < target_head_height).float()
    standup_phase_progress = torch.clamp(
        (base_height - standup_phase_start) / (target_head_height - standup_phase_start),
        0.0, 1.0
    )
    # Extra reward for making progress in stand-up phase
    standup_phase_reward = standup_phase_mask * standup_phase_progress
    rew_standup_phase = env.cfg.rew_scale_standup_phase * standup_phase_reward if hasattr(env.cfg, 'rew_scale_standup_phase') else 3.0 * standup_phase_reward
    
    rew_height_task = env.cfg.rew_scale_height_task * height_reward
    
    # Scale task rewards
    rew_task_total = rew_orientation_task + rew_height_task + rew_standup_phase
    
    # ========== Style Rewards (rstyle) - ADDED ==========
    # 1. Waist deviation penalty: 1(|qwaist| > 1.4)
    waist_joint_indices = []
    for i, name in enumerate(env._controlled_joint_names):
        if 'joint1' in name:  # hip_yaw joints (LL_joint1, RL_joint1)
            waist_joint_indices.append(i)
    if len(waist_joint_indices) > 0:
        waist_dof = current_pos[:, waist_joint_indices]
        waist_deviation = (torch.max(torch.abs(waist_dof), dim=1).values > 1.4).float()
    else:
        waist_deviation = (torch.abs(yaw) > 1.4).float()
    rew_waist_penalty = -env.cfg.rew_scale_waist_penalty * waist_deviation
    
    # 2. Knee deviation penalty: 1(max(|knee|) > 2.85) | (min(knee) < -0.06)
    knee_indices = [3, 8]  # LL_knee, RL_knee
    knee_angles = current_pos[:, knee_indices]
    knee_deviation = (
        (torch.max(torch.abs(knee_angles), dim=1).values > 2.85) |
        (torch.min(knee_angles, dim=1).values < -0.06)
    ).float()
    rew_knee_penalty = -env.cfg.rew_scale_knee_penalty * knee_deviation
    
    # 3. Feet distance penalty: 1(||feet|| > 0.9)
    try:
        left_ankle_ids, _ = env.robot.find_bodies("LL_ankle")
        right_ankle_ids, _ = env.robot.find_bodies("RL_ankle")
        if len(left_ankle_ids) > 0 and len(right_ankle_ids) > 0:
            left_ankle_pos = env.robot.data.body_pos_w[:, left_ankle_ids[0], :]
            right_ankle_pos = env.robot.data.body_pos_w[:, right_ankle_ids[0], :]
            feet_distance = torch.norm(left_ankle_pos - right_ankle_pos, dim=1)
        else:
            feet_distance = torch.zeros(env.scene.num_envs, device=env.device)
    except (AttributeError, IndexError, KeyError, RuntimeError, ValueError):
        feet_distance = torch.zeros(env.scene.num_envs, device=env.device)
    feet_distance_penalty = (feet_distance > 0.9).float()
    rew_feet_distance_penalty = -env.cfg.rew_scale_feet_distance_penalty * feet_distance_penalty
    
    # 4. Shank orientation reward: tolerance(mean(shank_z), [0.8, inf], 1.0, 0.1)
    # Calculate shank orientation (knee to ankle vector z-component)
    try:
        left_knee_ids, _ = env.robot.find_bodies("LL_knee")
        right_knee_ids, _ = env.robot.find_bodies("RL_knee")
        left_ankle_ids, _ = env.robot.find_bodies("LL_ankle")
        right_ankle_ids, _ = env.robot.find_bodies("RL_ankle")
        
        if (len(left_knee_ids) > 0 and len(right_knee_ids) > 0
                and len(left_ankle_ids) > 0 and len(right_ankle_ids) > 0):
            left_knee_pos = env.robot.data.body_pos_w[:, left_knee_ids[0], :]
            right_knee_pos = env.robot.data.body_pos_w[:, right_knee_ids[0], :]
            left_ankle_pos = env.robot.data.body_pos_w[:, left_ankle_ids[0], :]
            right_ankle_pos = env.robot.data.body_pos_w[:, right_ankle_ids[0], :]
            
            left_shank_vec = left_knee_pos - left_ankle_pos
            right_shank_vec = right_knee_pos - right_ankle_pos
            
            left_shank_z = left_shank_vec[:, 2] / torch.norm(left_shank_vec, dim=1)
            right_shank_z = right_shank_vec[:, 2] / torch.norm(right_shank_vec, dim=1)
            
            shank_z = (left_shank_z + right_shank_z) / 2.0
        else:
            shank_z = torch.ones(env.scene.num_envs, device=env.device)
    except (AttributeError, IndexError, KeyError, RuntimeError, ValueError):
        shank_z = torch.ones(env.scene.num_envs, device=env.device)
    
    shank_orientation_reward = tolerance(shank_z, [0.8, np.inf], 1.0, 0.1)
    # Only active when height > phase1_threshold
    shank_orientation_reward = shank_orientation_reward * (base_height > phase1_threshold).float()
    rew_shank_orientation = env.cfg.rew_scale_shank_orientation * shank_orientation_reward
    
    # 5. Leg extension reward: encourage straightening knees during stand-up phase
    # Knee joints should be extended (negative values for Qmini knee joints)
    # When standing, knees should be around -0.8 (from target_joint_pos)
    knee_indices = [3, 8]  # LL_knee, RL_knee
    knee_angles = current_pos[:, knee_indices]
    target_knee_angle = -0.8  # Target knee angle from config
    knee_extension_error = torch.abs(knee_angles - target_knee_angle)
    knee_extension_reward = torch.exp(-5.0 * torch.mean(knee_extension_error, dim=1))
    # Only active during stand-up phase (height >= 0.25m)
    knee_extension_reward = knee_extension_reward * (base_height >= 0.25).float()
    rew_knee_extension = env.cfg.rew_scale_knee_extension * knee_extension_reward if hasattr(env.cfg, 'rew_scale_knee_extension') else 2.0 * knee_extension_reward
    
    # ========== Regularization Rewards (rregu) - ADDED ==========
    # 1. Joint acceleration penalty: ||accel||^2
    rew_joint_accel = -env.cfg.rew_scale_joint_accel * torch.sum(torch.square(joint_accel), dim=1)
    
    # 2. Action rate penalty: ||action - last_action||^2
    rew_action_rate = -env.cfg.rew_scale_action_rate * torch.sum(torch.square(env.actions - env._prev_actions), dim=1)
    
    # 3. Torque penalty: ||torque||^2
    rew_torque = -env.cfg.rew_scale_torque * torch.sum(torch.square(joint_torques), dim=1) if hasattr(env.robot.data, 'applied_torque') else torch.zeros(env.scene.num_envs, device=env.device)
    
    # 4. Power penalty: |torque * velocity|
    rew_power = -env.cfg.rew_scale_power * power
    
    # ========== Post-task Rewards (rpost) - Only active in Phase 3 ==========
    # 1. Angular velocity: exp(-2 * ||ang_vel_xy||^2)
    ang_vel_xy = base_ang_vel[:, :2]
    ang_vel_xy_norm_sq = torch.sum(torch.square(ang_vel_xy), dim=1)
    rew_ang_vel_post = env.cfg.rew_scale_ang_vel_post * torch.exp(-2.0 * ang_vel_xy_norm_sq) * phase3_mask
    
    # 2. Linear velocity: exp(-5 * ||lin_vel_xy||^2)
    lin_vel_xy = base_lin_vel[:, :2]
    lin_vel_xy_norm_sq = torch.sum(torch.square(lin_vel_xy), dim=1)
    rew_lin_vel_post = env.cfg.rew_scale_lin_vel_post * torch.exp(-5.0 * lin_vel_xy_norm_sq) * phase3_mask
    
    # 3. Target base height: exp(-20 * |base_height - target|)
    base_height_target = env.cfg.base_height_target if hasattr(env.cfg, 'base_height_target') else 0.43
    height_error = torch.abs(base_height - base_height_target)
    rew_height_post = env.cfg.rew_scale_height_post * torch.exp(-20.0 * height_error) * phase3_mask
    
    # ========== Basic Rewards ==========
    rew_alive = env.cfg.rew_scale_alive * (1.0 - env.reset_terminated.float())
    rew_term = env.cfg.rew_scale_terminated * env.reset_terminated.float()
    
    # ========== Combine All Rewards ==========
    # Task reward is multiplied, others are added
    total_reward = (
        rew_task_total
        + rew_alive
        + rew_term
        + rew_waist_penalty
        + rew_knee_penalty
        + rew_feet_distance_penalty
        + rew_shank_orientation
        + rew_knee_extension
        + rew_joint_accel
        + rew_action_rate
        + rew_torque
        + rew_power
        + rew_ang_vel_post
        + rew_lin_vel_post
        + rew_height_post
    )
    
    # ========== Logging - Simplified ==========
    if env._tb_step % 32 == 0:
        # Only essential parameters
        env._tb_writer.add_scalar("debug/roll_deg", roll_deg.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/pitch_deg", pitch_deg.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/base_height", base_height.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_total", total_reward.mean().item(), env._tb_step)
    
    return total_reward
