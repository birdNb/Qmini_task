# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for Qmini task.
Based on HoST (Humanoid Standing-up Control) framework.
Implements 4-group reward structure: task, regu, style, target.
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
    """Compute rewards following HoST 4-group structure.
    
    Reward groups:
    1. task: Task rewards (multiplicative combination)
       - Orientation reward
       - Head height reward (relative to feet)
    2. regu: Regularization rewards (additive)
       - Joint acceleration penalty
       - Action rate penalty
       - Torque penalty
       - Joint power penalty
       - Joint velocity penalty
       - Joint tracking error penalty
       - Joint position limits penalty
       - Joint velocity limits penalty
    3. style: Style rewards (additive)
       - Knee deviation penalty
       - Feet distance penalty
       - Shank orientation reward
       - Ground parallel reward
    4. target: Post-task rewards (additive, only when standing)
       - Target angular velocity
       - Target linear velocity
       - Target orientation
       - Target base height (0.43m)
    
    Args:
        env: The environment instance
        
    Returns:
        Dictionary with reward groups: {'task': ..., 'regu': ..., 'style': ..., 'target': ...}
    """
    # ========== Extract State Information ==========
    root_state = env.robot.data.root_state_w
    base_height = root_state[:, 2]  # Z coordinate (height)
    base_lin_vel = root_state[:, 7:10]  # [vx, vy, vz]
    base_ang_vel = root_state[:, 10:13]  # [ωx, ωy, ωz]
    base_quat = root_state[:, 3:7]  # [w, x, y, z]
    
    # Get joint information
    current_pos = env.joint_pos[:, env._controlled_joint_indices]
    current_vel = env.joint_vel[:, env._controlled_joint_indices]
    target_pos = env._target_pos.unsqueeze(0).expand_as(current_pos)
    
    # Compute projected gravity
    gravity_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).expand(env.scene.num_envs, -1)
    quat_inv = base_quat.clone()
    quat_inv[:, 1:4] = -quat_inv[:, 1:4]
    projected_gravity = env._quat_apply(quat_inv, gravity_world)
    
    # Compute pitch angle for penalty
    roll, pitch, yaw = env._quat_to_euler(base_quat)
    pitch_deg = torch.rad2deg(pitch)
    
    # Phase detection (scaled for Qmini: 0.43m is normal standing height)
    phase1_threshold = getattr(env.cfg, 'target_base_height_phase1', 0.25)  # 0.25m (scaled from 0.45m)
    phase3_threshold = getattr(env.cfg, 'target_base_height_phase3', 0.35)  # 0.35m (scaled from 0.65m)
    phase1_mask = (base_height > phase1_threshold).float()
    phase3_mask = (base_height > phase3_threshold).float()
    
    # ========== TASK REWARDS (Multiplicative Combination) ==========
    # Initialize task reward to 1.0, then multiply by each task component
    rew_task = torch.ones(env.scene.num_envs, device=env.device)
    
    # 1. Orientation reward (only active when base_height > phase1_threshold)
    # Target: -projected_gravity[2] should be close to 1.0 (upright)
    orientation_reward = tolerance(
        -projected_gravity[:, 2],
        [0.99, np.inf],
        1.0,
        0.05
    )
    rew_task *= orientation_reward * phase1_mask + (1.0 - phase1_mask)  # Only active in phase1+
    
    # 2. Head height reward (relative to feet)
    # For Qmini, use base height as proxy for head height
    # Target: head height relative to feet should be ~0.43m (scaled from 1.0m)
    try:
        # Get feet height
        left_ankle_ids, _ = env.robot.find_bodies("LL_ankle")
        right_ankle_ids, _ = env.robot.find_bodies("RL_ankle")
        if len(left_ankle_ids) > 0 and len(right_ankle_ids) > 0:
            left_ankle_pos = env.robot.data.body_pos_w[:, left_ankle_ids[0], :]
            right_ankle_pos = env.robot.data.body_pos_w[:, right_ankle_ids[0], :]
            feet_height = (left_ankle_pos[:, 2] + right_ankle_pos[:, 2]) / 2.0
        else:
            feet_height = torch.zeros(env.scene.num_envs, device=env.device)
    except (AttributeError, IndexError, KeyError, RuntimeError, ValueError):
        feet_height = torch.zeros(env.scene.num_envs, device=env.device)
    
    head_height_rel = base_height - feet_height  # Head height relative to feet
    target_head_height = getattr(env.cfg, 'target_head_height', 0.43)  # Scaled from 1.0m
    head_height_reward = tolerance(
        head_height_rel,
        [target_head_height, np.inf],
        target_head_height,
        0.1
    )
    rew_task *= head_height_reward
    
    # ========== REGULARIZATION REWARDS (Additive) ==========
    rew_regu = torch.zeros(env.scene.num_envs, device=env.device)
    
    # 1. Joint acceleration penalty
    if not hasattr(env, '_prev_joint_vel'):
        env._prev_joint_vel = current_vel.clone()
    joint_accel = (current_vel - env._prev_joint_vel) / env.step_dt
    env._prev_joint_vel = current_vel.clone()
    rew_regu += -2.5e-7 * torch.sum(torch.square(joint_accel), dim=1)
    
    # 2. Action rate penalty
    if not hasattr(env, '_prev_actions'):
        env._prev_actions = torch.zeros_like(env.actions)
    action_rate = torch.sum(torch.square(env.actions - env._prev_actions), dim=1)
    rew_regu += -0.01 * action_rate
    env._prev_actions = env.actions.clone()
    
    # 3. Torque penalty
    if hasattr(env.robot.data, 'applied_torque'):
        joint_torques = env.robot.data.applied_torque[:, env._controlled_joint_indices]
        rew_regu += -2.5e-6 * torch.sum(torch.square(joint_torques), dim=1)
    else:
        joint_torques = torch.zeros_like(current_pos)
    
    # 4. Joint power penalty
    if hasattr(env.robot.data, 'applied_torque'):
        joint_power = torch.abs(current_vel * joint_torques)
        rew_regu += -2.5e-5 * torch.sum(joint_power, dim=1)
    
    # 5. Joint velocity penalty
    rew_regu += -1e-3 * torch.sum(torch.square(current_vel), dim=1)
    
    # 6. Joint tracking error penalty
    joint_tracking_error = torch.sum(torch.square(target_pos - current_pos), dim=1)
    rew_regu += -0.00025 * joint_tracking_error
    
    # 7. Joint position limits penalty
    joint_lower = env._joint_lower.unsqueeze(0).expand_as(current_pos)
    joint_upper = env._joint_upper.unsqueeze(0).expand_as(current_pos)
    pos_limits_violation = torch.sum(
        torch.clamp(joint_lower - current_pos, 0.0, None) ** 2
        + torch.clamp(current_pos - joint_upper, 0.0, None) ** 2,
        dim=1
    )
    rew_regu += -100.0 * pos_limits_violation  # Large penalty
    
    # 8. Joint velocity limits penalty (check if velocity exceeds reasonable limits)
    max_vel_limit = getattr(env.cfg, 'max_joint_velocity', 5.0)
    vel_limits_violation = torch.sum(
        torch.clamp(torch.abs(current_vel) - max_vel_limit, 0.0, None) ** 2,
        dim=1
    )
    rew_regu += -1.0 * vel_limits_violation
    
    # 9. Pitch deviation penalty (增大pitch偏离的惩罚)
    # Penalize pitch deviation from 0 (upright)
    pitch_penalty_scale = getattr(env.cfg, 'rew_scale_pitch_penalty', 5.0)  # Increased from default
    pitch_error_deg = torch.abs(pitch_deg)
    # Use exponential penalty: stronger penalty for larger deviations
    rew_regu += -pitch_penalty_scale * torch.exp(pitch_error_deg / 10.0)  # Exponential penalty
    
    # ========== STYLE REWARDS (Additive) ==========
    rew_style = torch.zeros(env.scene.num_envs, device=env.device)
    
    # 1. Knee deviation penalty
    knee_indices = [3, 8]  # LL_knee, RL_knee
    knee_angles = current_pos[:, knee_indices]
    knee_max = torch.max(torch.abs(knee_angles), dim=1).values
    knee_min = torch.min(knee_angles, dim=1).values
    knee_deviation = ((knee_max > 2.85) | (knee_min < -0.06)).float()
    rew_style += -0.25 * knee_deviation
    
    # 2. Feet distance penalty
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
    rew_style += -10.0 * feet_distance_penalty
    
    # 3. Shank orientation reward (only active in phase1+)
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
            
            left_shank_z = left_shank_vec[:, 2] / (torch.norm(left_shank_vec, dim=1) + 1e-6)
            right_shank_z = right_shank_vec[:, 2] / (torch.norm(right_shank_vec, dim=1) + 1e-6)
            
            shank_z = (left_shank_z + right_shank_z) / 2.0
            shank_orientation_reward = tolerance(shank_z, [0.8, np.inf], 1.0, 0.1)
            rew_style += 10.0 * shank_orientation_reward * phase1_mask
    except (AttributeError, IndexError, KeyError, RuntimeError, ValueError):
        pass
    
    # 4. Ground parallel reward (feet height variance)
    try:
        if len(left_ankle_ids) > 0 and len(right_ankle_ids) > 0:
            left_ankle_pos = env.robot.data.body_pos_w[:, left_ankle_ids[0], :]
            right_ankle_pos = env.robot.data.body_pos_w[:, right_ankle_ids[0], :]
            feet_height_diff = torch.abs(left_ankle_pos[:, 2] - right_ankle_pos[:, 2])
            ground_parallel_reward = torch.exp(-feet_height_diff * 2.0)
            rew_style += 20.0 * ground_parallel_reward
    except (AttributeError, IndexError, KeyError, RuntimeError, ValueError):
        pass
    
    # ========== TARGET/POST-TASK REWARDS (Additive, only in phase3) ==========
    rew_target = torch.zeros(env.scene.num_envs, device=env.device)
    
    # 1. Target angular velocity (XY plane)
    ang_vel_xy = base_ang_vel[:, :2]
    ang_vel_xy_norm_sq = torch.sum(torch.square(ang_vel_xy), dim=1)
    rew_target += 10.0 * torch.exp(-2.0 * ang_vel_xy_norm_sq) * phase3_mask
    
    # 2. Target linear velocity (XY plane)
    lin_vel_xy = base_lin_vel[:, :2]
    lin_vel_xy_norm_sq = torch.sum(torch.square(lin_vel_xy), dim=1)
    rew_target += 10.0 * torch.exp(-5.0 * lin_vel_xy_norm_sq) * phase3_mask
    
    # 3. Target orientation (projected gravity XY should be 0)
    projected_gravity_xy = projected_gravity[:, :2]
    projected_gravity_xy_norm_sq = torch.sum(torch.square(projected_gravity_xy), dim=1)
    rew_target += 10.0 * torch.exp(-5.0 * projected_gravity_xy_norm_sq) * phase3_mask
    
    # 4. Target base height (0.43m for Qmini)
    target_base_height = getattr(env.cfg, 'target_base_height', 0.43)
    base_height_error = torch.abs(base_height - target_base_height)
    rew_target += 10.0 * torch.exp(-20.0 * base_height_error) * phase3_mask
    
    # ========== Combine Reward Groups ==========
    # Get reward group weights from config
    w_task = getattr(env.cfg, 'rew_weight_task', 2.5)
    w_regu = getattr(env.cfg, 'rew_weight_regu', 0.1)
    w_style = getattr(env.cfg, 'rew_weight_style', 1.0)
    w_target = getattr(env.cfg, 'rew_weight_target', 1.0)
    
    total_reward = (
        w_task * rew_task
        + w_regu * rew_regu
        + w_style * rew_style
        + w_target * rew_target
    )
    
    # ========== Basic Rewards ==========
    rew_alive = env.cfg.rew_scale_alive * (1.0 - env.reset_terminated.float())
    rew_term = env.cfg.rew_scale_terminated * env.reset_terminated.float()
    
    total_reward = total_reward + rew_alive + rew_term
    
    # ========== Logging ==========
    if env._tb_step % 32 == 0:
        env._tb_writer.add_scalar("reward/task", rew_task.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("reward/regu", rew_regu.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("reward/style", rew_style.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("reward/target", rew_target.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("reward/total", total_reward.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/base_height", base_height.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/head_height_rel", head_height_rel.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/pitch_deg", pitch_deg.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/phase3_mask", phase3_mask.mean().item(), env._tb_step)
    
    return total_reward
