# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for Qmini task.
Based on HoST (Humanoid Standing-up Control) framework from the paper.
Implements multi-stage curriculum learning with multi-critic reward structure.
"""

import math
import torch


def ftol(value: torch.Tensor, range_val: tuple[float, float], center: float, tolerance: float) -> torch.Tensor:
    """Tolerance function for smooth reward shaping.
    
    Args:
        value: Input value tensor
        range_val: (min, max) range for the value
        center: Center value for the tolerance
        tolerance: Tolerance width
        
    Returns:
        Reward value in [0, 1] range
    """
    min_val, max_val = range_val
    # Clamp value to range
    value_clamped = torch.clamp(value, min_val, max_val)
    # Compute distance from center
    dist = torch.abs(value_clamped - center)
    # Apply tolerance: exp(-dist/tolerance)
    reward = torch.exp(-dist / tolerance)
    return reward


def compute_rewards(env) -> torch.Tensor:
    """Compute total reward based on HoST framework with multi-stage curriculum learning.
    
    Implements:
    - Multi-stage rewards (Stage 1: h<0.45m, Stage 2: 0.45m≤h<0.65m, Stage 3: h≥0.65m)
    - Multi-critic structure: Task, Style, Regularization, Post-task rewards
    
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
    
    # Compute base orientation: z-axis component (vertical component)
    # For quaternion [w, x, y, z], z-axis in base frame is [2*(w*z + x*y), 2*(y*z - w*x), 1-2*(x^2+y^2)]
    x, y = base_quat[:, 1], base_quat[:, 2]
    base_z_axis = 1 - 2 * (x * x + y * y)  # Z component of base z-axis in world frame
    # Negative because we want upright (z-axis pointing up = 1)
    orientation_z = -base_z_axis  # Range: [-1, 1], 1 = upright
    
    roll_deg = torch.rad2deg(roll)
    pitch_deg = torch.rad2deg(pitch)
    
    # Get joint information
    pos_error = current_pos - target_pos
    joint_error = torch.norm(pos_error, dim=1)
    action_rate = torch.norm(env.actions - env._prev_actions, dim=1)
    
    # Get joint torques if available
    if hasattr(env.robot.data, 'applied_torque'):
        joint_torques = env.robot.data.applied_torque[:, env._controlled_joint_indices]
        torque_norm = torch.norm(joint_torques, dim=1)
        # Power = torque * velocity
        power = torch.abs(torch.sum(joint_torques * current_vel, dim=1))
    else:
        torque_norm = torch.zeros(env.scene.num_envs, device=env.device)
        power = torch.zeros(env.scene.num_envs, device=env.device)
    
    # Compute joint acceleration (approximate from velocity change)
    if not hasattr(env, '_prev_joint_vel'):
        env._prev_joint_vel = current_vel.clone()
    joint_accel = (current_vel - env._prev_joint_vel) / env.step_dt
    env._prev_joint_vel = current_vel.clone()
    joint_accel_norm = torch.norm(joint_accel, dim=1)
    
    # ========== Multi-Stage Detection ==========
    # Stage thresholds (adjusted for Qmini robot scale)
    stage1_threshold = 0.2  # Stage 1: h < 0.25m (body correction)
    stage2_threshold = 0.3  # Stage 2: 0.25m ≤ h < 0.35m (body lifting)
    # Stage 3: h ≥ 0.35m (fully standing)
    
    stage1_mask = (base_height < stage1_threshold).float()
    stage2_mask = ((base_height >= stage1_threshold) & (base_height < stage2_threshold)).float()
    stage3_mask = (base_height >= stage2_threshold).float()
    
    # ========== Task Rewards (rtask) ==========
    target_height = env.cfg.target_base_height if hasattr(env.cfg, 'target_base_height') else 0.43
    
    # 1. Head/Base height reward: ftol(hhead, [1, ∞], 1, 0.1)
    # Using base height as proxy for head height
    rew_height_task = env.cfg.rew_scale_height_task * ftol(
        base_height,
        [0.0, target_height * 1.5],
        target_height,
        0.1
    )
    
    # Height penalty: strongly penalize when height is below target
    # This encourages reaching target height
    height_deficit = torch.clamp(target_height - base_height, 0.0, target_height)
    rew_height_penalty = -env.cfg.rew_scale_height_penalty * height_deficit

    # 2. Body orientation reward: ftol(-θz_base, [0.99, ∞], 1, 0.05)
    # orientation_z ranges from -1 to 1, we want it close to 1 (upright)
    rew_orientation_task = env.cfg.rew_scale_orientation_task * ftol(
        orientation_z,
        [0.0, 1.0],
        1.0,
        0.05
    )
    
    # ========== Style Rewards (rstyle) ==========
    # 1. Waist twist penalty: 1(|qwaist| > 1.4)
    # Using yaw as proxy for waist twist
    yaw_abs = torch.abs(yaw)
    waist_twist_penalty = (yaw_abs > 1.4).float()  # ~80 degrees
    rew_waist_penalty = -env.cfg.rew_scale_waist_penalty * waist_twist_penalty
    
    # 2. Knee angle penalty: 1(max(|ql,r_knee|) > 2.85)
    # Knee joints are indices 3 and 8 (LL_joint4, RL_joint4)
    knee_indices = [3, 8]  # LL_knee, RL_knee
    knee_angles = torch.abs(current_pos[:, knee_indices])
    max_knee_angle = torch.max(knee_angles, dim=1).values
    knee_penalty = (max_knee_angle > 2.85).float()  # ~163 degrees
    rew_knee_penalty = -env.cfg.rew_scale_knee_penalty * knee_penalty
    
    # 3. Feet distance penalty: ∥ql_feet − qr_feet∥2 > 0.9
    # Ankle positions (approximate from joint positions)
    # Using ankle joint indices 4 and 9 (LL_joint5, RL_joint5)
    ankle_positions = current_pos[:, [4, 9]]  # Simplified: using joint angles
    feet_distance = torch.abs(ankle_positions[:, 0] - ankle_positions[:, 1])
    feet_distance_penalty = (feet_distance > 0.9).float()
    rew_feet_distance_penalty = -env.cfg.rew_scale_feet_distance_penalty * feet_distance_penalty
    
    # 4. Shank orientation reward: ftol(mean(θl,r_shank[2]), [0.8, ∞], 1, 0.1)
    # Simplified: using knee and ankle angles to estimate shank orientation
    # Shank should be vertical (close to 0 in pitch)
    shank_angles = (current_pos[:, 3] + current_pos[:, 8]) / 2.0  # Average knee angles
    rew_shank_orientation = env.cfg.rew_scale_shank_orientation * ftol(
        torch.abs(shank_angles),
        [0.0, math.pi],
        0.0,  # Want shank vertical (angle = 0)
        0.1
    )
    
    # ========== Regularization Rewards (rregu) ==========
    # 1. Joint acceleration penalty: ∥p̈∥2
    rew_joint_accel = -env.cfg.rew_scale_joint_accel * joint_accel_norm
    
    # 2. Action rate penalty: ∥at − at−1∥2
    rew_action_rate = -env.cfg.rew_scale_action_rate * action_rate
    
    # 3. Torque penalty: ∥τ∥2
    rew_torque = -env.cfg.rew_scale_torque * torque_norm
    
    # 4. Power penalty: |τ∥ṗ|T
    rew_power = -env.cfg.rew_scale_power * power
    
    # ========== Post-task Rewards (rpost) - Only active in Stage 3 ==========
    # Only apply when standing (stage 3)
    # 1. Angular velocity: exp(-2 × ∥ωxy_base∥2)
    ang_vel_xy = base_ang_vel[:, :2]  # Only x, y components
    ang_vel_xy_norm = torch.norm(ang_vel_xy, dim=1)
    rew_ang_vel_post = env.cfg.rew_scale_ang_vel_post * torch.exp(-2.0 * ang_vel_xy_norm) * stage3_mask
    
    # 2. Linear velocity: exp(-5 × ∥vxy_base∥2)
    lin_vel_xy = base_lin_vel[:, :2]  # Only x, y components
    lin_vel_xy_norm = torch.norm(lin_vel_xy, dim=1)
    rew_lin_vel_post = env.cfg.rew_scale_lin_vel_post * torch.exp(-5.0 * lin_vel_xy_norm) * stage3_mask
    
    # 3. Height maintenance: exp(-20 × ∥hbase − htarget_base∥2)
    height_error_post = torch.abs(base_height - target_height)
    rew_height_post = env.cfg.rew_scale_height_post * torch.exp(-20.0 * height_error_post) * stage3_mask
    
    # ========== Stage-specific Joint Position Reward ==========
    # Only reward joint position accuracy when height >= 0.25m (Stage 2 and 3)
    height_ok_mask = (base_height >= 0.25).float()
    rew_joint = -env.cfg.rew_scale_joint * joint_error * height_ok_mask
    
    # ========== Basic Rewards ==========
    rew_alive = env.cfg.rew_scale_alive * (1.0 - env.reset_terminated.float())
    rew_term = env.cfg.rew_scale_terminated * env.reset_terminated.float()
    
    # ========== Success Detection ==========
    orientation_ok = (torch.abs(roll) < env.cfg.success_pitch_tol) & (
        torch.abs(pitch) < env.cfg.success_pitch_tol
    )
    height_ok = base_height >= (target_height - 0.05)  # Within 5cm of target
    success_mask = (
        (torch.max(torch.abs(pos_error), dim=1).values < env._success_joint_tol)
        & orientation_ok
        & height_ok
    )
    rew_success = env.cfg.rew_scale_success * success_mask.float()
    
    # ========== Combine All Rewards ==========
    total_reward = (
        rew_alive
        + rew_term
        + rew_height_task
        + rew_height_penalty
        + rew_orientation_task
        + rew_waist_penalty
        + rew_knee_penalty
        + rew_feet_distance_penalty
        + rew_shank_orientation
        + rew_joint_accel
        + rew_action_rate
        + rew_torque
        + rew_power
        + rew_ang_vel_post
        + rew_lin_vel_post
        + rew_height_post
        + rew_joint
        + rew_success
    )
    
    # ========== Logging - All in debug folder ==========
    if env._tb_step % 32 == 0:
        # Pose information
        env._tb_writer.add_scalar("debug/roll_deg", roll_deg.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/pitch_deg", pitch_deg.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/yaw_deg", torch.rad2deg(yaw).mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/base_height", base_height.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/target_height", target_height, env._tb_step)
        env._tb_writer.add_scalar("debug/height_error", (target_height - base_height).mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/orientation_z", orientation_z.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/base_lin_vel_x", base_lin_vel[:, 0].mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/base_lin_vel_y", base_lin_vel[:, 1].mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/base_lin_vel_z", base_lin_vel[:, 2].mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/base_ang_vel_x", base_ang_vel[:, 0].mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/base_ang_vel_y", base_ang_vel[:, 1].mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/base_ang_vel_z", base_ang_vel[:, 2].mean().item(), env._tb_step)
        
        # Joint information
        env._tb_writer.add_scalar("debug/joint_error", joint_error.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/joint_accel_norm", joint_accel_norm.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/action_rate", action_rate.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/torque_norm", torque_norm.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/power", power.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/max_knee_angle", max_knee_angle.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/feet_distance", feet_distance.mean().item(), env._tb_step)
        
        # Stage information
        env._tb_writer.add_scalar("debug/stage1_mask", stage1_mask.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/stage2_mask", stage2_mask.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/stage3_mask", stage3_mask.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/height_ok_mask", height_ok_mask.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/success_mask", success_mask.float().mean().item(), env._tb_step)
        
        # All rewards
        env._tb_writer.add_scalar("debug/reward_total", total_reward.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_alive", rew_alive.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_term", rew_term.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_height_task", rew_height_task.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_height_penalty", rew_height_penalty.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_orientation_task", rew_orientation_task.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_waist_penalty", rew_waist_penalty.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_knee_penalty", rew_knee_penalty.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_feet_distance_penalty", rew_feet_distance_penalty.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_shank_orientation", rew_shank_orientation.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_joint_accel", rew_joint_accel.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_action_rate", rew_action_rate.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_torque", rew_torque.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_power", rew_power.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_ang_vel_post", rew_ang_vel_post.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_lin_vel_post", rew_lin_vel_post.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_height_post", rew_height_post.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_joint", rew_joint.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_success", rew_success.mean().item(), env._tb_step)
    
    return total_reward
