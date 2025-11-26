# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import math
import torch
from torch.utils.tensorboard import SummaryWriter

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .qmini_task_env_cfg import QminiTaskEnvCfg
from .reward import compute_rewards


class QminiTaskEnv(DirectRLEnv):
    cfg: QminiTaskEnvCfg

    def __init__(self, cfg: QminiTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._controlled_joint_names = list(self.cfg.controlled_joints)
        self._controlled_joint_indices: list[int] = []
        for name in self._controlled_joint_names:
            joint_ids, _ = self.robot.find_joints(name)
            if len(joint_ids) == 0:
                raise RuntimeError(f"Failed to find joint '{name}' in Qmini articulation.")
            self._controlled_joint_indices.append(joint_ids[0])

        self._num_dofs = len(self._controlled_joint_indices)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        device = self.device
        self._joint_lower = torch.tensor(self.cfg.joint_lower_limits, device=device)
        self._joint_upper = torch.tensor(self.cfg.joint_upper_limits, device=device)
        self._target_pos = torch.tensor(
            [self.cfg.target_joint_pos[name] for name in self._controlled_joint_names],
            device=device,
        )
        self._action_mid = (self._joint_upper + self._joint_lower) * 0.5
        self._action_scale = (self._joint_upper - self._joint_lower) * 0.5

        self._upright_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
        self._failure_up_cos = math.cos(self.cfg.failure_tilt_angle)
        self._success_up_cos = self.cfg.success_upright_cos
        self._min_height = self.cfg.failure_min_height
        self._success_joint_tol = self.cfg.success_joint_tol
        self._orientation_noise = math.radians(self.cfg.orientation_noise_deg)

        self._action_filter_gain = float(self.cfg.action_filter_gain)
        self._prev_actions = torch.zeros((self.scene.num_envs, self._num_dofs), device=device)
        self._filtered_actions = torch.zeros((self.scene.num_envs, self._num_dofs), device=device)
        self._prev_targets = self._target_pos.unsqueeze(0).expand(self.scene.num_envs, -1).clone()

        log_dir = Path("logs/qmini_stand")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=str(log_dir))
        self._tb_step = 0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    @staticmethod
    def _quat_to_euler(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * math.pi / 2,
            torch.asin(sinp),
        )

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    @staticmethod
    def _euler_to_quat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Convert Euler angles to quaternion (w, x, y, z)."""
        half_roll = roll * 0.5
        half_pitch = pitch * 0.5
        half_yaw = yaw * 0.5

        cr = torch.cos(half_roll)
        sr = torch.sin(half_roll)
        cp = torch.cos(half_pitch)
        sp = torch.sin(half_pitch)
        cy = torch.cos(half_yaw)
        sy = torch.sin(half_yaw)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return torch.stack((w, x, y, z), dim=1)

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (w, x, y, z)."""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        raw_actions = torch.clamp(actions, -1.0, 1.0)
        gain = self._action_filter_gain
        if 0.0 < gain < 1.0:
            self._filtered_actions = self._filtered_actions + gain * (raw_actions - self._filtered_actions)
            self.actions = self._filtered_actions
        else:
            self.actions = raw_actions

    def _apply_action(self) -> None:
        targets = self._action_mid + self._action_scale * self.actions
        smoothing = max(0.0, min(1.0, float(self.cfg.action_smoothing_rate)))
        if smoothing > 0.0:
            targets = self._prev_targets + smoothing * (targets - self._prev_targets)

        max_delta = self.cfg.max_joint_velocity * self.step_dt
        if max_delta > 0.0:
            delta = torch.clamp(targets - self._prev_targets, min=-max_delta, max=max_delta)
            targets = self._prev_targets + delta

        targets = torch.minimum(torch.maximum(targets, self._joint_lower), self._joint_upper)

        self.robot.set_joint_position_target(targets, joint_ids=self._controlled_joint_indices)
        self._prev_targets = targets

    def _get_observations(self) -> dict:
        """Get 42-dimensional observations following walk_policy format.
        
        Observation breakdown:
        1. base_lin_vel (3): Base linear velocity [vx, vy, vz]
        2. base_ang_vel (3): Base angular velocity [ωx, ωy, ωz]
        3. projected_gravity (3): Gravity projected in base frame
        4. velocity_commands (3): Velocity commands [target_vx, target_vy, target_ωz] (zeros for stand-up)
        5. joint_pos_rel (10): Joint positions relative to target
        6. joint_vel_rel (10): Joint velocities relative to target
        7. last_action (10): Previous action
        Total: 42 dimensions
        """
        root_state = self.robot.data.root_state_w
        base_quat = root_state[:, 3:7]  # [w, x, y, z]
        base_lin_vel = root_state[:, 7:10]  # [vx, vy, vz]
        base_ang_vel = root_state[:, 10:13]  # [ωx, ωy, ωz]
        
        # 1. base_lin_vel (3 dims) - scaled
        obs_base_lin_vel = base_lin_vel * 0.2  # Scale down for stability
        
        # 2. base_ang_vel (3 dims) - scaled
        obs_base_ang_vel = base_ang_vel * 0.2
        
        # 3. projected_gravity (3 dims) - gravity vector in base frame
        # Gravity in world frame: [0, 0, -1] (normalized)
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0).expand(self.scene.num_envs, -1)
        # Rotate gravity to base frame (inverse quaternion rotation)
        # For quaternion [w, x, y, z], inverse is [w, -x, -y, -z]
        quat_inv = base_quat.clone()
        quat_inv[:, 1:4] = -quat_inv[:, 1:4]  # Negate x, y, z components
        projected_gravity = self._quat_apply(quat_inv, gravity_world)
        
        # 4. velocity_commands (3 dims) - zeros for stand-up task
        velocity_commands = torch.zeros((self.scene.num_envs, 3), device=self.device)
        
        # 5. joint_pos_rel (10 dims) - joint positions relative to target
        all_joint_pos = self.joint_pos[:, self._controlled_joint_indices]
        all_joint_pos_rel = all_joint_pos - self._target_pos.unsqueeze(0)
        
        # 6. joint_vel_rel (10 dims) - joint velocities (already relative)
        joint_vel = self.joint_vel[:, self._controlled_joint_indices]
        obs_joint_vel = joint_vel * 0.05  # Scale down
        
        # 7. last_action (10 dims)
        obs_actions = self._prev_actions
        
        # Concatenate all observations: 3+3+3+3+10+10+10 = 42 dims
        obs = torch.cat(
            (
                obs_base_lin_vel,      # 3
                obs_base_ang_vel,       # 3
                projected_gravity,      # 3
                velocity_commands,      # 3
                all_joint_pos_rel,      # 10
                obs_joint_vel,          # 10
                obs_actions,            # 10
            ),
            dim=1,
        )
        
        roll, pitch, _ = self._quat_to_euler(base_quat)
        if self._tb_step % 128 == 0:
            self._tb_writer.add_scalar("obs/roll_deg", torch.rad2deg(roll).mean().item(), self._tb_step)
            self._tb_writer.add_scalar("obs/pitch_deg", torch.rad2deg(pitch).mean().item(), self._tb_step)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(self)
        self._tb_step += 1
        self._prev_actions = self.actions.clone()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Only time_out, no pose-related resets
        out_of_limits = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return out_of_limits, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, 7:] = 0.0  # Zero velocities

        # Random rotation around X/Y axis: ±90 degrees
        # This allows exploring different fall directions
        num_envs = len(env_ids)
        device = joint_pos.device
        
        # Random roll (rotation around X axis): ±90 degrees
        random_roll = (torch.rand(num_envs, device=device) - 0.5) * 2.0 * math.pi / 2.0  # [-90°, +90°]
        
        # Random pitch (rotation around Y axis): ±90 degrees
        random_pitch = (torch.rand(num_envs, device=device) - 0.5) * 2.0 * math.pi / 2.0  # [-90°, +90°]
        
        # Random yaw (rotation around Z axis): full 360 degrees
        random_yaw = torch.rand(num_envs, device=device) * 2.0 * math.pi - math.pi  # [-180°, +180°]
        
        # Convert to quaternion
        initial_quat = self._euler_to_quat(random_roll, random_pitch, random_yaw)
        default_root_state[:, 3:7] = initial_quat
        
        # Set joints to target position with noise
        target = self._target_pos.unsqueeze(0).expand(len(env_ids), -1)
        noise_range = (self._joint_upper - self._joint_lower) * self.cfg.reset_noise_scale
        noise = (torch.rand_like(target) - 0.5) * 2.0 * noise_range
        sampled = torch.clamp(target + noise, self._joint_lower, self._joint_upper)
        joint_pos[:, self._controlled_joint_indices] = sampled
        joint_vel[:, self._controlled_joint_indices] = 0.0

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._prev_actions[env_ids] = 0.0
        self._filtered_actions[env_ids] = 0.0
        self._prev_targets[env_ids] = joint_pos[:, self._controlled_joint_indices]

    @staticmethod
    def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate vector(s) by quaternion(s)."""
        q_xyz = quat[:, :3]
        q_w = quat[:, 3].unsqueeze(1)
        t = 2.0 * torch.cross(q_xyz, vec, dim=1)
        return vec + q_w * t + torch.cross(q_xyz, t, dim=1)
