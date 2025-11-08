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
        current_pos = self.joint_pos[:, self._controlled_joint_indices]
        current_vel = self.joint_vel[:, self._controlled_joint_indices]
        root_state = self.robot.data.root_state_w
        base_quat = root_state[:, 3:7]
        base_lin_vel = root_state[:, 7:10]
        base_ang_vel = root_state[:, 10:13]
        roll, pitch, _ = self._quat_to_euler(base_quat)
        obs = torch.cat((current_pos, current_vel, base_quat, base_lin_vel, base_ang_vel), dim=1)

        if self._tb_step % 128 == 0:
            self._tb_writer.add_scalar("obs/roll_deg", torch.rad2deg(roll).mean().item(), self._tb_step)
            self._tb_writer.add_scalar("obs/pitch_deg", torch.rad2deg(pitch).mean().item(), self._tb_step)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        current_pos = self.joint_pos[:, self._controlled_joint_indices]
        current_vel = self.joint_vel[:, self._controlled_joint_indices]
        target_pos = self._target_pos.unsqueeze(0).expand_as(current_pos)

        root_state = self.robot.data.root_state_w
        base_quat = root_state[:, 3:7]
        base_lin_vel = root_state[:, 7:10]
        base_ang_vel = root_state[:, 10:13]
        roll, pitch, _ = self._quat_to_euler(base_quat)
        orientation_error = torch.sqrt(roll * roll + pitch * pitch)
        roll_deg = torch.rad2deg(roll)
        pitch_deg = torch.rad2deg(pitch)

        pos_error = current_pos - target_pos
        joint_error = torch.norm(pos_error, dim=1)
        joint_vel_norm = torch.norm(current_vel, dim=1)
        lin_vel_norm = torch.norm(base_lin_vel, dim=1)
        ang_vel_norm = torch.norm(base_ang_vel, dim=1)
        action_rate = torch.norm(self.actions - self._prev_actions, dim=1)

        rew_alive = self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float())
        rew_term = self.cfg.rew_scale_terminated * self.reset_terminated.float()
        rew_joint = -self.cfg.rew_scale_joint * joint_error
        rew_joint_vel = -self.cfg.rew_scale_joint_vel * joint_vel_norm
        rew_upright = -self.cfg.rew_scale_upright * orientation_error
        rew_base_lin = -self.cfg.rew_scale_base_lin_vel * lin_vel_norm
        rew_base_ang = -self.cfg.rew_scale_base_ang_vel * ang_vel_norm
        rew_action = -self.cfg.rew_scale_action_rate * action_rate

        orientation_ok = (torch.abs(roll) < self.cfg.success_pitch_tol) & (
            torch.abs(pitch) < self.cfg.success_pitch_tol
        )
        success_mask = (torch.max(torch.abs(pos_error), dim=1).values < self._success_joint_tol) & orientation_ok
        rew_success = self.cfg.rew_scale_success * success_mask.float()

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
        )

        if self._tb_step % 32 == 0:
            self._tb_writer.add_scalar("pose/roll_deg", roll_deg.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("pose/pitch_deg", pitch_deg.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/total", total_reward.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/upright_penalty", rew_upright.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/joint_penalty", rew_joint.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/action_rate_penalty", rew_action.mean().item(), self._tb_step)

        self._tb_step += 1
        self._prev_actions = self.actions.clone()
        self._tb_step += 1
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        root_state = self.robot.data.root_state_w
        base_quat = root_state[:, 3:7]
        base_height = root_state[:, 2]
        too_low = base_height < self._min_height

        roll, pitch, _ = self._quat_to_euler(base_quat)
        tilt_exceeded = (torch.abs(pitch) > self.cfg.failure_tilt_angle) | (
            torch.abs(roll) > self.cfg.failure_tilt_angle
        )

        out_of_limits = too_low | tilt_exceeded
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return out_of_limits, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        target = self._target_pos.unsqueeze(0).expand(len(env_ids), -1)
        noise_range = (self._joint_upper - self._joint_lower) * self.cfg.reset_noise_scale
        noise = (torch.rand_like(target) - 0.5) * 2.0 * noise_range
        sampled = torch.clamp(target + noise, self._joint_lower, self._joint_upper)

        joint_pos[:, self._controlled_joint_indices] = sampled
        joint_vel[:, self._controlled_joint_indices] = 0.0

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, 7:] = 0.0

        if self._orientation_noise > 0.0:
            noise_roll = (torch.rand(len(env_ids), device=joint_pos.device) - 0.5) * 2.0 * self._orientation_noise
            noise_pitch = (torch.rand(len(env_ids), device=joint_pos.device) - 0.5) * 2.0 * self._orientation_noise
            noise_yaw = torch.zeros_like(noise_roll)

            delta_quat = self._euler_to_quat(noise_roll, noise_pitch, noise_yaw)
            default_quat = default_root_state[:, 3:7]
            new_quat = self._quat_multiply(delta_quat, default_quat)
            default_root_state[:, 3:7] = new_quat

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
