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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

from .qmini_task_env_cfg import QminiTaskEnvCfg


class QminiTaskEnv(DirectRLEnv):
    cfg: QminiTaskEnvCfg

    def __init__(self, cfg: QminiTaskEnvCfg, render_mode: str | None = None, **kwargs):
        self.visualization_markers: VisualizationMarkers | None = None
        self.marker_offset: torch.Tensor | None = None
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
        self._min_height = self.cfg.failure_min_height
        self._success_joint_tol = self.cfg.success_joint_tol
        self._orientation_noise = math.radians(self.cfg.orientation_noise_deg)
        self._failure_pitch_angle = float(self.cfg.failure_pitch_angle)

        self._action_filter_gain = float(self.cfg.action_filter_gain)
        self._prev_actions = torch.zeros((self.scene.num_envs, self._num_dofs), device=device)
        self._filtered_actions = torch.zeros((self.scene.num_envs, self._num_dofs), device=device)
        self._prev_targets = self._target_pos.unsqueeze(0).expand(self.scene.num_envs, -1).clone()

        self._command = torch.zeros((self.scene.num_envs, 3), device=device)
        self._command_dir = torch.zeros((self.scene.num_envs, 2), device=device)
        self._command_timer = torch.zeros(self.scene.num_envs, device=device)
        self._command_change_interval = float(self.cfg.command_change_interval_s)

        self._gait_phase = torch.zeros(self.scene.num_envs, device=device)
        cycle = max(self.cfg.gait_cycle_duration, 1e-6)
        self._gait_phase_rate = 2.0 * math.pi / cycle

        log_dir = Path("logs/qmini_stand")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=str(log_dir))
        self._tb_step = 0

        self.marker_offset = torch.tensor([0.0, 0.0, 0.5], device=device, dtype=self.robot.data.root_pos_w.dtype)
        self._setup_visual_markers()
        self._visualize_markers()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)
        self._setup_visual_markers()

    def _setup_visual_markers(self) -> None:
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/qmini_arrows",
            markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.125, 0.125, 0.25),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.125, 0.125, 0.25),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        self.visualization_markers = VisualizationMarkers(cfg=marker_cfg)

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

    def _sample_commands(self, env_ids: torch.Tensor | Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids_t = torch.arange(self.scene.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)

        if env_ids_t.numel() == 0:
            return

        x_min, x_max = self.cfg.command_lin_vel_x_range
        y_min, y_max = self.cfg.command_lin_vel_y_range
        yaw_min, yaw_max = self.cfg.command_yaw_range

        rand_vals = torch.rand((env_ids_t.numel(), 3), device=self.device)
        self._command[env_ids_t, 0] = x_min + (x_max - x_min) * rand_vals[:, 0]
        self._command[env_ids_t, 1] = y_min + (y_max - y_min) * rand_vals[:, 1]
        self._command[env_ids_t, 2] = yaw_min + (yaw_max - yaw_min) * rand_vals[:, 2]

        self._command_timer[env_ids_t] = 0.0
        self._gait_phase[env_ids_t] = torch.rand(env_ids_t.numel(), device=self.device) * 2.0 * math.pi
        self._update_command_direction(env_ids_t)
        self._visualize_markers()

    def _update_command_direction(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        if env_ids is None:
            cmd_xy = self._command[:, :2]
            env_ids_t = slice(None)
        else:
            if isinstance(env_ids, torch.Tensor):
                env_ids_t = env_ids.to(device=self.device, dtype=torch.long)
            else:
                env_ids_t = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
            cmd_xy = self._command[env_ids_t, :2]

        norm = torch.norm(cmd_xy, dim=1, keepdim=True)
        dir_vec = torch.zeros_like(cmd_xy)
        mask = norm.squeeze(1) > 1e-5
        if mask.any():
            dir_vec[mask] = cmd_xy[mask] / norm[mask]

        if isinstance(env_ids_t, slice):
            self._command_dir = dir_vec
        else:
            self._command_dir[env_ids_t] = dir_vec

    def _compute_gait_targets(self) -> torch.Tensor:
        num_envs = self.scene.num_envs
        targets = self._target_pos.unsqueeze(0).expand(num_envs, -1).clone()

        cmd_xy = self._command[:, :2]
        cmd_speed = torch.norm(cmd_xy, dim=1)
        max_speed = max(
            1e-6,
            abs(self.cfg.command_lin_vel_x_range[0]),
            abs(self.cfg.command_lin_vel_x_range[1]),
            abs(self.cfg.command_lin_vel_y_range[0]),
            abs(self.cfg.command_lin_vel_y_range[1]),
        )
        speed_gain = torch.clamp(cmd_speed / max_speed, 0.0, 1.0)

        phase_left = self._gait_phase
        phase_right = (self._gait_phase + math.pi) % (2.0 * math.pi)

        hip_amp = self.cfg.gait_hip_amp * speed_gain
        knee_amp = self.cfg.gait_knee_amp * speed_gain
        ankle_amp = self.cfg.gait_ankle_amp * speed_gain

        targets[:, 0] = 0.25 * self._command[:, 1]
        targets[:, 5] = -0.25 * self._command[:, 1]

        targets[:, 2] = hip_amp * torch.sin(phase_left)
        targets[:, 7] = hip_amp * torch.sin(phase_right)

        targets[:, 3] = self.cfg.gait_knee_base + knee_amp * torch.sin(phase_left + self.cfg.gait_knee_phase)
        targets[:, 8] = self.cfg.gait_knee_base + knee_amp * torch.sin(phase_right + self.cfg.gait_knee_phase)

        targets[:, 4] = self.cfg.gait_ankle_base + ankle_amp * torch.sin(phase_left)
        targets[:, 9] = self.cfg.gait_ankle_base + ankle_amp * torch.sin(phase_right)

        lower = self._joint_lower.unsqueeze(0)
        upper = self._joint_upper.unsqueeze(0)
        return torch.clamp(targets, lower, upper)

    def _visualize_markers(self) -> None:
        if self.visualization_markers is None:
            return

        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        pos = root_pos + self.marker_offset

        cmd_xy = self._command[:, :2]
        cmd_speed = torch.norm(cmd_xy, dim=1)
        cmd_yaw = torch.atan2(cmd_xy[:, 1], cmd_xy[:, 0])

        zero_mask = cmd_speed < 1e-5
        cmd_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(cmd_yaw),
            torch.zeros_like(cmd_yaw),
            cmd_yaw,
        )
        if zero_mask.any():
            cmd_quat[zero_mask] = root_quat[zero_mask]

        positions = torch.cat([pos, pos], dim=0)
        rotations = torch.cat([root_quat, cmd_quat], dim=0)
        marker_ids = torch.cat(
            [
                torch.zeros(self.scene.num_envs, dtype=torch.long, device=self.device),
                torch.ones(self.scene.num_envs, dtype=torch.long, device=self.device),
            ],
            dim=0,
        )

        self.visualization_markers.visualize(positions, rotations, marker_indices=marker_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        raw_actions = torch.clamp(actions, -1.0, 1.0)
        gain = self._action_filter_gain
        if 0.0 < gain < 1.0:
            self._filtered_actions = self._filtered_actions + gain * (raw_actions - self._filtered_actions)
            self.actions = self._filtered_actions
        else:
            self.actions = raw_actions

        if self._command_change_interval > 0.0:
            self._command_timer += self.step_dt
            env_ids = torch.nonzero(self._command_timer >= self._command_change_interval, as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0:
                self._sample_commands(env_ids)

        self._gait_phase = (self._gait_phase + self._gait_phase_rate * self.step_dt) % (2.0 * math.pi)

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
        phase_features = torch.stack((torch.sin(self._gait_phase), torch.cos(self._gait_phase)), dim=1)
        obs = torch.cat(
            (
                current_pos,
                current_vel,
                base_quat,
                base_lin_vel,
                base_ang_vel,
                self._command,
                self._command_dir,
                phase_features,
            ),
            dim=1,
        )

        if self._tb_step % 128 == 0:
            self._tb_writer.add_scalar("obs/roll_deg", torch.rad2deg(roll).mean().item(), self._tb_step)
            self._tb_writer.add_scalar("obs/pitch_deg", torch.rad2deg(pitch).mean().item(), self._tb_step)

        self._visualize_markers()
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

        cmd_error = base_lin_vel[:, :2] - self._command[:, :2]
        rew_cmd_lin = -self.cfg.rew_scale_cmd_lin_vel * torch.norm(cmd_error, dim=1)
        yaw_error = base_ang_vel[:, 2] - self._command[:, 2]
        rew_cmd_yaw = -self.cfg.rew_scale_cmd_yaw_vel * torch.abs(yaw_error)
        gait_targets = self._compute_gait_targets()
        gait_error = torch.norm(current_pos - gait_targets, dim=1)
        rew_gait = -self.cfg.rew_scale_gait * gait_error

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
            + rew_cmd_lin
            + rew_cmd_yaw
            + rew_gait
        )

        if self._tb_step % 32 == 0:
            self._tb_writer.add_scalar("pose/roll_deg", roll_deg.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("pose/pitch_deg", pitch_deg.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/total", total_reward.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/upright_penalty", rew_upright.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/joint_penalty", rew_joint.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/action_rate_penalty", rew_action.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/cmd_lin", rew_cmd_lin.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/cmd_yaw", rew_cmd_yaw.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/gait", rew_gait.mean().item(), self._tb_step)

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
        # tilt_exceeded = torch.abs(pitch) > self._failure_pitch_angle

        out_of_limits = too_low  # | tilt_exceeded
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

        self._sample_commands(env_ids)

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
