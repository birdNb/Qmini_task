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
import isaaclab.utils.math as math_utils

from .gait_curriculum import OmniGaitCurriculum
from .gait_rewards import compute_gait_rewards

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

        left_bodies, _ = self.robot.find_bodies(["LL_ankle"])
        right_bodies, _ = self.robot.find_bodies(["RL_ankle"])
        if len(left_bodies) == 0 or len(right_bodies) == 0:
            raise RuntimeError("Failed to locate ankle bodies for gait metrics.")
        self._left_foot_body_idx = int(left_bodies[0])
        self._right_foot_body_idx = int(right_bodies[0])
        self._single_leg_margin = getattr(self.cfg, "single_leg_height_margin", 0.03)

        self._gait_phase = torch.zeros(self.scene.num_envs, device=device)
        cycle = max(self.cfg.gait_cycle_duration, 1e-6)
        self._gait_phase_rate = 2.0 * math.pi / cycle
        self._control_dt = self.step_dt
        self._joint_target_speed = getattr(self.cfg, "joint_target_speed", 1.0)
        self._joint_speed_scale = getattr(self.cfg, "rew_scale_joint_speed", 0.0)
        self._height_fail_scale = getattr(self.cfg, "rew_scale_height_fail", 0.0)
        self._foot_contact_threshold = max(getattr(self.cfg, "foot_contact_force_threshold", 1.0), 1e-3)
        self._desired_foot_clearance = getattr(self.cfg, "desired_foot_clearance", 0.05)
        self._foot_contact_sensor = None
        sensors = getattr(self.scene, "sensors", None)
        if sensors is not None:
            if isinstance(sensors, dict):
                self._foot_contact_sensor = sensors.get("foot_contact_sensor")
            else:
                self._foot_contact_sensor = getattr(sensors, "foot_contact_sensor", None)
        self._foot_contact_indices: list[int] | None = None
        if self._foot_contact_sensor is not None and hasattr(self._foot_contact_sensor, "body_names"):
            try:
                body_names = list(self._foot_contact_sensor.body_names)
                indices: list[int] = []
                for target in ("LL_ankle", "RL_ankle"):
                    match_idx = next((i for i, name in enumerate(body_names) if target in name), None)
                    if match_idx is not None:
                        indices.append(match_idx)
                if len(indices) == 2:
                    self._foot_contact_indices = indices
            except Exception:
                self._foot_contact_indices = None

        log_dir = Path("logs/qmini_stand")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=str(log_dir))
        self._tb_step = 0

        self.marker_offset = torch.tensor([0.0, 0.0, 0.5], device=device, dtype=self.robot.data.root_pos_w.dtype)
        self._setup_visual_markers()
        self._visualize_markers()

        self.curriculum = OmniGaitCurriculum(self.cfg, self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # Load environment from USD file instead of default ground plane
        environment_usd_path = "/home/bird/isaacSim/Learn/default_environment.usd"
        environment_cfg = sim_utils.UsdFileCfg(usd_path=environment_usd_path)
        environment_cfg.func("/World/environment", environment_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_visual_markers(self) -> None:
        arrow_usd_path = "/home/bird/isaacSim/Learn/arrow_x.usd"
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/qmini_arrows",
            markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=arrow_usd_path,
                    scale=(0.125, 0.125, 0.25),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=arrow_usd_path,
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

        x_range, y_range, yaw_range = self.curriculum.get_command_ranges()
        x_min, x_max = x_range
        y_min, y_max = y_range
        yaw_min, yaw_max = yaw_range

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
        x_range, y_range, _ = self.curriculum.get_command_ranges()
        max_speed = max(1e-6, abs(x_range[0]), abs(x_range[1]), abs(y_range[0]), abs(y_range[1]))
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
        base_height = root_state[:, 2]
        roll, pitch, _ = self._quat_to_euler(base_quat)
        orientation_error = torch.sqrt(roll * roll + pitch * pitch)
        roll_deg = torch.rad2deg(roll)
        pitch_deg = torch.rad2deg(pitch)
        tilt_exceeded = torch.abs(pitch) > self._failure_pitch_angle

        pos_error = current_pos - target_pos
        joint_error = torch.norm(pos_error, dim=1)
        joint_vel_norm = torch.norm(current_vel, dim=1)
        lin_vel_norm = torch.norm(base_lin_vel, dim=1)
        ang_vel_norm = torch.norm(base_ang_vel, dim=1)
        action_rate = torch.norm(self.actions - self._prev_actions, dim=1)
        forward_speed = torch.clamp(base_lin_vel[:, 0], min=0.0)
        cmd_lin_x = self._command[:, 0]
        cmd_yaw = self._command[:, 2]
        lin_vel_scale = torch.clamp(torch.abs(cmd_lin_x), min=0.3, max=2.0) + 0.2
        yaw_norm = torch.clamp(torch.abs(cmd_yaw), min=0.3, max=1.5) + 0.2
        forward_factor = torch.clamp(5.0 / lin_vel_scale, min=2.0, max=10.0)
        yaw_factor = torch.clamp(2.0 / yaw_norm, min=2.0, max=6.0)
        forward_track = torch.exp(-forward_factor * (cmd_lin_x - base_lin_vel[:, 0]) ** 2)
        yaw_track = torch.exp(-yaw_factor * (cmd_yaw - base_ang_vel[:, 2]) ** 2)
        orientation_norm_sq = roll * roll + pitch * pitch
        height_term = torch.exp(-70.0 * (base_height - self.cfg.desired_root_height) ** 2)
        balance_factor = torch.clamp(5.0 / lin_vel_scale, min=2.0, max=8.0)
        balance = 0.5 * (height_term * torch.exp(-balance_factor * orientation_norm_sq) + 1.0)

        foot_positions = self.robot.data.body_pos_w[
            :, (self._left_foot_body_idx, self._right_foot_body_idx), :
        ]
        foot_velocities = self.robot.data.body_vel_w[
            :, (self._left_foot_body_idx, self._right_foot_body_idx), :
        ]
        foot_heights = foot_positions[:, :, 2]
        foot_speed_xy = torch.norm(foot_velocities[:, :, :2], dim=-1)
        contact_mask: torch.Tensor
        if (
            self._foot_contact_sensor is not None
            and hasattr(self._foot_contact_sensor, "data")
            and getattr(self._foot_contact_sensor.data, "net_forces_w", None) is not None
        ):
            forces = self._foot_contact_sensor.data.net_forces_w
            try:
                if self._foot_contact_indices is not None:
                    index_tensor = torch.as_tensor(
                        self._foot_contact_indices, device=self.device, dtype=torch.long
                    )
                    forces = forces.index_select(1, index_tensor)
                contact_force_mag = torch.norm(forces[..., :3], dim=-1)
                contact_mask = contact_force_mag > self._foot_contact_threshold
            except Exception:
                contact_mask = foot_heights < 0.02
        else:
            contact_mask = foot_heights < 0.02
        contact_mask = contact_mask.float()
        swing_mask = 1.0 - contact_mask
        foot_clearance = torch.clamp(foot_heights - self._desired_foot_clearance, min=0.0)
        foot_clearance_bonus = (foot_clearance * swing_mask).sum(dim=1)
        foot_slip = (foot_speed_xy * contact_mask).sum(dim=1)

        cmd_error = base_lin_vel[:, :2] - self._command[:, :2]
        rew_cmd_lin = -self.cfg.rew_scale_cmd_lin_vel * torch.norm(cmd_error, dim=1)
        yaw_error = base_ang_vel[:, 2] - self._command[:, 2]
        rew_cmd_yaw = -self.cfg.rew_scale_cmd_yaw_vel * torch.abs(yaw_error)
        gait_targets = self._compute_gait_targets()
        gait_error = torch.norm(current_pos - gait_targets, dim=1)
        rew_gait = -self.cfg.rew_scale_gait * gait_error
        gait_stats = compute_gait_rewards(self, root_state)
        rew_height = gait_stats["reward_height"]
        rew_single = gait_stats["reward_single"]
        joint_speed_deficit = torch.clamp(self._joint_target_speed - torch.abs(current_vel), min=0.0)
        rew_joint_speed = -self._joint_speed_scale * torch.mean(joint_speed_deficit, dim=1)
        forward_speed_scale = getattr(self.cfg, "rew_scale_forward_speed", 0.0)
        rew_forward_speed = forward_speed_scale * forward_speed
        rew_forward_track = getattr(self.cfg, "rew_scale_forward_track", 0.0) * forward_track
        rew_yaw_track = getattr(self.cfg, "rew_scale_yaw_track", 0.0) * yaw_track
        rew_balance = getattr(self.cfg, "rew_scale_balance", 0.0) * balance
        rew_lateral_penalty = -getattr(self.cfg, "rew_scale_lateral_penalty", 0.0) * torch.abs(base_lin_vel[:, 1])
        rew_foot_clear = getattr(self.cfg, "rew_scale_foot_clear", 0.0) * foot_clearance_bonus
        rew_foot_slip = -getattr(self.cfg, "rew_scale_foot_slip", 0.0) * foot_slip
        height_fail = base_height < self._min_height
        rew_height_fail = -self._height_fail_scale * height_fail.float()
        rew_tilt_fail = -self.cfg.rew_scale_tilt_fail * tilt_exceeded.float()

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
            + rew_height
            + rew_single
            + rew_tilt_fail
            + rew_height_fail
            + rew_forward_speed
            + rew_forward_track
            + rew_yaw_track
            + rew_balance
            + rew_lateral_penalty
            + rew_foot_clear
            + rew_foot_slip
            + rew_joint_speed
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
            self._tb_writer.add_scalar("reward/height", rew_height.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/single_leg", rew_single.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/tilt_fail", rew_tilt_fail.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/height_fail", rew_height_fail.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/forward_speed", rew_forward_speed.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/forward_track", rew_forward_track.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/yaw_track", rew_yaw_track.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/balance", rew_balance.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/lateral_penalty", rew_lateral_penalty.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/foot_clear", rew_foot_clear.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/foot_slip", rew_foot_slip.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/joint_speed", rew_joint_speed.mean().item(), self._tb_step)

        cmd_error_mean = float(torch.norm(cmd_error, dim=1).mean().item())
        if self.curriculum.update(
            self._control_dt, gait_stats["height_mean"], gait_stats["single_rate"], cmd_error_mean
        ):
            self._sample_commands(None)

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
        out_of_limits = too_low  # tilt-based reset disabled per user request
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return out_of_limits, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if self.curriculum.enabled and len(env_ids) == self.scene.num_envs:
            self.curriculum.reset()

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
