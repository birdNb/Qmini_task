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
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.utils.math as math_utils

from .gait_curriculum import OmniGaitCurriculum
from .gait_rewards import compute_gait_rewards, compute_feet_air_time, compute_feet_slide

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

        # Joint indices for grouped observations
        # HR joints (hip_pos): joint1
        self._hip_joint_indices = [self._controlled_joint_indices[0], self._controlled_joint_indices[5]]
        # HAA+HFE+KFE joints (kfe_pos): joint2,3,4
        self._kfe_joint_indices = [
            self._controlled_joint_indices[1], self._controlled_joint_indices[2], self._controlled_joint_indices[3],
            self._controlled_joint_indices[6], self._controlled_joint_indices[7], self._controlled_joint_indices[8]
        ]
        # FFE joints (ffe_pos): joint5
        self._ffe_joint_indices = [self._controlled_joint_indices[4], self._controlled_joint_indices[9]]

        # Target positions for relative joint positions
        self._hip_target = torch.tensor(
            [self.cfg.target_joint_pos["LL_joint1"], self.cfg.target_joint_pos["RL_joint1"]],
            device=device
        )
        self._kfe_target = torch.tensor(
            [
                self.cfg.target_joint_pos["LL_joint2"], self.cfg.target_joint_pos["LL_joint3"], self.cfg.target_joint_pos["LL_joint4"],
                self.cfg.target_joint_pos["RL_joint2"], self.cfg.target_joint_pos["RL_joint3"], self.cfg.target_joint_pos["RL_joint4"]
            ],
            device=device
        )
        self._ffe_target = torch.tensor(
            [self.cfg.target_joint_pos["LL_joint5"], self.cfg.target_joint_pos["RL_joint5"]],
            device=device
        )

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
                self._foot_contact_sensor = sensors.get("contact_forces")
            else:
                self._foot_contact_sensor = getattr(sensors, "contact_forces", None)
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
        # Terrain is configured in __post_init__ of config class

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.6, 0.6, 0.6))
        light_cfg.func("/World/Light", light_cfg)
        # Register contact sensors following tutorial style:
        # - expose a combined key "contact_forces" (use left as primary to match example access)
        # - also register right ankle separately if provided
        self.scene.sensors["contact_forces_left"] = ContactSensor(self.cfg.contact_forces_left)
        self.scene.sensors["contact_forces_right"] = ContactSensor(self.cfg.contact_forces_right)

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
        """Get 42-dimensional observations following the reference implementation.

        Observation breakdown:
        1. base_lin_vel (3): Base linear velocity [vx, vy, vz]
        2. base_ang_vel (3): Base angular velocity [ωx, ωy, ωz]
        3. projected_gravity (3): Gravity projected in base frame
        4. velocity_commands (3): Velocity commands [target_vx, target_vy, target_ωz]
        5. hip_pos (2): HR joint positions relative to target (joint1)
        6. kfe_pos (6): HAA+HFE+KFE joint positions relative to target (joint2,3,4)
        7. ffe_pos (2): FFE joint positions relative to target (joint5)
        8. joint_vel (10): All joint velocities relative to target
        9. actions (10): Previous action
        Total: 42 dimensions
        """
        root_state = self.robot.data.root_state_w
        base_quat = root_state[:, 3:7]  # [w, x, y, z]
        base_lin_vel = root_state[:, 7:10]  # [vx, vy, vz]
        base_ang_vel = root_state[:, 10:13]  # [ωx, ωy, ωz]

        # 1. base_lin_vel (3 dims) - with noise ±0.1
        base_lin_vel_noise = torch.rand_like(base_lin_vel) * 0.2 - 0.1
        obs_base_lin_vel = base_lin_vel + base_lin_vel_noise

        # 2. base_ang_vel (3 dims) - with noise ±0.2
        base_ang_vel_noise = torch.rand_like(base_ang_vel) * 0.4 - 0.2
        obs_base_ang_vel = base_ang_vel + base_ang_vel_noise

        # 3. projected_gravity (3 dims) - gravity vector in base frame
        # Gravity in world frame: [0, 0, -1] (normalized)
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0).expand(self.scene.num_envs, -1)
        # Rotate gravity to base frame using quaternion
        projected_gravity = math_utils.quat_apply_inverse(base_quat, gravity_world)
        projected_gravity_noise = torch.rand_like(projected_gravity) * 0.1 - 0.05
        obs_projected_gravity = projected_gravity + projected_gravity_noise

        # 4. velocity_commands (3 dims) - [target_vx, target_vy, target_ωz]
        # Convert command from [vx, vy, yaw] to [vx, vy, ωz]
        velocity_commands = torch.zeros((self.scene.num_envs, 3), device=self.device)
        velocity_commands[:, 0] = self._command[:, 0]  # vx
        velocity_commands[:, 1] = self._command[:, 1]  # vy
        velocity_commands[:, 2] = self._command[:, 2]  # yaw (used as ωz)

        # 5. hip_pos (2 dims) - HR joints relative to target, noise ±0.03
        hip_pos = self.joint_pos[:, self._hip_joint_indices]
        hip_pos_rel = hip_pos - self._hip_target.unsqueeze(0)
        hip_pos_noise = torch.rand_like(hip_pos_rel) * 0.06 - 0.03
        obs_hip_pos = hip_pos_rel + hip_pos_noise

        # 6. kfe_pos (6 dims) - HAA+HFE+KFE joints relative to target, noise ±0.05
        kfe_pos = self.joint_pos[:, self._kfe_joint_indices]
        kfe_pos_rel = kfe_pos - self._kfe_target.unsqueeze(0)
        kfe_pos_noise = torch.rand_like(kfe_pos_rel) * 0.1 - 0.05
        obs_kfe_pos = kfe_pos_rel + kfe_pos_noise

        # 7. ffe_pos (2 dims) - FFE joints relative to target, noise ±0.08
        ffe_pos = self.joint_pos[:, self._ffe_joint_indices]
        ffe_pos_rel = ffe_pos - self._ffe_target.unsqueeze(0)
        ffe_pos_noise = torch.rand_like(ffe_pos_rel) * 0.16 - 0.08
        obs_ffe_pos = ffe_pos_rel + ffe_pos_noise

        # 8. joint_vel (10 dims) - All joint velocities, noise ±1.5
        joint_vel = self.joint_vel[:, self._controlled_joint_indices]
        joint_vel_noise = torch.rand_like(joint_vel) * 3.0 - 1.5
        obs_joint_vel = joint_vel + joint_vel_noise

        # 9. actions (10 dims) - Previous action (no noise)
        obs_actions = self._prev_actions
        
        # Concatenate all observations: 3+3+3+3+2+6+2+10+10 = 42 dims
        obs = torch.cat(
            (
                obs_base_lin_vel,      # 3
                obs_base_ang_vel,       # 3
                obs_projected_gravity,  # 3
                velocity_commands,      # 3
                obs_hip_pos,            # 2
                obs_kfe_pos,            # 6
                obs_ffe_pos,            # 2
                obs_joint_vel,          # 10
                obs_actions,            # 10
            ),
            dim=1,
        )

        roll, pitch, _ = self._quat_to_euler(base_quat)
        if self._tb_step % 128 == 0:
            self._tb_writer.add_scalar("obs/roll_deg", torch.rad2deg(roll).mean().item(), self._tb_step)
            self._tb_writer.add_scalar("obs/pitch_deg", torch.rad2deg(pitch).mean().item(), self._tb_step)

        self._visualize_markers()
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards following the reference gait training implementation.

        Reward breakdown:
        1. Task rewards: track_lin_vel_xy_exp, track_ang_vel_z_exp
        2. Gait rewards: feet_air_time, feet_slide
        3. Stability penalties: lin_vel_z_l2, ang_vel_xy_l2, flat_orientation_l2
        4. Action penalties: joint_torques_l2, action_rate_l2
        5. Contact penalties: undesired_contacts, joint_deviation_hip, joint_deviation_knee
        """
        root_state = self.robot.data.root_state_w
        base_quat = root_state[:, 3:7]
        base_lin_vel = root_state[:, 7:10]
        base_ang_vel = root_state[:, 10:13]
        roll, pitch, _ = self._quat_to_euler(base_quat)
        roll_deg = torch.rad2deg(roll)
        pitch_deg = torch.rad2deg(pitch)

        current_pos = self.joint_pos[:, self._controlled_joint_indices]
        action_rate = torch.norm(self.actions - self._prev_actions, dim=1)

        # ========== 1. Task Rewards (指数奖励) ==========
        # track_lin_vel_xy_exp: weight=1.0, std=sqrt(0.25)=0.5
        cmd_vel_xy = self._command[:, :2]
        vel_error_xy = base_lin_vel[:, :2] - cmd_vel_xy
        vel_error_xy_norm_sq = torch.sum(vel_error_xy ** 2, dim=1)
        rew_track_lin_vel_xy = self.cfg.rew_scale_track_lin_vel_xy * torch.exp(-vel_error_xy_norm_sq / 0.25)

        # track_ang_vel_z_exp: weight=0.5, std=sqrt(0.25)=0.5
        cmd_ang_vel_z = self._command[:, 2]
        ang_vel_error_z = base_ang_vel[:, 2] - cmd_ang_vel_z
        rew_track_ang_vel_z = self.cfg.rew_scale_track_ang_vel_z * torch.exp(-ang_vel_error_z ** 2 / 0.25)

        # ========== 2. Gait Rewards ==========
        # feet_air_time: weight=2.0
        rew_feet_air_time = self.cfg.rew_scale_feet_air_time * compute_feet_air_time(
            self, threshold_min=0.2, threshold_max=0.5
        )

        # feet_slide: weight=-0.25
        rew_feet_slide = self.cfg.rew_scale_feet_slide * compute_feet_slide(self)

        # ========== 3. Stability Penalties ==========
        # lin_vel_z_l2: weight=-2.0 (惩罚垂直速度)
        rew_lin_vel_z = self.cfg.rew_scale_lin_vel_z * (base_lin_vel[:, 2] ** 2)

        # ang_vel_xy_l2: weight=-0.05 (惩罚俯仰/滚转)
        rew_ang_vel_xy = self.cfg.rew_scale_ang_vel_xy * torch.sum(base_ang_vel[:, :2] ** 2, dim=1)

        # flat_orientation_l2: weight=-0.5 (惩罚倾斜)
        # Projected gravity should be [0, 0, -1] when upright
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0).expand(self.scene.num_envs, -1)
        projected_gravity = math_utils.quat_apply_inverse(base_quat, gravity_world)
        # Deviation from [0, 0, -1] in base frame
        gravity_error = projected_gravity - gravity_world
        rew_flat_orientation = self.cfg.rew_scale_flat_orientation * torch.sum(gravity_error ** 2, dim=1)

        # ========== 4. Action Penalties ==========
        # joint_torques_l2: weight=-1e-5
        joint_torques = self.robot.data.applied_torque[:, self._controlled_joint_indices]
        rew_joint_torques = self.cfg.rew_scale_joint_torques * torch.sum(joint_torques ** 2, dim=1)

        # action_rate_l2: weight=-0.01
        rew_action_rate = self.cfg.rew_scale_action_rate * action_rate

        # ========== 5. Contact Penalties ==========
        # undesired_contacts: weight=-1.0 (惩罚髋关节接触)
        rew_undesired_contacts = torch.zeros(self.scene.num_envs, device=self.device)
        if self._foot_contact_sensor is not None and hasattr(self._foot_contact_sensor, "data"):
            forces = self._foot_contact_sensor.data.net_forces_w
            # Find hip body indices (HFE, HAA)
            hip_body_names = ["LL_HFE", "RL_HFE", "LL_HAA", "RL_HAA"]
            hip_body_indices = []
            for name in hip_body_names:
                bodies, _ = self.robot.find_bodies([name])
                if len(bodies) > 0:
                    hip_body_indices.append(int(bodies[0]))
            if hip_body_indices and hasattr(self._foot_contact_sensor, "body_ids"):
                sensor_body_ids = self._foot_contact_sensor.body_ids
                hip_sensor_indices = [i for i, body_id in enumerate(sensor_body_ids) if body_id in hip_body_indices]
                if hip_sensor_indices:
                    hip_forces = forces[:, hip_sensor_indices, :]
                    contact_threshold = 1.0
                    hip_contacts = torch.norm(hip_forces, dim=-1) > contact_threshold
                    rew_undesired_contacts = self.cfg.rew_scale_undesired_contacts * torch.sum(hip_contacts.float(), dim=1)

        # joint_deviation_hip: weight=-0.1 (HR, HAA关节偏离，相对目标位置)
        # Use hip joint indices from observation setup
        hip_pos = current_pos[:, [self._controlled_joint_indices.index(i) for i in self._hip_joint_indices]]
        hip_target = self._hip_target.unsqueeze(0).expand_as(hip_pos)
        hip_deviation = torch.abs(hip_pos - hip_target)
        rew_joint_deviation_hip = self.cfg.rew_scale_joint_deviation_hip * torch.sum(hip_deviation, dim=1)

        # joint_deviation_knee: weight=-0.01 (KFE关节偏离，相对目标位置)
        # KFE joints are part of kfe_joint_indices, but we need only KFE (joint4)
        kfe_joint_names = ["LL_joint4", "RL_joint4"]
        kfe_only_indices = []
        for name in kfe_joint_names:
            joint_ids, _ = self.robot.find_joints([name])
            if len(joint_ids) > 0 and joint_ids[0] in self._controlled_joint_indices:
                kfe_only_indices.append(self._controlled_joint_indices.index(joint_ids[0]))
        if kfe_only_indices:
            knee_pos = current_pos[:, kfe_only_indices]
            knee_target = torch.tensor(
                [self.cfg.target_joint_pos[name] for name in kfe_joint_names],
                device=self.device
            ).unsqueeze(0).expand_as(knee_pos)
            knee_deviation = torch.abs(knee_pos - knee_target)
            rew_joint_deviation_knee = self.cfg.rew_scale_joint_deviation_knee * torch.sum(knee_deviation, dim=1)
        else:
            rew_joint_deviation_knee = torch.zeros(self.scene.num_envs, device=self.device)

        # ========== Total Reward ==========
        total_reward = (
            rew_track_lin_vel_xy
            + rew_track_ang_vel_z
            + rew_feet_air_time
            + rew_feet_slide
            + rew_lin_vel_z
            + rew_ang_vel_xy
            + rew_flat_orientation
            + rew_joint_torques
            + rew_action_rate
            + rew_undesired_contacts
            + rew_joint_deviation_hip
            + rew_joint_deviation_knee
        )

        # ========== Logging ==========
        if self._tb_step % 32 == 0:
            # Contact forces (left/right ankles) - log mean of per-env max-norm
            left_force_mean = None
            right_force_mean = None
            try:
                sensors = getattr(self.scene, "sensors", {})
                left_sensor = sensors.get("contact_forces_left") if isinstance(sensors, dict) else None
                right_sensor = sensors.get("contact_forces_right") if isinstance(sensors, dict) else None
                if left_sensor is not None and hasattr(left_sensor, "data") and left_sensor.data.net_forces_w is not None:
                    lf = left_sensor.data.net_forces_w  # [N, bodies(=1), 3] typically
                    left_force_mean = torch.norm(lf, dim=-1).amax(dim=1).mean().item()
                if right_sensor is not None and hasattr(right_sensor, "data") and right_sensor.data.net_forces_w is not None:
                    rf = right_sensor.data.net_forces_w
                    right_force_mean = torch.norm(rf, dim=-1).amax(dim=1).mean().item()
            except Exception:
                left_force_mean = None
                right_force_mean = None

            self._tb_writer.add_scalar("pose/roll_deg", roll_deg.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("pose/pitch_deg", pitch_deg.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/total", total_reward.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/track_lin_vel_xy", rew_track_lin_vel_xy.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/track_ang_vel_z", rew_track_ang_vel_z.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/feet_air_time", rew_feet_air_time.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/feet_slide", rew_feet_slide.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/lin_vel_z", rew_lin_vel_z.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/ang_vel_xy", rew_ang_vel_xy.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/flat_orientation", rew_flat_orientation.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/joint_torques", rew_joint_torques.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/action_rate", rew_action_rate.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/undesired_contacts", rew_undesired_contacts.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/joint_deviation_hip", rew_joint_deviation_hip.mean().item(), self._tb_step)
            self._tb_writer.add_scalar("reward/joint_deviation_knee", rew_joint_deviation_knee.mean().item(), self._tb_step)
            if left_force_mean is not None:
                self._tb_writer.add_scalar("contact/left_force_max_mean", left_force_mean, self._tb_step)
            if right_force_mean is not None:
                self._tb_writer.add_scalar("contact/right_force_max_mean", right_force_mean, self._tb_step)

        # ========== Curriculum Update ==========
        cmd_error = base_lin_vel[:, :2] - self._command[:, :2]
        cmd_error_mean = float(torch.norm(cmd_error, dim=1).mean().item())
        gait_stats = compute_gait_rewards(self, root_state)
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
