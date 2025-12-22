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
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
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
        # Reference joint positions for observation (standing pose)
        self._ref_joint_act = torch.tensor(
            [self.cfg.ref_joint_act[name] for name in self._controlled_joint_names],
            device=device,
        )
        self._action_mid = (self._joint_upper + self._joint_lower) * 0.5
        self._action_scale = (self._joint_upper - self._joint_lower) * 0.5
        
        # Action increment ranges (for 12-dim action: 2 freq + 10 joint increments)
        self._act_inc_high = torch.tensor(self.cfg.act_inc_high, device=device)
        self._act_inc_low = torch.tensor(self.cfg.act_inc_low, device=device)

        self._upright_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
        self._min_height = self.cfg.failure_min_height
        self._success_joint_tol = self.cfg.success_joint_tol
        self._orientation_noise = math.radians(self.cfg.orientation_noise_deg)
        self._failure_pitch_angle = float(self.cfg.failure_pitch_angle)

        self._action_filter_gain = float(self.cfg.action_filter_gain)
        # For 12-dim action: 2 freq + 10 joint increments
        self._prev_actions = torch.zeros((self.scene.num_envs, 12), device=device)
        self._filtered_actions = torch.zeros((self.scene.num_envs, 12), device=device)
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
        
        # Phase and frequency tracking for ONNX model (2 legs)
        self._pm_phase = torch.zeros((self.scene.num_envs, 2), device=device)  # [left_phase, right_phase]
        self._pm_f = torch.ones((self.scene.num_envs, 2), device=device) * 1.0  # [left_freq, right_freq] in Hz
        # Current joint target positions (for position error calculation)
        self._joint_act = self._ref_joint_act.unsqueeze(0).expand(self.scene.num_envs, -1).clone()

        # Observation history buffer for 3-frame stacking (43 dims × 3 frames = 129 dims)
        self._num_stacks = getattr(self.cfg, "num_stacks", 3)
        self._obs_history = torch.zeros(
            (self.scene.num_envs, self._num_stacks, 43), device=device
        )  # [N, 3, 43]
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

        # joint velocity for acceleration calculation
        self._prev_joint_vel = None

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # Terrain is configured in __post_init__ of config class

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        # Register sky light with HDR texture
        if hasattr(self.cfg, "sky_light") and self.cfg.sky_light is not None:
            # Spawn the sky light using the spawn configuration
            self.cfg.sky_light.spawn.func(self.cfg.sky_light.prim_path, self.cfg.sky_light.spawn)
        # Keep existing simple light as fallback (only if sky_light is not configured)
        else:
            light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.6, 0.6, 0.6))
            light_cfg.func("/World/Light", light_cfg)
        # Register sensors
        # Height scanner for terrain height detection
        if hasattr(self.cfg, "height_scanner"):
            self.scene.sensors["height_scanner"] = RayCaster(self.cfg.height_scanner)
        # Contact sensors for both ankles
        self.scene.sensors["contact_forces_left"] = ContactSensor(self.cfg.contact_forces_left)
        self.scene.sensors["contact_forces_right"] = ContactSensor(self.cfg.contact_forces_right)

    def _setup_visual_markers(self) -> None:
        arrow_usd_path = "/home/bird/isaacSim/Learn/arrow_x.usd"
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/qmini_arrows",
            markers={
                "root_velocity": sim_utils.UsdFileCfg(
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

    def _get_ground_height(self) -> torch.Tensor:
        """Get ground height relative to base_link using height_scanner.
        
        Returns ground height in world frame for each environment.
        If height_scanner is not available, returns zeros (assumes ground at z=0).
        """
        sensors = getattr(self.scene, "sensors", {})
        height_scanner = sensors.get("height_scanner", None)
        
        if height_scanner is None or not hasattr(height_scanner, "data"):
            # Fallback: assume ground at z=0
            return torch.zeros(self.scene.num_envs, device=self.device)
        
        try:
            # Get ray hit positions in world frame
            ray_hits_w = height_scanner.data.ray_hits_w  # [N, B, 3] where N=num_envs, B=num_rays
            if ray_hits_w is None:
                return torch.zeros(self.scene.num_envs, device=self.device)
            
            # Get minimum Z coordinate from all ray hits (closest ground point)
            # ray_hits_w[:, :, 2] is Z coordinate for all rays
            # Take minimum across rays to get ground height
            ground_height = torch.min(ray_hits_w[:, :, 2], dim=1)[0]  # [N]
            
            return ground_height
        except Exception:
            # Fallback: assume ground at z=0
            return torch.zeros(self.scene.num_envs, device=self.device)
    
    def _get_base_height_relative_to_ground(self) -> torch.Tensor:
        """Get base height relative to ground (not absolute world height).
        
        Returns base height above ground for each environment.
        """
        root_state = self.robot.data.root_state_w
        base_height_abs = root_state[:, 2]  # Absolute height in world frame
        ground_height = self._get_ground_height()  # Ground height in world frame
        base_height_rel = base_height_abs - ground_height  # Relative height above ground
        return base_height_rel

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

        # Use fixed command ranges (no curriculum)
        # Only 2D control: forward/backward (vx) and turning (yaw)
        x_min, x_max = self.cfg.command_lin_vel_x_range
        yaw_min, yaw_max = self.cfg.command_yaw_range

        rand_vals = torch.rand((env_ids_t.numel(), 2), device=self.device)
        self._command[env_ids_t, 0] = x_min + (x_max - x_min) * rand_vals[:, 0]  # vx: forward/backward
        self._command[env_ids_t, 1] = 0.0  # vy: always 0 (no lateral movement)
        self._command[env_ids_t, 2] = yaw_min + (yaw_max - yaw_min) * rand_vals[:, 1]  # yaw: turning

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
        x_range = self.cfg.command_lin_vel_x_range
        y_range = self.cfg.command_lin_vel_y_range
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
        root_state = self.robot.data.root_state_w
        root_lin_vel = root_state[:, 7:10]  # Linear velocity in world frame
        
        pos = root_pos + self.marker_offset

        # Command velocity direction (red arrow)
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

        # Root velocity direction (blue arrow)
        root_vel_xy = root_lin_vel[:, :2]  # X/Y velocity components
        root_vel_speed = torch.norm(root_vel_xy, dim=1)
        root_vel_yaw = torch.atan2(root_vel_xy[:, 1], root_vel_xy[:, 0])
        
        zero_vel_mask = root_vel_speed < 1e-5
        root_vel_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(root_vel_yaw),
            torch.zeros_like(root_vel_yaw),
            root_vel_yaw,
        )
        if zero_vel_mask.any():
            root_vel_quat[zero_vel_mask] = root_quat[zero_vel_mask]

        positions = torch.cat([pos, pos], dim=0)
        rotations = torch.cat([root_vel_quat, cmd_quat], dim=0)
        marker_ids = torch.cat(
            [
                torch.zeros(self.scene.num_envs, dtype=torch.long, device=self.device),  # root_velocity (blue)
                torch.ones(self.scene.num_envs, dtype=torch.long, device=self.device),   # command (red)
            ],
            dim=0,
        )

        self.visualization_markers.visualize(positions, rotations, marker_indices=marker_ids)

    def _transform_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Transform action from [-1, 1] to actual ranges.
        
        Args:
            raw_action: [N, 12] tensor with values in [-1, 1]
            
        Returns:
            transformed_action: [N, 12] tensor with:
                [0:2]: phase frequencies in [0.5, 3.5] Hz
                [2:12]: joint increments in [-15.0, 15.0] rad/s
        """
        # Normalize from [-1, 1] to [0, 1]
        net = (raw_action + 1.0) / 2.0
        
        # Transform to actual ranges
        transformed = torch.zeros_like(raw_action)
        
        # [0:2] Phase frequencies: [0.5, 3.5] Hz
        freq_high = self._act_inc_high[0]
        freq_low = self._act_inc_low[0]
        transformed[:, 0:2] = net[:, 0:2] * (freq_high - freq_low) + freq_low
        
        # [2:12] Joint increments: [-15.0, 15.0] rad/s
        joint_high = self._act_inc_high[1]
        joint_low = self._act_inc_low[1]
        transformed[:, 2:12] = net[:, 2:12] * (joint_high - joint_low) + joint_low
        
        return transformed
    
    def _update_phase_and_frequency(self, action_increment: torch.Tensor) -> None:
        """Update phase and frequency based on action increments.
        
        Args:
            action_increment: [N, 12] tensor with transformed actions
        """
        # Update phase frequencies (first 2 dims)
        self._pm_f = action_increment[:, 0:2]  # [N, 2]
        
        # Update phases based on frequencies
        # Phase increment = frequency * 2π * dt
        phase_increment = self._pm_f * 2.0 * math.pi * self.step_dt
        self._pm_phase = (self._pm_phase + phase_increment) % (2.0 * math.pi)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Actions should be 12-dim: [2 freq + 10 joint increments]
        raw_actions = torch.clamp(actions, -1.0, 1.0)
        gain = self._action_filter_gain
        if 0.0 < gain < 1.0:
            self._filtered_actions = self._filtered_actions + gain * (raw_actions - self._filtered_actions)
            filtered_raw = self._filtered_actions
        else:
            filtered_raw = raw_actions
        
        # Transform actions from [-1, 1] to actual ranges
        action_increment = self._transform_action(filtered_raw)
        self.actions = action_increment  # Store transformed actions for observation
        
        # Update phase and frequency
        self._update_phase_and_frequency(action_increment)
        
        # Update joint target positions based on increments
        # joint_act += increment * dt
        joint_increments = action_increment[:, 2:12]  # [N, 10]
        self._joint_act = self._joint_act + joint_increments * self.step_dt
        
        # Clamp joint_act to joint limits
        self._joint_act = torch.clamp(
            self._joint_act,
            self._joint_lower.unsqueeze(0),
            self._joint_upper.unsqueeze(0)
        )

        if self._command_change_interval > 0.0:
            self._command_timer += self.step_dt
            env_ids = torch.nonzero(self._command_timer >= self._command_change_interval, as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0:
                self._sample_commands(env_ids)

        # Keep legacy gait_phase for compatibility (if needed)
        self._gait_phase = (self._gait_phase + self._gait_phase_rate * self.step_dt) % (2.0 * math.pi)

    def _apply_action(self) -> None:
        # Use joint_act as targets (already updated in _pre_physics_step)
        targets = self._joint_act.clone()
        
        # Apply smoothing if configured
        smoothing = max(0.0, min(1.0, float(self.cfg.action_smoothing_rate)))
        if smoothing > 0.0:
            targets = self._prev_targets + smoothing * (targets - self._prev_targets)

        # Respect per-joint velocity limits if provided; else fall back to scalar
        vel_lim = getattr(self.cfg, "joint_velocity_limits", None)
        if vel_lim is not None:
            # build [num_dofs] tensor
            if isinstance(vel_lim, (list, tuple)):
                vel_lim_t = torch.tensor(vel_lim, device=self.device, dtype=targets.dtype)
            else:
                vel_lim_t = torch.full((self._num_dofs,), float(vel_lim), device=self.device, dtype=targets.dtype)
            max_delta = vel_lim_t * self.step_dt
            delta = torch.clamp(targets - self._prev_targets, min=-max_delta, max=max_delta)
            targets = self._prev_targets + delta
        else:
            max_delta = self.cfg.max_joint_velocity * self.step_dt
            if max_delta > 0.0:
                delta = torch.clamp(targets - self._prev_targets, min=-max_delta, max=max_delta)
                targets = self._prev_targets + delta

        targets = torch.minimum(torch.maximum(targets, self._joint_lower), self._joint_upper)

        self.robot.set_joint_position_target(targets, joint_ids=self._controlled_joint_indices)
        self._prev_targets = targets

    def _get_single_frame_observation(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get single 43-dimensional observation frame following ONNX model specification.

        Observation breakdown (43 dims):
        1. [0:2] target_command (2): vx_cmd, yr_cmd
        2. [2:4] base_rpy (2): roll, pitch
        3. [4:7] base_rpy_rate (3): roll_rate*0.5, pitch_rate*0.5, yaw_rate*0.5
        4. [7:17] joint_pos_deviation (10): joint_pos[i] - ref_joint_act[i]
        5. [17:27] joint_vel (10): joint_vel[i] * 0.1
        6. [27:37] joint_pos_error (10): joint_act[i] - joint_pos[i]
        7. [37:41] phase_info (4): sin/cos left phase, sin/cos right phase (with static_flag)
        8. [41:43] frequency_info (2): (pm_f[0]*0.3-1)*static_flag, (pm_f[1]*0.3-1)*static_flag

        Args:
            env_ids: Optional tensor of environment indices. If None, uses all environments.

        Returns:
            obs: [N, 43] tensor with single frame observation
        """
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.scene.num_envs
            use_slice = True
        else:
            num_envs = len(env_ids) if isinstance(env_ids, torch.Tensor) else len(env_ids)
            use_slice = False

        root_state = self.robot.data.root_state_w
        if use_slice:
            base_quat = root_state[:, 3:7]  # [w, x, y, z]
            base_ang_vel = root_state[:, 10:13]  # [ωx, ωy, ωz] = [roll_rate, pitch_rate, yaw_rate]
        else:
            base_quat = root_state[env_ids, 3:7]  # [w, x, y, z]
            base_ang_vel = root_state[env_ids, 10:13]  # [ωx, ωy, ωz] = [roll_rate, pitch_rate, yaw_rate]

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = self._quat_to_euler(base_quat)

        # 1. [0:2] target_command (2 dims): vx_cmd, yr_cmd
        # vx_cmd: 期望前进速度 (m/s), yr_cmd: 期望转向角速度 (rad/s)
        target_command = torch.zeros((num_envs, 2), device=self.device)
        if use_slice:
            cmd_slice = self._command
        else:
            cmd_slice = self._command[env_ids]
        target_command[:, 0] = cmd_slice[:, 0]  # vx_cmd: forward/backward velocity
        target_command[:, 1] = cmd_slice[:, 2]  # yr_cmd: yaw rate (turning)

        # 2. [2:4] base_rpy (2 dims): roll, pitch
        # roll: 基座横滚角 (rad), pitch: 基座俯仰角 (rad)
        base_rpy = torch.stack([roll, pitch], dim=1)

        # 3. [4:7] base_rpy_rate (3 dims): roll_rate*0.5, pitch_rate*0.5, yaw_rate*0.5
        # base_ang_vel = [roll_rate, pitch_rate, yaw_rate] in rad/s
        # Apply scaling factor 0.5 as per specification
        base_rpy_rate = base_ang_vel * 0.5

        # 4. [7:17] joint_pos_deviation (10 dims): joint_pos[i] - ref_joint_act[i]
        # 关节位置相对参考位置的偏差 (rad)
        if use_slice:
            joint_pos_all = self.joint_pos  # [N, num_joints]
        else:
            joint_pos_all = self.joint_pos[env_ids]  # [len(env_ids), num_joints]
        joint_pos_slice = joint_pos_all[:, self._controlled_joint_indices]  # [N, 10]
        joint_pos_deviation = joint_pos_slice - self._ref_joint_act.unsqueeze(0)

        # 5. [17:27] joint_vel (10 dims): joint_vel[i] * 0.1
        # 关节角速度（缩放）(rad/s), 缩放因子0.1
        if use_slice:
            joint_vel_all = self.joint_vel  # [N, num_joints]
        else:
            joint_vel_all = self.joint_vel[env_ids]  # [len(env_ids), num_joints]
        joint_vel_slice = joint_vel_all[:, self._controlled_joint_indices]  # [N, 10]
        joint_vel = joint_vel_slice * 0.1

        # 6. [27:37] joint_pos_error (10 dims): joint_act[i] - joint_pos[i]
        # 目标位置与实际位置的误差 (rad)
        if use_slice:
            joint_act_slice = self._joint_act
        else:
            joint_act_slice = self._joint_act[env_ids]
        joint_pos_error = joint_act_slice - joint_pos_slice

        # 7. [37:41] phase_info (4 dims): sin/cos left phase, sin/cos right phase
        # Compute static_flag: 1 if moving (cmd_speed >= 0.15), 0 if stationary
        cmd_speed = torch.norm(target_command, dim=1)
        static_flag = (cmd_speed >= 0.15).float().unsqueeze(1)  # [N, 1]

        if use_slice:
            phase_left_slice = self._pm_phase[:, 0:1]  # Left leg phase [N, 1]
            phase_right_slice = self._pm_phase[:, 1:2]  # Right leg phase [N, 1]
        else:
            phase_left_slice = self._pm_phase[env_ids, 0:1]
            phase_right_slice = self._pm_phase[env_ids, 1:2]

        # Phase encoding: sin/cos for left and right legs (with static_flag)
        phase_info = torch.cat([
            torch.sin(phase_left_slice) * static_flag,   # [37] sin(_pm_phase[0]) * static_flag
            torch.cos(phase_left_slice) * static_flag,   # [38] cos(_pm_phase[0]) * static_flag
            torch.sin(phase_right_slice) * static_flag,   # [39] sin(_pm_phase[1]) * static_flag
            torch.cos(phase_right_slice) * static_flag,   # [40] cos(_pm_phase[1]) * static_flag
        ], dim=1)  # [N, 4]

        # 8. [41:43] frequency_info (2 dims): (pm_f[0]*0.3-1)*static_flag, (pm_f[1]*0.3-1)*static_flag
        # 左腿步态频率, 右腿步态频率 (归一化)
        if use_slice:
            pm_f_slice = self._pm_f  # [N, 2] = [left_freq, right_freq] in Hz
        else:
            pm_f_slice = self._pm_f[env_ids]
        # Formula: (pm_f * 0.3 - 1) * static_flag
        freq_info = (pm_f_slice * 0.3 - 1.0) * static_flag  # [N, 2]

        # Concatenate all observations: 2+2+3+10+10+10+4+2 = 43 dims
        obs = torch.cat(
            (
                target_command,       # 2: [0:2]   target_command
                base_rpy,             # 2: [2:4]   base_rpy
                base_rpy_rate,        # 3: [4:7]   base_rpy_rate
                joint_pos_deviation,  # 10: [7:17]  joint_pos_deviation
                joint_vel,            # 10: [17:27] joint_vel
                joint_pos_error,      # 10: [27:37] joint_pos_error
                phase_info,           # 4: [37:41] phase_info
                freq_info,            # 2: [41:43] frequency_info
            ),
            dim=1,
        )

        # Clip observations to [-3, 3] range (as per ONNX model specification)
        obs = torch.clamp(obs, -3.0, 3.0)

        return obs

    def _get_observations(self) -> dict:
        """Get 129-dimensional observations (43 dims × 3 frames) following ONNX model specification.

        Observation breakdown:
        - Single frame (43 dims): target_command(2) + base_rpy(2) + base_rpy_rate(3) +
          joint_pos_deviation(10) + joint_vel(10) + joint_pos_error(10) + phase_info(4) + frequency_info(2)
        - Stacked frames (129 dims): [obs_t-2, obs_t-1, obs_t] - 3 frames of 43 dims each
        """
        # Get current single frame observation
        obs = self._get_single_frame_observation()
        
        # Update observation history (sliding window)
        # Shift history: remove oldest frame, add current frame
        self._obs_history[:, :-1] = self._obs_history[:, 1:].clone()  # Shift left
        self._obs_history[:, -1] = obs  # Add current observation
        
        # Stack 3 frames: [obs_t-2, obs_t-1, obs_t] -> [N, 129]
        obs_stacked = self._obs_history.view(self.scene.num_envs, -1)  # [N, 3*43] = [N, 129]

        self._visualize_markers()
        return {"policy": obs_stacked}

    def _get_rewards(self) -> torch.Tensor:
        """Delegates reward computation to reward.py for better modularity."""
        from .reward import compute_total_reward
        return compute_total_reward(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        root_state = self.robot.data.root_state_w
        base_quat = root_state[:, 3:7]
        # Use relative height (above ground) instead of absolute height
        base_height_rel = self._get_base_height_relative_to_ground()
        too_low = base_height_rel < self._min_height

        roll, pitch, _ = self._quat_to_euler(base_quat)
        out_of_limits = too_low  # tilt-based reset disabled per user request
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
        # Add random position offset to disperse robots during reset
        # Use 80% of env_spacing to ensure robots stay within their environment region
        position_noise_scale = self.cfg.scene.env_spacing * 0.4
        position_noise = (torch.rand(len(env_ids), 3, device=default_root_state.device) - 0.5) * 2.0 * position_noise_scale
        position_noise[:, 2] = 0.0  # Don't add noise to Z (height)
        default_root_state[:, :3] += self.scene.env_origins[env_ids] + position_noise

        # Ensure base height is high enough to prevent ground penetration
        # With joint positions (hip_pitch=0.3, knee=-0.8, ankle=0.5),
        # we need sufficient clearance. Increase base height to prevent penetration.
        base_height_offset = 0.15  # Additional safety margin to prevent ground penetration
        # Get initial height from robot config or use default (0.35m from QMINI_ROBOT_CFG)
        initial_height = getattr(self.robot.cfg.init_state, "pos", (0.0, 0.0, 0.35))[2]
        min_base_height = initial_height + base_height_offset  # 0.35 + 0.15 = 0.5m
        default_root_state[:, 2] = torch.maximum(
            default_root_state[:, 2],
            torch.full((len(env_ids),), min_base_height, device=default_root_state.device)
        )

        # Ensure all velocities are zero to prevent bouncing
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
        # Initialize joint velocity for acceleration calculation
        if self._prev_joint_vel is None:
            self._prev_joint_vel = torch.zeros((self.scene.num_envs, self._num_dofs), device=self.device)
        self._prev_joint_vel[env_ids] = joint_vel[:, self._controlled_joint_indices]
        
        # Initialize phase and frequency for ONNX model
        self._pm_phase[env_ids] = torch.rand(len(env_ids), 2, device=self.device) * 2.0 * math.pi  # Random initial phase
        self._pm_f[env_ids] = torch.ones(len(env_ids), 2, device=self.device) * 1.0  # Initial frequency: 1.0 Hz
        # Initialize joint_act to reference positions
        self._joint_act[env_ids] = self._ref_joint_act.unsqueeze(0).expand(len(env_ids), -1)
        
        # Initialize observation history buffer (fill with current observation)
        # Get initial observation to fill history
        if isinstance(env_ids, torch.Tensor):
            initial_obs = self._get_single_frame_observation(env_ids)
            for i in range(self._num_stacks):
                self._obs_history[env_ids, i] = initial_obs
        else:
            # For slice or all environments
            initial_obs = self._get_single_frame_observation(env_ids)
            if isinstance(env_ids, slice):
                self._obs_history[env_ids, :] = initial_obs.unsqueeze(1).expand(-1, self._num_stacks, -1)
            else:
                for i in range(self._num_stacks):
                    self._obs_history[env_ids, i] = initial_obs

    @staticmethod
    def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate vector(s) by quaternion(s)."""
        q_xyz = quat[:, :3]
        q_w = quat[:, 3].unsqueeze(1)
        t = 2.0 * torch.cross(q_xyz, vec, dim=1)
        return vec + q_w * t + torch.cross(q_xyz, t, dim=1)
