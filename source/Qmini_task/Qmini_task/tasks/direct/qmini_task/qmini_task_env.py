# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import math
import torch

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

        self._action_filter_gain = float(self.cfg.action_filter_gain)
        self._prev_actions = torch.zeros((self.scene.num_envs, self._num_dofs), device=device)
        self._filtered_actions = torch.zeros((self.scene.num_envs, self._num_dofs), device=device)
        self._prev_targets = self._target_pos.unsqueeze(0).expand(self.scene.num_envs, -1).clone()

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
        obs = torch.cat((current_pos, current_vel, base_quat, base_lin_vel, base_ang_vel), dim=1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        current_pos = self.joint_pos[:, self._controlled_joint_indices]
        current_vel = self.joint_vel[:, self._controlled_joint_indices]
        target_pos = self._target_pos.unsqueeze(0).expand_as(current_pos)

        root_state = self.robot.data.root_state_w
        base_quat = root_state[:, 3:7]
        base_lin_vel = root_state[:, 7:10]
        base_ang_vel = root_state[:, 10:13]

        base_up = self._quat_apply(base_quat, self._upright_axis.unsqueeze(0).expand(root_state.shape[0], -1))
        upright_error = 1.0 - torch.clamp(base_up[:, 2], max=1.0)

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
        rew_upright = -self.cfg.rew_scale_upright * upright_error
        rew_base_lin = -self.cfg.rew_scale_base_lin_vel * lin_vel_norm
        rew_base_ang = -self.cfg.rew_scale_base_ang_vel * ang_vel_norm
        rew_action = -self.cfg.rew_scale_action_rate * action_rate

        success_mask = (torch.max(torch.abs(pos_error), dim=1).values < self._success_joint_tol) & (
            base_up[:, 2] > self._success_up_cos
        )
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

        self._prev_actions = self.actions.clone()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        root_state = self.robot.data.root_state_w
        base_quat = root_state[:, 3:7]
        base_pos_z = root_state[:, 2]
        base_up = self._quat_apply(base_quat, self._upright_axis.unsqueeze(0).expand(root_state.shape[0], -1))
        base_height = root_state[:, 2]
        too_low = base_height < self._min_height


        _, pitch, _ = self._quat_to_euler(base_quat)
        tilt_exceeded = torch.abs(pitch) > self.cfg.failure_tilt_angle

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
