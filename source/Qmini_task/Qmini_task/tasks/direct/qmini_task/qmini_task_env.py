# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

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

        self._prev_actions = torch.zeros((self.scene.num_envs, self._num_dofs), device=device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        action_scale = (self._joint_upper - self._joint_lower) * 0.5
        action_mid = (self._joint_upper + self._joint_lower) * 0.5
        targets = action_mid + action_scale * self.actions
        self.robot.set_joint_position_target(targets, joint_ids=self._controlled_joint_indices)

    def _get_observations(self) -> dict:
        current_pos = self.joint_pos[:, self._controlled_joint_indices]
        current_vel = self.joint_vel[:, self._controlled_joint_indices]
        base_quat = self.robot.data.root_state_w[:, 3:7]
        obs = torch.cat((current_pos, current_vel, base_quat), dim=1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        current_pos = self.joint_pos[:, self._controlled_joint_indices]
        current_vel = self.joint_vel[:, self._controlled_joint_indices]
        target_pos = self._target_pos.unsqueeze(0).expand_as(current_pos)

        pos_error = current_pos - target_pos
        vel_norm = torch.norm(current_vel, dim=1)
        action_rate = torch.norm(self.actions - self._prev_actions, dim=1)

        rew_alive = self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float())
        rew_term = self.cfg.rew_scale_terminated * self.reset_terminated.float()
        rew_pos = -self.cfg.rew_scale_position * torch.norm(pos_error, dim=1)
        rew_vel = -self.cfg.rew_scale_velocity * vel_norm
        rew_action = -self.cfg.rew_scale_action_rate * action_rate

        total_reward = rew_alive + rew_term + rew_pos + rew_vel + rew_action

        self._prev_actions = self.actions.clone()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        current_pos = self.joint_pos[:, self._controlled_joint_indices]

        lower_bound = self._joint_lower - 0.1
        upper_bound = self._joint_upper + 0.1
        out_of_limits = torch.any((current_pos < lower_bound) | (current_pos > upper_bound), dim=1)

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
