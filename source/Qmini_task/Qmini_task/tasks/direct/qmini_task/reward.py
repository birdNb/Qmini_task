from __future__ import annotations

from typing import Dict

import torch
import isaaclab.utils.math as math_utils


def compute_gait_rewards(env, root_state: torch.Tensor) -> Dict[str, torch.Tensor | float]:
    """Compute auxiliary gait rewards and statistics for curriculum decisions."""
    cfg = env.cfg

    root_height = root_state[:, 2]

    contact_sensor = getattr(env, "_foot_contact_sensor", None)
    single_support: torch.Tensor
    if contact_sensor is not None and hasattr(contact_sensor, "data") and contact_sensor.data.net_forces_w is not None:
        forces = contact_sensor.data.net_forces_w  # shape [N, bodies, 3]
        contact_threshold = getattr(env, "_foot_contact_threshold", 1.0)
        contact_mask = torch.norm(forces, dim=-1) > contact_threshold
        contacts_per_env = contact_mask.sum(dim=1)
        single_support = (contacts_per_env == 1).float()
        single_rate = float(single_support.mean().item())
    else:
        left_height = env.robot.data.body_pos_w[:, env._left_foot_body_idx, 2]
        right_height = env.robot.data.body_pos_w[:, env._right_foot_body_idx, 2]
        margin = getattr(env, "_single_leg_margin", 0.03)
        single_left = (left_height - right_height) > margin
        single_right = (right_height - left_height) > margin
        single_support = torch.clamp(single_left.float() + single_right.float(), max=1.0)
        single_rate = float(single_support.mean().item())

    height_target = getattr(cfg, "desired_root_height", getattr(cfg, "curriculum_height_threshold", 0.35))
    height_ratio = torch.clamp(root_height / max(height_target, 1e-4), min=0.0)
    rew_height = getattr(cfg, "rew_scale_height", 0.0) * height_ratio
    rew_single = getattr(cfg, "rew_scale_single_leg", 0.0) * single_support

    return {
        "reward_height": rew_height,
        "reward_single": rew_single,
        "single_support": single_support,
        "single_rate": single_rate,
        "height_mean": float(root_height.mean().item()),
    }


def compute_feet_air_time(env, threshold_min: float = 0.2, threshold_max: float = 0.5) -> torch.Tensor:
    """Compute reward for feet air time (using left/right ankle contact sensors when available)."""
    device = env.device
    num_envs = env.scene.num_envs
    sensors_dict = getattr(env.scene, "sensors", {})
    left = sensors_dict.get("contact_forces_left", None)
    right = sensors_dict.get("contact_forces_right", None)
    combined = sensors_dict.get("contact_forces", None)

    rewards = torch.zeros(num_envs, device=device)
    sensor_list = []
    if left is not None:
        sensor_list.append(left)
    if right is not None:
        sensor_list.append(right)
    if not sensor_list and combined is not None:
        sensor_list.append(combined)
    if not sensor_list:
        return rewards

    for s in sensor_list:
        if not hasattr(s, "data"):
            continue
        # first_contact mask
        try:
            first_contact = s.compute_first_contact(env.step_dt)
        except Exception:
            # fallback: use force threshold at current step
            forces = getattr(s.data, "net_forces_w", None)
            if forces is None:
                continue
            first_contact = (torch.norm(forces, dim=-1) > max(getattr(env, "_foot_contact_threshold", 5.0), 1e-3)).float()
        # last_air_time
        lat = getattr(s.data, "last_air_time", None)
        if lat is None:
            continue
        # Sensors may report shape [N, 1] or [N, B]; reduce along bodies
        while lat.dim() < 2:
            lat = lat.unsqueeze(1)
        while first_contact.dim() < 2:
            first_contact = first_contact.unsqueeze(1)
        air_time = (lat - threshold_min) * first_contact
        air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
        rewards = rewards + air_time.sum(dim=1)

    # gate by commanded motion (optional)
    command = getattr(env, "_command", None)
    if command is not None:
        cmd_vel_xy = torch.norm(command[:, :2], dim=1)
        rewards = rewards * (cmd_vel_xy > 0.1).float()
    return rewards


def compute_feet_slide(env) -> torch.Tensor:
    """Compute penalty for feet sliding during contact using left/right sensors when available."""
    device = env.device
    num_envs = env.scene.num_envs
    sensors = getattr(env.scene, "sensors", {})
    left = sensors.get("contact_forces_left", None)
    right = sensors.get("contact_forces_right", None)
    combined = sensors.get("contact_forces", None)

    total_penalty = torch.zeros(num_envs, device=device)
    used = False
    for s in [left, right] if (left or right) else [combined]:
        if s is None or not hasattr(s, "data"):
            continue
        used = True
        # contact mask
        try:
            hist = s.data.net_forces_w_history  # [N, H, B, 3]
            contacts = hist.norm(dim=-1).max(dim=1)[0] > 1.0
        except Exception:
            forces = s.data.net_forces_w
            if forces is None:
                continue
            contacts = torch.norm(forces, dim=-1) > 1.0
        # velocities
        robot = env.robot
        try:
            sensor_body_ids = s.body_ids
            vel = robot.data.body_lin_vel_w[:, sensor_body_ids, :2]
        except Exception:
            # fallback: take the first B bodies
            b = contacts.shape[1]
            vel = robot.data.body_lin_vel_w[:, :b, :2]
        total_penalty = total_penalty + torch.sum(vel.norm(dim=-1) * contacts.float(), dim=1)
    if not used:
        return torch.zeros(num_envs, device=device)
    return total_penalty


def compute_leg_lift(env, clearance: float = 0.05) -> torch.Tensor:
    """Reward lifting swing legs above a clearance height when not in contact.

    Uses left/right ankle sensors (preferred) or combined sensor as fallback.
    """
    device = env.device
    num_envs = env.scene.num_envs
    sensors = getattr(env.scene, "sensors", {})
    left = sensors.get("contact_forces_left", None)
    right = sensors.get("contact_forces_right", None)
    combined = sensors.get("contact_forces", None)

    # Determine ankle body indices on robot
    ankle_names = ["LL_ankle", "RL_ankle"]
    ankle_indices: list[int] = []
    for name in ankle_names:
        ids, _ = env.robot.find_bodies([name])
        ankle_indices.append(int(ids[0]) if len(ids) > 0 else -1)

    body_pos_w = env.robot.data.body_pos_w  # [N, bodies, 3]
    reward = torch.zeros(num_envs, device=env.device)

    def sensor_to_contact_mask(sensor) -> torch.Tensor | None:
        if sensor is None or not hasattr(sensor, "data"):
            return None
        try:
            hist = sensor.data.net_forces_w_history
            return (hist.norm(dim=-1).max(dim=1)[0] > 1.0).float()  # [N, B]
        except Exception:
            forces = sensor.data.net_forces_w
            if forces is None:
                return None
            return (torch.norm(forces, dim=-1) > 1.0).float()

    # Build per-ankle swing mask
    left_mask = sensor_to_contact_mask(left)
    right_mask = sensor_to_contact_mask(right)
    if left_mask is None and right_mask is None and combined is not None:
        cmask = sensor_to_contact_mask(combined)
        if cmask is not None:
            # if combined has two ankles, split them; else treat as one
            if cmask.shape[1] >= 2:
                left_mask, right_mask = cmask[:, 0:1], cmask[:, 1:2]
            else:
                left_mask, right_mask = cmask, cmask

    # For each ankle, reward positive height above clearance when not in contact
    for idx, m in zip(ankle_indices, [left_mask, right_mask]):
        if idx < 0:
            continue
        z = body_pos_w[:, idx, 2]
        if m is None:
            swing = torch.ones(num_envs, device=device)
        else:
            swing = 1.0 - torch.clamp(m.squeeze(-1), 0.0, 1.0)
        lift = torch.clamp(z - clearance, min=0.0)
        reward = reward + lift * swing
    return reward


def compute_total_reward(env) -> torch.Tensor:
    """Full reward computation moved from env._get_rewards."""
    # root/base states
    root_state = env.robot.data.root_state_w
    base_quat = root_state[:, 3:7]
    base_lin_vel = root_state[:, 7:10]
    base_ang_vel = root_state[:, 10:13]

    current_pos = env.joint_pos[:, env._controlled_joint_indices]
    action_rate = torch.norm(env.actions - env._prev_actions, dim=1)

    # 1) Task rewards
    cmd_vel_xy = env._command[:, :2]
    vel_error_xy = base_lin_vel[:, :2] - cmd_vel_xy
    vel_error_xy_norm_sq = torch.sum(vel_error_xy ** 2, dim=1)
    rew_track_lin_vel_xy = env.cfg.rew_scale_track_lin_vel_xy * torch.exp(-vel_error_xy_norm_sq / 0.25)

    cmd_ang_vel_z = env._command[:, 2]
    ang_vel_error_z = base_ang_vel[:, 2] - cmd_ang_vel_z
    rew_track_ang_vel_z = env.cfg.rew_scale_track_ang_vel_z * torch.exp(-ang_vel_error_z ** 2 / 0.25)

    # 2) Gait rewards
    rew_feet_air_time = env.cfg.rew_scale_feet_air_time * compute_feet_air_time(
        env, threshold_min=0.05, threshold_max=0.5
    )
    rew_feet_slide = env.cfg.rew_scale_feet_slide * compute_feet_slide(env)

    leg_lift_raw = compute_leg_lift(env, clearance=env.cfg.desired_foot_clearance)
    rew_leg_lift = getattr(env.cfg, "rew_scale_leg_lift", 1.0) * leg_lift_raw

    # 3) Stability penalties
    rew_lin_vel_z = env.cfg.rew_scale_lin_vel_z * (base_lin_vel[:, 2] ** 2)
    rew_ang_vel_xy = env.cfg.rew_scale_ang_vel_xy * torch.sum(base_ang_vel[:, :2] ** 2, dim=1)

    gravity_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).expand(env.scene.num_envs, -1)
    projected_gravity = math_utils.quat_apply_inverse(base_quat, gravity_world)
    gravity_error = projected_gravity - gravity_world
    rew_flat_orientation = env.cfg.rew_scale_flat_orientation * torch.sum(gravity_error ** 2, dim=1)

    # 4) Action penalties
    joint_torques = env.robot.data.applied_torque[:, env._controlled_joint_indices]
    rew_joint_torques = env.cfg.rew_scale_joint_torques * torch.sum(joint_torques ** 2, dim=1)
    rew_action_rate = env.cfg.rew_scale_action_rate * action_rate

    # 5) Contact penalties
    rew_undesired_contacts = torch.zeros(env.scene.num_envs, device=env.device)
    if getattr(env, "_foot_contact_sensor", None) is not None and hasattr(env._foot_contact_sensor, "data"):
        forces = env._foot_contact_sensor.data.net_forces_w
        hip_body_names = ["LL_HFE", "RL_HFE", "LL_HAA", "RL_HAA"]
        hip_body_indices = []
        for name in hip_body_names:
            bodies, _ = env.robot.find_bodies([name])
            if len(bodies) > 0:
                hip_body_indices.append(int(bodies[0]))
        if hip_body_indices and hasattr(env._foot_contact_sensor, "body_ids"):
            sensor_body_ids = env._foot_contact_sensor.body_ids
            hip_sensor_indices = [i for i, body_id in enumerate(sensor_body_ids) if body_id in hip_body_indices]
            if hip_sensor_indices:
                hip_forces = forces[:, hip_sensor_indices, :]
                contact_threshold = 1.0
                hip_contacts = torch.norm(hip_forces, dim=-1) > contact_threshold
                rew_undesired_contacts = env.cfg.rew_scale_undesired_contacts * torch.sum(hip_contacts.float(), dim=1)

    # joint deviation penalties
    hip_pos = current_pos[:, [env._controlled_joint_indices.index(i) for i in env._hip_joint_indices]]
    hip_target = env._hip_target.unsqueeze(0).expand_as(hip_pos)
    hip_deviation = torch.abs(hip_pos - hip_target)
    rew_joint_deviation_hip = env.cfg.rew_scale_joint_deviation_hip * torch.sum(hip_deviation, dim=1)

    kfe_joint_names = ["LL_joint4", "RL_joint4"]
    kfe_only_indices = []
    for name in kfe_joint_names:
        joint_ids, _ = env.robot.find_joints([name])
        if len(joint_ids) > 0 and joint_ids[0] in env._controlled_joint_indices:
            kfe_only_indices.append(env._controlled_joint_indices.index(joint_ids[0]))
    if kfe_only_indices:
        knee_pos = current_pos[:, kfe_only_indices]
        knee_target = torch.tensor(
            [env.cfg.target_joint_pos[name] for name in kfe_joint_names],
            device=env.device
        ).unsqueeze(0).expand_as(knee_pos)
        knee_deviation = torch.abs(knee_pos - knee_target)
        rew_joint_deviation_knee = env.cfg.rew_scale_joint_deviation_knee * torch.sum(knee_deviation, dim=1)
    else:
        rew_joint_deviation_knee = torch.zeros(env.scene.num_envs, device=env.device)

    # Forward distance progress
    root_x = env.robot.data.root_pos_w[:, 0]
    delta_x = torch.clamp(root_x - env._prev_root_x, min=0.0)
    env._cum_forward_x = env._cum_forward_x + delta_x
    env._prev_root_x = root_x
    rew_forward_distance = getattr(env.cfg, "rew_scale_forward_distance", 0.0) * env._cum_forward_x

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
        + rew_leg_lift
        + rew_forward_distance
    )

    # Logging (two categories + progress)
    if env._tb_step % 32 == 0:
        task_reward = (rew_track_lin_vel_xy + rew_track_ang_vel_z + rew_feet_air_time + rew_leg_lift + rew_forward_distance)
        penalty_total = -(rew_lin_vel_z + rew_ang_vel_xy + rew_flat_orientation + rew_action_rate + rew_joint_torques + rew_undesired_contacts + rew_feet_slide)
        env._tb_writer.add_scalar("reward/total", total_reward.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("reward/task", task_reward.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("penalty/total", penalty_total.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("progress/forward_delta_x", delta_x.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("progress/forward_cumulative_x", env._cum_forward_x.mean().item(), env._tb_step)

    # curriculum update
    cmd_error = base_lin_vel[:, :2] - env._command[:, :2]
    cmd_error_mean = float(torch.norm(cmd_error, dim=1).mean().item())
    gait_stats = compute_gait_rewards(env, root_state)
    if env.curriculum.update(env._control_dt, gait_stats["height_mean"], gait_stats["single_rate"], cmd_error_mean):
        env._sample_commands(None)

    env._prev_actions = env.actions.clone()
    env._tb_step += 1
    return total_reward


