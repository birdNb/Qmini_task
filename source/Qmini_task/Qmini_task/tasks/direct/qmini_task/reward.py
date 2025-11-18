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
    Optimized with staged rewards and improved single support detection.
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

    # Get contact force threshold and height difference threshold
    force_threshold = getattr(env.cfg, "foot_contact_force_threshold", 1.0)
    height_diff_threshold = getattr(env.cfg, "single_support_height_diff", 0.03)
    exploration_threshold = getattr(env.cfg, "leg_lift_exploration_threshold", 0.02)

    def sensor_to_contact_mask(sensor) -> torch.Tensor | None:
        if sensor is None or not hasattr(sensor, "data"):
            return None
        try:
            hist = sensor.data.net_forces_w_history
            return (hist.norm(dim=-1).max(dim=1)[0] > force_threshold).float()  # [N, B]
        except Exception:
            forces = sensor.data.net_forces_w
            if forces is None:
                return None
            return (torch.norm(forces, dim=-1) > force_threshold).float()

    # Build per-ankle contact mask
    left_mask = sensor_to_contact_mask(left)
    right_mask = sensor_to_contact_mask(right)
    if left_mask is None and right_mask is None and combined is not None:
        cmask = sensor_to_contact_mask(combined)
        if cmask is not None:
            if cmask.shape[1] >= 2:
                left_mask, right_mask = cmask[:, 0:1], cmask[:, 1:2]
            else:
                left_mask, right_mask = cmask, cmask

    # Get ankle heights
    left_height = body_pos_w[:, ankle_indices[0], 2] if ankle_indices[0] >= 0 else torch.zeros(num_envs, device=device)
    right_height = body_pos_w[:, ankle_indices[1], 2] if ankle_indices[1] >= 0 else torch.zeros(num_envs, device=device)
    height_diff = torch.abs(left_height - right_height)

    # Improved single support detection: contact + small height difference
    for i, (idx, m) in enumerate(zip(ankle_indices, [left_mask, right_mask])):
        if idx < 0:
            continue
        z = body_pos_w[:, idx, 2]

        # Determine if this leg is in contact (convert to boolean)
        if m is None:
            is_contact = torch.zeros(num_envs, device=device, dtype=torch.bool)
        else:
            is_contact = (torch.clamp(m.squeeze(-1), 0.0, 1.0) > 0.5)

        # Improved swing detection: not in contact OR (in contact but height difference is large)
        # This prevents misclassifying swing leg as support leg
        is_support = is_contact & (height_diff < height_diff_threshold)
        swing = (~is_support).float()

        # Staged reward: from exploration_threshold (0.02m) start giving reward
        # Full reward at clearance (0.04m)
        lift_above_exploration = torch.clamp(z - exploration_threshold, min=0.0)
        lift_above_clearance = torch.clamp(z - clearance, min=0.0)

        # Linear interpolation: more reward as height increases
        # At exploration_threshold: reward = 0
        # At clearance: reward = lift_above_clearance
        # Between: linear interpolation
        if clearance > exploration_threshold:
            exploration_scale = lift_above_exploration / (clearance - exploration_threshold + 1e-6)
            exploration_scale = torch.clamp(exploration_scale, 0.0, 1.0)
            staged_lift = exploration_scale * lift_above_clearance + (1.0 - exploration_scale) * lift_above_exploration * 0.5
        else:
            staged_lift = lift_above_clearance

        reward = reward + staged_lift * swing
    return reward


def compute_leg_lift_velocity(env) -> torch.Tensor:
    """Reward upward velocity of swing legs to encourage fast lifting."""
    device = env.device
    num_envs = env.scene.num_envs
    sensors = getattr(env.scene, "sensors", {})
    left = sensors.get("contact_forces_left", None)
    right = sensors.get("contact_forces_right", None)
    combined = sensors.get("contact_forces", None)

    # Determine ankle body indices
    ankle_names = ["LL_ankle", "RL_ankle"]
    ankle_indices: list[int] = []
    for name in ankle_names:
        ids, _ = env.robot.find_bodies([name])
        ankle_indices.append(int(ids[0]) if len(ids) > 0 else -1)

    body_lin_vel_w = env.robot.data.body_lin_vel_w  # [N, bodies, 3]
    reward = torch.zeros(num_envs, device=device)

    force_threshold = getattr(env.cfg, "foot_contact_force_threshold", 1.0)

    def sensor_to_contact_mask(sensor) -> torch.Tensor | None:
        if sensor is None or not hasattr(sensor, "data"):
            return None
        try:
            forces = sensor.data.net_forces_w
            if forces is None:
                return None
            return (torch.norm(forces, dim=-1) > force_threshold).float()
        except Exception:
            return None

    left_mask = sensor_to_contact_mask(left)
    right_mask = sensor_to_contact_mask(right)
    if left_mask is None and right_mask is None and combined is not None:
        cmask = sensor_to_contact_mask(combined)
        if cmask is not None:
            if cmask.shape[1] >= 2:
                left_mask, right_mask = cmask[:, 0:1], cmask[:, 1:2]
            else:
                left_mask, right_mask = cmask, cmask

    # Reward upward velocity of swing legs
    for idx, m in zip(ankle_indices, [left_mask, right_mask]):
        if idx < 0:
            continue
        # Get upward velocity (z-direction)
        vel_z = body_lin_vel_w[:, idx, 2]
        # Determine if swinging (not in contact)
        if m is None:
            swing = torch.ones(num_envs, device=device)
        else:
            swing = 1.0 - torch.clamp(m.squeeze(-1), 0.0, 1.0)
        # Only reward positive (upward) velocity
        upward_vel = torch.clamp(vel_z, min=0.0)
        reward = reward + upward_vel * swing

    return reward


def compute_both_feet_contact_penalty(env, force_threshold: float = 1.0) -> torch.Tensor:
    """Penalty for both feet being in contact simultaneously (encourages alternating gait).

    Returns a penalty value that is 1.0 when both feet are in contact, 0.0 otherwise.
    """
    device = env.device
    num_envs = env.scene.num_envs
    sensors = getattr(env.scene, "sensors", {})
    left = sensors.get("contact_forces_left", None)
    right = sensors.get("contact_forces_right", None)
    combined = sensors.get("contact_forces", None)

    # Helper to get contact mask from sensor
    def get_contact_mask(sensor) -> torch.Tensor | None:
        if sensor is None or not hasattr(sensor, "data"):
            return None
        try:
            forces = sensor.data.net_forces_w
            if forces is None:
                return None
            # Check if any body in this sensor is in contact
            contact_mask = torch.norm(forces, dim=-1) > force_threshold  # [N, B]
            # Reduce along bodies dimension: at least one body in contact
            return contact_mask.any(dim=-1).float()  # [N]
        except Exception:
            return None

    # Get contact status for left and right feet
    left_contact = get_contact_mask(left)
    right_contact = get_contact_mask(right)

    # Fallback to combined sensor if separate sensors not available
    if left_contact is None or right_contact is None:
        if combined is not None:
            cmask = get_contact_mask(combined)
            if cmask is not None:
                # If combined sensor has multiple bodies, treat as both feet
                try:
                    forces = combined.data.net_forces_w
                    if forces is not None and forces.shape[1] >= 2:
                        # Split into left and right
                        left_forces = forces[:, 0, :]
                        right_forces = forces[:, 1, :]
                        left_contact = (torch.norm(left_forces, dim=-1) > force_threshold).float()
                        right_contact = (torch.norm(right_forces, dim=-1) > force_threshold).float()
                    else:
                        # Single body sensor - can't distinguish, assume both contact
                        left_contact = cmask
                        right_contact = cmask
                except Exception:
                    left_contact = cmask
                    right_contact = cmask
        else:
            # No sensors available - return zero penalty
            return torch.zeros(num_envs, device=device)

    # Both feet in contact: penalty = 1.0, otherwise 0.0
    both_contact = (left_contact > 0.5) & (right_contact > 0.5)
    penalty = both_contact.float()

    return penalty


def get_feet_air_time_stats(env) -> Dict[str, float]:
    """Get feet air time statistics for TensorBoard logging."""
    device = env.device
    num_envs = env.scene.num_envs
    sensors_dict = getattr(env.scene, "sensors", {})
    left = sensors_dict.get("contact_forces_left", None)
    right = sensors_dict.get("contact_forces_right", None)
    combined = sensors_dict.get("contact_forces", None)

    total_air_time = torch.zeros(num_envs, device=device)
    sensor_list = []
    if left is not None:
        sensor_list.append(left)
    if right is not None:
        sensor_list.append(right)
    if not sensor_list and combined is not None:
        sensor_list.append(combined)

    for s in sensor_list:
        if not hasattr(s, "data"):
            continue
        lat = getattr(s.data, "last_air_time", None)
        if lat is None:
            continue
        while lat.dim() < 2:
            lat = lat.unsqueeze(1)
        total_air_time = total_air_time + lat.sum(dim=1)

    return {
        "mean_air_time": float(total_air_time.mean().item()),
        "max_air_time": float(total_air_time.max().item()),
    }


def compute_ankle_gravity_projection_penalty(env) -> torch.Tensor:
    """Penalty for ankle link gravity projection deviation from vertical downward.

    Ensures RL_ankle and LL_ankle links have gravity projection pointing vertically downward
    in their local frame (i.e., gravity should project to [0, 0, -1] in local frame).
    """
    device = env.device
    num_envs = env.scene.num_envs

    # Find ankle body indices
    ankle_names = ["LL_ankle", "RL_ankle"]
    ankle_indices = []
    for name in ankle_names:
        ids, _ = env.robot.find_bodies([name])
        if len(ids) > 0:
            ankle_indices.append(int(ids[0]))
        else:
            # If ankle not found, return zero penalty
            return torch.zeros(num_envs, device=device)

    # Get ankle orientations (quaternions in world frame)
    body_quat_w = env.robot.data.body_quat_w  # [N, bodies, 4] (w, x, y, z)
    ankle_quats = body_quat_w[:, ankle_indices, :]  # [N, 2, 4]

    # Gravity vector in world frame: [0, 0, -1] (normalized)
    gravity_world = torch.tensor([0.0, 0.0, -1.0], device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
    gravity_world = gravity_world.expand(num_envs, 2, -1)  # [N, 2, 3]

    # Project gravity to ankle local frames
    # quat_apply_inverse rotates world vector to local frame
    gravity_local = math_utils.quat_apply_inverse(ankle_quats, gravity_world)  # [N, 2, 3]

    # Ideal gravity projection in local frame should be [0, 0, -1]
    ideal_gravity_local = torch.tensor([0.0, 0.0, -1.0], device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
    ideal_gravity_local = ideal_gravity_local.expand(num_envs, 2, -1)  # [N, 2, 3]

    # Compute deviation (L2 norm of difference)
    gravity_deviation = torch.norm(gravity_local - ideal_gravity_local, dim=-1)  # [N, 2]

    # Sum over both ankles
    total_deviation = torch.sum(gravity_deviation, dim=1)  # [N]

    return total_deviation


def compute_root_pitch_roll_penalty(env, base_quat: torch.Tensor) -> torch.Tensor:
    """Penalty for root pitch and roll angles to ensure stable upright posture.

    Computes pitch and roll angles from quaternion and penalizes deviations from zero.
    This ensures the robot maintains balance before attempting forward motion.
    """
    # Extract quaternion components (w, x, y, z)
    w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]

    # Compute pitch angle (rotation around Y-axis)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * 1.5707963267948966,  # pi/2
        torch.asin(sinp),
    )

    # Compute roll angle (rotation around X-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Penalize squared angles (stronger penalty for larger deviations)
    pitch_penalty = pitch ** 2
    roll_penalty = roll ** 2

    # Sum both penalties
    total_penalty = pitch_penalty + roll_penalty

    return total_penalty


def compute_imbalance_penalty(env, base_quat: torch.Tensor, base_height: torch.Tensor) -> torch.Tensor:
    """High penalty when robot is close to imbalance/reset conditions.

    Penalizes when pitch/roll angles exceed thresholds or height drops too low.
    This provides strong negative feedback before actual reset occurs.
    """
    # Extract quaternion components (w, x, y, z)
    w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]

    # Compute pitch and roll angles
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * 1.5707963267948966,  # pi/2
        torch.asin(sinp),
    )
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Get thresholds from config
    pitch_threshold = getattr(env.cfg, "imbalance_pitch_threshold", 0.523599)  # ~30 deg default
    roll_threshold = getattr(env.cfg, "imbalance_roll_threshold", 0.523599)   # ~30 deg default
    height_threshold = getattr(env.cfg, "imbalance_height_threshold", 0.3)     # 0.3m default

    # Check if exceeding thresholds
    pitch_imbalance = torch.abs(pitch) > pitch_threshold
    roll_imbalance = torch.abs(roll) > roll_threshold
    height_imbalance = base_height < height_threshold

    # Penalty: 1.0 if any imbalance condition is met, 0.0 otherwise
    any_imbalance = pitch_imbalance | roll_imbalance | height_imbalance
    penalty = any_imbalance.float()

    return penalty


def compute_total_reward(env) -> torch.Tensor:
    """Full reward computation moved from env._get_rewards."""
    # root/base states
    root_state = env.robot.data.root_state_w
    base_quat = root_state[:, 3:7]
    base_height = root_state[:, 2]
    base_lin_vel = root_state[:, 7:10]
    base_ang_vel = root_state[:, 10:13]

    current_pos = env.joint_pos[:, env._controlled_joint_indices]
    current_vel = env.joint_vel[:, env._controlled_joint_indices]
    action_rate = torch.norm(env.actions - env._prev_actions, dim=1)

    # 1) Task rewards - Following reference configuration
    # X/Y direction velocity tracking reward (reference style: exponential tracking)
    cmd_lin_vel_xy = env._command[:, :2]
    base_lin_vel_xy = base_lin_vel[:, :2]
    lin_vel_error_xy = base_lin_vel_xy - cmd_lin_vel_xy
    lin_vel_error_norm = torch.norm(lin_vel_error_xy, dim=1)
    rew_track_lin_vel_xy = getattr(env.cfg, "rew_scale_track_lin_vel_xy", 3.0) * torch.exp(-lin_vel_error_norm ** 2 / 0.25)

    # Z direction velocity penalty (penalize vertical motion)
    rew_lin_vel_z = getattr(env.cfg, "rew_scale_lin_vel_z", -2.0) * base_lin_vel[:, 2] ** 2

    cmd_ang_vel_z = env._command[:, 2]
    ang_vel_error_z = base_ang_vel[:, 2] - cmd_ang_vel_z
    rew_track_ang_vel_z = getattr(env.cfg, "rew_scale_track_ang_vel_z", 3.0) * torch.exp(-ang_vel_error_z ** 2 / 0.25)

    # Alive reward (reference: weight=0.3)
    rew_alive = getattr(env.cfg, "rew_scale_alive", 0.3) * torch.ones(env.scene.num_envs, device=env.device)

    # 2) Gait rewards
    rew_feet_air_time = env.cfg.rew_scale_feet_air_time * compute_feet_air_time(
        env, threshold_min=0.05, threshold_max=0.5
    )
    rew_feet_slide = env.cfg.rew_scale_feet_slide * compute_feet_slide(env)

    leg_lift_raw = compute_leg_lift(env, clearance=getattr(env.cfg, "desired_foot_clearance", 0.05))
    rew_leg_lift = getattr(env.cfg, "rew_scale_leg_lift", 0.99) * leg_lift_raw

    # Feet contact forces penalty (reference: weight=-0.2, threshold=100)
    rew_feet_contact_forces = torch.zeros(env.scene.num_envs, device=env.device)
    sensors = getattr(env.scene, "sensors", {})
    left = sensors.get("contact_forces_left", None)
    right = sensors.get("contact_forces_right", None)
    if left is not None and hasattr(left, "data") and left.data.net_forces_w is not None:
        forces = left.data.net_forces_w
        force_norm = torch.norm(forces, dim=-1)
        threshold = getattr(env.cfg, "foot_contact_force_threshold", 100.0)
        excess_forces = torch.clamp(force_norm - threshold, min=0.0)
        rew_feet_contact_forces = rew_feet_contact_forces + torch.sum(excess_forces ** 2, dim=1)
    if right is not None and hasattr(right, "data") and right.data.net_forces_w is not None:
        forces = right.data.net_forces_w
        force_norm = torch.norm(forces, dim=-1)
        threshold = getattr(env.cfg, "foot_contact_force_threshold", 100.0)
        excess_forces = torch.clamp(force_norm - threshold, min=0.0)
        rew_feet_contact_forces = rew_feet_contact_forces + torch.sum(excess_forces ** 2, dim=1)
    rew_feet_contact_forces = getattr(env.cfg, "rew_scale_feet_contact_forces", -0.2) * rew_feet_contact_forces

    # Leg lift velocity reward (encourage fast lifting)
    rew_leg_lift_velocity = getattr(env.cfg, "rew_scale_leg_lift_velocity", 0.0) * compute_leg_lift_velocity(env)

    # Penalty for both feet in contact (encourages alternating gait)
    rew_both_feet_contact = getattr(env.cfg, "rew_scale_both_feet_contact", -5.0) * compute_both_feet_contact_penalty(
        env, force_threshold=getattr(env.cfg, "foot_contact_force_threshold", 1.0)
    )

    # Ankle gravity projection penalty (ensures ankle links are vertical)
    rew_ankle_gravity = getattr(env.cfg, "rew_scale_ankle_gravity", -2.0) * compute_ankle_gravity_projection_penalty(env)

    # 3) Stability penalties - Following reference configuration
    # Root pitch and roll penalty
    rew_root_pitch_roll = getattr(env.cfg, "rew_scale_root_pitch_roll", -1.0) * compute_root_pitch_roll_penalty(env, base_quat)

    # Imbalance penalty (high penalty when close to reset conditions)
    rew_imbalance = getattr(env.cfg, "rew_scale_imbalance_penalty", -50.0) * compute_imbalance_penalty(env, base_quat, base_height)

    rew_ang_vel_xy = getattr(env.cfg, "rew_scale_ang_vel_xy", -0.5) * torch.sum(base_ang_vel[:, :2] ** 2, dim=1)

    gravity_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).expand(env.scene.num_envs, -1)
    projected_gravity = math_utils.quat_apply_inverse(base_quat, gravity_world)
    gravity_error = projected_gravity - gravity_world
    rew_flat_orientation = getattr(env.cfg, "rew_scale_flat_orientation", -1.0) * torch.sum(gravity_error ** 2, dim=1)

    # Base height penalty (reference: weight=-10.0, target_height=0.15)
    target_height = getattr(env.cfg, "desired_root_height", 0.15)
    height_error = base_height - target_height
    rew_base_height = getattr(env.cfg, "rew_scale_base_height", -10.0) * height_error ** 2

    # 4) Action penalties - Following reference configuration
    joint_torques = env.robot.data.applied_torque[:, env._controlled_joint_indices]
    rew_joint_torques = getattr(env.cfg, "rew_scale_joint_torques", -1.0e-5) * torch.sum(joint_torques ** 2, dim=1)
    rew_action_rate = getattr(env.cfg, "rew_scale_action_rate", -0.10) * action_rate

    # Joint acceleration penalty (reference: weight=-2.5e-7)
    if hasattr(env, "_prev_joint_vel"):
        joint_acc = (current_vel - env._prev_joint_vel) / env.step_dt
        rew_joint_acc = getattr(env.cfg, "rew_scale_joint_acc", -2.5e-7) * torch.sum(joint_acc ** 2, dim=1)
    else:
        rew_joint_acc = torch.zeros(env.scene.num_envs, device=env.device)
    env._prev_joint_vel = current_vel.clone()

    # DOF position limits penalty (reference: weight=-5.0)
    joint_lower = torch.tensor(env.cfg.joint_lower_limits, device=env.device)
    joint_upper = torch.tensor(env.cfg.joint_upper_limits, device=env.device)
    violation_lower = torch.clamp(joint_lower - current_pos, min=0.0)
    violation_upper = torch.clamp(current_pos - joint_upper, min=0.0)
    rew_dof_pos_limits = getattr(env.cfg, "rew_scale_dof_pos_limits", -5.0) * torch.sum(violation_lower ** 2 + violation_upper ** 2, dim=1)

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
        rew_track_lin_vel_xy  # Reference: velocity tracking
        + rew_track_ang_vel_z
        + rew_alive  # Reference: alive reward
        + rew_lin_vel_z  # Penalize vertical motion
        + rew_ang_vel_xy
        + rew_flat_orientation
        + rew_base_height  # Reference: base height penalty
        + rew_joint_acc  # Reference: joint acceleration penalty
        + rew_action_rate
        + rew_dof_pos_limits  # Reference: DOF position limits penalty
        + rew_joint_torques
        + rew_feet_air_time
        + rew_feet_slide
        + rew_leg_lift
        + rew_feet_contact_forces  # Reference: feet contact forces penalty
        + rew_undesired_contacts
        + rew_joint_deviation_hip
        + rew_joint_deviation_knee
        + rew_forward_distance
        + rew_both_feet_contact
        + rew_ankle_gravity
        + rew_root_pitch_roll
        + rew_imbalance  # Very high penalty for imbalance (reset condition)
    )

    # Logging (two categories + progress)
    if env._tb_step % 32 == 0:
        task_reward = (rew_track_lin_vel_xy + rew_track_ang_vel_z + rew_feet_air_time + rew_leg_lift + rew_leg_lift_velocity + rew_forward_distance)
        penalty_total = -(rew_lin_vel_z + rew_root_pitch_roll + rew_imbalance + rew_ang_vel_xy + rew_flat_orientation + rew_action_rate + rew_joint_torques + rew_undesired_contacts + rew_feet_slide + rew_both_feet_contact + rew_ankle_gravity)

        # Get feet air time statistics for logging
        air_time_stats = get_feet_air_time_stats(env)

        # Compute root pitch and roll angles for logging
        w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * 1.5707963267948966,
            torch.asin(sinp),
        )
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        env._tb_writer.add_scalar("reward/total", total_reward.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("reward/task", task_reward.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("penalty/total", penalty_total.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("progress/forward_delta_x", delta_x.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("progress/forward_cumulative_x", env._cum_forward_x.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("gait/feet_air_time_mean", air_time_stats["mean_air_time"], env._tb_step)
        env._tb_writer.add_scalar("gait/feet_air_time_max", air_time_stats["max_air_time"], env._tb_step)
        env._tb_writer.add_scalar("balance/root_pitch_rad", pitch.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("balance/root_roll_rad", roll.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("balance/root_pitch_deg", torch.rad2deg(pitch).mean().item(), env._tb_step)
        env._tb_writer.add_scalar("balance/root_roll_deg", torch.rad2deg(roll).mean().item(), env._tb_step)

    # curriculum update
    cmd_error = base_lin_vel[:, :2] - env._command[:, :2]
    cmd_error_mean = float(torch.norm(cmd_error, dim=1).mean().item())
    gait_stats = compute_gait_rewards(env, root_state)
    if env.curriculum.update(env._control_dt, gait_stats["height_mean"], gait_stats["single_rate"], cmd_error_mean):
        env._sample_commands(None)

    env._prev_actions = env.actions.clone()
    env._tb_step += 1
    return total_reward


