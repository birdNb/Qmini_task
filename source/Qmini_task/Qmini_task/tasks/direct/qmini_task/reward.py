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


def compute_feet_gait(env, period: float = 0.6, offset: list[float] = [0.0, 0.5], threshold: float = 0.55) -> torch.Tensor:
    """Compute gait reward based on contact pattern matching expected gait phase.
    
    Reference: mdp.feet_gait with period=0.6, offset=[0.0, 0.5], threshold=0.55
    """
    import math
    device = env.device
    num_envs = env.scene.num_envs
    
    # Get contact sensor (ankle bodies)
    sensors = getattr(env.scene, "sensors", {})
    left = sensors.get("contact_forces_left", None)
    right = sensors.get("contact_forces_right", None)
    
    # Get contact states for both feet
    def get_contact_state(sensor) -> torch.Tensor:
        if sensor is None or not hasattr(sensor, "data"):
            return torch.zeros(num_envs, device=device, dtype=torch.bool)
        try:
            forces = sensor.data.net_forces_w
            if forces is None:
                return torch.zeros(num_envs, device=device, dtype=torch.bool)
            # Sum over bodies if multiple
            force_norm = torch.norm(forces, dim=-1)  # [N, B] or [N]
            if force_norm.dim() > 1:
                force_norm = force_norm.sum(dim=1)  # [N]
            return force_norm > getattr(env.cfg, "foot_contact_force_threshold", 100.0)
        except Exception:
            return torch.zeros(num_envs, device=device, dtype=torch.bool)
    
    left_contact = get_contact_state(left)
    right_contact = get_contact_state(right)
    
    # Get normalized gait phase [0, 1)
    gait_phase = env._gait_phase  # [N] - already in [0, 2π)
    phase_normalized = (gait_phase / (2.0 * math.pi)) % 1.0  # [0, 1)
    
    # Expected contact pattern based on phase
    # offset[0] for left, offset[1] for right
    left_phase = (phase_normalized + offset[0]) % 1.0
    right_phase = (phase_normalized + offset[1]) % 1.0
    
    # Expected contact: 1 if in contact phase, 0 if in swing phase
    # Contact phase is when phase is in [0, 0.5) (half cycle)
    left_expected = (left_phase < 0.5).float()
    right_expected = (right_phase < 0.5).float()
    
    # Actual contact states
    left_actual = left_contact.float()
    right_actual = right_contact.float()
    
    # Match quality: how well actual matches expected
    left_match = 1.0 - torch.abs(left_actual - left_expected)
    right_match = 1.0 - torch.abs(right_actual - right_expected)
    
    # Average match quality
    match_quality = (left_match + right_match) / 2.0
    
    # Reward when match quality exceeds threshold
    reward = (match_quality > threshold).float()
    
    return reward


def compute_feet_slide(env) -> torch.Tensor:
    """Compute penalty for feet sliding during contact.
    
    Reference: mdp.feet_slide with ankle bodies
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
    
    if not ankle_indices:
        return torch.zeros(num_envs, device=device)
    
    # Get contact sensors
    sensors = getattr(env.scene, "sensors", {})
    left = sensors.get("contact_forces_left", None)
    right = sensors.get("contact_forces_right", None)
    
    def get_contact_mask(sensor) -> torch.Tensor:
        if sensor is None or not hasattr(sensor, "data"):
            return torch.zeros(num_envs, device=device, dtype=torch.bool)
        try:
            forces = sensor.data.net_forces_w
            if forces is None:
                return torch.zeros(num_envs, device=device, dtype=torch.bool)
            force_norm = torch.norm(forces, dim=-1)
            if force_norm.dim() > 1:
                force_norm = force_norm.sum(dim=1)
            return force_norm > getattr(env.cfg, "foot_contact_force_threshold", 100.0)
        except Exception:
            return torch.zeros(num_envs, device=device, dtype=torch.bool)
    
    left_contact = get_contact_mask(left)
    right_contact = get_contact_mask(right)
    
    # Get ankle velocities (horizontal plane only)
    body_vel = env.robot.data.body_lin_vel_w  # [N, bodies, 3]
    total_penalty = torch.zeros(num_envs, device=device)
    
    for i, (idx, contact_mask) in enumerate(zip(ankle_indices, [left_contact, right_contact])):
        if idx < 0:
            continue
        # Horizontal velocity (xy plane)
        vel_xy = body_vel[:, idx, :2]  # [N, 2]
        vel_norm = torch.norm(vel_xy, dim=1)  # [N]
        # Penalty only when in contact
        penalty = vel_norm * contact_mask.float()
        total_penalty = total_penalty + penalty
    
    return total_penalty


def compute_foot_clearance_reward(env, std: float = 0.05, tanh_mult: float = 2.0, target_height: float = 0.05) -> torch.Tensor:
    """Reward for foot clearance during swing phase.
    
    Reference: mdp.foot_clearance_reward with std=0.05, tanh_mult=2.0, target_height=0.05
    Uses tanh-based reward centered at target_height with std as scale.
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
    
    if not ankle_indices:
        return torch.zeros(num_envs, device=device)
    
    # Get contact sensors
    sensors = getattr(env.scene, "sensors", {})
    left = sensors.get("contact_forces_left", None)
    right = sensors.get("contact_forces_right", None)
    
    def get_contact_mask(sensor) -> torch.Tensor:
        if sensor is None or not hasattr(sensor, "data"):
            return torch.zeros(num_envs, device=device, dtype=torch.bool)
        try:
            forces = sensor.data.net_forces_w
            if forces is None:
                return torch.zeros(num_envs, device=device, dtype=torch.bool)
            force_norm = torch.norm(forces, dim=-1)
            if force_norm.dim() > 1:
                force_norm = force_norm.sum(dim=1)
            return force_norm > getattr(env.cfg, "foot_contact_force_threshold", 100.0)
        except Exception:
            return torch.zeros(num_envs, device=device, dtype=torch.bool)
    
    left_contact = get_contact_mask(left)
    right_contact = get_contact_mask(right)
    
    # Get ankle heights relative to ground (assuming ground at z=0)
    body_pos_w = env.robot.data.body_pos_w  # [N, bodies, 3]
    reward = torch.zeros(num_envs, device=device)
    
    for idx, contact_mask in zip(ankle_indices, [left_contact, right_contact]):
        if idx < 0:
            continue
        
        # Get ankle height (relative to ground, assuming ground at z=0)
        ankle_height = body_pos_w[:, idx, 2]  # [N]
        
        # Only reward when not in contact (swing phase)
        swing_mask = (~contact_mask).float()
        
        # Compute clearance reward using tanh function
        # Reference: mdp.foot_clearance_reward
        # Reward when foot height is close to target_height during swing
        height_diff = ankle_height - target_height
        # Normalize by std: reward peaks when height_diff is close to 0
        normalized_diff = height_diff / (std + 1e-6)
        # Apply tanh: creates smooth reward that peaks at target_height
        # tanh_mult controls the steepness of the reward curve
        clearance_reward = torch.tanh(tanh_mult * torch.exp(-normalized_diff ** 2))
        
        # Only apply reward during swing phase
        reward = reward + clearance_reward * swing_mask
    
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


def compute_feet_air_time_mean_reward(env) -> torch.Tensor:
    """High reward for average feet air time - encourages lifting legs.
    
    This reward encourages the robot to maintain good air time during swing phase,
    which is essential for proper walking gait.
    """
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
        # Sum air time across all bodies in the sensor
        total_air_time = total_air_time + lat.sum(dim=1)

    # Return mean air time per environment (average across both feet)
    mean_air_time = total_air_time / max(len(sensor_list), 1)
    return mean_air_time


def compute_target_air_time_reward(env, target_air_time: float = 0.3, tolerance: float = 0.1) -> torch.Tensor:
    """Reward for foot air time being close to target (0.3s).
    
    This reward encourages the robot to maintain a consistent air time of 0.3s
    during swing phase, which is essential for proper walking gait.
    
    Args:
        env: The environment instance
        target_air_time: Target air time in seconds (default: 0.3s)
        tolerance: Tolerance for air time deviation (default: 0.1s)
    
    Returns:
        Reward tensor that peaks when air time is close to target
    """
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
        # Sum air time across all bodies in the sensor
        total_air_time = total_air_time + lat.sum(dim=1)

    # Average air time per foot
    mean_air_time = total_air_time / max(len(sensor_list), 1)
    
    # Compute reward: exponential reward centered at target_air_time
    air_time_error = torch.abs(mean_air_time - target_air_time)
    # Normalize by tolerance: reward peaks when error is 0, decreases as error increases
    normalized_error = air_time_error / (tolerance + 1e-6)
    # Exponential reward: 1.0 when error=0, decreases exponentially
    reward = torch.exp(-normalized_error ** 2)
    
    return reward


def compute_feet_height_consistency_penalty(env, threshold: float = 0.02) -> torch.Tensor:
    """Penalty for inconsistent foot heights when both feet are in contact.
    
    This penalty ensures that when both feet are on the ground, they maintain
    similar heights (consistent posture), preventing one foot from being too
    low (pointing at ground) while the other is higher.
    
    Args:
        env: The environment instance
        threshold: Maximum allowed height difference when both feet are in contact (default: 0.02m)
    
    Returns:
        Penalty tensor: 0.0 when heights are consistent or only one foot is in contact,
                        positive value when both feet are in contact but heights differ significantly
    """
    device = env.device
    num_envs = env.scene.num_envs
    
    # Get contact sensors
    sensors = getattr(env.scene, "sensors", {})
    left = sensors.get("contact_forces_left", None)
    right = sensors.get("contact_forces_right", None)
    combined = sensors.get("contact_forces", None)
    
    # Helper to get contact mask
    def get_contact_mask(sensor) -> torch.Tensor | None:
        if sensor is None or not hasattr(sensor, "data"):
            return None
        try:
            forces = sensor.data.net_forces_w
            if forces is None:
                return None
            contact_mask = torch.norm(forces, dim=-1) > getattr(env.cfg, "foot_contact_force_threshold", 100.0)
            if contact_mask.dim() > 1:
                contact_mask = contact_mask.any(dim=-1)
            return contact_mask.float()
        except Exception:
            return None
    
    # Get contact status
    left_contact = get_contact_mask(left)
    right_contact = get_contact_mask(right)
    
    # Fallback to combined sensor
    if left_contact is None or right_contact is None:
        if combined is not None:
            cmask = get_contact_mask(combined)
            if cmask is not None:
                try:
                    forces = combined.data.net_forces_w
                    if forces is not None and forces.shape[1] >= 2:
                        left_forces = forces[:, 0, :]
                        right_forces = forces[:, 1, :]
                        force_threshold = getattr(env.cfg, "foot_contact_force_threshold", 100.0)
                        left_contact = (torch.norm(left_forces, dim=-1) > force_threshold).float()
                        right_contact = (torch.norm(right_forces, dim=-1) > force_threshold).float()
                    else:
                        left_contact = cmask
                        right_contact = cmask
                except Exception:
                    left_contact = cmask
                    right_contact = cmask
        else:
            return torch.zeros(num_envs, device=device)
    
    # Get foot heights
    body_pos_w = env.robot.data.body_pos_w  # [N, bodies, 3]
    
    # Find ankle body indices
    ankle_names = ["LL_ankle", "RL_ankle"]
    ankle_indices = []
    for name in ankle_names:
        ids, _ = env.robot.find_bodies([name])
        if len(ids) > 0:
            ankle_indices.append(int(ids[0]))
        else:
            return torch.zeros(num_envs, device=device)
    
    if len(ankle_indices) < 2:
        return torch.zeros(num_envs, device=device)
    
    left_height = body_pos_w[:, ankle_indices[0], 2]  # [N]
    right_height = body_pos_w[:, ankle_indices[1], 2]  # [N]
    
    # Height difference
    height_diff = torch.abs(left_height - right_height)  # [N]
    
    # Both feet in contact
    both_contact = (left_contact > 0.5) & (right_contact > 0.5)  # [N]
    
    # Penalty: when both feet are in contact, penalize height difference exceeding threshold
    excess_height_diff = torch.clamp(height_diff - threshold, min=0.0)  # [N]
    penalty = excess_height_diff * both_contact.float()  # [N]
    
    return penalty


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


def compute_imbalance_penalty(env, base_quat: torch.Tensor, base_height_rel: torch.Tensor) -> torch.Tensor:
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
    height_imbalance = base_height_rel < height_threshold

    # Penalty: 1.0 if any imbalance condition is met, 0.0 otherwise
    any_imbalance = pitch_imbalance | roll_imbalance | height_imbalance
    penalty = any_imbalance.float()

    return penalty


def compute_total_reward(env) -> torch.Tensor:
    """Full reward computation moved from env._get_rewards."""
    # root/base states
    root_state = env.robot.data.root_state_w
    base_quat = root_state[:, 3:7]
    # Use relative height (above ground) instead of absolute height
    base_height = env._get_base_height_relative_to_ground()
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
    
    # Penalty for velocity tracking error (encourages accurate velocity tracking)
    rew_vel_tracking_error = getattr(env.cfg, "rew_scale_vel_tracking_error", -2.0) * lin_vel_error_norm ** 2

    # Z direction velocity penalty (penalize vertical motion)
    rew_lin_vel_z = getattr(env.cfg, "rew_scale_lin_vel_z", -2.0) * base_lin_vel[:, 2] ** 2

    cmd_ang_vel_z = env._command[:, 2]
    ang_vel_error_z = base_ang_vel[:, 2] - cmd_ang_vel_z
    rew_track_ang_vel_z = getattr(env.cfg, "rew_scale_track_ang_vel_z", 3.0) * torch.exp(-ang_vel_error_z ** 2 / 0.25)
    
    # Penalty for angular velocity tracking error (encourages accurate rotation)
    rew_ang_vel_tracking_error = getattr(env.cfg, "rew_scale_ang_vel_tracking_error", -1.0) * ang_vel_error_z ** 2

    # Alive reward (increased to encourage survival)
    rew_alive = getattr(env.cfg, "rew_scale_alive", 2.0) * torch.ones(env.scene.num_envs, device=env.device)
    
    # Reset penalty (high penalty when robot is about to reset/terminate)
    reset_penalty_scale = getattr(env.cfg, "rew_scale_reset_penalty", -100.0)
    # Check if robot is about to reset (height too low)
    min_height = getattr(env.cfg, "failure_min_height", 0.25)
    is_resetting = (base_height < min_height).float()
    rew_reset_penalty = reset_penalty_scale * is_resetting

    # Penalty for stationary root (no velocity) - High penalty to encourage movement
    base_lin_vel_xy_norm = torch.norm(base_lin_vel[:, :2], dim=1)
    stationary_threshold = getattr(env.cfg, "stationary_velocity_threshold", 0.05)  # 0.05 m/s threshold
    is_stationary = (base_lin_vel_xy_norm < stationary_threshold).float()
    rew_stationary_penalty = getattr(env.cfg, "rew_scale_stationary_penalty", -5.0) * is_stationary

    # 2) Gait rewards - Following reference configuration
    # Gait reward (reference: weight=0.5, period=0.6, offset=[0.0, 0.5], threshold=0.55)
    gait_period = getattr(env.cfg, "gait_cycle_duration", 0.6)
    gait_offset = getattr(env.cfg, "gait_offset", [0.0, 0.5])
    gait_threshold = getattr(env.cfg, "gait_threshold", 0.55)
    rew_gait = getattr(env.cfg, "rew_scale_feet_air_time", 0.5) * compute_feet_gait(
        env, period=gait_period, offset=gait_offset, threshold=gait_threshold
    )
    
    # Feet slide penalty (reference: weight=-0.3)
    rew_feet_slide = getattr(env.cfg, "rew_scale_feet_slide", -0.3) * compute_feet_slide(env)

    # Feet clearance reward - Reduced weight to prevent excessive leg lifting
    clearance_std = getattr(env.cfg, "feet_clearance_std", 0.05)
    clearance_tanh_mult = getattr(env.cfg, "feet_clearance_tanh_mult", 2.0)
    clearance_target = getattr(env.cfg, "desired_foot_clearance", 0.05)
    rew_feet_clearance = getattr(env.cfg, "rew_scale_leg_lift", 0.3) * compute_foot_clearance_reward(
        env, std=clearance_std, tanh_mult=clearance_tanh_mult, target_height=clearance_target
    )
    
    # High reward for average feet air time (encourages lifting legs)
    rew_feet_air_time_mean = getattr(env.cfg, "rew_scale_feet_air_time_mean", 5.0) * compute_feet_air_time_mean_reward(env)
    
    # Reward for target air time (0.3s) - encourages consistent air time
    target_air_time = getattr(env.cfg, "target_foot_air_time", 0.3)
    air_time_tolerance = getattr(env.cfg, "air_time_tolerance", 0.1)
    rew_target_air_time = getattr(env.cfg, "rew_scale_target_air_time", 2.0) * compute_target_air_time_reward(
        env, target_air_time=target_air_time, tolerance=air_time_tolerance
    )
    
    # Penalty for inconsistent foot heights when both feet are in contact
    feet_height_threshold = getattr(env.cfg, "feet_height_consistency_threshold", 0.02)
    rew_feet_height_consistency = getattr(env.cfg, "rew_scale_feet_height_consistency", -5.0) * compute_feet_height_consistency_penalty(
        env, threshold=feet_height_threshold
    )

    # Feet contact forces penalty - Use linear penalty instead of squared to prevent excessive values
    rew_feet_contact_forces = torch.zeros(env.scene.num_envs, device=env.device)
    sensors = getattr(env.scene, "sensors", {})
    left = sensors.get("contact_forces_left", None)
    right = sensors.get("contact_forces_right", None)
    threshold = getattr(env.cfg, "foot_contact_force_threshold", 100.0)
    
    for sensor in [left, right]:
        if sensor is not None and hasattr(sensor, "data") and sensor.data.net_forces_w is not None:
            forces = sensor.data.net_forces_w
            force_norm = torch.norm(forces, dim=-1)
            if force_norm.dim() > 1:
                force_norm = force_norm.sum(dim=1)  # Sum over bodies
            # Use linear penalty instead of squared to prevent excessive values
            excess_forces = torch.clamp(force_norm - threshold, min=0.0)
            rew_feet_contact_forces = rew_feet_contact_forces + excess_forces
    
    rew_feet_contact_forces = getattr(env.cfg, "rew_scale_feet_contact_forces", -0.01) * rew_feet_contact_forces

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

    # Base height penalty and reward (encourage standing at target height)
    target_height = getattr(env.cfg, "desired_root_height", 0.35)
    height_error = base_height - target_height
    # Penalty for deviation from target (quadratic)
    rew_base_height_penalty = getattr(env.cfg, "rew_scale_base_height", -8.0) * height_error ** 2
    # Positive reward when close to target height (encourage standing)
    height_reward_scale = getattr(env.cfg, "rew_scale_base_height_reward", 2.0)
    height_tolerance = getattr(env.cfg, "base_height_reward_tolerance", 0.05)  # 5cm tolerance
    height_reward = height_reward_scale * torch.exp(-(height_error ** 2) / (2 * height_tolerance ** 2))
    rew_base_height = rew_base_height_penalty + height_reward

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

    total_reward = (
        rew_track_lin_vel_xy  # Reference: velocity tracking
        + rew_track_ang_vel_z
        + rew_alive  # Increased alive reward to encourage survival
        + rew_lin_vel_z  # Penalize vertical motion
        + rew_ang_vel_xy
        + rew_vel_tracking_error  # Penalty for velocity tracking error
        + rew_ang_vel_tracking_error  # Penalty for angular velocity tracking error
        + rew_flat_orientation
        + rew_base_height  # Reference: base height penalty
        + rew_joint_acc  # Reference: joint acceleration penalty
        + rew_action_rate
        + rew_dof_pos_limits  # Reference: DOF position limits penalty
        + rew_joint_torques
        + rew_gait  # Reference: gait reward
        + rew_feet_slide
        + rew_feet_clearance  # Reference: feet clearance reward
        + rew_feet_air_time_mean  # High reward for average feet air time
        + rew_target_air_time  # Reward for target air time (0.3s)
        + rew_feet_height_consistency  # Penalty for inconsistent foot heights when both feet are in contact
        + rew_feet_contact_forces  # Reference: feet contact forces penalty
        + rew_undesired_contacts
        + rew_joint_deviation_hip
        + rew_joint_deviation_knee
        + rew_root_pitch_roll
        + rew_imbalance  # Very high penalty for imbalance (reset condition)
        + rew_reset_penalty  # High penalty when resetting/terminating
        + rew_stationary_penalty  # High penalty for stationary root (no movement)
    )

    # Logging (two categories + progress)
    if env._tb_step % 32 == 0:
        task_reward = (rew_track_lin_vel_xy + rew_track_ang_vel_z + rew_gait + rew_feet_clearance + rew_feet_air_time_mean + rew_target_air_time)
        penalty_total = -(rew_lin_vel_z + rew_root_pitch_roll + rew_imbalance + rew_ang_vel_xy + rew_flat_orientation + rew_action_rate + rew_joint_torques + rew_undesired_contacts + rew_feet_slide + rew_feet_contact_forces + rew_vel_tracking_error + rew_ang_vel_tracking_error + rew_reset_penalty + rew_stationary_penalty + rew_joint_deviation_knee + rew_feet_height_consistency)

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

        # All parameters in debug category
        env._tb_writer.add_scalar("debug/reward_total", total_reward.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/reward_task", task_reward.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/penalty_total", penalty_total.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/feet_air_time_mean", air_time_stats["mean_air_time"], env._tb_step)
        env._tb_writer.add_scalar("debug/feet_air_time_max", air_time_stats["max_air_time"], env._tb_step)
        env._tb_writer.add_scalar("debug/root_pitch_rad", pitch.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/root_roll_rad", roll.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/root_pitch_deg", torch.rad2deg(pitch).mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/root_roll_deg", torch.rad2deg(roll).mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/root_height_mean", base_height.mean().item(), env._tb_step)
        
        # Stationary penalty logging
        base_lin_vel_xy_norm = torch.norm(base_lin_vel[:, :2], dim=1)
        env._tb_writer.add_scalar("debug/stationary_penalty", rew_stationary_penalty.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/base_lin_vel_xy_norm", base_lin_vel_xy_norm.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/stationary_ratio", is_stationary.mean().item(), env._tb_step)
        
        # Velocity tracking and air time rewards logging
        env._tb_writer.add_scalar("debug/vel_tracking_error_penalty", rew_vel_tracking_error.mean().item(), env._tb_step)
        env._tb_writer.add_scalar("debug/feet_air_time_mean_reward", rew_feet_air_time_mean.mean().item(), env._tb_step)
        
        # Knee joint deviation logging
        if kfe_only_indices:
            knee_pos_mean = knee_pos.mean().item()
            knee_deviation_mean = knee_deviation.mean().item()
            env._tb_writer.add_scalar("debug/knee_joint_pos_mean", knee_pos_mean, env._tb_step)
            env._tb_writer.add_scalar("debug/knee_joint_deviation_mean", knee_deviation_mean, env._tb_step)
            env._tb_writer.add_scalar("debug/knee_joint_deviation_penalty", rew_joint_deviation_knee.mean().item(), env._tb_step)

    # No curriculum learning - commands are sampled at fixed intervals

    env._prev_actions = env.actions.clone()
    env._tb_step += 1
    return total_reward


