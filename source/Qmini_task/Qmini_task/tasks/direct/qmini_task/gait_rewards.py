from __future__ import annotations

from typing import Dict

import torch


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
    """Compute reward for feet air time (gait formation).

    Following the reference implementation from isaaclab.envs.mdp.rewards.

    Args:
        env: The environment instance
        threshold_min: Minimum air time to get reward (seconds)
        threshold_max: Maximum air time to get reward (seconds)

    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Get contact sensor (configured to select ankles via prim path)
    contact_sensor = None
    if hasattr(env, "scene") and hasattr(env.scene, "sensors"):
        sensors = env.scene.sensors
        if isinstance(sensors, dict):
            contact_sensor = sensors.get("contact_forces") or sensors.get("foot_contact_sensor")
        else:
            contact_sensor = getattr(sensors, "contact_forces", None) or getattr(sensors, "foot_contact_sensor", None)
    if contact_sensor is None:
        contact_sensor = getattr(env, "_foot_contact_sensor", None)
    if contact_sensor is None or not hasattr(contact_sensor, "data"):
        return torch.zeros(env.scene.num_envs, device=env.device)

    # Use all bodies from the sensor (these are ankles per config)
    try:
        num_bodies = contact_sensor.data.last_air_time.shape[1]
    except Exception:
        num_bodies = contact_sensor.data.net_forces_w.shape[1]
    body_ids = torch.arange(num_bodies, device=env.device, dtype=torch.long)

    if body_ids is None or len(body_ids) == 0:
        return torch.zeros(env.scene.num_envs, device=env.device)

    # Compute first contact using the sensor's method
    try:
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, body_ids]
    except Exception:
        # Fallback: use contact forces to determine first contact
        forces = contact_sensor.data.net_forces_w[:, body_ids, :]
        contact_threshold = getattr(env, "_foot_contact_threshold", 5.0)
        first_contact = (torch.norm(forces, dim=-1) > contact_threshold).float()

    # Get last air time from sensor data
    try:
        last_air_time = contact_sensor.data.last_air_time[:, body_ids]
    except Exception:
        # If last_air_time doesn't exist, return zero
        return torch.zeros(env.scene.num_envs, device=env.device)

    # Negative reward for small steps (air time < threshold_min)
    air_time = (last_air_time - threshold_min) * first_contact

    # No reward for large steps (clamp to threshold_max - threshold_min)
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)

    # Sum over feet
    reward = torch.sum(air_time, dim=1)

    # No reward for zero command (only reward when moving)
    command = getattr(env, "_command", None)
    if command is not None:
        cmd_vel_xy = torch.norm(command[:, :2], dim=1)
        reward = reward * (cmd_vel_xy > 0.1).float()

    return reward


def compute_feet_slide(env) -> torch.Tensor:
    """Compute penalty for feet sliding during contact.

    Following the reference implementation from isaaclab.envs.mdp.rewards.

    Args:
        env: The environment instance

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get contact sensor (configured to ankles)
    contact_sensor = None
    if hasattr(env, "scene") and hasattr(env.scene, "sensors"):
        sensors = env.scene.sensors
        if isinstance(sensors, dict):
            contact_sensor = sensors.get("contact_forces") or sensors.get("foot_contact_sensor")
        else:
            contact_sensor = getattr(sensors, "contact_forces", None) or getattr(sensors, "foot_contact_sensor", None)
    if contact_sensor is None:
        contact_sensor = getattr(env, "_foot_contact_sensor", None)
    if contact_sensor is None or not hasattr(contact_sensor, "data"):
        return torch.zeros(env.scene.num_envs, device=env.device)

    # Use all sensor bodies (ankles)
    num_bodies = contact_sensor.data.net_forces_w.shape[1]
    body_ids = torch.arange(num_bodies, device=env.device, dtype=torch.long)

    if body_ids is None or len(body_ids) == 0:
        return torch.zeros(env.scene.num_envs, device=env.device)

    # Get contact mask from force history (following reference implementation)
    try:
        # Use history to get max force over recent steps
        forces_history = contact_sensor.data.net_forces_w_history  # [num_envs, history, bodies, 3]
        contacts = forces_history[:, :, body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    except Exception:
        # Fallback: use current forces
        forces = contact_sensor.data.net_forces_w[:, body_ids, :]
        contacts = torch.norm(forces, dim=-1) > 1.0

    # Get foot body velocities (horizontal only)
    # Map from sensor's body_ids (ankles) to robot body indices if available,
    # otherwise assume the first num_bodies correspond to ankles in robot order.
    robot = env.robot
    try:
        sensor_body_ids = contact_sensor.body_ids  # indices into robot bodies
        body_vel = robot.data.body_lin_vel_w[:, sensor_body_ids, :2]
    except Exception:
        body_vel = robot.data.body_lin_vel_w[:, :num_bodies, :2]  # fallback

    # Penalty = horizontal velocity * contact_mask (only penalize when in contact)
    penalty = torch.sum(body_vel.norm(dim=-1) * contacts.float(), dim=1)

    return penalty
