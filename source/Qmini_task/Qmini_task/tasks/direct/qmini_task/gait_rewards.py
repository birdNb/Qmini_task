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
