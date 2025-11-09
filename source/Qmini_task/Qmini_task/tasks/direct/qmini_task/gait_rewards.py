from __future__ import annotations

from typing import Dict

import torch


def compute_gait_rewards(env, root_state: torch.Tensor) -> Dict[str, torch.Tensor | float]:
    """Compute auxiliary gait rewards and statistics for curriculum decisions."""
    cfg = env.cfg

    left_height = env.robot.data.body_pos_w[:, env._left_foot_body_idx, 2]
    right_height = env.robot.data.body_pos_w[:, env._right_foot_body_idx, 2]
    root_height = root_state[:, 2]

    margin = getattr(env, "_single_leg_margin", 0.03)
    single_left = (left_height - right_height) > margin
    single_right = (right_height - left_height) > margin
    single_support = torch.clamp(single_left.float() + single_right.float(), max=1.0)

    height_target = getattr(cfg, "curriculum_height_threshold", 0.3)
    height_bonus = torch.clamp(root_height - height_target, min=0.0)
    rew_height = getattr(cfg, "rew_scale_height", 0.0) * height_bonus
    rew_single = getattr(cfg, "rew_scale_single_leg", 0.0) * single_support

    return {
        "reward_height": rew_height,
        "reward_single": rew_single,
        "single_support": single_support,
        "single_rate": float(single_support.mean().item()),
        "height_mean": float(root_height.mean().item()),
    }
