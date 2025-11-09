from __future__ import annotations

from typing import Tuple


class OmniGaitCurriculum:
    """Lightweight curriculum that gradually expands command ranges based on stability metrics."""

    def __init__(self, cfg, device):
        self.enabled = getattr(cfg, "curriculum_enabled", False)
        self.device = device

        self.phase = 0
        self.elapsed = 0.0

        self.lin_vel_x_schedule = list(getattr(cfg, "curriculum_lin_vel_x_schedule", ()))
        if not self.lin_vel_x_schedule:
            self.lin_vel_x_schedule.append(cfg.command_lin_vel_x_range)
        elif self.lin_vel_x_schedule[-1] != cfg.command_lin_vel_x_range:
            self.lin_vel_x_schedule.append(cfg.command_lin_vel_x_range)

        self.lin_vel_y_schedule = list(getattr(cfg, "curriculum_lin_vel_y_schedule", ()))
        if not self.lin_vel_y_schedule:
            self.lin_vel_y_schedule.append(cfg.command_lin_vel_y_range)
        elif self.lin_vel_y_schedule[-1] != cfg.command_lin_vel_y_range:
            self.lin_vel_y_schedule.append(cfg.command_lin_vel_y_range)
        self.lin_vel_y_schedule = [(0.0, 0.0) for _ in self.lin_vel_y_schedule]

        self.yaw_schedule = list(getattr(cfg, "curriculum_yaw_range_schedule", ()))
        if not self.yaw_schedule:
            self.yaw_schedule.append(cfg.command_yaw_range)
        elif self.yaw_schedule[-1] != cfg.command_yaw_range:
            self.yaw_schedule.append(cfg.command_yaw_range)
        self.yaw_schedule = [(0.0, 0.0) for _ in self.yaw_schedule]

        self.phase_durations = list(getattr(cfg, "curriculum_phase_durations", ()))
        self.height_threshold = getattr(cfg, "curriculum_height_threshold", 0.3)
        self.single_leg_threshold = getattr(cfg, "curriculum_single_leg_threshold", 0.2)
        self.cmd_error_threshold = getattr(cfg, "curriculum_cmd_error_threshold", 0.4)

        self._curr_lin_vel_x_range = cfg.command_lin_vel_x_range
        self._curr_lin_vel_y_range = cfg.command_lin_vel_y_range
        self._curr_yaw_range = cfg.command_yaw_range
        self._update_ranges()

    def reset(self) -> None:
        self.elapsed = 0.0

    def get_command_ranges(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        return self._curr_lin_vel_x_range, self._curr_lin_vel_y_range, self._curr_yaw_range

    def update(self, dt: float, height_mean: float, single_rate: float, cmd_error_mean: float) -> bool:
        if not self.enabled:
            return False

        self.elapsed += dt
        if self.phase < len(self.phase_durations):
            duration = self.phase_durations[self.phase]
            if duration > 0.0 and self.elapsed < duration:
                return False

        if (
            height_mean >= self.height_threshold
            and single_rate >= self.single_leg_threshold
            and cmd_error_mean <= self.cmd_error_threshold
        ):
            if self.phase < max(len(self.lin_vel_x_schedule), len(self.lin_vel_y_schedule), len(self.yaw_schedule)) - 1:
                self.phase += 1
                self.elapsed = 0.0
                self._update_ranges()
                return True
        return False

    def _update_ranges(self) -> None:
        idx = min(self.phase, len(self.lin_vel_x_schedule) - 1)
        self._curr_lin_vel_x_range = self.lin_vel_x_schedule[idx]

        idx_y = min(self.phase, len(self.lin_vel_y_schedule) - 1)
        self._curr_lin_vel_y_range = self.lin_vel_y_schedule[idx_y]

        idx_yaw = min(self.phase, len(self.yaw_schedule) - 1)
        self._curr_yaw_range = self.yaw_schedule[idx_yaw]
