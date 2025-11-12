# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

QMINI_USD_PATH = "/home/bird/isaacSim/Learn/Qmini/Qmini_1108.usd"

QMINI_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=QMINI_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=5.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            "LL_joint1": 0.0,
            "LL_joint2": 0.0,
            "LL_joint3": 0.0,
            "LL_joint4": 0.0,
            "LL_joint5": 0.0,
            "RL_joint1": 0.0,
            "RL_joint2": 0.0,
            "RL_joint3": 0.0,
            "RL_joint4": 0.0,
            "RL_joint5": 0.0,
        },
    ),
    actuators={
        "LL_hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint1"],
            effort_limit_sim=24.0,
            stiffness=40.0,
            damping=6.0,
        ),
        "LL_hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint2"],
            effort_limit_sim=24.0,
            stiffness=50.0,
            damping=6.0,
        ),
        "LL_hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint3"],
            effort_limit_sim=24.0,
            stiffness=60.0,
            damping=8.0,
        ),
        "LL_knee": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint4"],
            effort_limit_sim=24.0,
            stiffness=70.0,
            damping=10.0,
        ),
        "LL_ankle": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint5"],
            effort_limit_sim=24.0,
            stiffness=35.0,
            damping=6.0,
        ),
        "RL_hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint1"],
            effort_limit_sim=24.0,
            stiffness=40.0,
            damping=6.0,
        ),
        "RL_hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint2"],
            effort_limit_sim=24.0,
            stiffness=50.0,
            damping=6.0,
        ),
        "RL_hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint3"],
            effort_limit_sim=24.0,
            stiffness=60.0,
            damping=8.0,
        ),
        "RL_knee": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint4"],
            effort_limit_sim=24.0,
            stiffness=70.0,
            damping=10.0,
        ),
        "RL_ankle": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint5"],
            effort_limit_sim=24.0,
            stiffness=35.0,
            damping=6.0,
        ),
    },
)


@configclass
class QminiTaskEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition
    action_space = 10
    observation_space = 37
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_prim_path="/physicsScene",
        gravity=(0.0, 0.0, -9.81),
    )

    # robot(s)
    robot_cfg: ArticulationCfg = QMINI_ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Qmini")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=1.0,
        replicate_physics=True,
    )

    # custom parameters/scales
    controlled_joints = (
        "LL_joint1",
        "LL_joint2",
        "LL_joint3",
        "LL_joint4",
        "LL_joint5",
        "RL_joint1",
        "RL_joint2",
        "RL_joint3",
        "RL_joint4",
        "RL_joint5",
    )

    joint_lower_limits = (
        -0.349,
        -0.275,
        -0.61,
        -1.117,
        -1.396,
        -0.525,
        -0.698,
        -0.61,
        -1.117,
        -1.396,
    )

    joint_upper_limits = (
        0.525,
        0.698,
        1.483,
        1.047,
        1.221,
        0.349,
        0.275,
        1.483,
        1.047,
        1.221,
    )

    target_joint_pos = {
        "LL_joint1": 0.0,
        "LL_joint2": 0.0,
        "LL_joint3": 0.0,
        "LL_joint4": 0.0,
        "LL_joint5": 0.0,
        "RL_joint1": 0.0,
        "RL_joint2": 0.0,
        "RL_joint3": 0.0,
        "RL_joint4": 0.0,
        "RL_joint5": 0.0,
    }

    # reward scales
    rew_scale_alive = 0.1
    rew_scale_terminated = -1.0
    rew_scale_joint = 1.0
    rew_scale_joint_vel = 0.05      # 进一步降低 joint_vel 惩罚
    rew_scale_joint_speed = 0.5     # 低速惩罚
    rew_scale_upright = 5.0
    rew_scale_base_lin_vel = 0.5
    rew_scale_base_ang_vel = 0.5
    rew_scale_action_rate = 0.05
    rew_scale_success = 2.0
    rew_scale_cmd_lin_vel = 3.0
    rew_scale_cmd_yaw_vel = 0.5
    rew_scale_gait = 0.5
    rew_scale_height = 3.0         # 身高奖励，高度越高奖励越大
    rew_scale_single_leg = 8.0     # 单腿支撑接触高额奖励
    rew_scale_tilt_fail = 10.0     # 倾角超限惩罚
    rew_scale_height_fail = 12.0   # 高度过低重置惩罚
    rew_scale_forward_speed = 4.0  # X 方向速度奖励
    rew_scale_forward_track = 2.3  # 指令前向速度跟踪奖励
    rew_scale_yaw_track = 2.0      # 指令偏航速度跟踪奖励
    rew_scale_balance = 1.5        # 平衡奖励
    rew_scale_lateral_penalty = 0.7  # 侧向速度惩罚
    rew_scale_foot_clear = 1.2     # 摆动腿抬脚奖励
    rew_scale_foot_slip = 0.6      # 支撑腿滑移惩罚

    # success / failure thresholds
    success_joint_tol = 0.05
    success_upright_cos = 0.98
    success_pitch_tol = math.radians(5.0)
    failure_pitch_angle = math.radians(45.0)
    failure_min_height = 0.25  # [m] 低于该高度重置

    # reset sampling
    reset_noise_scale = 0.1
    orientation_noise_deg = 5.0     # 减小初始姿态噪声

    desired_root_height = 0.4       # 目标机身高度 [m]
    foot_contact_force_threshold = 5.0  # 足底接触判定阈值 [N]
    desired_foot_clearance = 0.05   # 摆动腿目标离地高度 [m]

    joint_target_speed = 1.0        # 目标关节速度 [rad/s]

    # command profile
    command_lin_vel_x_range = (0.0, 0.8)
    command_lin_vel_y_range = (0.0, 0.0)
    command_yaw_range = (0.0, 0.0)
    command_change_interval_s = 2.0

    # gait parameters
    gait_cycle_duration = 0.8
    gait_hip_amp = 0.35
    gait_knee_base = -0.6
    gait_knee_amp = 0.35
    gait_knee_phase = math.pi / 2
    gait_ankle_base = 0.25
    gait_ankle_amp = 0.15

    # action smoothing
    action_smoothing_rate = 0.1
    max_joint_velocity = 8.0  # [rad/s]
    action_filter_gain = 0.2

    # sensors
    foot_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Qmini/.*_ankle",
        update_period=0.0,
        history_length=2,
        track_air_time=True,
        force_threshold=5.0,
        debug_vis=False,
    )
