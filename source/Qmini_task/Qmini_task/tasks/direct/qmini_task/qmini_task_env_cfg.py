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

QMINI_USD_PATH = "/home/bird/isaacSim/Learn/Qmini/Qmini_1108.usd"

QMINI_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=QMINI_USD_PATH,
        activate_contact_sensors=False,
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
            effort_limit_sim=45.0,
            stiffness=40.0,
            damping=6.0,
        ),
        "LL_hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint2"],
            effort_limit_sim=45.0,
            stiffness=50.0,
            damping=6.0,
        ),
        "LL_hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint3"],
            effort_limit_sim=55.0,
            stiffness=60.0,
            damping=8.0,
        ),
        "LL_knee": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint4"],
            effort_limit_sim=55.0,
            stiffness=70.0,
            damping=10.0,
        ),
        "LL_ankle": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint5"],
            effort_limit_sim=45.0,
            stiffness=35.0,
            damping=6.0,
        ),
        "RL_hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint1"],
            effort_limit_sim=45.0,
            stiffness=40.0,
            damping=6.0,
        ),
        "RL_hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint2"],
            effort_limit_sim=45.0,
            stiffness=50.0,
            damping=6.0,
        ),
        "RL_hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint3"],
            effort_limit_sim=55.0,
            stiffness=60.0,
            damping=8.0,
        ),
        "RL_knee": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint4"],
            effort_limit_sim=55.0,
            stiffness=70.0,
            damping=10.0,
        ),
        "RL_ankle": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint5"],
            effort_limit_sim=45.0,
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
    rew_scale_joint_vel = 0.5
    rew_scale_upright = 5.0
    rew_scale_base_lin_vel = 0.5
    rew_scale_base_ang_vel = 0.5
    rew_scale_action_rate = 0.05
    rew_scale_success = 2.0
    rew_scale_cmd_lin_vel = 3.0
    rew_scale_cmd_yaw_vel = 0.5
    rew_scale_gait = 0.5

    # success / failure thresholds
    success_joint_tol = 0.05
    success_upright_cos = 0.98
    success_pitch_tol = math.radians(5.0)
    failure_pitch_angle = math.radians(45.0)
    failure_min_height = 0.12  # [m]

    # reset sampling
    reset_noise_scale = 0.1
    orientation_noise_deg = 15.0

    # command profile
    command_lin_vel_x_range = (-0.6, 0.8)
    command_lin_vel_y_range = (-0.6, 0.6)
    command_yaw_range = (-1.0, 1.0)
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
    max_joint_velocity = 5.0  # [rad/s]
    action_filter_gain = 0.2
