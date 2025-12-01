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
        pos=(0.0, 0.0, 0.25),  # Reference: pos=(0.0, 0.0, 0.3)
        joint_pos={
            "LL_joint1": 0.0,   # hip_yaw
            "LL_joint2": 0.0,   # hip_roll
            "LL_joint3": 0.3,   # hip_pitch (Reference: 0.3)
            "LL_joint4": -0.8,  # knee (Reference: -0.8)
            "LL_joint5": 0.5,   # ankle (Reference: 0.5)
            "RL_joint1": 0.0,   # hip_yaw
            "RL_joint2": 0.0,   # hip_roll
            "RL_joint3": 0.3,   # hip_pitch (Reference: 0.3)
            "RL_joint4": -0.8,  # knee (Reference: -0.8)
            "RL_joint5": 0.5,   # ankle (Reference: 0.5)
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
    observation_space = 37  # 37 dims: base_ang_vel(3) + projected_gravity(3) + dof_pos(10) + dof_vel(10) + last_action(10) + action_rescale(1)
    state_space = 0
    
    # Observation scales (following HoST design)
    obs_scale_ang_vel = 0.25  # Base angular velocity scale
    obs_scale_dof_pos = 1.0   # Joint position scale
    obs_scale_dof_vel = 0.05  # Joint velocity scale

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
        "LL_joint1": 0.0,   # hip_yaw - matches init_state
        "LL_joint2": 0.0,   # hip_roll - matches init_state
        "LL_joint3": 0.3,   # hip_pitch - matches init_state
        "LL_joint4": -0.8,  # knee - matches init_state
        "LL_joint5": 0.5,   # ankle - matches init_state
        "RL_joint1": 0.0,   # hip_yaw - matches init_state
        "RL_joint2": 0.0,   # hip_roll - matches init_state
        "RL_joint3": 0.3,   # hip_pitch - matches init_state
        "RL_joint4": -0.8,  # knee - matches init_state
        "RL_joint5": 0.5,   # ankle - matches init_state
    }

    # reward scales - HoST framework
    rew_scale_alive = 0.1
    rew_scale_terminated = -1.0
    
    # HoST 4-group reward weights
    # Reward groups: task (multiplicative), regu, style, target
    rew_weight_task = 2.5  # Task reward weight
    rew_weight_regu = 0.1  # Regularization reward weight
    rew_weight_style = 1.0  # Style reward weight
    rew_weight_target = 1.0  # Post-task reward weight
    
    # Pitch deviation penalty (in regu group)
    rew_scale_pitch_penalty = 5.0  # Pitch deviation penalty scale (增大pitch偏离的惩罚)
    
    # Phase thresholds (scaled for Qmini: 0.43m is normal standing)
    target_base_height_phase1 = 0.25  # Phase 1 threshold (scaled from 0.45m)
    target_base_height_phase3 = 0.35  # Phase 3 threshold (scaled from 0.65m)
    target_base_height = 0.43  # Target base height for Qmini (scaled from 0.75m)
    target_head_height = 0.43  # Target head height relative to feet (scaled from 1.0m)
    
    # Curriculum learning parameters
    enable_curriculum = True  # Enable curriculum learning
    initial_pull_force = 20.0  # Initial upward pull force (N) - scaled from 100N
    curriculum_force_decrement = 4.0  # Force decrement per update (scaled from 20N)
    curriculum_head_height_threshold = 0.39  # Head height threshold for curriculum (scaled from 0.9m)
    initial_action_rescale = 1.0  # Initial action scaling factor
    curriculum_action_rescale_decrement = 0.02  # Action rescale decrement per update
    min_action_rescale = 0.25  # Minimum action rescale
    
    # Task rewards (rtask) - added together (not multiplied to avoid gradient vanishing)
    rew_scale_task = 1.0  # Overall task reward scale (deprecated, use individual scales)
    rew_scale_orientation_task = 2.0  # Orientation reward scale (increased)
    rew_scale_height_task = 5.0  # Height reward scale (increased to encourage standing)
    rew_scale_standup_phase = 3.0  # Stand-up phase reward (0.25m to 0.43m) - encourages standing up after flipping
    target_base_height_phase1 = 0.25  # Phase 1 threshold
    target_base_height_phase3 = 0.35  # Phase 3 threshold
    orientation_threshold = 0.99  # Orientation threshold for tolerance
    target_head_margin = 0.43  # Head height margin (deprecated)
    base_height_target = 0.43  # Target base height for post-task reward
    
    # Style rewards (rstyle)
    rew_scale_waist_penalty = 10.0  # Waist twist penalty
    rew_scale_knee_penalty = 10.0  # Knee angle penalty
    rew_scale_feet_distance_penalty = 10.0  # Feet distance penalty
    rew_scale_shank_orientation = 10.0  # Shank orientation reward
    rew_scale_knee_extension = 2.0  # Knee extension reward during stand-up phase
    
    # Regularization rewards (rregu)
    rew_scale_joint_accel = 2.5e-7  # Joint acceleration penalty
    rew_scale_action_rate = 1e-2  # Action rate penalty
    rew_scale_torque = 2.5e-6  # Torque penalty
    rew_scale_power = 2.5e-5  # Power penalty
    
    # Post-task rewards (rpost) - only active when standing
    rew_scale_ang_vel_post = 10.0  # Angular velocity reward when standing
    rew_scale_lin_vel_post = 10.0  # Linear velocity reward when standing
    rew_scale_height_post = 10.0  # Height maintenance reward when standing
    
    # Joint position reward (only active when height >= 0.25m)
    rew_scale_joint = 1.0

    # success / failure thresholds
    success_joint_tol = 0.05
    success_upright_cos = 0.98
    success_pitch_tol = math.radians(5.0)
    # reset 阈值
    failure_tilt_angle = math.radians(45.0)
    failure_min_height = 0.12  # [m]

    # stand-up task specific parameters
    target_base_height = 0.43  # Target height for successful stand-up [m]
    standup_stage1_height = 0.15  # Stage 1: initial lying down (~35% of target height)
    standup_stage2_height = 0.15  # Stage 2: intermediate (~35% of target height)
    standup_stage3_height = 0.30  # Stage 3: near standing (~70% of target height)
    
    # reward scales for stand-up task
    rew_scale_height_progress = 5.0  # Reward for height progress (increased to encourage standing up)
    rew_scale_height_target = 10.0  # Reward for reaching target height (increased)
    rew_scale_standup_success = 20.0  # Large reward for successful stand-up (increased)
    
    # initial pose sampling for stand-up
    enable_standup_task = True  # Enable stand-up task mode
    initial_pose_types = ["supine", "prone", "left_side", "right_side"]  # Types of initial poses
    initial_pose_prob = [0.4, 0.3, 0.15, 0.15]  # Probability for each pose type
    initial_height_range = (0.05, 0.15)  # Initial height range when lying down [m]

    # reset sampling
    reset_noise_scale = 0.1
    orientation_noise_deg = 15.0

    # action smoothing
    action_smoothing_rate = 0.1
    max_joint_velocity = 5.0  # [rad/s]
    action_filter_gain = 0.2
