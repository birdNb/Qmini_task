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
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains import TerrainImporterCfg
# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # unused

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
            effort_limit_sim=20.0,
            stiffness=40.0,
            damping=6.0,
        ),
        "LL_hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint2"],
            effort_limit_sim=60.0,
            stiffness=50.0,
            damping=6.0,
        ),
        "LL_hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint3"],
            effort_limit_sim=20.0,
            stiffness=60.0,
            damping=8.0,
        ),
        "LL_knee": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint4"],
            effort_limit_sim=20.0,
            stiffness=70.0,
            damping=10.0,
        ),
        "LL_ankle": ImplicitActuatorCfg(
            joint_names_expr=["LL_joint5"],
            effort_limit_sim=20.0,
            stiffness=35.0,
            damping=6.0,
        ),
        "RL_hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint1"],
            effort_limit_sim=20.0,
            stiffness=40.0,
            damping=6.0,
        ),
        "RL_hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint2"],
            effort_limit_sim=60.0,
            stiffness=50.0,
            damping=6.0,
        ),
        "RL_hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint3"],
            effort_limit_sim=20.0,
            stiffness=60.0,
            damping=8.0,
        ),
        "RL_knee": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint4"],
            effort_limit_sim=20.0,
            stiffness=70.0,
            damping=10.0,
        ),
        "RL_ankle": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint5"],
            effort_limit_sim=20.0,
            stiffness=35.0,
            damping=6.0,
        ),
    },
)


ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),  # Reference: (8.0, 8.0)
    border_width=20.0,  # Reference: 20.0
    num_rows=9,  # Reference: 9
    num_cols=21,  # Reference: 21
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),  # Reference: (0.0, 1.0)
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),  # Reference: only flat terrain with proportion=0.5
    },
)


@configclass
class QminiTaskEnvCfg(DirectRLEnvCfg):
    # env - Following reference configuration
    decimation = 4  # Reference: 4 (increased from 2)
    episode_length_s = 20.0  # Reference: 20.0 (increased from 10.0)
    # - spaces definition
    action_space = 10
    observation_space = 43  # Reference: 3+3+3+3+10+10+10+1 = 43 (added gait_phase)
    state_space = 0

    # simulation - Following reference configuration
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,  # Reference: 0.005 (changed from 1/120 ≈ 0.0083)
        render_interval=decimation,
        physics_prim_path="/physicsScene",
        gravity=(0.0, 0.0, -9.81),
    )

    # robot(s)
    robot_cfg: ArticulationCfg = QMINI_ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Qmini")

    # terrain
    terrain_cfg: TerrainGeneratorCfg = ROUGH_TERRAINS_CFG

    # scene - Following reference configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,  # Reference: 4096, using 512 for reasonable training speed
        env_spacing=2.5,  # Reference: 2.5 (保持分散生成)
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

    # reward scales - Following reference configuration
    # 1. Task Rewards - Reference weights
    rew_scale_track_lin_vel_xy = 3.0      # Reference: 3.0 (track_lin_vel_xy_yaw_frame_exp)
    rew_scale_lin_vel_z = -2.0            # Reference: -2.0 (base_linear_velocity)
    rew_scale_track_ang_vel_z = 3.0       # Reference: 3.0 (track_ang_vel_z_exp)
    rew_scale_alive = 0.3                 # Reference: 0.3 (alive reward)

    # 2. Base Stability Penalties - Reference weights
    rew_scale_ang_vel_xy = -0.5           # Reference: -0.5 (base_angular_velocity)
    rew_scale_flat_orientation = -1.0     # Reference: -1.0 (flat_orientation_l2)
    rew_scale_base_height = -10.0         # Reference: -10.0 (base_height_l2, target_height: 0.15)
    rew_scale_root_pitch_roll = -1.0     # Combined with flat_orientation

    # 3. Action Penalties - Reference weights
    rew_scale_joint_acc = -2.5e-7         # Reference: -2.5e-7 (joint_acc_l2)
    rew_scale_action_rate = -0.10         # Reference: -0.10 (action_rate_l2)
    rew_scale_joint_torques = -1.0e-5     # Keep existing
    rew_scale_dof_pos_limits = -5.0       # Reference: -5.0 (dof_pos_limits)

    # 4. Gait Rewards - Reference weights
    rew_scale_feet_air_time = 0.5         # Reference: 0.5 (gait, period: 0.6)
    rew_scale_feet_slide = -0.3            # Reference: -0.3 (feet_slide)
    rew_scale_leg_lift = 0.99              # Reference: 0.99 (feet_clearance, target_height: 0.05)
    rew_scale_feet_contact_forces = -0.2  # Reference: -0.2 (feet_contact_forces, threshold: 100)
    
    # Gait parameters for compute_feet_gait
    gait_offset = [0.0, 0.5]              # Reference: offset=[0.0, 0.5]
    gait_threshold = 0.55                  # Reference: threshold=0.55
    feet_clearance_std = 0.05              # Reference: std=0.05
    feet_clearance_tanh_mult = 2.0        # Reference: tanh_mult=2.0

    # 5. Contact Penalties - Reference weights
    rew_scale_undesired_contacts = -1.0   # Reference: -1.0 (undesired_contacts)
    rew_scale_joint_deviation_hip = -0.5  # Reference: -0.5 (joint_deviation_hips)
    rew_scale_joint_deviation_knee = 0.0  # Not in reference
    rew_scale_ankle_gravity = 0.0         # Not in reference

    # success / failure thresholds - Following reference configuration
    success_joint_tol = 0.05
    success_upright_cos = 0.98
    success_pitch_tol = math.radians(5.0)
    failure_pitch_angle = math.radians(45.0)
    failure_min_height = 0.10  # Reference: 0.10 (base_height termination)
    imbalance_pitch_threshold = math.radians(30.0)  # [rad] 失衡惩罚阈值（pitch角度）
    imbalance_roll_threshold = math.radians(30.0)    # [rad] 失衡惩罚阈值（roll角度）
    imbalance_height_threshold = 0.15  # Reference: 0.15 (base_height target)

    # reset sampling
    reset_noise_scale = 0.1
    orientation_noise_deg = 5.0     # 减小初始姿态噪声

    desired_root_height = 0.15       # Reference: 0.15 (base_height target)
    foot_contact_force_threshold = 100.0  # Reference: 100 (feet_contact_forces threshold)
    desired_foot_clearance = 0.05    # Reference: 0.05 (feet_clearance target_height)
    leg_lift_exploration_threshold = 0.02  # 抬腿探索奖励阈值 [m]
    single_support_height_diff = 0.03  # 单腿支撑判断：高度差阈值 [m]

    joint_target_speed = 1.0        # 目标关节速度 [rad/s]
    rew_scale_forward_distance = 0.5     # 累积前向距离奖励（G1 style: lower priority than velocity tracking）
    rew_scale_imbalance_penalty = -50.0   # 机身失衡高额惩罚（接近reset条件时）
    # 每关节最高角/线速度（来自 URDF velocity 字段；第二关节更低）
    joint_velocity_limits = (
        1.0,   # LL_joint1
        0.3,   # LL_joint2
        1.0,   # LL_joint3
        1.0,   # LL_joint4
        1.0,   # LL_joint5
        1.0,   # RL_joint1
        0.3,   # RL_joint2
        1.0,   # RL_joint3
        1.0,   # RL_joint4
        1.0,   # RL_joint5
    )

    # command profile - Following reference configuration (no curriculum)
    command_lin_vel_x_range = (-0.5, 0.5)  # Reference: ranges.lin_vel_x=(-0.5, 0.5)
    command_lin_vel_y_range = (-0.2, 0.2)  # Reference: ranges.lin_vel_y=(-0.2, 0.2)
    command_yaw_range = (-0.1, 0.1)  # Reference: ranges.ang_vel_z=(-0.1, 0.1)
    command_change_interval_s = 10.0  # Reference: resampling_time_range=(10.0, 10.0)

    # gait parameters - Following reference configuration
    gait_cycle_duration = 0.6  # Reference: period=0.6 (gait reward)
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

    # sensors - register separate contact sensors for both ankles
    contact_forces_left: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Qmini/Qmini/LL_ankle",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        debug_vis=True,
    )
    contact_forces_right: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Qmini/Qmini/RL_ankle",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        debug_vis=True,
    )

    def __post_init__(self):
        """Post initialization to set up terrain."""
        super().__post_init__()
        # Set up terrain in scene
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,  # Reference: 1.0
                dynamic_friction=1.0,  # Reference: 1.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.4, 0.2),  # Gray color for terrain
                roughness=0.8,
                metallic=0.0,
            ),
            debug_vis=False,  # 显示地形坐标系（关闭）
        )
