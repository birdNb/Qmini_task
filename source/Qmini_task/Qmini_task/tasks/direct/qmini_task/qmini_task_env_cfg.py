# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
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
        pos=(0.0, 0.0, 0.35),  # Reference: pos=(0.0, 0.0, 0.3)
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
        # Reference: Unified actuator config for all joints
        # effort_limit_sim=10, velocity_limit_sim=30.0, stiffness=40.0, damping=1.0, armature=0.01
        # Using slightly higher effort_limit for better performance while keeping other params similar
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[".*L_joint.*"],  # All joints (LL and RL)
            effort_limit_sim=25.0,  # Reference: 10, increased for better torque
            velocity_limit_sim=30.0,  # Reference: 30.0
            stiffness=40.0,  # Reference: 40.0
            damping=1.0,  # Reference: 1.0
            armature=0.01,  # Reference: 0.01
        ),
    },
)


ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.0, 0.2), num_waves=4, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.0, 0.06), noise_step=0.02, border_width=0.25
        ),
    },
)


@configclass
class QminiTaskEnvCfg(DirectRLEnvCfg):
    # env - Following reference configuration
    decimation = 4  # Reference: 4 (increased from 2)
    episode_length_s = 30.0  # Reference: 20.0 (increased from 10.0)
    # - spaces definition
    action_space = 12  # 2维相位频率 + 10维关节增量
    observation_space = 129  # 43维单帧观测 × 3帧历史堆叠 = 129维
    num_stacks = 3  # 历史帧堆叠数量
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
        "LL_joint4": -0.2,  # Knee joint target: -0.8 (Reference: -0.8)
        "LL_joint5": 0.0,
        "RL_joint1": 0.0,
        "RL_joint2": 0.0,
        "RL_joint3": 0.0,
        "RL_joint4": -0.2,  # Knee joint target: -0.8 (Reference: -0.8)
        "RL_joint5": 0.0,
    }

    # Reference joint positions for observation (standing pose reference)
    ref_joint_act = {
        "LL_joint1": 0.0,
        "LL_joint2": 0.0,
        "LL_joint3": 0.3,   # Reference standing pose
        "LL_joint4": -0.8,  # Reference standing pose
        "LL_joint5": 0.5,   # Reference standing pose
        "RL_joint1": 0.0,
        "RL_joint2": 0.0,
        "RL_joint3": 0.3,   # Reference standing pose
        "RL_joint4": -0.8,  # Reference standing pose
        "RL_joint5": 0.5,   # Reference standing pose
    }

    # Action increment ranges (for 12-dim action: 2 freq + 10 joint increments)
    act_inc_high = [3.5, 15.0]  # [频率上限(Hz), 关节增量上限(rad/s)]
    act_inc_low = [0.5, -15.0]  # [频率下限(Hz), 关节增量下限(rad/s)]

    # reward scales - Adjusted to prevent jumping and ensure reasonable reward values
    # 1. Task Rewards - Scaled down by 1000x to target ~10 reward at convergence
    rew_scale_track_lin_vel_xy = 0.0015      # Scaled down from 1.5
    rew_scale_lin_vel_z = -0.001            # Scaled down from -1.0
    rew_scale_track_ang_vel_z = 0.003       # Scaled down from 3.0
    rew_scale_vel_tracking_error = -0.002   # Scaled down from -2.0
    rew_scale_ang_vel_tracking_error = -0.001  # Scaled down from -1.0
    rew_scale_alive = 0.0005                 # Scaled down from 0.5
    rew_scale_reset_penalty = -0.001        # Scaled down from -1.0
    rew_scale_stationary_penalty = -0.005   # Scaled down from -5.0
    stationary_velocity_threshold = 0.05  # Velocity threshold below which robot is considered stationary [m/s]

    # 2. Base Stability Penalties - Scaled down by 1000x
    rew_scale_ang_vel_xy = -0.0003            # Scaled down from -0.3
    rew_scale_flat_orientation = -0.0005     # Scaled down from -0.5
    rew_scale_base_height = -0.008          # Scaled down from -8.0
    rew_scale_base_height_reward = 0.002     # Scaled down from 2.0
    base_height_reward_tolerance = 0.05    # 5cm tolerance for height reward
    rew_scale_root_pitch_roll = -0.0005      # Scaled down from -0.5

    # 3. Action Penalties - Scaled down by 1000x
    rew_scale_joint_acc = -2.5e-10         # Scaled down from -2.5e-7
    rew_scale_action_rate = -0.00005         # Scaled down from -0.05
    rew_scale_joint_torques = -1.0e-8     # Scaled down from -1.0e-5
    rew_scale_dof_pos_limits = -0.002       # Scaled down from -2.0

    # 4. Gait Rewards - Scaled down by 1000x
    rew_scale_feet_air_time = 0.0003         # Scaled down from 0.3
    rew_scale_feet_air_time_mean = 0.005     # Scaled down from 5.0
    rew_scale_target_air_time = 0.002        # Scaled down from 2.0
    rew_scale_feet_slide = -0.0002            # Scaled down from -0.2
    rew_scale_leg_lift = 0.0003              # Scaled down from 0.3
    rew_scale_feet_contact_forces = -0.00001  # Scaled down from -0.01
    rew_scale_feet_height_consistency = -0.005  # Scaled down from -5.0

    # Gait parameters for compute_feet_gait
    gait_offset = [0.0, 0.5]              # Reference: offset=[0.0, 0.5]
    gait_threshold = 0.55                  # Reference: threshold=0.55
    feet_clearance_std = 0.05              # Reference: std=0.05
    feet_clearance_tanh_mult = 2.0        # Reference: tanh_mult=2.0
    
    # Foot air time target parameters
    target_foot_air_time = 0.3            # Target foot air time: 0.3s
    air_time_tolerance = 0.1              # Tolerance for air time deviation: 0.1s
    feet_height_consistency_threshold = 0.02  # Maximum allowed height difference when both feet are in contact: 0.02m (2cm)

    # 5. Contact Penalties - Scaled down by 1000x
    rew_scale_undesired_contacts = -0.0005   # Scaled down from -0.5
    rew_scale_joint_deviation_hip = -0.0002  # Scaled down from -0.2
    rew_scale_joint_deviation_knee = -0.0005  # Scaled down from -0.5
    rew_scale_ankle_gravity = 0.0         # Not in reference

    # success / failure thresholds - Following reference configuration
    success_joint_tol = 0.05
    success_upright_cos = 0.98
    success_pitch_tol = math.radians(5.0)
    failure_pitch_angle = math.radians(45.0)
    # 增加重置高度阈值
    failure_min_height = 0.30  # Reset when root height below 0.25m

    imbalance_pitch_threshold = math.radians(30.0)  # [rad] 失衡惩罚阈值（pitch角度）
    imbalance_roll_threshold = math.radians(30.0)    # [rad] 失衡惩罚阈值（roll角度）
    imbalance_height_threshold = 0.15  # Reference: 0.15 (base_height target)

    # reset sampling
    reset_noise_scale = 0.1
    orientation_noise_deg = 5.0     # 减小初始姿态噪声

    desired_root_height = 0.43       # Base height target: 0.4m
    foot_contact_force_threshold = 100.0  # Reference: 100 (feet_contact_forces threshold)
    desired_foot_clearance = 0.09    # Target foot clearance height: 0.05m (抬腿目标高度)
    leg_lift_exploration_threshold = 0.02  # 抬腿探索奖励阈值 [m]
    single_support_height_diff = 0.03  # 单腿支撑判断：高度差阈值 [m]

    joint_target_speed = 1.0        # 目标关节速度 [rad/s]
    rew_scale_imbalance_penalty = -0.002   # Scaled down from -2.0 (target ~10 reward at convergence)
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

    # command profile - Following reference configuration
    command_lin_vel_x_range = (-0.5, 0.5)  # Reference: ranges.lin_vel_x=(-0.5, 0.5) - allow forward/backward
    command_lin_vel_y_range = (-0.2, 0.2)  # Reference: ranges.lin_vel_y=(-0.2, 0.2) - allow lateral movement
    command_yaw_range = (-0.5, 0.5)  # Increased from (-0.1, 0.1) to allow faster rotation/turning
    command_change_interval_s = 10.0  # Reference: resampling_time_range=(10.0, 10.0)

    # gait parameters - Following reference configuration
    gait_cycle_duration = 0.8  # Set gait cycle to 0.8s so both legs have 0.8s swing/stance
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

    # sensors - register height scanner and contact sensors
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Qmini/Qmini/base_link",  # Qmini使用base_link作为基座
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # 高度扫描偏移
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 较小的扫描范围
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
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

    # lights
    sky_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file="/home/bird/isaacSim/Learn/kloofendal_43d_clear_puresky_4k.hdr",
        ),
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
                diffuse_color=(0.4, 0.4, 0.4),  # Gray color for terrain
                roughness=0.8,
                metallic=0.0,
            ),
            debug_vis=False,  # 显示地形坐标系（关闭）
        )
        # Update sensor update periods (following reference configuration)
        # Height scanner updates at decimation rate (every 4 steps)
        if hasattr(self, "height_scanner"):
            self.height_scanner.update_period = self.decimation * self.sim.dt
