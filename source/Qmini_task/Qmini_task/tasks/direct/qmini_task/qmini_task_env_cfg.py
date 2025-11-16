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
    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition
    action_space = 10
    observation_space = 42  # Updated to 42 dimensions
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

    # terrain
    terrain_cfg: TerrainGeneratorCfg = ROUGH_TERRAINS_CFG

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

    # reward scales - Gait Training Rewards (following reference implementation)
    # 1. Task Rewards
    rew_scale_track_lin_vel_xy = 1.0      # track_lin_vel_xy_exp: weight=1.0
    rew_scale_track_ang_vel_z = 0.5      # track_ang_vel_z_exp: weight=0.5

    # 2. Gait Rewards
    rew_scale_feet_air_time = 2.0        # feet_air_time: weight=2.0 (核心步态奖励)
    rew_scale_feet_slide = -0.25         # feet_slide: weight=-0.25 (惩罚滑动)

    # 3. Stability Penalties
    rew_scale_lin_vel_z = -2.0          # lin_vel_z_l2: weight=-2.0 (惩罚垂直速度)
    rew_scale_ang_vel_xy = -0.05        # ang_vel_xy_l2: weight=-0.05 (惩罚俯仰/滚转)
    rew_scale_flat_orientation = -0.5    # flat_orientation_l2: weight=-0.5 (惩罚倾斜)

    # 4. Action Penalties
    rew_scale_joint_torques = -1.0e-5    # joint_torques_l2: weight=-1e-5 (惩罚力矩)
    rew_scale_action_rate = -0.01       # action_rate_l2: weight=-0.01 (惩罚动作变化)

    # 5. Contact Penalties
    rew_scale_undesired_contacts = -1.0  # undesired_contacts: weight=-1.0 (惩罚不当接触)
    rew_scale_joint_deviation_hip = -0.1  # joint_deviation_hip: weight=-0.1 (惩罚髋关节偏离)
    rew_scale_joint_deviation_knee = -0.01  # joint_deviation_knee: weight=-0.01 (惩罚膝关节偏离)

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
                static_friction=2.0,  # 增大摩擦力
                dynamic_friction=2.0,  # 增大摩擦力
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.4, 0.2),  # Gray color for terrain
                roughness=0.8,
                metallic=0.0,
            ),
            debug_vis=False,  # 显示地形坐标系（关闭）
        )
