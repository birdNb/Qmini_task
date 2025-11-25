# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# Try to import pygame for gamepad support
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame not available. Gamepad control will be disabled.")
    print("[INFO] Install pygame with: pip install pygame")

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--gamepad", action="store_true", default=False, help="Enable gamepad control for velocity commands.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Qmini_task.tasks  # noqa: F401

# Import for camera control
try:
    from omni.isaac.core.utils.viewports import set_camera_view
    from pxr import Gf
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("[WARNING] Camera control not available. Install omni.isaac.core for camera tracking.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # Initialize gamepad if requested
    gamepad = None
    if args_cli.gamepad:
        if not PYGAME_AVAILABLE:
            print("[ERROR] pygame is not available. Cannot enable gamepad control.")
            print("[INFO] Install pygame with: pip install pygame")
            args_cli.gamepad = False
        else:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                gamepad = pygame.joystick.Joystick(0)
                gamepad.init()
                print(f"[INFO] Gamepad connected: {gamepad.get_name()}")
                print("[INFO] Gamepad controls:")
                print("  - Left stick Y: Forward/Backward velocity (vx)")
                print("  - Left stick X: Lateral velocity (vy)")
                print("  - Right stick X: Angular velocity (yaw)")
                print("  - Press START to reset environment")
            else:
                print("[WARNING] No gamepad detected. Gamepad control disabled.")
                args_cli.gamepad = False

    # Command velocity ranges from config
    vx_range = env_cfg.command_lin_vel_x_range if hasattr(env_cfg, 'command_lin_vel_x_range') else (-0.5, 0.5)
    vy_range = env_cfg.command_lin_vel_y_range if hasattr(env_cfg, 'command_lin_vel_y_range') else (-0.2, 0.2)
    yaw_range = env_cfg.command_yaw_range if hasattr(env_cfg, 'command_yaw_range') else (-0.1, 0.1)

    # Disable automatic command sampling if using gamepad
    if args_cli.gamepad and gamepad is not None:
        # Set command_change_interval to a very large value to disable auto-sampling
        if hasattr(env.unwrapped, '_command_change_interval'):
            env.unwrapped._command_change_interval = 1e6  # Effectively disable auto-sampling
        if hasattr(env.unwrapped, '_command_timer'):
            env.unwrapped._command_timer.zero_()  # Reset timer

    # reset environment
    obs = env.get_observations()
    timestep = 0
    last_debug_time = time.time()
    
    # Setup camera to follow robot (track first environment's robot)
    camera_follow = True
    camera_offset = np.array([3.0, 0.0, 2.0])  # Camera position offset: [x, y, z] in meters
    camera_target_offset = np.array([0.0, 0.0, 0.5])  # Look at point offset relative to robot base
    
    if camera_follow and CAMERA_AVAILABLE:
        print("[INFO] Camera tracking enabled. Camera will follow the first robot.")
        print(f"[INFO] Camera offset: {camera_offset}, Target offset: {camera_target_offset}")
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        
        # Handle gamepad input BEFORE env.step() so command is set before physics step
        if args_cli.gamepad and gamepad is not None:
            pygame.event.pump()  # Process pygame events
            
            # Read gamepad axes (normalized to [-1, 1])
            # Left stick: axes 1 (Y, inverted), 0 (X)
            # Right stick: axis 3 (X)
            left_stick_y_raw = -gamepad.get_axis(1) if gamepad.get_numaxes() > 1 else 0.0  # Forward/Backward
            left_stick_x_raw = -gamepad.get_axis(0) if gamepad.get_numaxes() > 0 else 0.0   # Lateral (inverted to fix direction)
            right_stick_x_raw = -gamepad.get_axis(3) if gamepad.get_numaxes() > 3 else 0.0  # Yaw (inverted to fix direction)
            
            # Map to velocity commands
            # Dead zone to avoid drift
            dead_zone = 0.1
            left_stick_y = 0.0 if abs(left_stick_y_raw) < dead_zone else left_stick_y_raw
            left_stick_x = 0.0 if abs(left_stick_x_raw) < dead_zone else left_stick_x_raw
            right_stick_x = 0.0 if abs(right_stick_x_raw) < dead_zone else right_stick_x_raw
            
            # Map stick values to velocity ranges
            # Scale from [-1, 1] to [min, max], centered at 0
            vx = left_stick_y * (vx_range[1] - vx_range[0]) / 2.0
            vy = left_stick_x * (vy_range[1] - vy_range[0]) / 2.0
            yaw = right_stick_x * (yaw_range[1] - yaw_range[0]) / 2.0
            
            # Clamp to ranges
            vx = max(vx_range[0], min(vx_range[1], vx))
            vy = max(vy_range[0], min(vy_range[1], vy))
            yaw = max(yaw_range[0], min(yaw_range[1], yaw))
            
            # Set command in environment using torch tensor
            if hasattr(env.unwrapped, '_command'):
                # Convert to torch tensor and set command for all environments
                device = env.unwrapped.device
                num_envs = env.unwrapped.scene.num_envs
                
                # Set command for all environments
                env.unwrapped._command[:, 0] = torch.full((num_envs,), vx, device=device, dtype=torch.float32)
                env.unwrapped._command[:, 1] = torch.full((num_envs,), vy, device=device, dtype=torch.float32)
                env.unwrapped._command[:, 2] = torch.full((num_envs,), yaw, device=device, dtype=torch.float32)
                
                # Reset command timer to prevent auto-sampling
                if hasattr(env.unwrapped, '_command_timer'):
                    env.unwrapped._command_timer.zero_()
                
                # Update command direction for visualization
                if hasattr(env.unwrapped, '_update_command_direction'):
                    env_ids = torch.arange(num_envs, device=device)
                    env.unwrapped._update_command_direction(env_ids)
                
                # Update visualization markers if available
                if hasattr(env.unwrapped, '_visualize_markers'):
                    env.unwrapped._visualize_markers()
                
                # Debug output every 0.5 seconds
                current_time = time.time()
                if current_time - last_debug_time > 0.5:
                    print(f"[GAMEPAD] vx={vx:.3f}, vy={vy:.3f}, yaw={yaw:.3f} | "
                          f"Stick: ({left_stick_x:.2f}, {left_stick_y:.2f}), RightX={right_stick_x:.2f}")
                    last_debug_time = current_time
            
            # Check for reset button (START button, typically button 7)
            if gamepad.get_numbuttons() > 7 and gamepad.get_button(7):
                print("[INFO] Reset button pressed. Resetting environment...")
                obs = env.get_observations()
                timestep = 0
        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping (command should be set before this)
            obs, _, _, _ = env.step(actions)
        
        # Update camera to follow robot
        if camera_follow and CAMERA_AVAILABLE and hasattr(env.unwrapped, 'robot'):
            try:
                # Get robot base position and orientation (first environment)
                robot = env.unwrapped.robot
                if hasattr(robot, 'data') and hasattr(robot.data, 'root_state_w'):
                    root_state = robot.data.root_state_w[0]  # First environment
                    base_pos = root_state[:3].cpu().numpy()  # Position [x, y, z]
                    base_quat = root_state[3:7].cpu().numpy()  # Quaternion [w, x, y, z]
                    
                    # Convert quaternion to rotation matrix for direction
                    # For simplicity, we'll use the robot's forward direction (assuming +X is forward)
                    # Calculate forward direction from quaternion
                    w, x, y, z = base_quat
                    # Forward vector in robot frame is [1, 0, 0], transform to world frame
                    forward_x = 1 - 2 * (y * y + z * z)
                    forward_y = 2 * (x * y + w * z)
                    forward_z = 2 * (x * z - w * y)
                    forward_dir = np.array([forward_x, forward_y, forward_z])
                    forward_dir = forward_dir / (np.linalg.norm(forward_dir) + 1e-6)
                    
                    # Calculate camera position (behind and above robot)
                    camera_pos = base_pos + camera_offset[0] * forward_dir + np.array([0, 0, camera_offset[2]])
                    # Also offset perpendicular to forward direction
                    right_dir = np.cross(forward_dir, np.array([0, 0, 1]))
                    right_dir = right_dir / (np.linalg.norm(right_dir) + 1e-6)
                    camera_pos = camera_pos + camera_offset[1] * right_dir
                    
                    # Calculate target position (robot base + offset)
                    target_pos = base_pos + camera_target_offset
                    
                    # Set camera view
                    eye = Gf.Vec3d(camera_pos[0], camera_pos[1], camera_pos[2])
                    target = Gf.Vec3d(target_pos[0], target_pos[1], target_pos[2])
                    set_camera_view(eye=eye, target=target, camera_path="/OmniverseKit_Persp")
            except Exception:
                # Silently fail if camera update fails (might happen during initialization)
                pass
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    
    # Cleanup gamepad
    if gamepad is not None:
        gamepad.quit()
    if args_cli.gamepad and PYGAME_AVAILABLE:
        pygame.joystick.quit()
        pygame.quit()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
