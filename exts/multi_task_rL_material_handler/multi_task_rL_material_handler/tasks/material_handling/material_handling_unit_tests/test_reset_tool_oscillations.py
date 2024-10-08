# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to test the oscillation at reset"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher
import matplotlib.pyplot as plt
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.num_envs = 10
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv

from omni.isaac.lab_tasks.manager_based.material_handling.material_handler_env import MaterialHandlerRLEnv
from omni.isaac.lab_tasks.manager_based.material_handling.material_handler_env_cfg import MaterialHandlerEnvCfg

def main():
    """Main function."""
    # create environment configuration
    env_cfg = MaterialHandlerEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = MaterialHandlerRLEnv(cfg=env_cfg)
    asset = env.scene['robot']
    # simulate physics
    count = 0
    test_len = 50
    vel_norms = torch.zeros((env.num_envs, test_len), device = env.device)
    
    while count < test_len:
        with torch.inference_mode():
            # reset
            env.reset()
            print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.zeros_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            #print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            vel_norms[:,count] = torch.norm(asset.data.body_vel_w[:,-1], dim=-1)
            count += 1

            print(-asset.data.joint_pos[:,[3]]-torch.pi/2-asset.data.joint_pos[:,[1]]-asset.data.joint_pos[:,[2]])

            vel_norms_cpu = vel_norms.cpu().numpy()

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10), constrained_layout=True)

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Plot histograms for each environment
    for i in range(env.num_envs):
        axes[i].hist(vel_norms_cpu[i], bins=20, alpha=0.75)
        axes[i].set_title(f'Env {i+1}')
        axes[i].set_xlabel('Velocity Norm')
        axes[i].set_ylabel('Frequency')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
