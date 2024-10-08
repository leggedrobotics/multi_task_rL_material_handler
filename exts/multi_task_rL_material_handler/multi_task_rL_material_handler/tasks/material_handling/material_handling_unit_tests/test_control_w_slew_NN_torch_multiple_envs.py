# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Script to test how to control the material handler with the nn model for the slew in torch for multiple envs"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher
import numpy as np
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.num_envs = 2
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import matplotlib.pyplot as plt
from omni.isaac.lab.envs import ManagerBasedRLEnv

from omni.isaac.lab_tasks.manager_based.material_handling.material_handler_env import MaterialHandlerRLEnv
from omni.isaac.lab_tasks.manager_based.material_handling.material_handler_env_cfg import MaterialHandlerEnvCfg

def plot_scalar_over_time(data, label):
    
    data = torch.tensor(data)
    data_np = data.cpu().numpy()
    plt.figure()
    plt.plot(data_np, color ='g')
    plt.xlabel('steps')
    plt.ylabel(label)
    plt.legend()

def main():
    """Main function."""
    # create environment configuration
    env_cfg = MaterialHandlerEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = MaterialHandlerRLEnv(cfg=env_cfg)
    env.reset()
    # simulate physics
    count = 0
    pressureL = []
    pressureR = []
    slew_vel  = []
    while simulation_app.is_running():
        #with torch.inference_mode():
            # reset
            if count % 300 == 0 and count!= 0:
                count = 0
                #env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                plot_scalar_over_time(pressureL, 'Left pressure')
                plot_scalar_over_time(pressureR, 'Right pressure')
                plot_scalar_over_time(slew_vel, 'Slew Velocity')
                plt.show()
            # sample random actions
            actions = torch.zeros_like(env.action_manager.action)
            joystick_command_raw = np.sin(count*2*np.pi/200) * 0.1
            actions[0,0] = joystick_command_raw
            actions[0,1] = joystick_command_raw
            actions[1,0] = -joystick_command_raw
            # step the environment
            obs, rew, terminated, truncated, info = env.step(actions)
            # print current orientation of pole
            #print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1
            pressureL.append(env.press_l[0].clone())
            pressureR.append(env.press_r[0].clone())
            slew_vel.append(env.target_joint_vel[0,0].clone())

    # close the environment
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
