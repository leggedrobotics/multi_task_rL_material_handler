# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script allows to plot the observation for a MH env"""

"""Launch Isaac Sim Simulator first."""

import argparse
import matplotlib.pyplot as plt
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.num_envs = 1
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
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
    env.reset()
    # simulate physics
    test_len = 200
    count = 0
    obs_log = torch.zeros((env.observation_space['policy'].shape[1], test_len),device=env.device)
    rewards_log = torch.zeros((len(env.reward_manager.active_terms), test_len),device=env.device)
    error_target = torch.zeros((3, test_len),device=env.device)
    while simulation_app.is_running() and count <test_len:
        with torch.inference_mode():
            actions = torch.zeros_like(env.action_manager.action)
            joystick_command_raw = np.sin(count*2*np.pi/200) * 0.8
            actions[:,0] = joystick_command_raw
            # step the environment
            obs, rew, terminated, truncated, info = env.step(actions)
            obs_log[:,count] = obs['policy'].squeeze()
            error_target[:,count] = env.error_to_target.squeeze()
            rew_count = 0
            for key in env.reward_manager.one_step_rew.keys():
                rewards_log[rew_count, count] = env.reward_manager.one_step_rew[key]
                rew_count += 1
            # print current orientation of pole
            #print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    

    slew_history = 5
    arm_history = 3

    where_boom = 3*slew_history
    where_stick = where_boom +3*arm_history
    where_tool = where_stick + 3*arm_history
    where_inertia = where_tool + 4*arm_history
    where_ee_target = where_inertia + 1

    obs_log = obs_log.cpu().numpy()
    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(obs_log[0,:], label='slew action')
    plt.plot(obs_log[2*slew_history,:], label='slew vel')
    plt.plot(obs_log[slew_history,:], label='slew pos')
    plt.plot(obs_log[where_boom,:], label='boom action')
    plt.plot(obs_log[where_boom+2*arm_history,:], label='boom vel')
    plt.plot(obs_log[where_boom+arm_history,:], label='boom pos')
    plt.plot(obs_log[where_stick,:], label='stick action')
    plt.plot(obs_log[where_stick+2*arm_history,:], label='stick vel')
    plt.plot(obs_log[where_stick+arm_history,:], label='stick pos')
    plt.title('Observations Joints Scaled')
    plt.grid()

    plt.legend()
    plt.figure()
    plt.subplot(2,1,2)
    plt.plot(obs_log[where_inertia,:], label='inertia')
    plt.plot(obs_log[where_ee_target,:], label='target x')
    plt.plot(obs_log[where_ee_target+2,:], label='target z')
    plt.plot(obs_log[where_tool+2*arm_history,:], label='toolx vel')
    plt.plot(obs_log[where_tool,:], label='toolx pos')
    plt.plot(obs_log[where_tool+3*arm_history,:], label='tooly vel')
    plt.plot(obs_log[where_tool+arm_history,:], label='tooly pos')

    



    plt.title('Observations Other Scaled')
    plt.grid()
    plt.legend()

    plt.figure()
    rewards_log = rewards_log.cpu().numpy()
    plt.subplot()
    plt.plot(rewards_log[0, :], label='balance')
#    plt.plot(rewards_log[2, :], label='target')
#    plt.plot(rewards_log[3, :], label='overhsoot')
    plt.plot(rewards_log[1, :], label='action')
#    plt.plot(rewards_log[4, :], label='coupling') 
#    plt.plot(rewards_log[5, :], label='oscillation')
#    plt.plot(rewards_log[6, :], label='oneshot')
    plt.title('Rewards')
    plt.grid()
    plt.legend()


    plt.figure()
    error_target = error_target.cpu().numpy()
    plt.subplot()
    plt.plot(error_target[0, :], label='x')
    plt.plot(error_target[1, :], label='y')
    plt.plot(error_target[2, :], label='z')
    plt.title('error_to_target')
    plt.grid()
    plt.legend()

    plt.show()

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()