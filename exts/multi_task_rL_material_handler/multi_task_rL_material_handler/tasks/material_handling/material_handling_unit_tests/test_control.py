# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to test how to control the material handler without NN model for the slew"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script is meant to ")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.controllers.inverse_dynamics import InverseDynamicsController, InverseDynamicsControllerCfg
from omni.isaac.lab_tasks.manager_based.material_handling.material_handler_env_cfg import MaterialHandlerSceneCfg

from omni.isaac.lab_assets.material_handler import MATERIAL_HANDLER_CFG  # isort:skip
import torch
import numpy as np

import matplotlib.pyplot as plt

joint_vel_limits = np.zeros((5, 2))
joint_vel_limits[0] = np.array([-1, 1])
joint_vel_limits[1] = np.array([-1, 1])
joint_vel_limits[2] = np.array([-1, 1])
joint_vel_limits[3] = np.array([-1, 1])
joint_vel_limits[4] = np.array([-1, 1])
joint_efforts_limits = np.zeros((5, 2))
joint_efforts_limits[0] = np.array([-2e10, 2e10])
joint_efforts_limits[1] = np.array([-2e10, 2e10])
joint_efforts_limits[2] = np.array([-2e10, 2e10])
joint_efforts_limits[3] = np.array([-2e10, 2e10])
joint_efforts_limits[4] = np.array([-2e10, 2e10])

# Config
INV_DYN_CFG = InverseDynamicsControllerCfg(
    command_type="vel",
    k_p=[0, 0, 0, 0, 0],
    k_d=[0, 15, 12, 0, 0],
    dof_limits= joint_vel_limits,
    dof_efforts_limits= joint_efforts_limits,
)


def main():

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cuda")
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([12, 12, 12], [0.0, 0.0, 0.0])

    #-- Spawn things into stage, use Scence from Cfg
    scene_cfg = MaterialHandlerSceneCfg(num_envs = 1, env_spacing = 10)
    scene = InteractiveScene(scene_cfg)
    robot = scene['robot']
    #robot = Articulation(cfg=robot_cfg.replace(prim_path="/World/Robot"))

    sim.reset()
    # Declare utils
    joint_ids, _ = robot.find_joints(joint_names)
    sim_dt = sim.get_physics_dt()
    actuation_forces = robot.root_physx_view.get_dof_actuation_forces()
    # Declare Controller
    # Set up control
    inverse_dyn_controller = InverseDynamicsController(INV_DYN_CFG, robot.root_physx_view.count, robot.device)

    count = 0
    # Logging data
    commanded_vel = []
    achieved_sim_vel = []
    achieved_manual_vel = []
    torques = []
    # Simulation loop
    while simulation_app.is_running():
        # Logging previous position
        prev_joint_pos = robot.data.joint_pos.clone()
        # Reset
        if count % 500 == 0:  
            # reset counter
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            #root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            #scene.reset()
            print("[INFO]: Resetting robot state...")
            if count != 0:
                plot_trajectories(commanded_vel, achieved_sim_vel)
                plot_joint_data(torques)
        # Apply random action
        target_vel = torch.zeros(robot.root_physx_view.count, robot.num_joints,device = robot.device)
        if count >= 0:
            target_vel[:, :] = 0
            for i in range(len(joint_to_tune)):
                target_vel[:, joint_dict[joint_to_tune[i]]] = np.sin((count*sim_dt - 100 * sim_dt) * 0.25* 2 * np.pi) * 0.1
        inverse_dyn_controller.set_command(target_vel)
        # -- generate random joint efforts
        #efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- Inverse Dynamics control to track joint velocity
        masses = robot.root_physx_view.get_masses().to(robot.device)[:,1:]# 1 bc of fixed base
        gravity = torch.tensor([0, 0, -9.81], device=robot.device)
        gravity_expanded = gravity.view(1, 3)
        masses_expanded = masses.view(robot.num_bodies-1, 1)
        jacobians_reduced = robot.root_physx_view.get_jacobians()[:,:,:,:]
        gravity_forces = masses_expanded * gravity_expanded
        jac_lin = jacobians_reduced[:, :, 0:3, :].to(robot.device)
        jac_rot = jacobians_reduced[:, :, 3:6, :].to(robot.device)
        jac_lin_T = jac_lin.transpose(-2, -1)
        jac_rot_T = jac_rot.transpose(-2, -1)
        inertias = robot.root_physx_view.get_inertias().to(robot.device)
        inertias = torch.reshape(inertias, (robot.num_bodies,3,3))[1:,:,:]# 1 bc of fixed base
        mass_matrix_manual = torch.sum(jac_lin_T.matmul(masses.view(-1, 1, 1) *jac_lin)
                                                + jac_rot_T.matmul(inertias.matmul(jac_rot))
                                                , dim=1, 
                                                )[:, -robot.num_joints :, -robot.num_joints :]
        mass_matrix_sim = robot.root_physx_view.get_mass_matrices().shape
        gravity_genco_tau = torch.sum(
                        torch.matmul(
                            jac_lin_T[:, :, :, :],
                            gravity_forces[ :, :].unsqueeze(-1),
                        ),
                        dim=1,
                    )
        gravity_torques_manual = gravity_genco_tau.squeeze()
        joint_vel = robot.data.joint_vel.clone()
        joint_ids_to_actuate, _ = robot.find_joints(joint_to_actuate)
        id_torques = inverse_dyn_controller.compute(
                        robot.data.joint_pos,
                        joint_vel,
                        mass_matrix_manual,
                        gravity_torques_manual[:],
                        robot.root_physx_view.get_coriolis_and_centrifugal_forces()
                    )
        # Only actuate the desired joints
        #joint_ids_to_actuate, _ = robot.find_joints(joint_to_actuate)
        actuation_forces = torch.zeros_like(actuation_forces)
        actuation_forces[:,joint_ids_to_actuate] = (id_torques)[:, joint_ids_to_actuate]

        # -- apply action to the robot
        robot.set_joint_velocity_target(target_vel[0,0], joint_ids=[0])
        robot.set_joint_effort_target((actuation_forces[0,[1,2]]), joint_ids=[1,2])
        
        #robot.root_physx_view.set_dof_velocities(target_vel[:, joint_ids], indices=torch.tensor([0,1,2], device=sim.device))
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)

        # Logging
        commanded_vel.append(target_vel[:, :].clone())
        achieved_sim_vel.append(robot.data.joint_vel.clone()[:,:])
        manual_joint_vel = (robot.data.joint_pos[:, :]-prev_joint_pos[:, :])/sim_dt
        achieved_manual_vel.append(manual_joint_vel[:,:])
        torques.append(robot.root_physx_view.get_dof_actuation_forces().clone())

def plot_trajectories(commanded_velocities, achieved_velocities):
    """
    commanded_velocities: tensor of shape (num_steps, num_joints)
    achieved_velocities: tensor of shape (num_steps, num_joints)
    """
    commanded_velocities = torch.cat(commanded_velocities, dim=0)
    achieved_velocities = torch.cat(achieved_velocities, dim=0)
    
    fig, axs = plt.subplots(len(joint_to_actuate), 1, figsize=(10, 10))
    for i in range(len(joint_to_actuate)):
        axs[i].plot(commanded_velocities.cpu().numpy()[:, i], label="Commanded")
        axs[i].plot(achieved_velocities.cpu().numpy()[:, i], label="Achieved")
        axs[i].set_ylim([-1, 1])
        axs[i].legend()
    plt.show()


def plot_joint_data(data):
    """
    commanded_velocities: tensor of shape (num_steps, num_joints)
    achieved_velocities: tensor of shape (num_steps, num_joints)
    """
    data = torch.cat(data, dim=0)
    
    fig, axs = plt.subplots(len(joint_to_actuate), 1, figsize=(10, 10))
    for i in range(len(joint_to_actuate)):
        axs[i].plot(data.cpu().numpy()[:, i], label=f"Torqu_{i}")
        axs[i].legend()
    plt.show()

if __name__ == "__main__":
    # For Ref
    joint_names = ["joint_slew", "joint_boom", "joint_stick", "joint_hanger", "joint_bearingfork"]
    joint_dict = {
        "joint_slew": 0,
        "joint_boom": 1,
        "joint_stick": 2,
        "joint_hanger": 3,
        "joint_bearingfork": 4,
    }
    # Joint to actuate
    joint_to_actuate = ["joint_slew", "joint_boom", "joint_stick"]
    # Joint to tune
    joint_to_tune = ["joint_slew", "joint_boom", "joint_stick"]
    # run the main function
    main()
    # close sim app
    simulation_app.close()