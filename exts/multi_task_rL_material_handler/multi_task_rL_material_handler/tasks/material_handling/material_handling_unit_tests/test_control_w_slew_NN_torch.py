# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Script to test how to control the material handler with the nn model for the slew in torch"""

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
args_cli.num_envs = 1

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
from omni.isaac.lab_tasks.manager_based.material_handling.NN_Slew.models import model_press, model_vel
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
    k_d=[0, 15, 20, 0, 0],
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
    scene_cfg = MaterialHandlerSceneCfg(num_envs = args_cli.num_envs, env_spacing = 10)
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
    joint_poses = []
    achieved_sim_vel = []
    body_vel = []
    achieved_manual_vel = []
    torques = []

    machine_inertias = []
    scaled_actions = []
    raw_actions = []
    slew_vel = []
    pressureL = []
    pressureR = []
    slew_vel = []

    sim_joint_vel_comp = []


    # Order is joy, pL, pR, vel, inertia
    means = torch.tensor([0., 7.78003071e+01, 7.78003071e+01, 0., 4.03212056e+05], device=robot.device)  # new model
    stds = torch.tensor([4.10256637e+02, 9.77388791e+01, 9.77388791e+01, 3.45157123e-01, 1.50740475e+05], device=robot.device)
    
    history = 10
    stride = 5

    inertia = means[4]
    press_l = 0.
    press_r = 0.
    speed = torch.zeros((scene.num_envs, 5), device = scene.device)  # stores slew, boom, stick, tooly, toolx
    position = np.zeros(5)
    position[1] = -0.8
    position[2] = 1.5
    old_speed = np.zeros(5)  # needed to simulate the pendulum

    joystick_buff = torch.ones((scene.num_envs, history*stride+1), device = robot.device)*(-means[0])/stds[0]
    pressl_buff = torch.ones((scene.num_envs, history*stride+1), device = robot.device)*(-means[1])/stds[1]
    pressr_buff = torch.ones((scene.num_envs, history*stride+1), device = robot.device)*(-means[2])/stds[2]
    speed_buff = torch.ones((scene.num_envs, history*stride+1), device = robot.device)*(-means[3])/stds[3]

    prev_joint_pos = robot.data.joint_pos.clone()

    
    
    # Simulation loop
    while simulation_app.is_running():
        # Logging previous position
       
        # Reset
        if count % 300 == 0:  
            # reset counter
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            
            #root_state[:, :3] += scene.env_origins
            
            # set joint positions with some noise
            #joint_pos += torch.rand_like(joint_pos) * 0.1
            
            # clear internal buffers
            #scene.reset()
            print("[INFO]: Resetting robot state...")
            if count != 0:
                #plot_trajectories(commanded_vel, achieved_sim_vel)
                #plot_vel_comp(achieved_manual_vel, sim_joint_vel_comp)
                #plot_joint_data(achieved_sim_vel, label="Joint_vel")
                #plot_joint_data(joint_poses, "joint_pos")
                #plot_joint_data(achieved_manual_vel, "manual_joint_vel")
                #plot_body_data(body_vel, label="body_vel")
                #plt.show()
                #plot_scalar_over_time(scaled_actions, 'Scaled actions')
                #plot_scalar_over_time(raw_actions, 'Raw actions')
                #plot_scalar_over_time(machine_inertias, 'Inertia')
                plot_scalar_over_time(pressureL, 'Left pressure')
                plot_scalar_over_time(pressureR, 'Right pressure')
                plot_scalar_over_time(slew_vel, 'Slew Velocity')
                plt.show()

            else:

                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                root_state = robot.data.default_root_state.clone()
                robot.write_root_state_to_sim(root_state)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
        # Action for Boom and Stick
        target_vel = torch.zeros(robot.root_physx_view.count, robot.num_joints,device = robot.device)
        if count >= 0:
            target_vel[:, :] = 0
            #for i in range(len(joint_inverse_dyn)):
            #    target_vel[:, joint_dict[joint_inverse_dyn[i]]] = np.sin((count*sim_dt - 100 * sim_dt) * 0.25* 2 * np.pi) * 0.1
        
        
        #-------- Action for slew
        joystick_command_raw = np.sin(count*2*np.pi/200) * 0.8 
        joystick_command = joystick_command_raw*800
        
        for i in range(5):
            # Shift all values to the right
            slew = (joystick_command- means[0])/stds[0]
            
            joystick_buff = torch.roll(joystick_buff, shifts=1, dims=1)
            joystick_buff[:,0] = slew
            pressl_buff = torch.roll(pressl_buff, shifts=1, dims=1)
            pressl_buff[:,0] = 0
            pressr_buff = torch.roll(pressr_buff, shifts=1, dims=1)
            pressr_buff[:,0] = 0
            speed_buff = torch.roll(speed_buff, shifts=1, dims=1)
            speed_buff[:,0] = 0


            machine_inertia = compute_inertia_from_angles(robot.data.joint_pos[:,1], robot.data.joint_pos[:,2], 0, scene.num_envs)
            #machine_inertia = torch.tensor([449225], device =robot.device).reshape(1,1)
            inertia = (machine_inertia - means[4])/stds[4]
            inertia.reshape(scene.num_envs,1)
            input_press = torch.cat((joystick_buff[:,0:history*stride:stride], \
                pressl_buff[:,stride:history*stride+1:stride], \
                pressr_buff[:,stride:history*stride+1:stride], \
                speed_buff[:,stride:history*stride+1:stride], \
                inertia), dim=1)
            
            # Inference for pressure
            press_out = model_press(input_press.float())


            pressl_buff[:,0] = press_out[:,0]
            pressr_buff[:,0] = press_out[:,1]

            input_speed = torch.cat((joystick_buff[:,0:history*stride:stride], \
                pressl_buff[:,0:history*stride:stride], \
                pressr_buff[:,0:history*stride:stride], \
                speed_buff[:,stride:history*stride+1:stride], \
                inertia), dim=1)

            vel_out = model_vel(input_speed.float())
            speed_buff[:,0] = vel_out[:,0]

            press_l = (press_out[:,0]*stds[1] + means[1])
            press_r = (press_out[:,1]*stds[2] + means[2])
            #old_speed[0] = speed[0]
            speed[:,0] = (vel_out[:,0]*stds[3] + means[3])
            #position[0] = (position[0] + speed[0]*0.02)

            # Inference for Velocity
            target_vel[:, joint_dict[joint_joystick[0]]] = speed[:,0]
            
            # Set command for boom and stick
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
        
            # Perfect tracking for all, adjust dampings!!
            robot.set_joint_velocity_target(target_vel[0,:3], joint_ids=[0,1,2])
            # Trackin +ID
            #robot.set_joint_velocity_target(target_vel[0,0], joint_ids=[0])
            #robot.set_joint_effort_target((actuation_forces[0,[1,2]]), joint_ids=[1,2])
            
            #robot.root_physx_view.set_dof_velocities(target_vel[:, joint_ids], indices=torch.tensor([0,1,2], device=sim.device))
            # -- write data to sim
            scene.write_data_to_sim()
            # Perform step
            sim.step()
            # Increment counter
            
            # Update buffers
            scene.update(sim_dt)


            sim_joint_vel_comp.append(robot.data.joint_vel[:, :].clone())
            manual_joint_vel = (robot.data.joint_pos[:, :]-prev_joint_pos[:, :])/sim_dt
            achieved_manual_vel.append(manual_joint_vel[:,:])
            prev_joint_pos = robot.data.joint_pos.clone()
        
        count += 1

        # Logging
        commanded_vel.append(target_vel[:, :].clone())
        achieved_sim_vel.append(robot.data.joint_vel.clone()[:,:])
        body_vel.append(robot.data.body_vel_w.clone()[:,:])
        #manual_joint_vel = (robot.data.joint_pos[:, :]-prev_joint_pos[:, :])/sim_dt
        #achieved_manual_vel.append(manual_joint_vel[:,:])
        torques.append(robot.root_physx_view.get_dof_actuation_forces().clone())

        machine_inertias.append(float(machine_inertia))
        scaled_actions.append(joystick_command)
        raw_actions.append(joystick_command_raw)
        slew_vel.append(speed[0,0].clone())
        pressureL.append(press_l.clone())
        pressureR.append(press_r.clone())
        joint_poses.append(-robot.data.joint_pos[:, :])

def plot_scalar_over_time(data, label):
    data = torch.tensor(data)
    data_np = data.cpu().numpy()
    plt.figure()
    plt.plot(data, color ='g')
    plt.xlabel('steps')
    plt.ylabel(label)
    plt.legend()
    

def plot_trajectories(commanded_velocities, achieved_velocities):
    """
    commanded_velocities: tensor of shape (num_steps, num_joints)
    achieved_velocities: tensor of shape (num_steps, num_joints)
    """
    commanded_velocities = torch.cat(commanded_velocities, dim=0)
    achieved_velocities = torch.cat(achieved_velocities, dim=0)
    
    fig, axs = plt.subplots(len(joint_to_actuate), 1, figsize=(10, 10))
    for i in range(len(joint_to_actuate)):
        axs[i].plot(commanded_velocities.detach().cpu().numpy()[:, i], label="Commanded")
        axs[i].plot(achieved_velocities.cpu().numpy()[:, i], label="Achieved")
        axs[i].set_ylim([-1, 1])
        axs[i].legend()
    plt.show()

def plot_vel_comp(commanded_velocities, achieved_velocities):
    """
    commanded_velocities: tensor of shape (num_steps, num_joints)
    achieved_velocities: tensor of shape (num_steps, num_joints)
    """
    commanded_velocities = torch.cat(commanded_velocities, dim=0)
    achieved_velocities = torch.cat(achieved_velocities, dim=0)
    
    fig, axs = plt.subplots(len(joint_names), 1, figsize=(10, 10))
    for i in range(len(joint_names)):
        axs[i].plot(commanded_velocities.detach().cpu().numpy()[5:, i], label="Manual")
        axs[i].plot(achieved_velocities.cpu().numpy()[5:, i], label="Sim")
        axs[i].set_ylim([-1, 1])
        axs[i].legend()
    plt.show()


def plot_joint_data(data, label):
    """
    commanded_velocities: tensor of shape (num_steps, num_joints)
    achieved_velocities: tensor of shape (num_steps, num_joints)
    """
    data = torch.cat(data, dim=0)
    
    fig, axs = plt.subplots(len(joint_names), 1, figsize=(10, 10))
    for i in range(len(joint_names)):
        axs[i].plot(data.cpu().numpy()[:, i], label=f"{label}_{i}")
        axs[i].legend()
    plt.show()

def plot_body_data(data, label):
    """
    commanded_velocities: tensor of shape (num_steps, num_joints)
    achieved_velocities: tensor of shape (num_steps, num_joints)
    """
    data = torch.cat(data, dim=0)
    
    fig, axs = plt.subplots(6, 1, figsize=(10, 10))
    for i in range(6):
        axs[i].plot(data.cpu().numpy()[:, i], label=f"{label}_{i}")
        axs[i].legend()
    plt.show()


    

def compute_inertia_from_angles(boomAngle, stickAngle, load, num_envs):
    
    boomAngle = -boomAngle
    stickAngle = -stickAngle

    base_mass = 9710+6007
    base_dim = torch.tensor([3.2, 3.0, 1.5],device = "cuda")

    cabin_mass = 1055+898
    cabin_dim = torch.tensor([1.8, 1.0, 1.7],device = "cuda")
    

    arm1_mass = 3848+(154+103+176+117)*2
    arm1_dim = torch.tensor([10.0, 1.0, 1.0],device = "cuda")

    arm2_mass = 1745
    arm2_dim = torch.tensor([7.0, 1.0, 0.5],device = "cuda")

    gripper_mass = 1720 + load

    boomAngle = - boomAngle
    stickAngle = np.pi - stickAngle
    # Inertia computatioon
    I_com = base_mass*(base_dim[0]/2)**2 + cabin_mass*(cabin_dim[0]/2)**2 + arm1_mass*(arm1_dim[0]/2*torch.cos(boomAngle))**2 + arm2_mass*(arm1_dim[0]*torch.cos(boomAngle) - arm2_dim[0]/2*torch.cos(boomAngle+stickAngle))**2 + gripper_mass*(arm1_dim[0]*torch.cos(boomAngle) - arm2_dim[0]*torch.cos(boomAngle+stickAngle))**2
    I_cub = cuboid_Icom(base_mass, base_dim, torch.zeros((num_envs,1), device ="cuda")) + cuboid_Icom(cabin_mass, cabin_dim, torch.zeros((num_envs,1), device ="cuda")) + cuboid_Icom(arm1_mass, arm1_dim, boomAngle) + cuboid_Icom(arm2_mass, arm2_dim, boomAngle+stickAngle); 
    
    return I_com + I_cub



def cuboid_Icom(mass, dim, angle):
    temp = (dim[0]**2)*(torch.cos(angle)**2)+(dim[2]**2)*(torch.sin(angle)**2)+(dim[1]**2)
    return mass*temp/12







def scale_input_vel():
    pass



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
    # Joint to control with joystick command
    joint_joystick = ["joint_slew"]
    # Joint to perform inverse dynamics on
    joint_inverse_dyn = ["joint_boom", "joint_stick"]
    # run the main function
    main()
    # close sim app
    simulation_app.close()