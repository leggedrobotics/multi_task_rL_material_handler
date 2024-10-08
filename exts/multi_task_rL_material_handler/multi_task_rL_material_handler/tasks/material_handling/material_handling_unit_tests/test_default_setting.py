# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to test the default setting of the material handler"""

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

    # Simulation loop
    count=0
    while simulation_app.is_running():
        # Logging previous position


        # Only actuate the desired joints
        actuation_forces = torch.zeros_like(actuation_forces)
        if count < 1000:
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos = torch.zeros_like(joint_pos)
            joint_pos[:,1] = 1
            joint_pos[:,2] = -2
            # -- apply action to the robot
            #robot.set_joint_effort_target(actuation_forces, joint_ids)
            # -- write data to sim
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

        else:
            #joint_pos[:,:] = 0.0
            robot.write_joint_state_to_sim(joint_pos[:,:2], joint_vel[:,:2], joint_ids = [0,1])
        print("Joint pos init:", joint_pos)
        print("Joint pos:", robot.data.joint_pos)
        print("Body pos", robot.data.body_pos_w[:,-1,:])
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


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
    # run the main function
    main()
    # close sim app
    simulation_app.close()