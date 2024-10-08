# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from omni.isaac.version import get_version

from omni.isaac.lab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.envs.manager_based_env import ManagerBasedEnv
from omni.isaac.lab_tasks.manager_based.material_handling.material_handler_shared_cfg import MaterialHandlerEnvCfg
from omni.isaac.lab.envs.common import VecEnvStepReturn
from omni.isaac.lab_tasks.manager_based.material_handling.agents.rsl_rl_cfg import MaterialHandlerPPORunnerCfg

from omni.isaac.lab_tasks.manager_based.material_handling.NN_Slew.models import model_press, model_vel

from omni.isaac.lab.managers import ActionManager, EventManager, ObservationManager

class MaterialHandlerRLEnv(ManagerBasedRLEnv):
    
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: MaterialHandlerEnvCfg
    """Configuration for the environment."""
    def __init__(self, cfg: MaterialHandlerEnvCfg, render_mode: str | None = None, **kwargs): 
        # Initialization of the buffers needed to initialize observations and rewards
        self.analytic_inertia = torch.zeros(cfg.scene.num_envs, device = cfg.sim.device)
        self.analytic_inertia_scaled = torch.zeros((cfg.scene.num_envs, 1), device = cfg.sim.device)
        self.load = torch.ones(cfg.scene.num_envs, device = cfg.sim.device)*cfg.materialhandlerquantities.ee_load
        self.ee_target = torch.tensor(cfg.materialhandlerquantities.ee_target,device=cfg.sim.device).repeat(cfg.scene.num_envs).reshape(cfg.scene.num_envs, 3)
        self.error_to_target = torch.zeros((cfg.scene.num_envs, 3), device = cfg.sim.device)
        self.pre_processed_action = torch.zeros((cfg.scene.num_envs, 3), device = cfg.sim.device)
        # Initialize RL Env
        super().__init__(cfg)

        # Initialize additional buffers and quantities
        self.num_ppo_update = 0 # For Curriculum
        self.num_steps_per_env = 3# MaterialHandlerPPORunnerCfg.num_steps_per_env

        # Slew Model
        self.history = cfg.materialhandlerquantities.slew_nn_history
        self.stride = cfg.materialhandlerquantities.slew_nn_stride
        self.nn_means = torch.tensor(cfg.materialhandlerquantities.slew_nn_means, device=self.device)  # new model
        self.nn_stds = torch.tensor(cfg.materialhandlerquantities.slew_nn_stds, device=self.device)
    
        self.target_joint_vel = torch.zeros((self.num_envs, 3), device = self.device)
        self.joystick_buff = torch.ones((self.num_envs, self.history*self.stride+1), device = self.device)*(-self.nn_means[0])/self.nn_stds[0]
        self.pressl_buff = torch.ones((self.num_envs, self.history*self.stride+1), device = self.device)*(-self.nn_means[1])/self.nn_stds[1]
        self.pressr_buff = torch.ones((self.num_envs, self.history*self.stride+1), device = self.device)*(-self.nn_means[2])/self.nn_stds[2]
        self.speed_buff = torch.ones((self.num_envs, self.history*self.stride+1), device = self.device)*(-self.nn_means[3])/self.nn_stds[3]
        
        self.press_l = torch.zeros(self.num_envs, device = self.device)
        self.press_r = torch.zeros(self.num_envs, device = self.device)

        # Action Preprocessing
        self.slew_scale = cfg.materialhandlerquantitiesPython Debugger: Current Fileon manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # -- action manager
        self.action_manager_3DArm = ActionManager(self.cfg.actions_3DArm, self)
        print("[INFO] Action Manager: ", self.action_manager_3DArm)
        self.action_manager_waypoint = ActionManager(self.cfg.actions_waypoint, self)
        print("[INFO] Action Manager: ", self.action_manager_waypoint)

        # -- observation manager
        self.observation_manager_3DArm = ObservationManager(self.cfg.observations_3DArm, self)
        print("[INFO] Observation Manager:", self.observation_manager_3DArm)
        self.observation_manager_waypoint = ObservationManager(self.cfg.observations_waypoint, self)
        print("[INFO] Observation Manager:", self.observation_manager_waypoint)

        # -- event manager
        self.event_manager_3DArm = EventManager(self.cfg.events_3DArm, self)
        print("[INFO] Event Manager: ", self.event_manager_3DArm)
        self.event_manager_waypoint = EventManager(self.cfg.events_waypoint, self)
        print("[INFO] Event Manager: ", self.event_manager_waypoint)

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager_3DArm = RewardManager(self.cfg.rewards_3DArm, self)
        print("[INFO] Reward Manager: ", self.reward_manager_3DArm)
        self.reward_manager_waypoint = RewardManager(self.cfg.rewards_waypoint, self)
        print("[INFO] Reward Manager: ", self.reward_manager_waypoint)

        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager_3DArm.available_modes:
            self.event_manager.apply(mode="startup")
        if "startup" in self.event_manager_waypoint.available_modes:
            self.event_manager.apply(mode="startup")

    def step(self, action: torch.Tensor, task: str) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        #self.action_manager.process_action(action)
        # perform physics stepping

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # Pre-physics step
            self.pre_physics_step(action,task)
            # set actions into buffers
            if task == "3DArm":
                self.action_manager_3DArm.apply_action()
            if task == "waypoint":
                self.action_manager_waypoint.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # Post-physics step
            self.post_physics_step()


        # post-step:
        self.post_step(task)
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (commdataon for all envs)
        #self.num_ppo_update = self.common_step_counter*self.num_steps_per_env
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        if task == "3DArm":
            self.reward_buf = self.reward_manager_3DArm.compute(dt=self.step_dt)
        if task == "waypoint":
            self.reward_buf = self.reward_manager_waypoint.compute(dt=self.step_dt)

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids,task)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager_3DArm.available_modes:
            self.event_manager_3DArm.apply(mode="interval", dt=self.step_dt)
        if "interval" in self.event_manager_waypoint.available_modes:
            self.event_manager_waypoint.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    
    def pre_physics_step(self, action, task: str):
        '''
            Pre Physics Step
        '''
        # Action Preprocessing
        joystick_command_scaled = torch.clamp(action[:,0]*self.slew_scale, min=-self.slew_clamp, max=self.slew_clamp).reshape(self.num_envs,1)
        target_boom_vel =  torch.clamp(action[:,1]*self.boom_stick_scale, min=-self.boom_stick_clamp, max=self.boom_stick_clamp).reshape(self.num_envs,1)
        target_stick_vel =  torch.clamp(action[:,2]*self.boom_stick_scale, min=-self.boom_stick_clamp, max=self.boom_stick_clamp).reshape(self.num_envs,1)
        self.pre_processed_action[:,:]= torch.cat((joystick_command_scaled, target_boom_vel, target_stick_vel), dim=1)

        slew_command_scaled = (joystick_command_scaled[:,0]-self.nn_means[0])/self.nn_stds[0]
        #-- NN inference
        # Buffer
        self.joystick_buff = torch.roll(self.joystick_buff, shifts=1, dims=1)
        self.joystick_buff[:,0] = slew_command_scaled
        self.pressl_buff = torch.roll(self.pressl_buff, shifts=1, dims=1)
        self.pressl_buff[:,0] = 0 # keep alignement as in training
        self.pressr_buff = torch.roll(self.pressr_buff, shifts=1, dims=1)
        self.pressr_buff[:,0] = 0 # keep alignement as in training
        self.speed_buff = torch.roll(self.speed_buff, shifts=1, dims=1)
        self.speed_buff[:,0] = 0 # keep alignement as in training
        # Pressure input
        input_press = torch.cat((self.joystick_buff[:,0:self.history*self.stride:self.stride], \
                self.pressl_buff[:,self.stride:self.history*self.stride+1:self.stride], \
                self.pressr_buff[:,self.stride:self.history*self.stride+1:self.stride], \
                self.speed_buff[:,self.stride:self.history*self.stride+1:self.stride], \
                self.analytic_inertia_scaled), dim=1)
        # Inference for pressure
        press_out_scaled = model_press(input_press.float())
        self.pressl_buff[:,0] = press_out_scaled[:,0]
        self.pressr_buff[:,0] = press_out_scaled[:,1]
        # Velocity Input
        input_speed = torch.cat((self.joystick_buff[:,0:self.history*self.stride:self.stride], \
                self.pressl_buff[:,0:self.history*self.stride:self.stride], \
                self.pressr_buff[:,0:self.history*self.stride:self.stride], \
                self.speed_buff[:,self.stride:self.history*self.stride+1:self.stride], \
                self.analytic_inertia_scaled), dim=1)

        vel_out_scaled = model_vel(input_speed.float())
        self.speed_buff[:,0] = vel_out_scaled[:,0]
        # Unscale useful quantities
        self.press_l = (press_out_scaled[:,0]*self.nn_stds[1] + self.nn_means[1])
        self.press_r = (press_out_scaled[:,1]*self.nn_stds[2] + self.nn_means[2])
        # Slew: target velocity from the NN
        self.target_joint_vel[:,0] = (vel_out_scaled[:,0]*self.nn_stds[3] + self.nn_means[3])
        # Boom and stick: Target velocity from the action and scaled
        self.target_joint_vel[:,[1]] = target_boom_vel
        self.target_joint_vel[:,[2]] = target_stick_vel
        # need process action in every control step (Filipo's model works like this)
        if task == "3DArm":
            self.action_manager_3DArm.process_action(self.target_joint_vel)
        if task == "waypoint":
            self.action_manager_waypoint.process_action(self.target_joint_vel)
    
    def post_physics_step(self):
        '''
            Post Physics Step
        '''
        # Inertia computation
        self.analytic_inertia = compute_inertia_from_angles(self.scene['robot'].data.joint_pos[:,1], self.scene['robot'].data.joint_pos[:,2], self.load, self.num_envs)
        self.analytic_inertia_scaled = ((self.analytic_inertia - self.nn_means[4])/self.nn_stds[4]).reshape(self.num_envs,1)
    
    def post_step(self,task: str):
        '''
            Post step
        '''
        # Update error to target
        self.error_to_target = self.ee_target-(self.scene['robot'].data.body_pos_w[:,4,:]-self.scene.env_origins)

    def _reset_idx(self, env_ids: Sequence[int], task: str):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            if task == "3DArm":
                self.event_manager_3DArm.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)
            if task == "waypoint":
                self.event_manager_waypoint.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)
                
        # Recompute the inertia        
        self.analytic_inertia = compute_inertia_from_angles(self.scene['robot'].data.joint_pos[:,1], self.scene['robot'].data.joint_pos[:,2], self.load, self.num_envs)
        self.analytic_inertia_scaled = ((self.analytic_inertia - self.nn_means[4])/self.nn_stds[4]).reshape(self.num_envs,1)
        # reset inertia

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0


    def draw_debug_vis(self):
        """
            Draws visualizations for debugging (slows down simulation when not headless).
        """
        #-------- Markers
        #indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        zero_orientation = torch.tensor([1,0,0,0],device=self.device).expand(self.num_envs, -1)
        # Target
        indices_target = torch.zeros(self.num_envs, device = self.device)
        marker_pos_bucket_target = self.ee_target+self.scene.env_origins
        marker_orientations_target = zero_orientation.clone()
        # enf effect
        indices_ee = torch.ones(self.num_envs, device = self.device)
        marker_pos_ee = self.scene['robot'].data.body_pos_w[:,4,:]
        marker_orientations_ee = zero_orientation.clone()

        # Concatenate everything
        marker_indices =  torch.cat((indices_target, indices_ee), dim=0)
        marker_locations = torch.cat((marker_pos_bucket_target, marker_pos_ee), dim=0)
        marker_orientations = torch.cat((marker_orientations_target, marker_orientations_ee), dim=0)
        
        # Visualization tool
        self.visualizer.visualize(marker_locations, marker_orientations, marker_indices=marker_indices)


def compute_inertia_from_angles(boomAngle, stickAngle, load, num_envs):
    
    # Inverse since URDF convention got changed
    boomAngle = -boomAngle
    stickAngle = -stickAngle

    # Inertia Formula
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
    I_cub = cuboid_Icom(base_mass, base_dim, torch.zeros((num_envs), device ="cuda")) + cuboid_Icom(cabin_mass, cabin_dim, torch.zeros((num_envs), device ="cuda")) + cuboid_Icom(arm1_mass, arm1_dim, boomAngle) + cuboid_Icom(arm2_mass, arm2_dim, boomAngle+stickAngle) 
    
    return I_com + I_cub

def cuboid_Icom(mass, dim, angle):
    temp = (dim[0]**2)*(torch.cos(angle)**2)+(dim[2]**2)*(torch.sin(angle)**2)+(dim[1]**2)
    return mass*temp/12



