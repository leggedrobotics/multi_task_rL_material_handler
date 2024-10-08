# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def balancing_reward(
    env: ManagerBasedRLEnv, balancingrewardscale: float, balancingrewardcoeff: float) -> torch.Tensor:
    result = (torch.exp(((0*env.num_ppo_update)/6000. + 1)*balancingrewardcoeff*(abs(env.error_to_target[:,0])+abs(env.error_to_target[:,1])+abs(env.error_to_target[:,2])) ) -1)
    result *= balancingrewardscale
    return result

def action_reward(env: ManagerBasedRLEnv, actionrewardcoeff:float)-> torch.Tensor:
    scaled_delta_ = (env.action_manager.action-env.action_manager.prev_action)
    #action_std = torch.tensor([800, 0.25, 0.25], device = env.device)
    #quotient = scaled_delta_/action_std
    # Dot product
    return actionrewardcoeff*torch.sum(scaled_delta_ * scaled_delta_, dim=1) #dactionrewardcoeff*torch.sum(quotient * quotient, dim=1)
 
def target_reward(env: ManagerBasedRLEnv, targetrewardscale: float, targetrewardcoeff: float)-> torch.Tensor:
    result = (1. + 0.5*((env.num_ppo_update)/1500.))
    result *= torch.exp((10**((env.num_ppo_update)/3000))*targetrewardcoeff*torch.sum(env.error_to_target * env.error_to_target, dim=1))
    result *= targetrewardscale
    return result

def overshoot_reward(env: ManagerBasedRLEnv, overshootrewardscale: float, overshootrewardcoeff: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))-> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # Slew target is always 0
    overshoot = torch.zeros(env.num_envs, device = env.device)
    condition = asset.data.default_joint_pos[:,0]> 0
    # Overshoot based on condition
    overshoot[condition]  = torch.min( torch.zeros(env.num_envs, device=env.device) , asset.data.joint_pos[:,0])[condition]
    overshoot[~condition] = torch.max( torch.zeros(env.num_envs, device=env.device) , asset.data.joint_pos[:,0])[~condition]
    return overshootrewardscale*(env.num_ppo_update/2000)*(torch.exp(overshootrewardcoeff * abs(overshoot))-1)

def decoupling_reward(env: ManagerBasedRLEnv, decouplingrewardcoeff: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))->torch.tensor: # no motion of slew and arm at the same time
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_slew = asset.data.joint_vel[:,0]
    joint_vel_boom = asset.data.joint_vel[:,1]
    joint_vel_stick = asset.data.joint_vel[:,2]
    return decouplingrewardcoeff*(torch.abs(joint_vel_slew*joint_vel_boom)+torch.abs(joint_vel_slew*joint_vel_stick))

def oscillation_reward(env: ManagerBasedRLEnv, oscillation_reward_coeff, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))->torch.tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tool_vx = asset.data.body_vel_w[:,-1,0]
    tool_vy = asset.data.body_vel_w[:,-1,1]
    return  oscillation_reward_coeff*(1. + 0.25*((env.num_ppo_update)/3000.))*(torch.abs(tool_vx) + torch.abs(tool_vy))

def one_shot_reward(env: ManagerBasedRLEnv, oneshotrewardcoef:float, oneshotinit:float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))->torch.tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # Condtion
    condition = (torch.sum(env.error_to_target*env.error_to_target, dim=1) < 0.25) & (torch.abs(asset.data.joint_vel[:,0]) < 0.2)
    result = torch.zeros(env.num_envs, device=env.device)
    # If condition is satisfied, give reward
    factor = (math.atan(5e-3*(env.num_ppo_update - oneshotinit)) + torch.pi/2)/torch.pi
    result[condition] = oneshotrewardcoef*factor*torch.sum(env.action_manager.action*env.action_manager.action, dim=1)[condition]
    
    return result
    


    
    