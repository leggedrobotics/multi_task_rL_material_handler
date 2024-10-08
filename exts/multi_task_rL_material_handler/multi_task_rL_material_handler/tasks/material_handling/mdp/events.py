# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""


from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from omni.isaac.lab.utils.math import sample_uniform
import numpy as np

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

def reset_joints_within_limits_with_tool_straight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_pos_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

  
    # ranges
    range_list = [joint_pos_range.get(key, (0.0, 0.0)) for key in ["slew", "boom", "stick"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)
    
    # Slew
    slew_pos = 2*torch.pi*(rand_samples[:,0]-0.5).unsqueeze(-1)
    # Arm
    boom_pos = -(0.6*(rand_samples[:,1]-0.5)-1.1).unsqueeze(-1) # - Because orientation of URDF changed
    stick_pos = -(-boom_pos.squeeze()*(rand_samples[:,2]-0.5)+1.2).unsqueeze(-1)
    # Tool
    hanger_pos = -torch.pi/2-(boom_pos+stick_pos)+0.255096326794896 # This offset is needed to keep the tool straight
    bearing_fork_pos = torch.zeros_like(hanger_pos)
    # Overall
    joint_pos = torch.cat([slew_pos, boom_pos, stick_pos, hanger_pos, bearing_fork_pos], dim=-1)
    joint_vel = asset.data.default_joint_vel[env_ids].clone()# 0
    
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)