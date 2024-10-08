# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
import numpy as np
import omni.isaac.lab_tasks.manager_based.material_handling.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.lab_assets.material_handler import MATERIAL_HANDLER_CFG  # isort:skip


##
# Scene definition
##

# Custom class to store observations for more than 1 timestep
OBSERVATION_HISTORY = mdp.ObservationHistory()
 

@configclass
class MaterialHandlerSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # Ground plane, here if needed, removed for now
    #ground = AssetBaseCfg(
    #    prim_path="/World/ground",
    #    spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
    #)

    # Material Handler
    robot: ArticulationCfg = MATERIAL_HANDLER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_velocity = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["joint_slew", "joint_boom", "joint_stick"], scale=1.0)
    
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # Slew history, history_size = 5
        joystick_action_history = ObsTerm(func=OBSERVATION_HISTORY.joystick_action, params={"history_indices": [-1, -2, -3,-4,-5]}, scale=1/850)
        slew_pos_history = ObsTerm(func=OBSERVATION_HISTORY.joint_pos_slew, params={"history_indices": [-1, -2, -3,-4,-5]}, scale=1/np.pi)
        slew_vel_history = ObsTerm(func=OBSERVATION_HISTORY.joint_vel_slew, params={"history_indices": [-1, -2, -3,-4,-5]}, scale = 1/0.3) 
        # Boom history, history_size = 3
        boom_action_history = ObsTerm(func=OBSERVATION_HISTORY.boom_action, params={"history_indices": [-1, -2, -3]}, scale = 1/0.15)
        boom_pos_history = ObsTerm(func=OBSERVATION_HISTORY.joint_pos_boom, params={"history_indices": [-1, -2, -3]}, mean = 0.8, scale = 1/0.5)
        boom_vel_history = ObsTerm(func=OBSERVATION_HISTORY.joint_vel_boom, params={"history_indices": [-1, -2, -3]}, scale = 1/0.15)
        # Stick history, history_size = 3
        stick_action_history = ObsTerm(func=OBSERVATION_HISTORY.stick_action, params={"history_indices": [-1, -2, -3]}, scale = 1/0.15)
        stick_pos_history = ObsTerm(func=OBSERVATION_HISTORY.joint_pos_stick, params={"history_indices": [-1, -2, -3]}, mean = -1.5, scale=1/0.8)
        stick_vel_history = ObsTerm(func=OBSERVATION_HISTORY.joint_vel_stick, params={"history_indices": [-1, -2, -3]}, scale =1/0.15 )
        # Tool history, histor_size = 3
        px_tool_history = ObsTerm(func=OBSERVATION_HISTORY.tool_px, params={"history_indices": [-1, -2, -3]}, scale = 4/np.pi)
        py_tool_history = ObsTerm(func=OBSERVATION_HISTORY.tool_py, params={"history_indices": [-1, -2, -3]}, scale = 4/np.pi)
        vx_tool_history = ObsTerm(func=OBSERVATION_HISTORY.tool_vx, params={"history_indices": [-1, -2, -3]}, scale = 1/0.8)
        vy_tool_history = ObsTerm(func=OBSERVATION_HISTORY.tool_vy, params={"history_indices": [-1, -2, -3]}, scale = 1/0.8)
        # Inertia and Target, no history needed 
        slew_analytical_inertia = ObsTerm(func=mdp.analytical_inertia, mean = 4e5, scale =1/2e5)
        ee_target_x = ObsTerm(func=mdp.ee_target_x, mean=9.0, scale= 1/2.0)
        ee_target_y = ObsTerm(func=mdp.ee_target_y)
        ee_target_z = ObsTerm(func=mdp.ee_target_z, mean=6.0, scale= 1/2.0)

        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    '''reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )'''

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_within_limits_with_tool_straight,
        mode="reset",
        params={
            "joint_pos_range": {
                "slew": (0, 1),
                "boom": (0, 1),
                "stick": (0,1),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # TODO: remove coeff param and change them into weights
    balancing_reward = RewTerm(func=mdp.balancing_reward, weight=1., params={"balancingrewardscale": 1.5, "balancingrewardcoeff": -0.1 })
    #action_reward = RewTerm(func=mdp.action_reward, weight=1., params={"actionrewardcoeff": -0.1, })
    target_reward = RewTerm(func=mdp.target_reward, weight=1., params={"targetrewardscale": 0.4, "targetrewardcoeff": -0.1 })
    #overshoot_reward = RewTerm(func=mdp.overshoot_reward, weight=1., params={"overshootrewardscale": 0.3, "overshootrewardcoeff": -10 })
    #decoupling_reward = RewTerm(func=mdp.decoupling_reward, weight=1., params={"decouplingrewardcoeff": 0.0 })
    #oscillation_reward = RewTerm(func=mdp.oscillation_reward, weight=1., params={"oscillation_reward_coeff": -0.2})
    one_shot_reward = RewTerm(func=mdp.one_shot_reward, weight = 1., params={"oneshotrewardcoef": -50, "oneshotinit": 2500})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # Only timeout termination for MH right now
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""
    # Curriculum is done via the reward
    pass


@configclass
class MaterialHandlerQuantitiesCfg:
    """Additional Quantities related to the Material Handler"""
    # Action Preprocessing
    action_slew_scale = 800
    action_boom_stick_scale = 0.25
    action_slew_clamp = 850
    action_boom_stick_clamp = 0.2
    # Scaling for the NN slew model
    slew_nn_means = [0., 7.78003071e+01, 7.78003071e+01, 0., 4.03212056e+05]
    slew_nn_stds = [4.10256637e+02, 9.77388791e+01, 9.77388791e+01, 3.45157123e-01, 1.50740475e+05]
    slew_nn_history = 10
    slew_nn_stride = 5
    # End Effector Target
    ee_target = [11.08,0,9.6]
    # Load on the end effector
    ee_load = 0

##
# Environment configuration
##

@configclass
class MaterialHandlerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MaterialHandlerSceneCfg = MaterialHandlerSceneCfg(num_envs=4096, env_spacing=18.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()
    # Material Hnadler related quantities 
    materialhandlerquantities: MaterialHandlerQuantitiesCfg =  MaterialHandlerQuantitiesCfg()
        


    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (12.0, 12.0, 12.0)
        # simulation settings
        self.sim.dt = 0.02

        ## Slew Model
        # Scaling
        self.nn_means = [0.1]