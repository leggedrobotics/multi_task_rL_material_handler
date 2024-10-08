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

import task_env_configs.mh_3DArm_env_cfg as mh_3DArm_env_cfg
import task_env_configs.mh_waypoint_env_cfg as mh_waypoint_env_cfg

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
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()

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

@configclass
class MaterialHandlerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MaterialHandlerSceneCfg = MaterialHandlerSceneCfg(num_envs=4096, env_spacing=18.0)
    # Basic settings
    observations_3DArm: mh_3DArm_env_cfg.ObservationsCfg = mh_3DArm_env_cfg.ObservationsCfg()
    observations_waypoint: mh_waypoint_env_cfg.ObservationsCfg = mh_waypoint_env_cfg.ObservationsCfg()
    actions_3DArm: mh_3DArm_env_cfg.ActionsCfg = mh_3DArm_env_cfg.ActionsCfg()
    actions_waypoint: mh_waypoint_env_cfg.ActionsCfg = mh_waypoint_env_cfg.ActionsCfg()
    events_3DArm: mh_3DArm_env_cfg.EventCfg = mh_3DArm_env_cfg.EventCfg()
    events_waypoint: mh_waypoint_env_cfg.EventCfg = mh_waypoint_env_cfg.EventCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards_3DArm: mh_3DArm_env_cfg.RewardsCfg = mh_3DArm_env_cfg.RewardsCfg()
    rewards_waypoint: mh_waypoint_env_cfg.RewardsCfg = mh_waypoint_env_cfg.RewardsCfg()
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

