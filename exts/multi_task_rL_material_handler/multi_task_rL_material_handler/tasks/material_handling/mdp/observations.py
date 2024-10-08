

from __future__ import annotations


import torch
from omni.isaac.lab.envs import ManagerBasedEnv
import omni.isaac.lab_tasks.manager_based.material_handling.mdp as mdp 
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation



# Define all observations of the MDP and put them in a history class later
# Not just using mdp.joint_vel etc. since different joint need different history size 


#-- Slew
def joystick_action(env: ManagerBasedEnv)-> torch.Tensor:
    
    return env.pre_processed_action[:,[0]]

def joint_pos_slew(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, [0]]

def joint_vel_slew(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))-> torch.Tensor:
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, [0]]

#-- Boom
def boom_action(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))-> torch.Tensor:

    return env.pre_processed_action[:,[1]]

def joint_pos_boom(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, [1]]

def joint_vel_boom(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))-> torch.Tensor:
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, [1]]

#-- Stick
def stick_action(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))-> torch.Tensor:

    return env.pre_processed_action[:,[2]]

def joint_pos_stick(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, [2]]

def joint_vel_stick(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))-> torch.Tensor:
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, [2]]

#-- Tool
def tool_px(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The body x position of the last body, i.e. the tool.

    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    return asset.data.joint_pos[:,[4]]#(asset.data.body_pos_w[:,-1,:] - env.scene.env_origins)[:,[0]]

def tool_py(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The body y position of the last body, i.e. the tool.

    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return -asset.data.joint_pos[:,[3]]-torch.pi/2-asset.data.joint_pos[:,[1]]-asset.data.joint_pos[:,[2]]# (asset.data.body_pos_w[:,-1,:] - env.scene.env_origins)[:,[1]]

def tool_vx(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The body x velocity of the last body, i.e. the tool.

    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:,[4]]


def tool_vy(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The body y velocity of the last body, i.e. the tool.

    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return -(asset.data.joint_vel[:,[1]]+asset.data.joint_vel[:,[2]]+asset.data.joint_vel[:,[3]])

def analytical_inertia(env: ManagerBasedEnv):
    """ Analytical inertia of the slew computed in the env
    
    """
    return env.analytic_inertia.reshape(env.num_envs,1)

def ee_target_x(env: ManagerBasedEnv):
    """ End effector Target computed in the env    
    """
    return env.ee_target[:,[0]]

def ee_target_y(env: ManagerBasedEnv):
    """ End effector Target computed in the env    
    """
    return env.ee_target[:,[1]]

def ee_target_z(env: ManagerBasedEnv):
    """ End effector Target computed in the env    
    """
    return env.ee_target[:,[2]]


"""
History of information - imitate Recurrency for certain observations
"""
class ObservationHistory:
    def __init__(self) -> None:
        """Initialize the buffers for the history of observations.
        NOTE: the histroy includes past information excluding the latest one!
        NOTE: The buffer is updated with step frequency (e.g. 50Hz) - only history group is updated with 200Hz!
        """
        self.buffers = {} # buffer container data and last update step

    def _reset(self, env: ManagerBasedEnv, buffer_names: list = None):
        """Reset the buffers for terminated episodes.

        Args:
            env: The environment object.
        """
        # Initialize & find terminated episodes
        try:
            terminated_mask = env.termination_manager.dones
        except AttributeError:
            terminated_mask = torch.zeros((env.num_envs), dtype=int).to(env.device)
        for key in buffer_names:
            # Initialize buffer if empty
            if self.buffers[key]['data'].shape[0] == 0:
                self.buffers[key]['data'] = torch.zeros((env.num_envs, *list(self.buffers[key].shape[1:]))).to(env.device)
            # Reset buffer for terminated episodes
            self.buffers[key]['data'][terminated_mask, :, :] = 0.0
    
    def _process(self, env: ManagerBasedEnv, buffer_name: str, history_indices: list[int], data: torch.Tensor, asset_cfg: SceneEntityCfg):
        """Update the bufers and return new buffer.
        Args:
            env: The environment object.
            buffer_name: The name of the buffer.
            history_indices: The history indices. -1 is the most recent entry in the history.
            data: The data to be stored in the buffer.
        """   
        # Extract the history of the data
        history_idx = torch.tensor(history_indices).to(env.device) # history_idx-1 to not include the current step, here we want it though

        # Extract env step
        try:
            env_step = env.common_step_counter
        except AttributeError:
            env_step = 0
        
        # Add new buffer if fist call
        if buffer_name not in self.buffers:
            buffer_length = max(abs(index) for index in history_indices) + 2 # +1 for the current step
            self.buffers[buffer_name] = {}
            self.buffers[buffer_name]['data'] = torch.zeros((env.num_envs, buffer_length, data.shape[1])).to(env.device)
            self.buffers[buffer_name]['last_update_step'] = env_step
            
        # Check if buffer is already updated
        if not self.buffers[buffer_name]['last_update_step'] == env_step:
            # Reset buffer for terminated episodes
            self._reset(env, [buffer_name])
            # Updates buffer
            self.buffers[buffer_name]['data'] = self.buffers[buffer_name]['data'].roll(shifts=-1, dims=1)
            self.buffers[buffer_name]['data'][:, -1, :] = data
            self.buffers[buffer_name]['last_update_step'] = env_step

        # Extract the history of the data
        obs_history = self.buffers[buffer_name]['data'][:, history_idx, :]
        return obs_history[:,:,asset_cfg.joint_ids].reshape(env.num_envs, -1) 

    def joystick_action(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of joystick action.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "joystick_action"
        data = mdp.joystick_action(env)
        return self._process(env, name, history_indices, data, asset_cfg)
    
    def joint_pos_slew(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of slew positions.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "joint_pos_slew"
        data = mdp.joint_pos_slew(env)
        return self._process(env, name, history_indices, data, asset_cfg)

    def joint_vel_slew(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of slew velocities.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "joint_vel_slew"
        data = mdp.joint_vel_slew(env)
        return self._process(env, name, history_indices, data, asset_cfg)
    
    def boom_action(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of boom action.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "boom_action"
        data = mdp.boom_action(env)
        return self._process(env, name, history_indices, data, asset_cfg)
    
    def joint_pos_boom(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of boom position.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "joint_pos_boom"
        data = mdp.joint_pos_boom(env)
        return self._process(env, name, history_indices, data, asset_cfg)
    
    def joint_vel_boom(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of boom velocity.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "joint_vel_boom"
        data = mdp.joint_vel_boom(env)
        return self._process(env, name, history_indices, data, asset_cfg)
    

    def stick_action(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of stick action.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "stick_action"
        data = mdp.stick_action(env)
        return self._process(env, name, history_indices, data, asset_cfg)
    
    def joint_pos_stick(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of stick position.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "joint_pos_stick"
        data = mdp.joint_pos_stick(env)
        return self._process(env, name, history_indices, data, asset_cfg)
    
    def joint_vel_stick(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of stick velocity.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "joint_vel_stick"
        data = mdp.joint_vel_stick(env)
        return self._process(env, name, history_indices, data, asset_cfg)
    
    def tool_px(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of tool x position.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "tool_px"
        data = mdp.tool_px(env)
        return self._process(env, name, history_indices, data, asset_cfg)
    
    def tool_py(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of tool y position.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "tool_py"
        data = mdp.tool_py(env)
        return self._process(env, name, history_indices, data, asset_cfg)
    
    def tool_vx(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of tool x velocity.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "tool_vx"
        data = mdp.tool_vx(env)
        return self._process(env, name, history_indices, data, asset_cfg)
    
    def tool_vy(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]):
        """Get the history of tool y velocity.
        
        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "tool_vy"
        data = mdp.tool_vy(env)
        return self._process(env, name, history_indices, data, asset_cfg)
    
    
