# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, material_handler_env_cfg, material_handler_env

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Material-Handler-v0",
    entry_point="omni.isaac.lab_tasks.manager_based.material_handling.material_handler_env:MaterialHandlerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": material_handler_env_cfg.MaterialHandlerEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.MaterialHandlerPPORunnerCfg,
    },
)

