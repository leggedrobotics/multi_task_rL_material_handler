import torch
import torch.nn as nn
import os

import argparse
from omni.isaac.lab.app import AppLauncher

""" Test the slew NN """


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
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from omni.isaac.lab_tasks.manager_based.material_handling.NN_Slew.models import model_press, model_vel

def main():
    """
    Main function to load models and perform inference on zero input.
    """

    zero_input = torch.zeros(41)

    output_press = model_press(zero_input)
    print("Pressure output for zero input: ", output_press)

    output_vel = model_vel(zero_input)
    print("Velocity output for zero input: ", output_vel)

if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
