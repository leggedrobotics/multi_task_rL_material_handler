import torch
import torch.nn as nn
import os

class MLPPress(nn.Module):
    """
    Multilayer Perceptron for regression, 2 outputs.
    """
    def __init__(self):
        super(MLPPress, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(41, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        """
        Forward pass.
        """
        return self.layers(x)

class MLPVel(nn.Module):
    """
    Multilayer Perceptron for regression.
    """
    def __init__(self):
        super(MLPVel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(41, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Forward pass.
        """
        return self.layers(x)

def load_model_weights(model, weights_path):
    """
    Load model weights from a given path.
    """
    model.load_state_dict(torch.load(weights_path))
    model.train()  # Set the model to training mode

# Set the paths for the model weights
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRESS_MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'NN_Slew', 'full_large_set_press_new.pth')
VEL_MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'NN_Slew', 'full_large_set_new.pth')

# Instantiate the models
model_press = MLPPress()
model_vel = MLPVel()

model_press.eval()
model_vel.eval()

# Load the weights and set the models to training mode
load_model_weights(model_press, PRESS_MODEL_WEIGHTS_PATH)
load_model_weights(model_vel, VEL_MODEL_WEIGHTS_PATH)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to the GPU if available
model_press.to(device)
model_vel.to(device)


# Make models accessible when importing this module
__all__ = ['model_press', 'model_vel']
