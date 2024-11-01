"""Pipeline to train, test, and analyze model with command-line arguments."""
import os
import torch
import random
import numpy as np
import argparse

from src.task import MorrisWaterMaze
from src.models import LSTM
from src.train import train
from src.test import test
from src.analysis import plot_error_reward, neuron_activation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Pipeline for training, testing, and analyzing a model.")
parser.add_argument('--seed', type=int, default=6942069, help='Seed for reproducibility')
parser.add_argument('--save_loc', type=str, required=True, help='Location to save the outputs')
parser.add_argument('--hidden_neurons', type=int, default=500, help='Number of hidden neurons in the model')
parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')

args = parser.parse_args()

# Set the seed
set_seed(args.seed)

# Create and set up the environment
enviro = MorrisWaterMaze(args.seed)
enviro.ReAssignReward((10, 10))

# Create the model
model = LSTM(
    49,  # Input neurons
    args.hidden_neurons,  # Hidden neurons
    3  # Output neurons
)

# Train the model
rewardHistory, errorHistory = train(
    enviro,
    model,
    args.epochs,  # Epochs
    25,  # Trials
    100,  # Steps allowed per trial
    10,  # Blackbox steps
)

# Save the environment and model state
enviro.Save(os.path.join(args.save_loc, f'environment_{args.seed}.pt'))
torch.save(
    model.state_dict(),
    os.path.join(args.save_loc, f'model_{args.seed}.pt'),
)
torch.save(
    {
        'rewardHistory': rewardHistory,
        'errorHistory': errorHistory
    },
    os.path.join(args.save_loc, f'data_{args.seed}.pt'),
)

# Plot and save error-reward analysis
plot_error_reward(
    rewardHistory,
    errorHistory,
    args.seed,
    args.save_loc,
)

# Analyze neuron activation
neuron_activation(
    enviro,
    model,
    50,  # Trials
    100,  # Steps allowed per trial
    10,  # Blackbox steps
    420,  # Neuron number
    args.seed,
    args.save_loc,
)

# Test the model
test(
    enviro,
    model,
    3,  # Trials
    100,  # Steps allowed per trial
    10,  # Blackbox steps
    args.seed,
    args.save_loc,
)