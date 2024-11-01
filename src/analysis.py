"""Analysis functions."""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

def plot_error_reward(rewardHistory, errorHistory, seed, save_loc):
    """Simple plotting function for errors and rewards over epochs."""
    fig, ax = plt.subplots()
    ax.plot(rewardHistory)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Reward')
    ax.set_title('Accumulated reward over training epochs')
    plt.savefig(os.path.join(save_loc, f'accumulated_reward_{seed}.png'))
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(errorHistory)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Distance Error')
    ax.set_title('Accumulated distance error over epochs')
    plt.savefig(os.path.join(save_loc, f'accumulated_error_{seed}.png'))
    plt.close()

def neuron_activation(environment, model, trials, steps, blackBox, neuron, seed, save_loc):
    """
    For a particular neuron, this function creates a heatmap of the neurons activations
    across the spatial grid of the `MorrisWaterMaze` environment.
    """
    model.reset()
    model.eval()
    dark = torch.zeros(1, 49)
    
    walls = {}
    for i in range(22):
        for j in range(22):
            walls[(i,j)] = []
    
    for j in tqdm(range(trials)):
        environment.PlaceAgent()
        
        for k in range(steps):
            sightVals, sightInd = environment.GetVision(locations=True)
            out = model(sightVals)
            environment.UpdateAgent(out[0], out[1], out[2])
            
            agnLoc = environment.posAgn
            agnLen = int(agnLoc[0].item())
            agnWid = int(agnLoc[1].item())
            walls[(agnLen, agnWid)].append(model.state[0,neuron])
            
            if environment.CheckReward():
                break
        
        for k in range(blackBox):
            out = model(dark)
            
    wallHeatmap = np.zeros((22,22))
    for i in range(22):
        for j in range(22):
            activations = walls[(i,j)]
            if len(activations) == 0:
                wallHeatmap[i, j] = 0
            else:
                wallHeatmap[i, j] = sum(activations)/len(activations)
    
    fig, ax = plt.subplots()
    ax = sns.heatmap(wallHeatmap, vmin=-1, vmax=1)
    ax.set_title(f'Neuron {neuron} activations')
    plt.savefig(os.path.join(save_loc, f'neuron_{neuron}_heatmap_{seed}.png'))
    plt.close()