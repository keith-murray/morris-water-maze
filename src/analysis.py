"""Function to visualize neuron receptive fields."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

def neuron_activation(environment, model, trials, steps, blackBox, neuron, figure_title, save_loc):
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
            sightVals, sightInd = enviro.GetVision(locations=True)
            out = model(sightVals)
            enviro.UpdateAgent(out[0], out[1], out[2])
            
            agnLoc = enviro.posAgn
            agnLen = int(agnLoc[0].item())
            agnWid = int(agnLoc[1].item())
            walls[(agnLen, agnWid)].append(model.state[0,neuron])
            
            if enviro.CheckReward():
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
    ax.set_title(figure_title)
    plt.savefig(save_loc)
    plt.show()