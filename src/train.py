"""Training functions."""
import torch
import numpy as np
from task import MorrisWaterMaze
from models import LSTM
import matplotlib.pyplot as plt
from tqdm import tqdm

def CalculateReward(step):
    denominator = 1/(1+np.exp(-0.06*50))
    reward = 1/(1+np.exp(0.06*(step-50)))
    return reward/denominator

def train(environment, model, epochs, trials, steps, blackBox):
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    rewardHistory = []
    errorHistory = []
    dark = torch.zeros(1, 49)
    
    for i in tqdm(range(epochs)):
        model.reset()
        optimizer.zero_grad()
        cumulativeReward = 0
        cumulativeError = 0
        
        for j in range(trials):
            reward = None
            environment.PlaceAgent()
            
            for k in range(steps):
                sightVals = enviro.GetVision()
                out = model(sightVals)
                enviro.UpdateAgent(out[0], out[1], out[2])
                cumulativeError += enviro.CalculateRewardDistance()*k/steps
                if enviro.CheckReward():
                    reward = CalculateReward(k)
                    break
            
            if reward is not None:
                cumulativeReward += reward
            
            for k in range(blackBox):
                out = model(dark)
        
        cumulativeError.backward()
        optimizer.step()
        rewardHistory.append(cumulativeReward)
        errorHistory.append(cumulativeError.item())

    return rewardHistory, errorHistory


if __name__ == "__main__":
    enviro = MorrisWaterMaze()
    enviro.ReAssignReward((10,10))
    model = LSTM(49, 500, 3)
    rewardHistory, errorHistory = train(enviro, model, 200, 25, 100, 10)
    
    fig, ax = plt.subplots()
    ax.plot(rewardHistory)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Reward')
    ax.set_title('Accumulated reward over training epochs')
    plt.savefig('accumulated_reward.png')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(errorHistory)
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Distance Error')
    ax.set_title('Accumulated distance error over epochs')
    plt.savefig('accumulated_error.png')
    plt.show()