"""Training functions."""
import torch
import numpy as np

def CalculateReward(step):
    """
    I do not remember my logic behind this reward function.
    It's only for evaluating the performance of the agent, not
    for backpropagation to train the agent.
    """
    denominator = 1/(1+np.exp(-0.06*50))
    reward = 1/(1+np.exp(0.06*(step-50)))
    return reward/denominator

def train(environment, model, epochs, trials, steps, blackBox):
    """Function to simulate and train an agent performing in `MorrisWaterMaze`."""
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
                # Error is modulated by number of steps.
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