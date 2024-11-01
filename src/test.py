"""Testing functions."""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from celluloid import Camera

black = np.array([0,0,0])
grey = np.array([195,200,205])
red = np.array([255,17,0])
green = np.array([0,176,24])
yellow = np.array([255,237,0])

def test(environment, model, trials, steps, blackBox, seed, save_loc):
    """
    Simulates the agent performing in the `MorrisWaterMaze` task and 
    records a video.
    """
    model.reset()
    model.eval()
    dark = torch.zeros(1, 49)
    
    time = trials*(steps+blackBox)
    arena = np.zeros((time,22,22,3))
    
    arena[:,:,:,0] = grey[0]*np.ones((time,22,22))
    arena[:,:,:,1] = grey[1]*np.ones((time,22,22))
    arena[:,:,:,2] = grey[2]*np.ones((time,22,22))
    
    for i in environment.rewards:
        arena[:,i[0],i[1],0] = yellow[0]*np.ones(time)
        arena[:,i[0],i[1],1] = yellow[1]*np.ones(time)
        arena[:,i[0],i[1],2] = yellow[2]*np.ones(time)
    
    for i in environment.wallList:
        arena[:,i[0],i[1],0] = black[0]*np.ones(time)
        arena[:,i[0],i[1],1] = black[1]*np.ones(time)
        arena[:,i[0],i[1],2] = black[2]*np.ones(time)
    
    timeCount = 0
    for j in tqdm(range(trials)):
        environment.PlaceAgent()
        
        for k in range(steps):
            agnLoc = environment.posAgn
            agnLen = int(agnLoc[0].item())
            agnWid = int(agnLoc[1].item())
            
            arena[timeCount,agnLen,agnWid,0] = red[0]
            arena[timeCount,agnLen,agnWid,1] = red[1]
            arena[timeCount,agnLen,agnWid,2] = red[2]
            
            sightVals, sightInd = environment.GetVision(locations=True)
            
            for i in sightInd:
                arena[timeCount,i[0],i[1],0] = green[0]
                arena[timeCount,i[0],i[1],1] = green[1]
                arena[timeCount,i[0],i[1],2] = green[2]
            
            out = model(sightVals)
            environment.UpdateAgent(out[0], out[1], out[2])
            timeCount += 1
            
            if environment.CheckReward():
                break
        
        for k in range(blackBox):
            out = model(dark)
            
            arena[timeCount,:,:,0] = black[0]*np.ones((22,22))
            arena[timeCount,:,:,1] = black[1]*np.ones((22,22))
            arena[timeCount,:,:,2] = black[2]*np.ones((22,22))
            timeCount += 1
    
    arena = arena[0:timeCount].astype('int32')
    fig = plt.figure()
    camera = Camera(fig)
    
    for i in range(timeCount):
        plt.imshow(arena[i,:,:,:])
        camera.snap()

    animation = camera.animate()
    animation.save(
        os.path.join(save_loc, f'simulated_tests_{seed}.gif'), 
        writer = 'pillow', 
        fps=15
    )