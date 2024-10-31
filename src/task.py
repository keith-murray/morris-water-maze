"""Morris water maze task."""
import torch
import numpy as np

class MorrisWaterMaze():
    """
    A `MorrisWaterMaze` object is an environment that a simulated agent can interact with.
    The `GetVision` method is called to create a vision vector of what the agent sees.
    This vision vector is passed to the agent, and the agent returns actions.
    The `UpdateAgent` method takes in the agents actions and updates their location in the object.
    The `CheckReward` method returns a boolean indicating if the agent is in the reward zone.

    This code isn't the most optimal, but the logic for creating the vision vector is rather unique.
    It works by particles coming out of the peripheral of the agents sight, and the `ScourWalls` 
    function grabs all the wall values inbetween the peripheral particles. Neat right?
    """
    def __init__(self, seed):
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        self.length = 20
        self.width = 20
        self.arena = torch.zeros(self.length+2, self.width+2)
        self.lower = 1
        self.upper = 6

        self.FillSides()
        self.AssignReward()
        self.CreateWallList()
        self.PlaceAgent()
        
    def FillSides(self,):
        """Generate values for the sides of the maze."""
        top = torch.randint(self.lower, self.upper, (self.width+2,), generator=self.rng)
        bottom = torch.randint(self.lower, self.upper, (self.width+2,), generator=self.rng)
        left = torch.randint(self.lower, self.upper, (self.length+2,), generator=self.rng)
        right = torch.randint(self.lower, self.upper, (self.length+2,), generator=self.rng)
        
        self.arena[0,:] = top
        self.arena[-1,:] = bottom
        self.arena[:,0] = left
        self.arena[:,-1] = right
        self.arena[0,0] = 0
        self.arena[-1,0] = 0
        self.arena[0,-1] = 0
        self.arena[-1,-1] = 0
        
    def CreateWallList(self,):
        """
        Constructs a dictionary of wall positions.
        Dictionary is used in grabing the values of the walls the agent is looking at.
        """
        right = [i for i in range(self.width + 2)]
        left = [i for i in reversed(right)]
        down = [21 for i in range(self.length + 2)]
        up = [0 for i in range(self.length + 2)]
        
        length = right + down + left + up
        width = up + right + down + left
        
        self.wallList = list(zip(width, length))
        self.wallList.remove((0, 0))
        self.wallList.remove((0, 0))
        self.wallList.remove((self.length+1, 0))
        self.wallList.remove((self.length+1, 0))
        self.wallList.remove((0, self.width+1))
        self.wallList.remove((0, self.width+1))
        self.wallList.remove((self.length+1, self.width+1))
        self.wallList.remove((self.length+1, self.width+1))
        
        self.wallDict = {}
        for i in range(len(self.wallList)):
            self.wallDict[self.wallList[i]] = i
            
        self.wallVals = []
        for i in self.wallList:
            self.wallVals.append(self.arena[i[0], i[1]].item())
                
    def AssignReward(self,):
        """Create a random reward location."""
        lenPos = torch.randint(1,self.length+1,(1,), generator=self.rng).item()
        widPos = torch.randint(1,self.width+1,(1,), generator=self.rng).item()
        rewards = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                rewards.append((lenPos+i, widPos+j))
        
        self.reward = (lenPos, widPos)
        self.rewards = set(rewards)
        
    def ReAssignReward(self, reward):
        """Assign reward to a fixed location."""
        rewards = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                rewards.append((reward[0]+i, reward[1]+j))
        
        self.reward = reward
        self.rewards = set(rewards)
    
    def PlaceAgent(self,):
        """Randomly place agent in maze."""
        lenAgn = torch.randint(1,self.length+1,(1,), generator=self.rng)[0]
        widAgn = torch.randint(1,self.width+1,(1,), generator=self.rng)[0]
        posAgn = (lenAgn, widAgn)
        
        if posAgn in self.rewards:
            self.PlaceAgent()
        else:
            self.posAgn = posAgn
            
        self.headAgn = 2*np.pi*torch.rand(1, generator=self.rng)[0]
        
        sight = self.GetVision()
        sightRange = torch.where(sight > 0, 1, 0)
        sightRange = torch.sum(sightRange)
        if sightRange < 10:
            self.PlaceAgent()
        
    def ReachWall(self, ang, above=True):
        """
        Simulates particles coming from the agents peripheral to grab
        the wall locations needed for the `ScourWalls` function.
        """
        for r in np.linspace(1, 30, num=50):
            length = int(r*torch.cos(ang) + self.posAgn[0])
            width = int(r*torch.sin(ang) + self.posAgn[1])
            if length > self.length+1:
                length = self.length+1
            elif length < 0:
                length = 0
            if width > self.length+1:
                width = self.length+1
            elif width < 0:
                width = 0
            if self.arena[length, width] != 0:
                return (length, width)
        
        if above:
            return self.ReachWall(ang+np.pi/20, above=True)
        else:
            return self.ReachWall(ang-np.pi/20, above=False)            
    
    def ScourWalls(self, wallAbove, wallBellow, location=False):
        """Grabs values along walls to construct agents vision."""
        firstInd = self.wallDict[wallAbove]
        secondInd = self.wallDict[wallBellow]
        
        if not location:
            if secondInd >= firstInd:
                return self.wallVals[firstInd:secondInd+1]
            else:
                return self.wallVals[firstInd:] + self.wallVals[:secondInd+1]
        else:
            if secondInd >= firstInd:
                return self.wallVals[firstInd:secondInd+1], self.wallList[firstInd:secondInd+1]
            else:
                return self.wallVals[firstInd:] + self.wallVals[:secondInd+1], self.wallList[firstInd:] + self.wallList[:secondInd+1]
        
    def GetVision(self, locations=False):
        """Returns a vector of wall values depending on the position and head direction of the agent."""
        visAbv = self.headAgn + np.pi/6
        visBlw = self.headAgn - np.pi/6
        
        wallAbv = self.ReachWall(visAbv, above=True)
        wallBlw = self.ReachWall(visBlw, above=False)
        
        if not locations:
            sightVals = self.ScourWalls(wallAbv, wallBlw)
        else:
            sightVals, sightInd = self.ScourWalls(wallAbv, wallBlw, location=True)
        
        if len(sightVals) > 49:
            print(self.posAgn)
            self.PlaceAgent()
            return self.GetVision()
        
        if len(sightVals) < 49:
            zeros = int((49 - len(sightVals))/2)
            sightVals = [0.0 for i in range(zeros)] + sightVals + [0.0 for i in range(zeros)]
        
        if len(sightVals) < 49:
            sightVals = sightVals + [0.0]
        
        if not locations:
            return torch.unsqueeze(torch.tensor(sightVals), 0)
        else:
            return torch.unsqueeze(torch.tensor(sightVals), 0), sightInd
    
    def CheckReward(self,):
        """Function to check if agent has reached the reward zone."""
        realPos = tuple([int(i) for i in self.posAgn])
        if realPos in self.rewards:
            return True
        else:
            return False
        
    def UpdateAgent(self, newLength, newWidth, newAngle):
        """Update agent's position based on their output."""
        posLength = self.posAgn[0] + newLength
        posWidth = self.posAgn[1] + newWidth
        if posLength > 21:
            extraLength = posLength - 21
            posLength -= extraLength
        elif posLength < 1:
            extraLength = 1 - posLength
            posLength += extraLength
        if posWidth > 21:
            extraWidth = posWidth - 21
            posWidth -= extraWidth
        elif posWidth < 1:
            extraWidth = 1 - posWidth
            posWidth += extraWidth
        self.posAgn = (posLength, posWidth)
        self.headAgn += newAngle
        
    def CalculateRewardDistance(self,):
        """Reward is the agent's Euclidean distance from the reward zone."""
        return torch.sqrt((self.posAgn[0]-self.reward[0])**2 + (self.posAgn[1]-self.reward[1])**2)
    
    def Save(self, name):
        saveObj = {'arena': self.arena,
                   'reward': self.reward,
                   'rewards': self.rewards,
                   'wallList': self.wallList,
                   'wallDict': self.wallDict,
                   'wallVals': self.wallVals}
        torch.save(saveObj, name)

    def Load(self, name):
        saveObj = torch.load(name)
        self.arena = saveObj['arena']
        self.reward = saveObj['reward']
        self.rewards = saveObj['rewards']
        self.wallList = saveObj['wallList']
        self.wallDict = saveObj['wallDict']
        self.wallVals = saveObj['wallVals']