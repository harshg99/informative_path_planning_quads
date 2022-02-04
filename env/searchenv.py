import gym
import time
import numpy as np
from gym.envs.classic_control import rendering
from enum import Enum
from render import circle,rectangle
import math
import matplotlib.pyplot as plt
import GPy
'''
Reward Class
'''
class REWARD(Enum):
    STEP        = -1.0
    STEPDIAGONAL= -1.0*np.sqrt(2)
    STAY      = -0.5
    MAP  = +1.0
    TARGET  = +15.0
    COLLISION = -2.0

'''
Constant Envrionment Variables for rendering
'''
ACTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]
ZEROREWARDCOLOR = [0.,0.,0.1]
MAXREWARDCOLOR = [0.,1.,1.0]
AGENT_MINCOL = [0.,0.,0.]
AGENT_MAXCOL = [1.,0.,0.]

'''
Environment DEbug Variables
'''
DEBUG = True

class Agent():
    def __init__(self,ID,row,col,map_size,pad,world_size):
        self.ID = ID
        self.pos = (row,col)
        self.reward_map_size = map_size
        self.pad = pad
        self.world_size = world_size
        self.worldMap = None

    def updateMap(self,worldMap):
        self.worldMap= worldMap

    def updatePos(self,action):
        next_pos = self.pos + action
        is_action_valid = self.isValidPos(next_pos)
        if is_action_valid:
            self.pos = next_pos
        return is_action_valid

    def isValidPos(self, pos):
        is_valid = (np.array(pos - self.pad) > -1).all()
        is_valid = is_valid and (np.array(pos + self.pad) < self.world_size.shape).all()
        return is_valid


class SearchEnv(gym.Env):
    def __init__(self,numAgents=None,rewardMap=None,initialPos=None,num_centers = [5,10],max_var = [10.0],mapSize = 30,seed = 45):
        self.worldMap = rewardMap
        self.numAgents = numAgents


        self.worldMap = None
        self.rewardMap = None
        self.trajMap = None
        self.agentMap = None

        # Parameters to create training maps
        self.centers = num_centers
        self.max_var = max_var
        self.reward_map_size = mapSize
        self.pad_size = mapSize - 1  # TODO clean up these
        self.world_map_size = self.reward_map_size + 2 * (self.pad_size)
        self.curr_r_map_size = self.reward_map_size + self.pad_size
        self.curr_r_pad = (self.curr_r_map_size - 1) / 2
        self.seed = seed

    def createWorld(self,rewardMap=None):
        if rewardMap is None:
            #Create random multimodels gaussian here



            # Sub-test to check the gaussian with very sparse data
            X = []
            Y = []
            self.seed+=1
            np.random.seed(self.seed)
            num_centers = np.random.randint(self.centers[0], self.centers[1])

            for j in range(num_centers):
                X.append([np.random.randint(0,self.reward_map_size),np.random.randint(0,self.reward_map_size)])
                Y.append(np.random.rand(1)*self.max_var)

            X = np.array(X)
            Y = np.clip(np.array(Y),1,self.max_var)
                       # m = GPy.models.GPRegression(X,Y)
            m = GPy.models.SparseGPRegression(X, Y, num_inducing=10)
            m.rbf.variance = 1
            m.rbf.lengthscale = 3
            #print(m.rbf)
            # m.optimize()

            x = np.array([np.linspace(0, self.reward_map_size-1,self.reward_map_size), np.linspace(0,self.reward_map_size-1, self.reward_map_size)]).T  # np.random.uniform(-3.,3.,(200,2))

            x1Mesh, x2Mesh = np.meshgrid(x[:, 0:1], x[:, 1:2])
            xPred = np.array([np.reshape(x1Mesh, (900,)), np.reshape(x2Mesh, (900,))]).T

            yPred, Var = m.predict(xPred)
            x1len = math.floor(np.max(x[:, 0:1]) - np.min(x[:, 0:1])) + 1
            x2len = math.floor(np.max(x[:, 1:2]) - np.min(x[:, 1:2])) + 1

            yMesh = np.reshape(yPred, (np.size(x, 0), np.size(x, 0))).T
            print(yMesh.shape)
            levels = np.linspace(np.min(yMesh), np.max(yMesh), 1000)
            levels1 = np.linspace(np.min(yMesh), np.max(yMesh), 10)
            # yMesh[:] = 0.5
            rewardMap = yMesh
            if DEBUG:
                plt.contourf(x1Mesh, x2Mesh, yMesh, levels, cmap='viridis')

        self.worldmap = np.zeros((self.world_map_size, self.world_map_size))
        self.worldmap[self.pad_size:self.pad_size + self.reward_map_size,\
        self.pad_size:self.pad_size + self.reward_map_size] = rewardMap # capped b/w 0 and 1
        self.orig_worldmap = np.copy(self.worldMap)
        self.rewardMap = rewardMap

        # Creating the agents
        self.agents = [Agent(j,self.reward_map_size,self.reward_map_size,self.reward_map_size,self.pad_size,self.world_map_size) for j in range(self.numAgents)]

    def reset(self, state):
        self.createWorld()

    def stepall(self,action_dict):
        rewards = []
        for j in range(self.numAgents):
            r = self.step(agentID=j,action=action_dict[j])
            rewards.append(r)
        return rewards

    def step(self,agentID,action):

        """
        Given the current state and action, return the next state
        """
        valid = self.agents[agentID].updatePos(action)
        reward = 0
        reward += REWARD.MAP*self.worldMap[self.agents[agentID].pos[0],self.agents[agentID].pos[1]]
        if valid:
            reward+=REWARD.STEP
        else:
            reward+=REWARD.COLLISION

        self.worldMap[self.agents[agentID].pos[0],self.agents[agentID].pos[1]] = 0
        self.agents[agentID].updateMap(self.worldMap)
        return reward

    def phi_from_map_coords(self,r, c):
        map_section = self.worldmap[r[0]:r[1], c[0]:c[1]]
        size = (r[1] - r[0]) * (c[1] - c[0])
        return np.sum(map_section) / size

    def get_obs(self,agentID):

        r = self.agents[agentID].pos[1]
        c = self.agents[agentID].pos[0]
        phi_prime = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not (i == 0 and j == 0):
                    phi_prime.append(self.worldmap[r + i, c + j])
                    phi_prime.append(self.phi_from_map_coords((r - 1 + 3 * i, r - 1 + 3 * (i + 1)),
                                                         (c - 1 + 3 * j, c - 1 + 3 * (j + 1))))

        for i in ((0, r - 4), (r - 4, r + 5), (r + 5, self.world_map_size)):
            for j in ((0, c - 4), (c - 4, c + 5), (c + 5, self.world_map_size)):
                if not (i == (r - 4, r + 5) and j == (c - 4, c + 5)):
                    phi_prime.append(self.phi_from_map_coords(i, j))

        phi_prime = np.squeeze(np.array([phi_prime]))
        return phi_prime

    def render(self, mode='visualise',W=400, H=400):
        self.viewer = rendering.Viewer(W, H)
        size_x = W/self.worldMap.shape[0]
        size_y = H/self.worldMap.shape[1]
        for i in range(self.worldMap.shape[0]):
            for j in range(self.worldMap.shape[1]):
                # rending the infoMap
                shade = ZEROREWARDCOLOR + (MAXREWARDCOLOR-ZEROREWARDCOLOR)*self.worldMap[i,j]
                self.viewer.add_onetime(rectangle(i * size_x, j * size_y, size_x, size_y, shade, False))
                for agentID in range(self.num_agents):
                    agentColor = AGENT_MINCOL + (AGENT_MAXCOL-AGENT_MINCOL)*((agentID+1)/self.numAgents)
                    if i == self.agents[agentID].pos[0] and j ==self.agents[agentID].pos[1]:
                        self.viewer.add_onetime(circle(i * size_x, j * size_y, size_x, size_y, agentColor))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


