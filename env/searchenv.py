import gym
import time
import numpy as np
from gym.envs.classic_control import rendering
from enum import Enum

class REWARD(Enum):
    STEP        = -1.0
    STEPDIAGONAL= -1.0*np.sqrt(2)
    STAY      = -0.5
    MAP  = +1.0
    TARGET  = +15.0
    COLLISION = -2.0

ACTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]

class Agent():
    def __init__(self,ID,row,col,max_size):
        self.ID = ID
        self.pos = (row,col)
        self.max_size = max_size

    def updatePos(self,action):
        next_pos = self.pos + action
        is_action_valid = self.isValidPos(next_pos)
        if is_action_valid:
            self.pos = next_pos
        return is_action_valid

    def isValidPos(self, pos):
        is_valid = (np.array(pos - self.curr_r_pad) > -1).all()
        is_valid = is_valid and (np.array(pos + self.curr_r_pad) < self.orig_worldmap.shape).all()
        return is_valid


class SearchEnv(gym.Env):
    def __init__(self,numAgents=None,rewardMap=None,initialPos=None):
        self.worldMap = rewardMap
        self.numAgents = numAgents
        self.agents = [Agent(j) for j in range(numAgents)]
        self.worldMap = None
        self.rewardMap = None

    def createWorld(self,rewardMap):
        self.worldmap = np.zeros((self.world_map_size, self.world_map_size))
        self.worldmap[self.pad_size:self.pad_size + self.reward_map_size,\
        self.pad_size:self.pad_size + self.reward_map_size] = rewardMap
        self.orig_worldmap = np.copy(self.worldMap)
        self.rewardMap = rewardMap

    def reset(self, state):
        pass

    def stepall(self,action_dict):
        rewards = []
        for j in range(self.numAgents):
            r = self.step(agentID=j,action=action_dict[j])
            rewards.append(r)
        return rewards

    def step(self,agentID,action):

        """
        Given the current state and action, return the next state
        Ensures that next_pos is still in the reward map area
        """
        valid = self.agents[agentID].updatePos(action)
        reward = 0
        reward += REWARD.MAP*self.worldMap[self.agents[agentID].pos[0],self.agents[agentID].pos[1]]
        if valid:
            reward+=REWARD.STEP
        else:
            reward+=REWARD.COLLISION

        self.worldMap[self.agents[agentID].pos[0],self.agents[agentID].pos[1]] = 0
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

    def render(self, screenWidth=400, screenHeight=400):
        self.viewer = rendering.Viewer(screenWidth, screenHeight)

