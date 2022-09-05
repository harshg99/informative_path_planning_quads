import gym
import time
import numpy as np
from gym.envs.classic_control import rendering
from enum import Enum
from env.render import *
import math
import matplotlib.pyplot as plt
import GPy
from params import *
from skimage.measure import block_reduce
from env.agents import Agent
from copy import deepcopy
'''
Reward Class
'''
class REWARD(Enum):
    STEP        = -0.1
    STEPDIAGONAL= -0.1*np.sqrt(2)
    STAY      = -0.5
    MAP  = +100
    TARGET  = +200.0
    COLLISION = -1.0
    MP = 10000

'''
Constant Envrionment Variables for rendering
'''
ACTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]
ZEROREWARDCOLOR = (0.05,0.1,0.1)
MAXREWARDCOLOR = (0.8,1.0,1.0)
AGENT_MINCOL = (0.5,0.3,0.3)
AGENT_MAXCOL = (1.,0.3,0.3)

'''
Environment DEbug Variables
'''
DEBUG = False


class SearchEnv(gym.Env):

    '''
    Initialising the gym environemnt:

    @params
    numAgents: Number of agents
    numCentres: number of random gaussian centres for the reward map
    max_var = Maximum vairance for the gaussians
    min_var = Minumum variance for the gaussians
    mapSize: size of the reward map
    seed: environemnt seed to reproduce the training results if necessary6
    '''
    def __init__(self,params_dict,args_dict):

        self.numAgents = params_dict['numAgents']
        self.worldMap = None
        self.rewardMap = None
        self.trajMap = None
        self.agentMap = None
        self.obstacle_map = None
        self.args_dict = args_dict

        # Multiple Reward Map Size choices
        self.reward_map_size_list = params_dict['rewardMapSizeList']
        self.random_map_size = params_dict['randomMapSize']

        if self.random_map_size:
            self.reward_map_size = np.random.choice(self.reward_map_size_list)
        else:
            self.reward_map_size = self.reward_map_size_list[params_dict['defaultMapChoice']]


        # Parameters to create training maps
        self.centers = params_dict['num_centers']
        self.max_var = params_dict['max_var']
        self.min_var = params_dict['min_var']
        self.pad_size = params_dict['pad_size']
        self.scale = params_dict['scale']
        self.world_map_size = self.reward_map_size + 2*self.pad_size

        self.action_size = len(ACTIONS)
        self.episode_length = params_dict['episode_length']
        self.sensor_range = params_dict['sensor_range']
        self.max_steps = params_dict['episode_length']


        if SET_SEED:
            self.seed = params_dict['seed']
        self.viewer = None

        if OBSERVER == 'RANGE':
            self.input_size = [2*RANGE,2*RANGE,1]
        elif OBSERVER =='TILED':
            self.input_size = [24,1,1]
        elif OBSERVER == 'TILEDwOBS':
            self.input_size = [48,1,1]
        elif OBSERVER == 'RANGEwOBS':
            self.input_size = [2*RANGE,2*RANGE,2]
        elif OBSERVER == 'RANGEwOBSwNEIGH':
            self.input_size = [2*(RANGE+1),2*(RANGE+1),2]
        elif OBSERVER == 'RANGEwOBSwPENC':
            self.input_size = [2*RANGE,2*RANGE,4]
        elif OBSERVER == 'RANGEwOBSwMULTI':
            self.map_length = 2
            self.input_size = [2*RANGE, 2*RANGE, len(self.scale)*2]
        elif OBSERVER == 'RANGEwOBSwMULTIwCOV':
            self.map_length = 2
            self.input_size = [2*RANGE, 2*RANGE, len(self.scale)*3]

    '''
    Adds a Gaussian
    '''
    def Gaussian(self,mean,cov):
        x = np.array([np.linspace(0, self.reward_map_size - 1, self.reward_map_size),
                      np.linspace(0, self.reward_map_size - 1, self.reward_map_size)]).T

        x1Mesh, x2Mesh = np.meshgrid(x[:, 0:1], x[:, 1:2])
        # Making the gaussians circular

        cov1 = np.copy(cov)
        cov[0][0] = np.clip(cov[0][0], self.min_var, self.max_var)
        cov[1][1] = np.clip(cov[1][1], self.min_var, self.max_var)
        cov[0][1] = np.clip(cov[0][1],0, self.min_var / 2)
        cov[1][0] = cov[0][1]

        if np.linalg.det(cov)<0:
            cov[0][0] = np.clip(cov1[0][1], self.min_var, self.max_var)
            cov[1][1] = np.clip(cov1[1][0], self.min_var, self.max_var)
            cov[0][1] = np.clip(cov1[1][1], self.min_var / 1.5, self.max_var / 1.5)
            cov[1][0] = cov[0][1]
        elif np.linalg.det(cov)==0:
            cov[0][0] = np.clip(cov1[0][1]+np.random.rand(1), self.min_var, self.max_var)
            cov[1][1] = np.clip(cov1[1][0]+np.random.rand(1), self.min_var, self.max_var)


        xPred = np.array([np.reshape(x1Mesh, (self.reward_map_size*self.reward_map_size,))\
                             , np.reshape(x2Mesh, (self.reward_map_size*self.reward_map_size,))])
        gaussian = np.diag(1/np.sqrt(2*np.pi*np.abs(np.linalg.det(cov)))*\
                           np.exp(-(xPred - mean.reshape((mean.shape[0],1))).T@np.linalg.inv(cov)@\
                                  (xPred-mean.reshape((mean.shape[0],1)))))
        gaussian = gaussian.reshape((self.reward_map_size,self.reward_map_size))
        return gaussian

    '''
    Creates a reward map based on the number of Gaussians
    '''
    def createRewardMap(self,X,var):
        # Sub-test to check the gaussian with very sparse data
        gaussians = np.array([self.Gaussian(X[j],var[j]) for j in range(X.shape[0])])
        rewardMap  = np.sum(gaussians,axis = 0)
        rewardMap/=rewardMap.sum()
        return rewardMap
        # m = GPy.models.GPRegression(X,Y

    '''
    Reward Map from Gpy framework based on measurements and expected locations of sources
    '''
    def createRewardMapGP(self,X,var):

        m = GPy.models.SparseGPRegression(X, var, num_inducing=10)
        m.rbf.variance = 1
        m.rbf.lengthscale = 3
        #print(m.rbf)
        # m.optimize()

        x = np.array([np.linspace(0, self.reward_map_size-1,self.reward_map_size), np.linspace(0,self.reward_map_size-1, self.reward_map_size)])  # np.random.uniform(-3.,3.,(200,2))

        x1Mesh, x2Mesh = np.meshgrid(x[:, 0:1], x[:, 1:2])
        xPred = np.array([np.reshape(x1Mesh, (900,)), np.reshape(x2Mesh, (900,))])

        yPred, Var = m.predict(xPred)
        x1len = math.floor(np.max(x[:, 0:1]) - np.min(x[:, 0:1])) + 1
        x2len = math.floor(np.max(x[:, 1:2]) - np.min(x[:, 1:2])) + 1

        yMesh = np.reshape(yPred, (np.size(x, 0), np.size(x, 0))).T
        #print(yMesh.shape)
        levels = np.linspace(np.min(yMesh), np.max(yMesh), 1000)
        levels1 = np.linspace(np.min(yMesh), np.max(yMesh), 10)
        # yMesh[:] = 0.5
        rewardMap = yMesh


    '''
    Creates the world with reward map and agents
    '''
    def createWorld(self,rewardMap=None):
        if rewardMap is None:
            #Create random multimodels gaussian here
            # Sub-test to check the gaussian with very sparse data
            X = []
            Y = []
            if not SAME_MAP:
                self.seed+=1
                np.random.seed(self.seed)
            num_centers = np.random.randint(self.centers[0], self.centers[1])

            for j in range(num_centers):
                X.append([np.random.randint(0,self.reward_map_size),np.random.randint(0,self.reward_map_size)])
                y = np.zeros((2,2))
                y[0][0] = np.random.rand(1)*(self.max_var - self.min_var) + self.min_var
                y[1][1] = np.random.rand(1)*(self.max_var - self.min_var) + self.min_var
                y[0][1] = np.random.rand(1)*self.min_var/2
                y[1][0] = y[0][1]
                Y.append(y)

            X = np.array(X)
            self.prior_centres = X
            self.prior_vars = Y
            # Y = np.clip(np.array(Y),self.min_var,self.max_var)
            rewardMap = self.createRewardMap(X,Y)
            #Using GPy for Maps, deosnt work
                       # m = GPy.models.GPRegression(X,Y)

            # if DEBUG:
            #     plt.contourf(x1Mesh, x2Mesh, yMesh, levels, cmap='viridis')
            if DEBUG:
                x = np.array([np.linspace(0, self.reward_map_size - 1, self.reward_map_size),
                              np.linspace(0, self.reward_map_size - 1,
                                          self.reward_map_size)]).T  # np.random.uniform(-3.,3.,(200,2))

                x1Mesh, x2Mesh = np.meshgrid(x[:, 0:1], x[:, 1:2])
                yMesh = rewardMap
                levels = np.linspace(0, np.max(yMesh), 1000)
                plt.contourf(x1Mesh, x2Mesh, yMesh, levels, cmap='viridis')

        self.obstacle_map = np.zeros((self.world_map_size,self.world_map_size))+1
        self.worldMap = np.zeros((self.world_map_size, self.world_map_size)) #for boundaries
        self.worldMap[self.pad_size:self.pad_size + self.reward_map_size,\
        self.pad_size:self.pad_size + self.reward_map_size] = rewardMap*REWARD.MAP.value # capped b/w 0 and 1
        self.obstacle_map[self.pad_size:self.pad_size + self.reward_map_size,\
        self.pad_size:self.pad_size + self.reward_map_size]= np.zeros((self.reward_map_size,self.reward_map_size))
        self.orig_worldMap = deepcopy(self.worldMap)

        self.orig_target_distribution_map = deepcopy(self.worldMap)
        self.rewardMap = rewardMap

        # Creating the agents
        if SPAWN_RANDOM_AGENTS:
            row = np.random.randint(self.pad_size,self.reward_map_size+self.pad_size,(self.numAgents,))
            col = np.random.randint(self.pad_size, self.reward_map_size + self.pad_size, (self.numAgents,))
            self.agents = [
                Agent(j, row[j],col[j], \
                      self.reward_map_size, self.pad_size, self.world_map_size) for j in range(self.numAgents)]
        else:
            self.agents = [Agent(j,self.reward_map_size+int(j/(int(j/2))),self.reward_map_size+(j%(int(j/2))),\
                                 self.reward_map_size,self.pad_size,self.world_map_size) for j in range(self.numAgents)]

        # For rendering
        self.maxDensity = self.worldMap.max()

    '''
    resets the world for a new reward map and new training epoch
    '''
    def reset(self, state=None):
        if state is not None:
            self.reward_map_size = state.shape[0]
            self.world_map_size =2*self.pad_size+self.reward_map_size
        self.createWorld(rewardMap=state)

    def step_all(self,action_dict):
        rewards = []
        for j in range(self.numAgents):
            r = self.step(agentID=j,action=ACTIONS[action_dict[j]])
            rewards.append(r)
        done=False
        if np.all(np.abs(self.worldMap[self.pad_size:self.pad_size + self.reward_map_size, \
        self.pad_size:self.pad_size + self.reward_map_size])<0.1):
            done = True
        return rewards,done

    def step(self,agentID,action):

        """
        Given the current state and action, return the next state
        """
        valid = self.agents[agentID].updatePos(action)
        reward = 0
        reward += self.worldMap[self.agents[agentID].pos[0],self.agents[agentID].pos[1]]
        if valid:
            reward+=REWARD.STEP.value
        else:
            reward+=REWARD.COLLISION.value

        self.worldMap[self.agents[agentID].pos[0],self.agents[agentID].pos[1]] = 0
        self.agents[agentID].updateMap(self.worldMap)
        return reward

    def updateInfoMap(self,agentID,action):
        valid = self.agents[agentID].updatePos(action)
        reward = 0
        reward += self.worldMap[self.agents[agentID].pos[0], self.agents[agentID].pos[1]]
        if valid:
            reward += REWARD.STEP.value
        else:
            reward += REWARD.COLLISION.value

        self.worldMap[self.agents[agentID].pos[0], self.agents[agentID].pos[1]] = 0
        self.agents[agentID].updateMap(self.worldMap)
        return reward

    def phi_from_map_coords(self,r, c, map=None):
        if map is None:
            map = self.worldMap
        map_section = map[r[0]:r[1], c[0]:c[1]]
        size = (r[1] - r[0]) * (c[1] - c[0])
        return np.sum(map_section) / size

    def phi_from_map_coords_max(self, r, c,map=None):
        if map is None:
            map = self.worldMap
        map_section = map[r[0]:r[1], c[0]:c[1]]
        size = (r[1] - r[0]) * (c[1] - c[0])
        return np.max(map_section)

    def get_obs_tiled_wobs(self,agentID):

        r = self.agents[agentID].pos[0]
        c = self.agents[agentID].pos[1]
        phi_prime = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not (i == 0 and j == 0):
                    phi_prime.append(self.worldMap[r + i, c + j])
                    phi_prime.append(self.obstacle_map[r+i,c+j])
                    phi_prime.append(self.phi_from_map_coords((r - 1 + 3 * i, r - 1 + 3 * (i + 1)),
                                                         (c - 1 + 3 * j, c - 1 + 3 * (j + 1))))
                    phi_prime.append(self.phi_from_map_coords((r - 1 + 3 * i, r - 1 + 3 * (i + 1)),
                                                              (c - 1 + 3 * j, c - 1 + 3 * (j + 1)),map=self.obstacle_map))

        for i in ((0, r - 4), (r - 4, r + 5), (r + 5, self.world_map_size)):
            for j in ((0, c - 4), (c - 4, c + 5), (c + 5, self.world_map_size)):
                if not (i == (r - 4, r + 5) and j == (c - 4, c + 5)):
                    phi_prime.append(self.phi_from_map_coords(i, j))
                    phi_prime.append(self.phi_from_map_coords(i, j,map=self.obstacle_map))

        phi_prime = np.squeeze(np.array([phi_prime]))
        phi_prime = np.expand_dims(np.expand_dims(phi_prime, axis=-1), axis=-1)
        return phi_prime

    def get_obs_tiled(self, agentID):

        r = self.agents[agentID].pos[0]
        c = self.agents[agentID].pos[1]
        phi_prime = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not (i == 0 and j == 0):
                    phi_prime.append(self.worldMap[r + i, c + j])
                    phi_prime.append(self.phi_from_map_coords_max((r - 1 + 3 * i, r - 1 + 3 * (i + 1)),
                                                                  (c - 1 + 3 * j, c - 1 + 3 * (j + 1))))

        for i in ((0, r - 4), (r - 4, r + 5), (r + 5, self.world_map_size)):
            for j in ((0, c - 4), (c - 4, c + 5), (c + 5, self.world_map_size)):
                if not (i == (r - 4, r + 5) and j == (c - 4, c + 5)):
                    phi_prime.append(self.phi_from_map_coords_max(i, j))

        phi_prime = np.squeeze(np.array([phi_prime]))
        phi_prime = np.expand_dims(np.expand_dims(phi_prime,axis=-1),axis=-1)
        return phi_prime

    def get_obs_all(self):
        obs = []
        for j in range(self.numAgents):
            if OBSERVER == 'TILED':
                obs.append(self.get_obs_tiled(agentID=j))
            elif OBSERVER == 'RANGE':
                obs.append(self.get_obs_ranged(agentID=j))
            elif OBSERVER == 'TILEDwOBS':
                obs.append(self.get_obs_tiled_wobs(agentID=j))
            elif OBSERVER == 'RANGEwOBS':
                obs.append(self.get_obs_ranged_wobs(agentID=j))
            elif OBSERVER == 'RANGEwOBSwPENC':
                obs.append(self.get_obs_ranged_wobspenc(agentID=j))
        obs_dict = dict()
        obs_dict['obs'] = obs
        obs_dict['valids'] = None
        return obs_dict

    def get_obs_ranged(self,agentID):
        r = self.agents[agentID].pos[0]
        c = self.agents[agentID].pos[1]
        min_x = np.max(r - RANGE,0)
        min_y = np.max(c - RANGE,0)
        max_x = np.min(r + RANGE,self.worldMap.shape[0])
        max_y = np.min(c + RANGE,self.worldMap.shape[1])

        infomap_feature = np.zeros((2*RANGE,2*RANGE))
        infomap_feature[min_x-(r-RANGE):2*RANGE - (r+RANGE-max_x),\
                        min_y-(c-RANGE):2*RANGE - (c+RANGE-max_y)] = self.worldMap[min_x:max_x, min_y:max_y]
        #print("%d %d %d %d".format(min_x,min_y,max_x,max_y))

        infomap_feature = np.expand_dims(infomap_feature,axis=-1)

        return infomap_feature


    def get_obs_ranged_wobs(self,agentID):
        r = self.agents[agentID].pos[0]
        c = self.agents[agentID].pos[1]
        min_x = np.max([r - RANGE,0])
        min_y = np.max([c - RANGE,0])
        max_x = np.min([r + RANGE,self.worldMap.shape[0]])
        max_y = np.min([c + RANGE,self.worldMap.shape[1]])

        infomap_feature = np.zeros((2 * RANGE, 2 * RANGE))
        obsmap_feature = np.ones((2 * RANGE, 2 * RANGE))
        infomap_feature[min_x - (r - RANGE):2 * RANGE - (r + RANGE - max_x), \
        min_y - (c - RANGE):2 * RANGE - (c + RANGE - max_y)] = self.worldMap[min_x:max_x, min_y:max_y]

        obsmap_feature[min_x - (r - RANGE):2 * RANGE - (r + RANGE - max_x), \
        min_y - (c - RANGE):2 * RANGE - (c + RANGE - max_y)] = self.obstacle_map[min_x:max_x, min_y:max_y]

        features = np.expand_dims(infomap_feature, axis=-1)
        features = np.concatenate((features, np.expand_dims(obsmap_feature, axis=-1)), axis=-1)

        return np.array(features)

    def get_obs_ranged_wobspenc(self,agentID):
        r = self.agents[agentID].pos[0]
        c = self.agents[agentID].pos[1]
        min_x = np.max([r - RANGE,0])
        min_y = np.max([c - RANGE,0])
        max_x = np.min([r + RANGE,self.worldMap.shape[0]])
        max_y = np.min([c + RANGE,self.worldMap.shape[1]])

        infomap_feature = np.zeros((2*RANGE,2*RANGE))
        obsmap_feature = np.ones((2 * RANGE, 2 * RANGE))
        infomap_feature[min_x-(r-RANGE):2*RANGE - (r+RANGE-max_x),\
                        min_y-(c-RANGE):2*RANGE - (c+RANGE-max_y)] = self.worldMap[min_x:max_x, min_y:max_y]

        obsmap_feature[min_x-(r-RANGE):2*RANGE - (r+RANGE-max_x),\
                        min_y-(c-RANGE):2*RANGE - (c+RANGE-max_y)] = self.obstacle_map[min_x:max_x,min_y:max_y]

        penc_x = np.expand_dims(np.arange(start=0,stop=1,step=1/(2*RANGE))-0.5,axis=1)\
            .repeat(repeats=2*RANGE,axis=1)
        penc_y = np.expand_dims(np.arange(start=0, stop=1, step=1 / (2 * RANGE))-0.5, axis=0)\
            .repeat(repeats=2 * RANGE,axis=0)

        features = np.expand_dims(infomap_feature,axis=-1)
        features = np.concatenate((features,np.expand_dims(obsmap_feature,axis=-1),\
                         np.expand_dims(penc_x,axis=-1),np.expand_dims(penc_y,axis=-1)),axis=-1)
        #print("{:d} {:d} {:d} {:d}".format(min_x, min_y, max_x, max_y))
        return np.array(features)


    def get_obs_range_wobs_multi(self,agentID):
        r = self.agents[agentID].pos[0]
        c = self.agents[agentID].pos[1]

        range = RANGE
        for s in self.scale:
            range = s*RANGE
            min_x = np.max([r - range, 0])
            min_y = np.max([c - range, 0])
            max_x = np.min([r + range, self.worldMap.shape[0]])
            max_y = np.min([c + range, self.worldMap.shape[1]])

            infomap_feature = np.zeros((2 * range, 2 * range))
            obsmap_feature = np.zeros((2 * range, 2 * range))
            infomap_feature[min_x - (r - range):2 * range - (r + range - max_x), \
            min_y - (c - range):2 * range - (c + range - max_y)] = self.worldMap[min_x:max_x, min_y:max_y]

            obsmap_feature[min_x - (r - range):2 * range - (r + range - max_x), \
            min_y - (c - range):2 * range - (c + range - max_y)] = self.obstacle_map[min_x:max_x, min_y:max_y]

            if s==1:
                features = np.expand_dims(infomap_feature, axis=-1)
                features = np.concatenate((features, np.expand_dims(obsmap_feature, axis=-1)), axis=-1)
            else:
                infomap_feature = block_reduce(infomap_feature,(s,s),np.max)
                obsmap_feature = block_reduce(obsmap_feature,(s,s),np.max)
                features = np.concatenate((features,np.expand_dims(infomap_feature, axis=-1)), axis=-1)
                features = np.concatenate((features,np.expand_dims(obsmap_feature, axis=-1)), axis=-1)



        # print("{:d} {:d} {:d} {:d}".format(min_x, min_y, max_x, max_y))
        return np.array(features)

    def get_obs_range_coverage_multifov(self,agentID):
        r = self.agents[agentID].pos[0]
        c = self.agents[agentID].pos[1]

        range = RANGE
        for s in self.scale:
            range = s * RANGE
            min_x = np.max([r - range, 0])
            min_y = np.max([c - range, 0])
            max_x = np.min([r + range, self.worldMap.shape[0]])
            max_y = np.min([c + range, self.worldMap.shape[1]])

            infomap_feature = np.zeros((2 * range, 2 * range))
            obsmap_feature = np.zeros((2 * range, 2 * range))

            coverage_feature = np.zeros((2 * range, 2 * range))

            coverage_feature[min_x - (r - range):2 * range - (r + range - max_x), \
            min_y - (c - range):2 * range - (c + range - max_y)] = self.agents[agentID].coverageMap[min_x:max_x,min_y:max_y]

            infomap_feature[min_x - (r - range):2 * range - (r + range - max_x), \
            min_y - (c - range):2 * range - (c + range - max_y)] = self.worldMap[min_x:max_x, min_y:max_y]

            obsmap_feature[min_x - (r - range):2 * range - (r + range - max_x), \
            min_y - (c - range):2 * range - (c + range - max_y)] = self.obstacle_map[min_x:max_x, min_y:max_y]

            if s == 1:
                features = np.expand_dims(infomap_feature, axis=-1)
                features = (features,
                            np.expand_dims(obsmap_feature, axis=-1),
                            np.expand_dims(coverage_feature, axis=-1))

                features = np.concatenate(features, axis=-1)

            else:
                infomap_feature = block_reduce(infomap_feature, (s, s), np.max)
                obsmap_feature = block_reduce(obsmap_feature, (s, s), np.max)
                coverage_feature = block_reduce(coverage_feature, (s, s), np.max)

                features = (features,
                            np.expand_dims(infomap_feature, axis=-1),
                            np.expand_dims(obsmap_feature, axis=-1),
                            np.expand_dims(coverage_feature, axis=-1))
                features = np.concatenate(features, axis=-1)


        # print("{:d} {:d} {:d} {:d}".format(min_x, min_y, max_x, max_y))
        return np.array(features)

    def render(self, mode='visualise',W=800, H=800):

        if self.viewer is None:
            self.viewer = rendering.Viewer(W, H)
        size_x = W/self.worldMap.shape[0]
        size_y = H/self.worldMap.shape[1]
        min = self.worldMap
        for i in range(self.worldMap.shape[0]):
            for j in range(self.worldMap.shape[1]):
                # rending the infoMap
                shade = np.array(ZEROREWARDCOLOR) + (np.array(MAXREWARDCOLOR)-np.array(ZEROREWARDCOLOR))*(self.worldMap[i,j]/self.maxDensity)
                isAgent  = False
                for agentID in range(self.numAgents):
                    agentColor = np.array(AGENT_MINCOL) + (np.array(AGENT_MAXCOL)-np.array(AGENT_MINCOL))*(float(agentID+1)/float(self.numAgents))
                    if i == self.agents[agentID].pos[0] and j ==self.agents[agentID].pos[1]:
                        self.viewer.add_onetime(circle(i * size_x, j * size_y, size_x, size_y, agentColor))
                        isAgent = True
                if not isAgent:
                    self.viewer.add_onetime(rectangle(i * size_x, j * size_y, size_x, size_y, shade, False))


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


