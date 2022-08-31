import gym
import time
import numpy as np
from gym.envs.classic_control import rendering
from enum import Enum
from env.render import *
from env.searchenv import *
from env.searchenvMP import *
import math
import matplotlib.pyplot as plt
import GPy
from params import *
from motion_primitives_py import MotionPrimitiveLattice
from copy import deepcopy
from env.agents import AgentGP
import os
from env.Metrics import Metrics

ACTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]
ZEROREWARDCOLOR = (1.0,1.0,1.0)
MAXREWARDCOLOR = (0.0,0.0,0.8)
ZEROENTROPYCOLOR = (1.0,1.0,1.0)
MAXENTROPYCOLOR = (0.0,0.5,0.5)
TARGET_FOUND = (0.5,1.0,1.0)
TARGET_COLOR = (0.0,0.0,0.0)
AGENT_MINCOL = (0.5,0.3,0.3)
AGENT_MAXCOL = (1.,0.3,0.3)
TRAJECTORY_COLOR = (0.6,0.0,0.8)


'''
Environment for target search
'''
class GPEnvMP(SearchEnvMP):
    def __init__(self,params_dict,args_dict):
        self.params_dict = params_dict
        super().__init__(params_dict,args_dict)
        self.defaultBelief = params_dict['defaultBelief']
        self.sensor_params = params_dict['sensor_params']
        self.numrand_targets = params_dict['num_targets']
        self.targetBeliefThresh = params_dict['targetBeliefThresh']
        self.noise_max_var = params_dict['noise_max_var']
        self.noise_min_var = params_dict['noise_min_var']
        self.metrics = Metrics()


    def reset(self,rewardMap=None,targetMap = None,original_target_dismap =None):
        self.createWorld(rewardMap,targetMap,original_target_dismap)

    def createWorld(self, rewardMap=None,targetMap = None,original_target_dismap =None):
        # this is the proabbility map
        super().createWorld(rewardMap)
        num_targets = np.random.randint(self.numrand_targets[0], self.numrand_targets[1])

        '''
        Randomly select a few targets as per reward map  
        '''


        # Create targets on the belief
        flat_reward_map = self.rewardMap.flatten()

        self.target_locations = np.random.choice(a=flat_reward_map.size,size = num_targets,p = flat_reward_map/flat_reward_map.sum()).tolist()

        if targetMap is None:
            self.targetMap = np.zeros(self.rewardMap.shape)
            self.targetList = []
            for j,target in enumerate(self.target_locations):
                target_index= np.unravel_index(target,self.rewardMap.shape)
                self.targetMap[target_index[0],target_index[1]] = 1
                self.targetList.append(list(target_index))
        else:
            self.targetMap = targetMap
            self.targetList = []
            for j in range(targetMap.shape[0]):
                for k in range(targetMap.shape[1]):
                    if targetMap[j,k]==1:
                        target_index = [j,k]
                        self.targetList.append(list(target_index))


        self.worldTargetMap = np.zeros((self.world_map_size, self.world_map_size))  # for boundaries
        self.worldTargetMap[self.pad_size:self.pad_size + self.reward_map_size, \
        self.pad_size:self.pad_size + self.reward_map_size] = self.targetMap

        self.orig_worldTargetMap = np.copy(self.worldTargetMap)


        # controls how randomnly placed the targets are
        target_randomiser = np.clip(np.random.normal(0.15,0.5),0,1) * self.params_dict['TARGET_RANDOM_SCALE']

        if rewardMap is None:
            random_noise = np.clip(np.abs(np.random.normal(size=(self.reward_map_size,self.reward_map_size))*\
                              self.params_dict['TARGET_RANDOM_SCALE']),1e-6,0.02)

            # Setting up the original prior
            self.orig_target_distribution_map = np.clip(self.orig_target_distribution_map,0.10,0.90)

            #Noising the prior centres and variances

            for j,_ in enumerate(self.prior_centres):
                self.prior_centres[j] += int(np.random.randint(-self.params_dict['MAXDISP'],self.params_dict['MAXDISP']))
                self.prior_centres[j] = np.clip(self.prior_centres[j],[0,0],self.worldMap.shape)
                self.prior_vars[j][0][0] = (1-target_randomiser)* self.prior_vars[j][0][0] +\
                                           (target_randomiser)*np.random.rand(1)*(self.max_var - self.min_var) + self.min_var
                self.prior_vars[j][1][1] = (1-target_randomiser)* self.prior_vars[j][1][1] +\
                                           target_randomiser*np.random.rand(1) * (self.max_var - self.min_var) + self.min_var
                self.prior_vars[j][0][1] = (target_randomiser)*np.random.rand(1) * self.min_var / 2 + \
                                           (1-target_randomiser)*self.prior_vars[j][0][1]
                self.prior_vars[j][1][0] = self.prior_vars[j][0][1]

            randX = []
            randY = []
            prob = 1 - flat_reward_map/flat_reward_map.max()
            random_locations = np.random.choice(a=flat_reward_map.size, size=self.params_dict['RANDOM_CENTRES'],
                                                     p=prob/prob.sum()).tolist()
            for j in range(self.params_dict['RANDOM_CENTRES']):
                random_loc = np.unravel_index(random_locations[j], self.rewardMap.shape)
                #randX.append([random_loc,random_locations])
                randX.append(random_loc)
                y = np.zeros((2,2))
                y[0][0] = np.random.rand(1)*(self.noise_max_var - self.noise_min_var) + self.noise_min_var
                y[1][1] = np.random.rand(1)*(self.noise_max_var - self.noise_min_var) + self.noise_min_var
                y[0][1] = np.random.rand(1)*self.noise_min_var/2
                y[1][0] = y[0][1]
                self.prior_vars.append(y)

            randX = np.array(randX)
            # Y = np.clip(np.array(Y),self.min_var,self.max_var)
            Xs = np.concatenate((self.prior_centres,randX),axis=0)


            noiseMap = self.createRewardMap(Xs,self.prior_vars)

            self.rewardMap = np.power(target_randomiser,1)*self.rewardMap + (1-np.power(target_randomiser,1))*noiseMap

            self.obstacle_map = np.zeros((self.world_map_size, self.world_map_size)) + 1
            self.worldMap = np.zeros((self.world_map_size, self.world_map_size))  # for boundaries

            # ensuring that reward map is scaled to represent an I.I.D.
            self.worldMap[self.pad_size:self.pad_size + self.reward_map_size, \
            self.pad_size:self.pad_size + self.reward_map_size] = \
                self.rewardMap/(self.rewardMap.max() - self.rewardMap.min()) # capped b/w 0 and 1


            self.obstacle_map[self.pad_size:self.pad_size + self.reward_map_size, \
                              self.pad_size:self.pad_size + self.reward_map_size] =\
                np.zeros((self.reward_map_size, self.reward_map_size))
            self.orig_worldMap = deepcopy(self.worldMap)

        self.worldBeliefMap = self.worldMap

        # self.worldBeliefMap = np.clip(np.clip(self.worldBeliefMap, 0, 1) / 1.5 + self.defaultBelief, 0, 1
        self.worldBeliefMap = np.clip(self.worldBeliefMap, 0.10, 0.90)
        # For observations
        self.worldMap = self.worldBeliefMap

        if original_target_dismap is not None:
            self.orig_target_distribution_map = deepcopy(original_target_dismap)

        if self.args_dict['FIXED_BUDGET']:
            agentBudget = self.args_dict['BUDGET'] * REWARD.MP.value
        else:
            agentBudget = None

        # Creating the agents
        if SPAWN_RANDOM_AGENTS:
            row = np.random.randint(self.pad_size,self.reward_map_size+self.pad_size,(self.numAgents,))
            col = np.random.randint(self.pad_size, self.reward_map_size + self.pad_size, (self.numAgents,))
            self.agents = [
                AgentGP(j, row[j],col[j], \
                      self.reward_map_size, self.pad_size, self.world_map_size,\
                        self.minimum_action_mp_graph,self.lookup_dictionary,\
                        self.spatial_dim,self.mp_graph.num_tiles,self.sensor_params,agentBudget) for j in range(self.numAgents)]
        else:
            self.agents = [AgentGP(j,self.reward_map_size+int(j/(int(j/2))),self.reward_map_size+(j%(int(j/2))),\
                                 self.reward_map_size,self.pad_size,self.world_map_size,\
                        self.minimum_action_mp_graph,self.lookup_dictionary,self.spatial_dim,\
                                   self.mp_graph.num_tiles,self.sensor_params,agentBudget) for j in range(self.numAgents)]

        for agent in self.agents:
            agent.initBeliefMap(self.worldBeliefMap)

        self.metrics.update(self.worldBeliefMap,self.worldTargetMap)

    def step_all(self, action_dict):
        rewards = []
        for j in range(self.numAgents):
            r = self.step(agentID=j, action=action_dict[j])
            rewards.append(r)
        done = False
        if np.all(np.abs(self.worldMap[self.pad_size:self.pad_size + self.reward_map_size, \
                         self.pad_size:self.pad_size + self.reward_map_size]) < 0.1):
            done = True

        # If no agent has valid motion primitives terminate
        if self.args_dict['FIXED_BUDGET']:
            agentsDone = False
        for agent_idx, _ in enumerate(self.agents):
            _, valids,_ = self.get_mps(agent_idx)
            if np.array(valids).sum() == 0:
                done = True
            if self.args_dict['FIXED_BUDGET']:
                if self.agents[agent_idx].agentBudget != None and self.agents[agent_idx].agentBudget < 0:
                    agentsDone = agentsDone or True

        if self.args_dict['FIXED_BUDGET']:
            done = done or agentsDone

        return rewards, done

    def get_final_metrics(self):
        return self.metrics.compute_metrics(self.worldBeliefMap,self.worldTargetMap)

    def step(self, agentID, action):
        """
        Given the current state and action, return the next state
        """
        valid, visited_states, cost = self.agents[agentID].updatePos(action)
        reward = 0
        initialBeliefMap = self.worldBeliefMap.copy()
        initialTargetMap = self.worldTargetMap.copy()

        if valid:
            measurements = self.agents[agentID].updateInfoTarget(visited_states.T,self.worldTargetMap,self.targetBeliefThresh)
            self.updateRewardTarget(visited_states.T,measurements,agentID)
            reward += self.getReward(initialBeliefMap,self.worldBeliefMap)
            reward -= cost / REWARD.MP.value
            targets_found = (self.worldTargetMap==2).sum() - (initialTargetMap==2).sum()
            reward += targets_found*REWARD.TARGET.value/len(self.targetList)

        elif visited_states is not None:
            # reward += REWARD.COLLISION.value*(visited_states.shape[0]+1)
            reward += REWARD.COLLISION.value
        else:
            reward += REWARD.COLLISION.value * 1.5


        #reward += self.worldMap[int(self.agents[agentID].pos[0]), int(self.agents[agentID].pos[1])]
        #self.worldMap[self.agents[agentID].pos[0], self.agents[agentID].pos[1]] = 0
        self.worldMap = self.worldBeliefMap.copy()
        #self.agents[agentID].updateMap(self.worldMap)
        return reward

    def getEntropy(self,belief):
        entropy = belief*np.log(np.clip(belief,1e-7,1)) + (1-belief)*np.log(np.clip(1-belief,1e-7,1))
        return entropy

    def getReward(self,oldBelief,newBelief):
        oldentropy = self.getEntropy(oldBelief).mean()
        newEntropy = self.getEntropy(newBelief).mean()
        # Normalising total entropy reward to between 0 and 100
        return (newEntropy-oldentropy)/(np.log(2))*REWARD.MAP.value

    '''
    Returns the total entropy at the desired locations
    '''
    def updateRewardTarget(self,visited_states,measurement_list,agentID):
        for state,measurement in zip(visited_states.tolist(),measurement_list):
            r = state[0]
            c = state[1]
            range_ = int(self.sensor_params['sensor_range']/2)
            min_x = np.max([r - range_, 0])
            min_y = np.max([c - range_, 0])
            max_x = np.min([r + range_+1, self.worldBeliefMap.shape[0]])
            max_y = np.min([c + range_+1, self.worldBeliefMap.shape[1]])
            for j in range(min_x,max_x):
                for k in range(min_y,max_y):
                    logodds_b_map = np.log(self.worldBeliefMap[j,k]/(1-self.worldBeliefMap[j,k]))
                    #log_odds_prior_map = np.log(0.5*self.orig_worldTargetMap[j,k]/(1-0.5*self.orig_worldTargetMap[j,k]))
                    log_odds_prior_map = np.log(1),
                    sensor_log_odds = np.log((1-self.sensor_params['sensor_unc'][j-(r-range_),k-(c-range_)])/ \
                                            self.sensor_params['sensor_unc'][j-(r-range_),k-(c-range_)])
                    #print(sensor_log_odds)
                    if measurement[j-(r-range_),k-(c-range_)]==0:
                        logodds_b_map -= (sensor_log_odds + log_odds_prior_map)
                    else:
                        logodds_b_map += (sensor_log_odds - log_odds_prior_map)
                    self.worldBeliefMap[j,k] = 1/(np.exp(-logodds_b_map)+1)

                # Update whether target is found

                    if self.worldBeliefMap[j,k]>=self.targetBeliefThresh and self.worldTargetMap[j,k]>0:
                        self.worldTargetMap[j,k]=2


    def render(self, mode='visualise', W=400, H=400):

        if self.viewer is None:
            self.viewer = rendering.Viewer(3*W, H)
        size_x = W / self.worldMap.shape[0]
        size_y = H / self.worldMap.shape[1]
        min = self.worldMap

        # which map is rendered
        MAP = 0
        for i in range(self.worldMap.shape[0]):
            for j in range(self.worldMap.shape[1]):
                # rending the infoMap
                shade = np.array(ZEROREWARDCOLOR) + (np.array(MAXREWARDCOLOR) - np.array(ZEROREWARDCOLOR)) * (
                            self.worldMap[i, j] / self.worldMap.max())
                isAgent = False
                for agentID in range(self.numAgents):
                    agentColor = np.array(AGENT_MINCOL) + (np.array(AGENT_MAXCOL) - np.array(AGENT_MINCOL)) * (
                                float(agentID + 1) / float(self.numAgents))

                    if i == self.agents[agentID].pos[0] and j == self.agents[agentID].pos[1]:
                        self.viewer.add_onetime(circle(i * size_x, j * size_y, size_x, size_y, agentColor))
                        isAgent = True
                    # if self.agents[agentID].trajectory[i,j]==1:
                    #     self.viewer.add_onetime(rectangle(i * size_x, j * size_y, size_x, size_y, TRAJECTORY_COLOR))
                    #     isAgent =True
                if not isAgent:
                    self.viewer.add_onetime(rectangle(i * size_x, j * size_y, size_x, size_y, shade, False))

                if self.worldTargetMap[i,j]==1:
                    self.viewer.add_onetime(rectangle(i * size_x, j * size_y, size_x, size_y, TARGET_COLOR, False))
                elif self.worldTargetMap[i,j]==2:
                    self.viewer.add_onetime(rectangle(i * size_x, j * size_y, size_x, size_y, TARGET_FOUND, False))

        # which map is rendered
        MAP = 1
        entropyMap = - self.getEntropy(self.worldBeliefMap)

        for i in range(self.worldMap.shape[0]):
            for j in range(self.worldMap.shape[1]):
                # rending the infoMap
                shade = np.array(ZEROENTROPYCOLOR) + (np.array(MAXENTROPYCOLOR) - np.array(ZEROENTROPYCOLOR)) * (
                        entropyMap[i, j] / entropyMap.max())
                isAgent = False
                for agentID in range(self.numAgents):
                    agentColor = np.array(AGENT_MINCOL) + (np.array(AGENT_MAXCOL) - np.array(AGENT_MINCOL)) * (
                            float(agentID + 1) / float(self.numAgents))

                    if i == self.agents[agentID].pos[0] and j == self.agents[agentID].pos[1]:
                        self.viewer.add_onetime(circle(W + i * size_x, j * size_y, size_x, size_y, agentColor))
                        isAgent = True
                    # if self.agents[agentID].trajectory[i,j]==1:
                    #     self.viewer.add_onetime(rectangle(i * size_x, j * size_y, size_x, size_y, TRAJECTORY_COLOR))
                    #     isAgent =True
                if not isAgent:
                    self.viewer.add_onetime(rectangle(W + i * size_x, j * size_y, size_x, size_y, shade, False))

                if self.worldTargetMap[i, j] == 1:
                    self.viewer.add_onetime(rectangle(W + i * size_x, j * size_y, size_x, size_y, TARGET_COLOR, False))
                elif self.worldTargetMap[i, j] == 2:
                    self.viewer.add_onetime(rectangle(W + i * size_x, j * size_y, size_x, size_y, TARGET_FOUND, False))

        # original prior
        for i in range(self.worldMap.shape[0]):
            for j in range(self.worldMap.shape[1]):
                # rending the infoMap
                shade = np.array(ZEROREWARDCOLOR) + (np.array(MAXREWARDCOLOR) - np.array(ZEROREWARDCOLOR)) * (
                        self.orig_target_distribution_map[i, j] / self.orig_target_distribution_map.max())
                isAgent = False
                for agentID in range(self.numAgents):
                    agentColor = np.array(AGENT_MINCOL) + (np.array(AGENT_MAXCOL) - np.array(AGENT_MINCOL)) * (
                            float(agentID + 1) / float(self.numAgents))

                    if i == self.agents[agentID].pos[0] and j == self.agents[agentID].pos[1]:
                        self.viewer.add_onetime(circle(2*W + i * size_x, j * size_y, size_x, size_y, agentColor))
                        isAgent = True
                    # if self.agents[agentID].trajectory[i,j]==1:
                    #     self.viewer.add_onetime(rectangle(i * size_x, j * size_y, size_x, size_y, TRAJECTORY_COLOR))
                    #     isAgent =True
                if not isAgent:
                    self.viewer.add_onetime(rectangle(2*W + i * size_x, j * size_y, size_x, size_y, shade, False))

                if self.worldTargetMap[i, j] == 1:
                    self.viewer.add_onetime(rectangle(2*W + i * size_x, j * size_y, size_x, size_y, TARGET_COLOR, False))
                elif self.worldTargetMap[i, j] == 2:
                    self.viewer.add_onetime(rectangle(2*W + i * size_x, j * size_y, size_x, size_y, TARGET_FOUND, False))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')