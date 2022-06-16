import gym
import time
import numpy as np
from gym.envs.classic_control import rendering
from enum import Enum
from env.render import *
from env.searchenv import *
import math
import matplotlib.pyplot as plt
import GPy
from params import *
from motion_primitives_py import MotionPrimitiveLattice
from copy import deepcopy
import os

class AgentMP():
    def __init__(self,ID,row,col,map_size,pad,world_size,mp_graph,lookup,spatial_dim,tiles):
        self.ID = ID
        self.pos = np.array([row,col])
        self.reward_map_size = map_size
        self.pad = pad
        self.tiles = tiles
        self.world_size = world_size
        self.worldMap = None
        self.visited_states = None
        self.mp_graph = mp_graph
        self.lookup = lookup
        self.index = 0
        self.spatial_dim = spatial_dim
        self.prev_action = 0

    def updateMap(self,worldMap):
        self.worldMap= worldMap

    def updatePos(self,action):
        mp = deepcopy(self.mp_graph[self.index, action])
        if mp is not None:
            #mp.translate_start_position(self.pos)
            #_, sp = mp.get_sampled_position()
            # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
            #visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
            is_valid,visited_states = self.isValidMP(mp)
            if is_valid:
                next_index = self.lookup[self.index, action]
                next_index = int(np.floor(next_index / self.tiles))
                self.index = next_index
                self.pos = np.round(mp.end_state[:self.spatial_dim]).astype(int)
                #print("{:d} {:d} {:d} {:d}".format(self.pos[0], self.pos[1], visited_states[0,0], visited_states[1,0]))
                self.visited_states = visited_states
                return is_valid, visited_states, mp.cost / mp.subclass_specific_data.get('rho', 1) / 10
            return False,visited_states,None
        self.prev_action = action
        return False,None, None

    def isValidMP(self,mp):
        is_valid = mp.is_valid
        mp.translate_start_position(self.pos)
        _, sp = mp.get_sampled_position()
        # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
        visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
        is_valid = is_valid and self.isValidPoses(visited_states)
        return is_valid,visited_states

    def isValidPoses(self, poses):
        is_valid = True
        for state in poses.T:
            is_valid = is_valid and self.isValidPos(state)
        return is_valid

    def isValidPos(self, pos):
        is_valid = (np.array(pos - self.pad) > -1).all()
        is_valid = is_valid and (np.array(pos + self.pad) < self.world_size).all()
        return is_valid


class SearchEnvMP(SearchEnv):

    '''
    Handles the motion primitive search
    '''
    def __init__(self, params_dict):
        super().__init__(params_dict)
        if 'home_dir' not in params_dict.keys():
            self.mp_graph_file_name = os.getcwd()+ '/env/'+params_dict['graph_file_name']
        else:
            self.mp_graph_file_name = params_dict['home_dir'] + '/env/' + params_dict['graph_file_name']
        self.load_graph()
        self.spatial_dim = self.mp_graph.num_dims

    def load_graph(self):
        self.mp_graph = MotionPrimitiveLattice.load(self.mp_graph_file_name)
        self.mp_graph.edges = self.mp_graph.edges.T
        self.action_size = max([len([j for j in i if j != None]) for i in self.mp_graph.edges])
        self.num_graph_nodes = self.mp_graph.edges.shape[0]
        #print(self.num_actions)
        self.minimum_action_mp_graph = np.empty((self.mp_graph.edges.shape[0], self.action_size), dtype=object)
        self.lookup_dictionary = np.ones_like(self.minimum_action_mp_graph)*-1
        for i in range(self.mp_graph.edges.shape[0]):
            k = 0
            for j in range(self.mp_graph.edges.shape[1]):
                if self.mp_graph.edges[i, j] is not None:
                    self.minimum_action_mp_graph[i, k] = self.mp_graph.edges[i, j]
                    self.lookup_dictionary[i, k] = j
                    k += 1
        self.num_vertices = len(self.mp_graph.vertices)
        self.num_other_states = self.minimum_action_mp_graph.shape[0]
        self.num_actions_per_state = [len([j for j in i if j != None]) for i in self.minimum_action_mp_graph]

    ''' Get future states from the primitive for encoding'''
    def get_states(self,pos,action,index):
        mp = deepcopy(self.minimum_action_mp_graph[index, action])
        if mp is not None:
            mp.translate_start_position(pos)
            _, sp = mp.get_sampled_position()
            visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
            is_valid = mp.is_valid
            for state in visited_states.T:
                is_valid = is_valid and self.isValidPos(state)
            if is_valid:
                return visited_states - pos
            else:
                return np.zeros(visited_states.shape)

        return None

    ''' Get future states from the primitive for encoding'''

    def get_mps(self, agent_idx):
        action_coeffs =[]
        index = self.agents[agent_idx].index
        valids = []
        poly_order = 0
        for mp in self.minimum_action_mp_graph[index,:]:
            mp = deepcopy(mp)
            if mp is not None:
                action_coeffs.append(np.array(mp.poly_coeffs).reshape(-1))
                if self.agents[agent_idx].isValidMP(mp):
                    valids.append(1)
                else:
                    valids.append(0)
                poly_order = 2*(mp.poly_order+1)
            else:
                action_coeffs.append(np.zeros(poly_order))
                valids.append(0)
        return np.array(action_coeffs),valids

    def get_obs_all(self):
        obs = []
        agents_actions = []
        valids = []
        position = []
        previous_actions = []
        agent_idx = []
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
            elif OBSERVER == 'RANGEwOBSwMULTI':
                obs.append(self.get_obs_range_wobs_multi(agentID=j))

            coeffs,valid = self.get_mps(j)
            agents_actions.append(coeffs)
            valids.append(valid)
            position.append([self.agents[j].pos[0]/self.reward_map_size,\
                             self.agents[j].pos[1]/self.reward_map_size])
            previous_actions.append(self.agents[j].prev_action)
            agent_idx.append(self.agents[j].index)

        obs_dict = dict()
        obs_dict['obs'] = obs
        obs_dict['mps'] = np.array(agents_actions)
        obs_dict['valids'] = np.array(valids)
        obs_dict['position'] = np.array(position)
        obs_dict['previous_actions'] = np.array(previous_actions)
        obs_dict['node'] = np.array(agent_idx)
        return obs_dict


    def get_next_state(self, pos, action, index):
        # reset_map_index = int(np.floor(self.curr_state_index / self.mp_graph.num_tiles))
        mp = deepcopy(self.minimum_action_mp_graph[index, action])
        if mp is not None:
            mp.translate_start_position(pos)
            _, sp = mp.get_sampled_position()
            # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
            visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
            is_valid = mp.is_valid
            for state in visited_states.T:
                is_valid = is_valid and self.isValidPos(state)
            if is_valid:
                next_index = self.lookup_dictionary[index, action]
                next_index = int(np.floor(next_index/self.mp_graph.num_tiles))
                return mp.end_state[:self.spatial_dim], next_index, is_valid, visited_states, mp.cost/mp.subclass_specific_data.get('rho', 1)/10
        # else:
        #     print('Warning: invalid MP is being selected')
        return pos, index, False, None, None

    '''
        Creates the world with reward map and agents
        '''

    def createWorld(self, rewardMap=None):
        super().createWorld(rewardMap)
        # Creating the agents
        if SPAWN_RANDOM_AGENTS:
            row = np.random.randint(self.pad_size,self.reward_map_size+self.pad_size,(self.numAgents,))
            col = np.random.randint(self.pad_size, self.reward_map_size + self.pad_size, (self.numAgents,))
            self.agents = [
                AgentMP(j, row[j],col[j], \
                      self.reward_map_size, self.pad_size, self.world_map_size,\
                        self.minimum_action_mp_graph,self.lookup_dictionary,self.spatial_dim,self.mp_graph.num_tiles) for j in range(self.numAgents)]
        else:
            self.agents = [AgentMP(j,self.reward_map_size+int(j/(int(j/2))),self.reward_map_size+(j%(int(j/2))),\
                                 self.reward_map_size,self.pad_size,self.world_map_size,\
                        self.minimum_action_mp_graph,self.lookup_dictionary,self.spatial_dim,self.mp_graph.num_tiles) for j in range(self.numAgents)]

    def step_all(self,action_dict):
        rewards = []
        for j in range(self.numAgents):
            r = self.step(agentID=j, action=action_dict[j])
            rewards.append(r)
        done = False
        if np.all(np.abs(self.worldMap[self.pad_size:self.pad_size + self.reward_map_size, \
                         self.pad_size:self.pad_size + self.reward_map_size]) < 0.1):
            done = True
        return rewards, done

    def step(self,agentID,action):
        """
        Given the current state and action, return the next state
        """
        valid,visited_states,cost = self.agents[agentID].updatePos(action)
        reward = 0
        if valid:
            for state in visited_states.T:
                reward += self.worldMap[state[0], state[1]]
                self.worldMap[state[0], state[1]] = 0
            reward -= cost/10000
        elif visited_states is not None:
            #reward += REWARD.COLLISION.value*(visited_states.shape[0]+1)
            reward += REWARD.COLLISION.value*2
        else:
            reward += REWARD.COLLISION.value*3

        reward += self.worldMap[int(self.agents[agentID].pos[0]), int(self.agents[agentID].pos[1])]
        self.worldMap[self.agents[agentID].pos[0], self.agents[agentID].pos[1]] = 0
        self.agents[agentID].updateMap(self.worldMap)
        return reward

    def render(self, mode='visualise', W=800, H=800):

        if self.viewer is None:
            self.viewer = rendering.Viewer(W, H)
        size_x = W / self.worldMap.shape[0]
        size_y = H / self.worldMap.shape[1]
        min = self.worldMap
        for i in range(self.worldMap.shape[0]):
            for j in range(self.worldMap.shape[1]):
                # rending the infoMap
                shade = np.array(ZEROREWARDCOLOR) + (np.array(MAXREWARDCOLOR) - np.array(ZEROREWARDCOLOR)) * (
                            self.worldMap[i, j] / self.maxDensity)
                isAgent = False
                for agentID in range(self.numAgents):
                    agentColor = np.array(AGENT_MINCOL) + (np.array(AGENT_MAXCOL) - np.array(AGENT_MINCOL)) * (
                                float(agentID + 1) / float(self.numAgents))

                    if i == self.agents[agentID].pos[0] and j == self.agents[agentID].pos[1]:
                        self.viewer.add_onetime(circle(i * size_x, j * size_y, size_x, size_y, agentColor))
                        isAgent = True
                if not isAgent:
                    self.viewer.add_onetime(rectangle(i * size_x, j * size_y, size_x, size_y, shade, False))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


