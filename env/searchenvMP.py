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
from env.agents import AgentMP

class REWARD(Enum):
    PMAP = 15.0
    MAP  = 10.0
    TARGET  = +35.0
    COLLISION = -1.0
    MP = 10000

class SearchEnvMP(SearchEnv):

    '''
    Handles the motion primitive search
    '''
    def __init__(self, params_dict,args_dict):
        super().__init__(params_dict,args_dict)
        if 'home_dir' not in params_dict.keys():
            self.mp_graph_file_name = os.getcwd()+ '/env/'+params_dict['graph_file_name']
        else:
            self.mp_graph_file_name = params_dict['home_dir'] + '/env/' + params_dict['graph_file_name']
        self.load_graph()
        self.create_mp_graph_encodings()
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

    def create_mp_graph_encodings(self,bits= True):
        if bits:
            self.motionprim_tokensize = 20
        else:
            self.motionprim_tokensize = self.num_graph_nodes + self.mp_graph.edges.shape[1]
        self.mp_graph_embeddings = np.zeros((self.mp_graph.edges.shape[0],self.mp_graph.edges.shape[1],self.motionprim_tokensize))
        for j in range(self.mp_graph.edges.shape[0]):
            for k in range(self.action_size):
                if not bits:
                    node_embed = np.zeros((self.mp_graph.edges.shape[0],))
                    node_embed[j] = 1
                    edge_embed = np.zeros((self.mp_graph.edges.shape[1],))
                    if self.lookup_dictionary[j,k]>0:
                        edge_embed[self.lookup_dictionary[j,k]] = 1
                    else:
                        edge_embed -=1
                else:
                    node_embed = np.unpackbits(np.uint8([j]),count=8)
                    edge_embed = np.zeros((12,))
                    if self.lookup_dictionary[j, k] > 0:
                        edge_embed = np.unpackbits(np.uint8([k]),count=12)
                    else:
                        edge_embed -= 1
                self.mp_graph_embeddings[j,k] = np.array(node_embed.tolist()+edge_embed.tolist())

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
        mp_embeds = []
        poly_order = 0
        for j,mp in enumerate(self.minimum_action_mp_graph[index,:]):
            mp = deepcopy(mp)
            mp_embeds.append(self.mp_graph_embeddings[index, j])
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
        return np.array(action_coeffs),valids,np.array(mp_embeds)

    def get_obs_all(self):
        obs = []
        agents_actions = []
        valids = []
        position = []
        previous_actions = []
        agent_idx = []
        mp_embeds = []
        agent_budgets = []
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
            elif OBSERVER == 'RANGEwOBSwMULTIwCOV':
                obs.append(self.get_obs_range_coverage_multifov(agentID=j))

            coeffs,valid,mp_embed = self.get_mps(j)
            agents_actions.append(coeffs)
            mp_embeds.append(mp_embed)
            valids.append(valid)
            position.append([self.agents[j].pos[0]/self.reward_map_size,\
                             self.agents[j].pos[1]/self.reward_map_size])
            previous_actions.append(self.agents[j].prev_action)
            agent_idx.append(self.agents[j].index)
            if self.args_dict['FIXED_BUDGET']:
                agent_budgets.append([self.agents[j].agentBudget/self.args_dict['BUDGET']/REWARD.MP.value])

        obs_dict = dict()
        obs_dict['obs'] = obs
        obs_dict['mps'] = np.array(agents_actions)
        obs_dict['mp_embeds'] = np.stack(mp_embeds)
        obs_dict['valids'] = np.array(valids)
        obs_dict['position'] = np.array(position)
        obs_dict['previous_actions'] = np.array(previous_actions)
        obs_dict['node'] = np.array(agent_idx)
        if self.args_dict['FIXED_BUDGET']:
            obs_dict['budget'] = np.array(agent_budgets)
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
                return mp.end_state[:self.spatial_dim], next_index, is_valid, \
                       visited_states, mp.cost/mp.subclass_specific_data.get('rho', 1)/10
        # else:
        #     print('Warning: invalid MP is being selected')
        return pos, index, False, None, None

    '''
        Creates the world with reward map and agents
        '''

    def createWorld(self, rewardMap=None):
        super().createWorld(rewardMap)
        # Creating the agents
        if self.args_dict['FIXED_BUDGET']:
            agentBudget = self.args_dict['BUDGET']*REWARD.MP.value
        else:
            agentBudget = None

        if SPAWN_RANDOM_AGENTS:
            row = np.random.randint(self.pad_size,self.reward_map_size+self.pad_size,(self.numAgents,))
            col = np.random.randint(self.pad_size, self.reward_map_size + self.pad_size, (self.numAgents,))

            self.agents = [
                AgentMP(j, row[j],col[j], \
                      self.reward_map_size, self.pad_size, self.world_map_size,\
                        self.minimum_action_mp_graph,self.lookup_dictionary,\
                        self.spatial_dim,self.mp_graph.num_tiles,agentBudget) for j in range(self.numAgents)]
        else:
            self.agents = [AgentMP(j,self.reward_map_size+j+1,self.reward_map_size+j+1,\
                                 self.reward_map_size,self.pad_size,self.world_map_size,\
                        self.minimum_action_mp_graph,self.lookup_dictionary,\
                                   self.spatial_dim,self.mp_graph.num_tiles,agentBudget) for j in range(self.numAgents)]

    def step_all(self,action_dict):
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
        for agent_idx,_ in enumerate(self.agents):
            _,valids,_ = self.get_mps(agent_idx)
            if np.array(valids).sum()==0:
                done = True
            if self.args_dict['FIXED_BUDGET']:
                if self.agents[agent_idx].agentBudget != None and self.agents[agent_idx].agentBudget<0:
                    agentsDone = agentsDone or True

        if self.args_dict['FIXED_BUDGET']:
            done = done or agentsDone

        return rewards, done

    def step(self,agentID,action):
        """
        Given the current state and action, return the next state
        """
        if not self.args_dict['FIXED_BUDGET'] or (self.args_dict['FIXED_BUDGET']\
                                              and self.agents[agentID].agentBudget>0):
            valid,visited_states,cost = self.agents[agentID].updatePos(action)
            reward = 0

            if valid:
                for state in visited_states.T:
                    reward += self.worldMap[state[0], state[1]]
                    self.worldMap[state[0], state[1]] = 0
                reward -= cost/REWARD.MP.value
            elif visited_states is not None:
                #reward += REWARD.COLLISION.value*(visited_states.shape[0]+1)
                reward += REWARD.COLLISION.value
            else:
                reward += REWARD.COLLISION.value*1.5

            reward += self.worldMap[int(self.agents[agentID].pos[0]), int(self.agents[agentID].pos[1])]
            self.worldMap[self.agents[agentID].pos[0], self.agents[agentID].pos[1]] = 0
            self.agents[agentID].updateMap(self.worldMap)
            return reward
        return 0

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


