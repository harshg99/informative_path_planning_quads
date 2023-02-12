
import gym
import time
from gym.envs.classic_control import rendering
from enum import Enum
from env.render import *
import math
import matplotlib.pyplot as plt
import GPy
from params import *
from skimage.measure import block_reduce
from env.agents import AgentSemantic
from skimage.transform import resize
from copy import deepcopy

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
from env.SemanticMap import GPSemanticMap
from env.Metrics import *
from skimage.transform import resize
import PIL
import cupy
from typing import  *
# Reward definitions

'''
Reward Class
'''
class REWARD(Enum):
    STEP        = -0.1
    STEPDIAGONAL= -0.1*np.sqrt(2)
    STAY      = -0.5
    PMAP = 15.0
    MAP  = 10.0
    TARGET  = +35.0
    COLLISION = -1.0
    MP = 10000

'''
Constant Envrionment Variables for rendering
'''
ACTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]

#DEBUG = True



# TODO: Define the semantic mapping class : THis should  be able to store a semantic image
#   The class has a helper function that provides a random semantic map on initialization
#   This class should basically store a map of any kind


class MPObject:

    def __init__(self,minimum_action_mp_grpah = None,
                 lookup_dict = None,
                 spatial_dim = None,
                 num_tiles = None):
        self.mp_graph = minimum_action_mp_grpah
        self.lookup = lookup_dict
        self.spatial_dim = spatial_dim
        self.num_tiles = num_tiles


class VisBuffer:
    def __init__(self,num_agents):
        self.num_agents = num_agents
        self.agent_pos_buff = {}
        self.belief_semantic_map_buff = {}
        self.global_semantic_map_buff = None
        self.clear_buffer()

    def clear_buffer(self):

        for idx in range(self.num_agents):
            self.agent_pos_buff[idx] = None
            self.belief_semantic_map_buff[idx] = None
        self.globalSemanticMappBuff = None

    def add_buffer(self,agentID,agent_pos,beliefMap):
        if self.agent_pos_buff[agentID] is None:
            self.agent_pos_buff[agentID] = []
        self.agent_pos_buff[agentID].append(deepcopy(agent_pos))

        if self.belief_semantic_map_buff[agentID] is None:
            self.belief_semantic_map_buff[agentID] = []
        self.belief_semantic_map_buff[agentID].append(deepcopy(beliefMap.detected_semantic_map.get()))

    def add_buffer_global(self, beliefMap):

        if self.globalSemanticMappBuff is None:
            self.global_semantic_map_buff = []
        self.global_semantic_map_buff.append(deepcopy(beliefMap.detected_semantic_map))

    def get_buffer_size(self):
        return len(self.global_semantic_map_buff)

    def get_buffer_element(self,idx):

        positions = []
        for agent in self.agent_pos_buff:
            positions.append(self.agent_pos_buff[agent][idx])

        return positions,self.globalSemanticMappBuff[idx]

class GPSemanticGym(gym.Env):

    '''
    Handles the motion primitive search
    '''
    def __init__(self, params_dict,args_dict):
        super(GPSemanticGym, self).__init__()

        # Necessary parameters for semantic environments
        self.numAgents = params_dict['numAgents']
        self.args_dict = args_dict

        self.env_params = params_dict
        # Parameters to create training maps
        self.semantic_map_size = params_dict['rewardMapSize']  # m m size of the environment
        self.resolution = params_dict['resolution']
        self.pad_size = params_dict['pad_size']

        self.world_map_size = self.semantic_map_size + 2 * self.pad_size

        self.action_size = len(ACTIONS)
        self.episode_length = params_dict['episode_length']
        self.sensor_range = params_dict['sensor_range']
        self.max_steps = params_dict['episode_length']
        self.scale = args_dict['SCALE']

        #Observation space setting
        if self.args_dict['SET_SEED']:
            self.seed = params_dict['seed']
        self.viewer = None

        self.RANGE = self.args_dict['RANGE'] * self.args_dict['RESOLUTION']

        if self.args_dict['OBSERVER'] == 'RANGE':
            self.input_size = [2 * self.RANGE, 2 * self.RANGE, 1]
        elif self.args_dict['OBSERVER'] == 'RANGEwOBS':
            self.input_size = [2 * self.RANGE, 2 * self.RANGE, 2]
        elif self.args_dict['OBSERVER'] == 'RANGEwOBSwNEIGH':
            self.input_size = [2 * (self.RANGE + 1), 2 * (self.RANGE + 1), 2]
        elif self.args_dict['OBSERVER'] == 'RANGEwOBSwPENC':
            self.input_size = [2 * self.RANGE, 2 * self.RANGE, 4]
        elif self.args_dict['OBSERVER'] == 'RANGEwOBSwMULTI':
            self.map_length = 2
            self.input_size = [2 * self.RANGE, 2 * self.RANGE, len(self.scale) * 2]
        elif self.args_dict['OBSERVER'] == 'RANGEwOBSwMULTIwCOV':
            self.map_length = 2
            self.input_size = [2 * self.RANGE, 2 * self.RANGE, len(self.scale) * 3]

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=self.input_size, dtype=np.float)

        # Motion primitve library parameters
        if 'home_dir' not in params_dict.keys():
            self.mp_graph_file_name = os.getcwd()+ '/env/'+params_dict['graph_file_name']
        else:
            self.mp_graph_file_name = params_dict['home_dir'] + '/env/' + params_dict['graph_file_name']
        self.load_graph()
        self.create_mp_graph_encodings()
        self.spatial_dim = self.mp_graph.num_dims

        self.sensor_params = params_dict['sensor_params']
        self.target_noise_scale = params_dict['TARGET_NOISE_SCALE']  # scale for setting random locations
        self.random_centres = params_dict['RANDOM_CENTRES']
        self.centre_size = params_dict['CENTRE_SIZE']


        # Initialize the maps
        map_config_dict = {
            'resolution':self.env_params['resolution'],
            'world_map_size':self.world_map_size,
            'padding':self.pad_size,
            'num_semantics':self.env_params['num_semantics'],
            'target_belief_thresh':self.env_params['TargetBeliefThresh']
        }
        self.ground_truth_semantic_map = GPSemanticMap(map_config_dict,isGroundTruth=True)
        self.belief_semantic_map = GPSemanticMap(map_config_dict)

        # Initlize the metrics
        self.metrics = SemanticMetrics()

        # Visualize buffer
        self.buffer = VisBuffer(self.numAgents)
        self.renderer = QuadRender()

        # Randommly chhosing trainignand tagret maps
        shuffled_indices = np.arange(0,self.env_params['TOTAL_MAPS'])
        np.random.shuffle(shuffled_indices)
        self.training_map_indices = shuffled_indices[0:int(self.env_params['TOTAL_MAPS']*self.env_params['TRAIN_PROP'])]
        self.test_map_indices = shuffled_indices[int(self.env_params['TOTAL_MAPS']*self.env_params['TRAIN_PROP']):]

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
            if self.args_dict['OBSERVER'] == 'RANGE':
                obs.append(self.get_obs_ranged(agentID=j))
            elif self.args_dict['OBSERVER'] == 'RANGEwOBS':
                obs.append(self.get_obs_ranged_wobs(agentID=j))
            elif self.args_dict['OBSERVER'] == 'RANGEwOBSwPENC':
                obs.append(self.get_obs_ranged_wobspenc(agentID=j))
            elif self.args_dict['OBSERVER'] == 'RANGEwOBSwMULTI':
                obs.append(self.get_obs_range_wobs_multi(agentID=j))
            elif self.args_dict['OBSERVER'] == 'RANGEwOBSwMULTIwCOV':
                obs.append(self.get_obs_range_coverage_multifov(agentID=j))

            coeffs,valid,mp_embed = self.get_mps(j)
            agents_actions.append(coeffs)
            mp_embeds.append(mp_embed)
            valids.append(valid)
            position.append([self.agents[j].pos[0]/self.semantic_map_size,\
                             self.agents[j].pos[1]/self.semantic_map_size])
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


    def get_obs_ranged(self,agentID):
        semantic_features,_ = self.belief_semantic_map.get_observations(pos =self.agents[agentID].pos,
                                                fov = self.sensor_range,
                                                scale = 1,
                                                resolution=self.resolution,
                                                type= 'semantic' )
        #print("%d %d %d %d".format(min_x,min_y,max_x,max_y))

        infomap_feature = np.expand_dims(semantic_features,axis=-1)

        return infomap_feature


    def get_obs_ranged_wobs(self,agentID):
        semantic_features,_ = self.belief_semantic_map.get_observations(pos =self.agents[agentID].pos,
                                                fov = self.sensor_range,
                                                scale = 1,
                                                resolution=self.resolution,
                                                type= 'semantic' )

        obstacle_features, _ = self.belief_semantic_map.get_observations(pos=self.agents[agentID].pos,
                                                              fov=self.sensor_range,
                                                              scale=1,
                                                              resolution=self.resolution,
                                                              type='obstacle')
        features = np.expand_dims(semantic_features,axis=-1)
        features = np.concatenate((features,np.expand_dims(obstacle_features,axis=-1)),axis=-1)
        return np.array(features)

    def get_obs_ranged_wobspenc(self,agentID):

        semantic_features, _ = self.belief_semantic_map.get_observations(pos=self.agents[agentID].pos,
                                                              fov=self.sensor_range,
                                                              scale=1,
                                                              resolution=self.resolution,
                                                              type='semantic')

        obstacle_features, _ = self.belief_semantic_map.get_observations(pos=self.agents[agentID].pos,
                                                              fov=self.sensor_range,
                                                              scale=1,
                                                              resolution=self.resolution,
                                                              type='obstacle')


        _,penc_x,penc_y = self.belief_semantic_map.distances(range=self.sensor_range,
                                                            scale= 1.0,
                                                            resolution=self.resolution)


        features = np.expand_dims(semantic_features,axis=-1)
        features = np.concatenate((features,np.expand_dims(obstacle_features,axis=-1),\
                         np.expand_dims(penc_x,axis=-1),np.expand_dims(penc_y,axis=-1)),axis=-1)
        #print("{:d} {:d} {:d} {:d}".format(min_x, min_y, max_x, max_y))
        return np.array(features)


    def get_obs_range_wobs_multi(self,agentID):

        for s in self.scale:
            semantic_features, _ = self.belief_semantic_map.get_observations(pos=self.agents[agentID].pos,
                                                                           fov=self.sensor_range,
                                                                           scale=s,
                                                                           resolution=self.resolution,
                                                                           type='semantic')

            obstacle_features, _ = self.belief_semantic_map.get_observations(pos=self.agents[agentID].pos,
                                                                           fov=self.sensor_range,
                                                                           scale=s,
                                                                           resolution=self.resolution,
                                                                           type='obstacle')

            if s==1:
                features = np.expand_dims(semantic_features, axis=-1)
                features = np.concatenate((features, np.expand_dims(obstacle_features, axis=-1)), axis=-1)
            else:
                features = np.concatenate((features,np.expand_dims(semantic_features, axis=-1)), axis=-1)
                features = np.concatenate((features,np.expand_dims(obstacle_features, axis=-1)), axis=-1)

        # print("{:d} {:d} {:d} {:d}".format(min_x, min_y, max_x, max_y))
        return np.array(features)

    def get_obs_range_coverage_multifov(self,agentID):

        for s in self.scale:
            semantic_features, _ = self.belief_semantic_map.get_observations(pos=self.agents[agentID].pos,
                                                                           fov=self.sensor_range,
                                                                           scale=s,
                                                                           resolution=self.resolution,
                                                                           type='semantic')

            obstacle_features, _ = self.belief_semantic_map.get_observations(pos=self.agents[agentID].pos,
                                                                           fov=self.sensor_range,
                                                                           scale=s,
                                                                           resolution=self.resolution,
                                                                           type='obstacle')

            coverage_features, _ = self.belief_semantic_map.get_observations(pos=self.agents[agentID].pos,
                                                                           fov=self.sensor_range,
                                                                           scale=s,
                                                                           resolution=self.resolution,
                                                                           type='coverage')

            if s == 1:
                features = np.expand_dims(semantic_features, axis=-1)
                features = (features,
                            np.expand_dims(obstacle_features, axis=-1),
                            np.expand_dims(coverage_features, axis=-1))

                features = np.concatenate(features, axis=-1)
            else:
                features = (features,
                            np.expand_dims(semantic_features, axis=-1),
                            np.expand_dims(obstacle_features, axis=-1),
                            np.expand_dims(coverage_features, axis=-1))
                features = np.concatenate(features, axis=-1)


        # print("{:d} {:d} {:d} {:d}".format(min_x, min_y, max_x, max_y))
        return np.array(features)


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
         Creates and loads the ground turth semantics world and intializes a prior        '''
    def reset(self,test = False, test_indices = None):
        self.create_world(test,test_indices)

    #TODO : Add something to the buffer here
    def create_world(self, test, test_indices = None):

        if test:
            # GENERATING AND INITIALISING SEMANTIC MAP PRIOR
            test_indices = self.test_map_indices if test_indices is None else test_indices
            map_index = np.random.choice(test_indices)
            #map_index= 10
            load_dict = {
                'semantic_file_path': self.env_params['assets_folder'] +'sem{}.npy'.format(map_index),
                'map_image_file_path':self.env_params['assets_folder'] +'gmap{}.png'.format(map_index)
            }

            self.ground_truth_semantic_map.init_map(load_dict)
            #TODO: appropirate path referenced
            params_dict = {
                "semantic_file_path":np.random.random()*self.target_noise_scale,
            }
            self.proximity = self.belief_semantic_map.init_prior_semantics(params_dict=params_dict,ground_truth_map=self.ground_truth_semantic_map)

        else:
            # Generating and initialising semantic map prior
            map_index = np.random.choice(self.training_map_indices)

            #map_index= 10
            load_dict = {
                'semantic_file_path': self.env_params['assets_folder'] +'sem{}.npy'.format(map_index),
                'map_image_file_path':self.env_params['assets_folder'] +'gmap{}.png'.format(map_index)
            }

            self.ground_truth_semantic_map.init_map(load_dict)
            params_dict = {
                "randomness":np.random.random()*self.target_noise_scale,
                "num_centres":self.random_centres,
                "sigma":self.centre_size,
                "clip": self.env_params['MAX_CLIP']
            }
            self.proximity = self.belief_semantic_map.init_prior_semantics(params_dict=params_dict,ground_truth_map=self.ground_truth_semantic_map)

        # Generating and initialising belief semantic map

        # Creating the agents
        if self.args_dict['FIXED_BUDGET']:
            agent_budget = self.args_dict['BUDGET']*REWARD.MP.value
        else:
            agent_budget = None

        self.mp_object = MPObject(minimum_action_mp_grpah=self.minimum_action_mp_graph,
                         lookup_dict=self.lookup_dictionary,
                         spatial_dim=self.spatial_dim,
                         num_tiles=self.mp_graph.num_tiles)

        if self.args_dict['SPAWN_RANDOM_AGENTS']:
            row = np.random.randint(self.pad_size, self.semantic_map_size + self.pad_size, (self.numAgents,))
            col = np.random.randint(self.pad_size, self.semantic_map_size + self.pad_size, (self.numAgents,))
            # TODO:  MODIFY ONCE AGENT BASE CLASS IS MODIFIED
            self.agents = [
                AgentSemantic(ID = j, pos = [row[j], col[j]],
                        ground_truth= self.ground_truth_semantic_map,
                        mp_object = self.mp_object, \
                        sensor_params = self.sensor_params,
                        budget = agent_budget) for j in range(self.numAgents)]
        else:
            self.agents = [
                AgentSemantic(ID = j, pos = [15,15],
                        ground_truth= self.ground_truth_semantic_map,
                        mp_object = self.mp_object, \
                        sensor_params = self.sensor_params,
                        budget = agent_budget) for j in range(self.numAgents)]

        # Initialising the semantic map for each agent
        for agent in self.agents:
            agent.init_belief_map(self.belief_semantic_map)

    def step_all(self, action_dict):
        rewards = []
        for j in range(self.numAgents):
            r = self.step(agentID=j, action=action_dict[j])
            rewards.append(r)
        done = False

        # TODO Change termination condition of finding all semantics
        # if np.all(np.abs(self.worldMap[self.pad_size:self.pad_size + self.reward_map_size, \
        #                  self.pad_size:self.pad_size + self.reward_map_size]) < 0.1):
        #     done = True

        # If no agent has valid motion primitives terminate
        if self.args_dict['FIXED_BUDGET']:
            agents_done = False
        for agent_idx, _ in enumerate(self.agents):
            _, valids,_ = self.get_mps(agent_idx)
            if np.array(valids).sum() == 0:
                done = True
            if self.args_dict['FIXED_BUDGET']:
                if self.agents[agent_idx].agentBudget != None and self.agents[agent_idx].agentBudget < 0:
                    agents_done = agents_done or True

        if self.args_dict['FIXED_BUDGET']:
            done = done or agents_done

        return rewards, done

    def _coverage(self, initial_belief,final_belief):
        return (final_belief.coverageMap.sum() - initial_belief.coverage_map.sum())\
               /(np.prod(final_belief.coverage_map.shape))


    def _get_reward(self,initial_belief,final_belief):
        oldentropy = initial_belief.get_entropy().mean()
        newEntropy = final_belief.get_entropy().mean()
        # Normalising total entropy reward to between 0 and 100
        return (newEntropy-oldentropy)/(np.log(2))*REWARD.MAP.value

    def _get_semantic_reward(self,final_belief,initial_belief):

        detected_initial_semantics = self.ground_truth_semantic_map.detected_semantic_map
        semantic_change_proportion = {}
        for sem in self.ground_truth_semantic_map.semantic_list:
            semantic_change_proportion[sem] = np.sum(self.ground_truth_semantic_map.detected_semantic_map
                                                     [final_belief.detected_semantic_map == sem])\
                                              /self.ground_truth_semantic_map.semantic_proportion[sem]
            semantic_change_proportion[sem] -= np.sum(self.ground_truth_semantic_map.detected_semantic_map
                                                     [initial_belief.detected_semantic_map == sem])\
                                              /self.ground_truth_semantic_map.semantic_proportion[sem]

        return semantic_change_proportion

    def step(self, agentID, action):
        """
        Given the current state and action, return the next state
        """
        valid, visited_states, cost = self.agents[agentID].updatePos(action)
        reward = 0
        initialBelief = deepcopy(self.belief_semantic_map)

        if valid:
            measurements = self.agents[agentID].update_semantics(visited_states.T)

            for measurement,state in zip(measurements,visited_states.T):
                self.belief_semantic_map.update_semantics(state,measurement,self.sensor_params)
                self.buffer.add_buffer(agentID,state,self.agents[agentID].belief_semantic_map)

            reward_coverage = self._coverage(initialBelief,self.belief_semantic_map)

            reward += self._get_reward(initialBelief, self.belief_semantic_map)
            reward += reward_coverage

            reward -= cost / REWARD.MP.value

            semantics_found = self._get_semantic_reward(self.belief_semantic_map,initialBelief)
            semantics_found_total = 0
            for j in semantics_found:
                semantics_found_total+=semantics_found[j]

            reward += semantics_found_total * REWARD.TARGET.value

        elif visited_states is not None:
            # reward += REWARD.COLLISION.value*(visited_states.shape[0]+1)
            reward += REWARD.COLLISION.value
        else:
            reward += REWARD.COLLISION.value * 1.5

        return reward

    def render(self, mode='visualise', W=800, H=800):
        # TODO: Get rid of this crap, need to render via the GPU
        frame_list = []
        for idx in range(self.buffer.get_buffer_size()):
            positions,map = self.buffer.get_buffer_element(idx)
            semanticBelief = map.detected_semantic_map
            entropy = map.get_entropy()
            semanticGT = self.ground_truth_semantic_map.detected_semantic_map
            mapImage = self.ground_truth_semantic_map.map_image
            frame_list.append(self.renderer.render_image(semantic_image=semanticBelief,
                                       ground_truth=semanticGT,
                                       entropy_image=entropy,
                                       background_image=mapImage,
                                       quad_poses=positions,
                                       config_dict={"num_semantics":self.env_params['num_semantics'],
                                                    "alpha":0.4}))

        return frame_list




class QuadRender:
    def __init__(self,quad_img_path='env/assets/quad.jpg',render_H=600,render_W=600):
        # Load the quadrotor image as a numpy array
        self.image = np.array(PIL.Image.open(os.getcwd() + "/"+quad_img_path))
        self.aspect_Ratio = self.image.shape[0]/self.image.shape[1]

        self.quad_image = resize(self.image,output_shape=(self.aspect_Ratio*30,30))

        # Define an exhaustive list of semantic colours
        self.SEMANTIC_PALLETE = np.array([[1.0,0.3,0.3],[0.3,0.3,1.0],[0.3,1.0,0.3],[1.0,1,0,0.3],[1.0,0.3,1.0],[0.3,1.0,1.0]])
        self.NO_SEMANTIC = np.array([(0.9,0.9,0.9)])

        self.max_entropy = np.array([0.9,0.2,0.2])
        self.min_entropy = np.array([0.5,0.2,0.2])

        # Rendering sizes
        self.render_w = render_W
        self.render_h = render_H


    def render_image(self,semantic_image,ground_truth,entropy_image,quad_poses,background_image,config_dict):
        '''
        @params:
        semantic_image: the belief semantic maps
        ground_truth: the ground truth semantic map
        entropy_image : entropy image regularised beeten 0 and 1
        quad_position: poistiuon of quad in image corrdinatges
        background_image: the background map to overlay
        config_dict: consists of number of semantics , additional details
                        (number of semantics,
        '''

        frame = np.zeros(background_image.shape)
        frame = resize(frame,output_shape=(self.render_w,self.render_h,background_image.shape[-1]))

        # For semantics
        semantic_image = resize(semantic_image,output_shape=(self.render_w,self.render_h))
        semantic_pallete = np.concatenate((self.SEMANTIC_PALLETE[:config_dict.num_semantics-1],self.NO_SEMANTIC),axis = 0)
        semantic_image_frame =  semantic_pallete[semantic_image]
        semantic_image_frame = (1 - config_dict.alpha)*frame + config_dict.alpha*semantic_image_frame

        # For entropy
        entropy_image = resize(entropy_image, output_shape=(self.render_w, self.render_h))
        entropy_image_frame = self.min_entropy + entropy_image*(self.max_entropy - self.min_entropy)
        entropy_image_frame = (1 - config_dict.alpha) * frame + config_dict.alpha * entropy_image_frame

        # For ground truth
        ground_truth = resize(ground_truth, output_shape=(self.render_w, self.render_h))
        gt_image_frame = semantic_pallete[ground_truth]
        gt_image_frame = (1 - config_dict.alpha) * frame + config_dict.alpha * gt_image_frame


        # Add quadrotor image to the position
        for quad_position in quad_poses:
            quad_shape = self.quad_image.shape
            semantic_image_frame[quad_position[0]-int(quad_shape[0]/2):quad_position[0]+quad_shape[0]-int(quad_shape[0]/2),
                                 quad_position[1]-int(quad_shape[1]/2):quad_position[1]+quad_shape[1]-int(quad_shape[1]/2),:]\
                = self.quad_image

            entropy_image_frame[
            quad_position[0] - int(quad_shape[0] / 2):quad_position[0] + quad_shape[0] - int(quad_shape[0] / 2),
            quad_position[1] - int(quad_shape[1] / 2):quad_position[1] + quad_shape[1] - int(quad_shape[1] / 2), :] \
                = self.quad_image

            gt_image_frame[
            quad_position[0] - int(quad_shape[0] / 2):quad_position[0] + quad_shape[0] - int(quad_shape[0] / 2),
            quad_position[1] - int(quad_shape[1] / 2):quad_position[1] + quad_shape[1] - int(quad_shape[1] / 2), :] \
                = self.quad_image

        frame = np.concatenate((semantic_image_frame,entropy_image_frame,gt_image_frame),axis = 0)

        return frame
