
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
ZEROREWARDCOLOR = (0.05,0.1,0.1)
MAXREWARDCOLOR = (0.8,1.0,1.0)
AGENT_MINCOL = (0.5,0.3,0.3)
AGENT_MAXCOL = (1.,0.3,0.3)



# TODO: Define the semantic mapping class : THis should  be able to store a semantic image
#   The class has a helper function that provides a random semantic map on initialization
#   This class should basically store a map of any kind


class GPSemanticMap:

    def __init__(self, config_dict , isGroundTruth = False):
        '''
        GP Semantic Map models the necessary transitions and the updates to the sematic map based on the
        quadrotors position
        Maintains

        @param
            config_dict: Consists of the parameters for setting up tbe environment
            - num_semantics : number of semantics
            - world_map_size : tuple(int,int)
            - resolution :  tuple(int,int)
        '''
        self.config = config_dict
        self.resolution = self.config['resolution']

        self.world_map_size = self.config.world_map_size

        self.map_size = (self.config.world_map_size[0]*self.resolution[0],
                         self.config.world_map_size[1]*self.resolution[1],
                         self.config.num_semantics)


        self.semantic_map = None
        self.coverage_map = None
        self.obstacle_map = None
        self.detected_semantic_map = None # current semantic category

        self.padding = self.config.padding

        self.isGroundTruth = isGroundTruth

    def get_row_col(self,pos: Union(Tuple, np.Array,List)):
        '''
        @params:
        pos: tuple, list np.array (position of agent)
        '''

        return [int(pos[0]*self.resolution[0]),int(pos[1]*self.resolution[1])]

    def get_pos(self,row_col):
        '''
        @params
        row_col : the row and column to convert the position into
        '''

        return int(row_col[0]/self.resolution[0]),int(row_col[1]/self.resolution[1])

    def get_observations(self, pos, fov, scale = None, type='semantic',return_distance = False):
        '''
        Returns the necessary observations (semantic,coverage or obstacle)
        To return the distance map based on the field of view
        @params:
        pos :  Position of the agent
        fov :  field of view in Gloval coordinate system
        type: type of the map
        scale: Integer
        return_distance:  returns distance mebeddings

        @ return:
        feature, distance array:
        '''

        if type=='semantic':
            map = deepcopy(self.semantic_map)
        elif type=='obstacle':
            map = deepcopy(self.obstacle_map)
        else:
            map = deepcopy(self.coverage_map)

        r,c = self.get_row_col(pos)
        if scale is None:
            scale = 1


        range = list(self.get_row_col(fov))

        range[0] = scale*range[0]
        range[1] = scale*range[1]

        min_x = np.max([r - range[0], 0])
        min_y = np.max([c - range[1], 0])
        max_x = np.min([r + range[0], self.map_size[0]])
        max_y = np.min([c + range[1], self.map_size[1]])

        feature = np.zeros((2 * range[0], 2 * range[1]))
        feature[min_x - (r - range[0]):2 * range[0] - (r + range[0] - max_x), \
        min_y - (c - range[1]):2 * range[1] - (c + range[1] - max_y)] = map[min_x:max_x, min_y:max_y]

        feature = block_reduce(feature, (scale, scale), np.max)

        distances = self.distances(range,scale)

            # print("{:d} {:d} {:d} {:d}".format(min_x, min_y, max_x, max_y))
        return np.array(feature),distances

    def distances(self,range,scale):

        distances_x = np.repeat(np.expand_dims(
            np.arange(2*range[1])/self.resolution[1],axis=0),
                repeats=2*range[0],axis=0)
        distances_x = np.square(distances_x - range[1])
        distances_y = np.repeat(np.expand_dims(
            np.arange(2*range[0]/self.resolution[0]),
            axis=1),repeats=2*range[1],axis=1)
        distances_y = np.square(distances_y-range[0])

        distances = distances_x + distances_y
        distances = block_reduce(distances, (scale, scale), np.max)
        return distances

    def init_map(self, load_dict = None):

        '''
        @ params: Initializes the semantic, coverage and obstacle maps
        '''
        # TODO detected semantic map to change accordingly,stores only the semantic label
        if load_dict is None:
            self.semantic_map = cupy.array(np.zeros(shape=self.map_size))
            self.coverage_map = cupy.array(np.zeros(shape=(self.map_size[0], self.map_size[1])))
            self.obstacle_map = cupy.array(np.zeros(shape=(self.map_size[0], self.map_size[1])))
            self.detected_semantic_map =  cupy.array(np.zeros(shape=(self.map_size[0], self.map_size[1]))) -1
        else:
            # Load the semantic map from the file path
            self.semantic_map = cupy.array(np.load(load_dict['semantic_file_path']))
            self.coverage_map = cupy.array(np.zeros(shape=(self.map_size[0], self.map_size[1])))
            self.obstacle_map = cupy.array(np.zeros(shape=(self.map_size[0], self.map_size[1])))
            self.detected_semantic_map = cupy.array(np.zeros(shape=(self.map_size[0], self.map_size[1]))) -1



    def init_prior_semantics(self, similarity_index = None):
        '''
        Assigns a prior to the semantic map if the model is th
        '''

        if self.isGroundTruth:
            raise AttributeError
        # TODO: Need to complete how the prior is initialized
        raise NotImplementedError

    def getEntropy(self):
        entropy = self.semantic_map * np.log(np.clip(self.semantic_map, 1e-7, 1))
        return entropy
    '''
    Returns the total entropy at the desired locations
    '''

    def updateSemantics(self, visited_states, measurement_list,sensor_params):
        '''
            Measurement sites
        '''
        # reward_coverage = 0
        coverage = deepcopy(self.coverage_map)
        initial_belief = deepcopy(self.semantic_map)

        sensor_max_unc = sensor_params['sensor_max_acc']
        sensor_range = sensor_params['sensor_range']
        coeff = sensor_params['sensor_decay_coeff']
        sensor_range_map = self.get_row_col(sensor_range)

        distances = self.distances(sensor_range,scale = 1)

        # asseeting if the measurement shape is equivalent to the sensor shape
        assert 2*sensor_range_map[0] == measurement_list[0].shape[0]

        for state, measurement in zip(visited_states.tolist(), measurement_list):
            r,c = self.get_row_col(state)
            min_x = np.max([r - sensor_range_map[0], 0])
            min_y = np.max([c - sensor_range_map[1], 0])
            max_x = np.min([r + sensor_range_map[0] + 1, self.semantic_map.shape[0]])
            max_y = np.min([c + sensor_range_map[1] + 1, self.semantic_map.shape[1]])

            self.coverage_map[min_x:max_x,min_y:max_y] = 1.

            sensor_odds =  np.log(sensor_max_unc *(1-coeff*distances)/(1-sensor_max_unc *(1-coeff*distances)))
            semantic_map_log_odds = np.log(self.semantic_map[min_x:max_x, min_y:max_y,:]\
                                           / (1 - self.semantic_map[min_x:max_x,min_y:max_y,:]))
            semantic_map_log_odds -= sensor_odds[min_x- (r - sensor_range_map[0]) : max_x - (r-sensor_range_map[0])]\
                                                [min_y - (c - sensor_range_map[1]): max_y - (c - sensor_range_map[1])]
            semantic_map_log_odds[min_x:max_x,min_y:max_y,measurement] += 2*sensor_odds[min_x- (r - sensor_range_map[0]) : max_x - (r-sensor_range_map[0])]\
                                                [min_y - (c - sensor_range_map[1]): max_y - (c - sensor_range_map[1])]
            self.semantic_map[min_x:max_x,min_y:max_y] =  1 / (np.exp(-semantic_map_log_odds) + 1)

            self.detected_semantic_map[min_x:max_x,min_y:max_y,np.max(self.semantic_map
                                    [min_x:max_x,min_y:max_y,:],axis=-1)>self.targetBeliefThresh] = \
                np.argmax(self.semantic_map[min_x:max_x,min_y:max_y,np,max(
                          self.semantic_map[min_x:max_x,min_y:max_y,:],axis=-1)>self.targetBeliefThresh],axis=-1)

    def compute_entropy(self):
        '''
        returns the entropy in the semantic map classification results
        '''
        return np.sum(np.log(np.clip(self.semantic_map,1e-6,1.0))*self.semantic_map)

# TODO: New gym environment with observation structure

class MPObject:

    def __init__(self,minimum_action_mp_grpah = None,
                 lookup_dict = None,
                 spatial_dim = None,
                 num_tiles = None):
        self.mp_graph = minimum_action_mp_grpah
        self.lookup = lookup_dict
        self.spatial_dim = spatial_dim
        self.num_tiles = num_tiles


class GPSemanticGym(gym.Env):

    '''
    Handles the motion primitive search
    '''
    def __init__(self, params_dict,args_dict):
        super(GPSemanticGym, self).__init__()

        # Necessary parameters for semantic environments
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
        self.world_map_size = self.reward_map_size + 2 * self.pad_size

        self.action_size = len(ACTIONS)
        self.episode_length = params_dict['episode_length']
        self.sensor_range = params_dict['sensor_range']
        self.max_steps = params_dict['episode_length']

        if SET_SEED:
            self.seed = params_dict['seed']
        self.viewer = None

        if OBSERVER == 'RANGE':
            self.input_size = [2 * RANGE, 2 * RANGE, 1]
        elif OBSERVER == 'TILED':
            self.input_size = [24, 1, 1]
        elif OBSERVER == 'TILEDwOBS':
            self.input_size = [48, 1, 1]
        elif OBSERVER == 'RANGEwOBS':
            self.input_size = [2 * RANGE, 2 * RANGE, 2]
        elif OBSERVER == 'RANGEwOBSwNEIGH':
            self.input_size = [2 * (RANGE + 1), 2 * (RANGE + 1), 2]
        elif OBSERVER == 'RANGEwOBSwPENC':
            self.input_size = [2 * RANGE, 2 * RANGE, 4]
        elif OBSERVER == 'RANGEwOBSwMULTI':
            self.map_length = 2
            self.input_size = [2 * RANGE, 2 * RANGE, len(self.scale) * 2]
        elif OBSERVER == 'RANGEwOBSwMULTIwCOV':
            self.map_length = 2
            self.input_size = [2 * RANGE, 2 * RANGE, len(self.scale) * 3]

        if 'home_dir' not in params_dict.keys():
            self.mp_graph_file_name = os.getcwd()+ '/env/'+params_dict['graph_file_name']
        else:
            self.mp_graph_file_name = params_dict['home_dir'] + '/env/' + params_dict['graph_file_name']
        self.load_graph()
        self.create_mp_graph_encodings()
        self.spatial_dim = self.mp_graph.num_dims

        self.defaultBelief = params_dict['defaultBelief']
        self.sensor_params = params_dict['sensor_params']
        self.numrand_targets = params_dict['num_targets']
        self.targetBeliefThresh = params_dict['targetBeliefThresh']
        self.noise_max_var = params_dict['noise_max_var']
        self.noise_min_var = params_dict['noise_min_var']
        self.metrics = SemanticMetrics()

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

    # TODO
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
    #TODO
    def createWorld(self, rewardMap=None):
        super().createWorld(rewardMap)
        # Creating the agents
        if self.args_dict['FIXED_BUDGET']:
            agentBudget = self.args_dict['BUDGET']*REWARD.MP.value
        else:
            agentBudget = None

        self.mp_object = MPObject(minimum_action_mp_grpah=self.minimum_action_mp_graph,
                         lookup_dict=self.lookup_dictionary,
                         spatial_dim=self.spatial_dim,
                         num_tiles=self.mp_graph.num_tiles)

        if SPAWN_RANDOM_AGENTS:
            row = np.random.randint(self.pad_size, self.reward_map_size + self.pad_size, (self.numAgents,))
            col = np.random.randint(self.pad_size, self.reward_map_size + self.pad_size, (self.numAgents,))
            # TODO:  mODIFY ONCE AGENT BASE CLASS IS MODIFIED
            self.agents = [
                AgentSemantic(ID = j, pos = [row[j], col[j]],
                        groundTruth= self.groundTruthMap,
                        mp_object = self.mp_object, \
                        sensor_params = self.sensor_params,
                        budget = agentBudget) for j in range(self.numAgents)]
        else:
            self.agents = [
                AgentSemantic(ID = j, pos = [10,10],
                        groundTruth= self.groundTruthMap,
                        mp_object = self.mp_object, \
                        sensor_params = self.sensor_params,
                        budget = agentBudget) for j in range(self.numAgents)]

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

    #TODO
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

    #todo
    def render(self, mode='visualise', W=800, H=800):
        # TODO: Get rid of this crap, need to renbder via the GPU
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





# TODO: Newer way to render the gym semantic maps