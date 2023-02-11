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
from env.sensors import *
from env.SemanticMap import GPSemanticMap

class Agent():
    def __init__(self,ID,row,col,map_size,pad,world_size):
        self.ID = ID
        self.pos = np.array([row,col])
        self.reward_map_size = map_size
        self.pad = pad
        self.world_size = world_size
        self.worldMap = None
        self.prev_action = 0


    def updateMap(self,worldMap):
        self.worldMap= worldMap

    def updatePos(self,action):
        next_pos = self.pos + np.array(action)
        is_action_valid = self.isValidPos(next_pos)
        if is_action_valid:
            self.pos = next_pos
        self.prev_action =  action
        return is_action_valid



    def isValidPos(self, pos):
        is_valid = (np.array(pos - self.pad) > -1).all()
        is_valid = is_valid and (np.array(pos + self.pad) < self.world_size).all()
        return is_valid

class AgentMP():
    def __init__(self,ID,row,col,map_size,pad,world_size,mp_graph,lookup,spatial_dim,tiles,budget = None):
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
        self.pos_actual = self.pos.copy()
        self.trajectory = np.zeros([self.world_size,self.world_size])
        if budget is not None:
            self.agentBudget = budget
        self.current_primitive = None
        self.coverageMap = np.zeros([self.world_size,self.world_size])


    def updateMap(self,worldMap):
        self.worldMap= worldMap

    def get_mp(self,action):
        mp = deepcopy(self.mp_graph[self.index, action])
        return mp

    def updatePos(self,action):
        mp = deepcopy(self.mp_graph[self.index, action])
        mpcost = 5000
        if mp is not None:
            #mp.translate_start_position(self.pos)
            #_, sp = mp.get_sampled_position()
            # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
            #visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
            is_valid,visited_states = self.isValidMP(mp)
            mask = [np.all(visited_states[:,j]==self.pos) for j in range(visited_states.shape[-1])]
            visited_states = np.delete(visited_states,np.where(mask),axis= -1)
            self.prev_action = action
            self.current_primitive = deepcopy(mp)
            if is_valid:
                next_index = self.lookup[self.index, action]
                next_index = int(np.floor(next_index / self.tiles))
                self.index = next_index
                self.pos = np.round(mp.end_state[:self.spatial_dim]).astype(int)
                self.pos_actual = mp.end_state[:self.spatial_dim]
                #print("{:d} {:d} {:d} {:d}".format(self.pos[0], self.pos[1], visited_states[0,0], visited_states[1,0]))
                self.visited_states = visited_states
                for s in visited_states.T:
                    self.trajectory[s[0],s[1]] = 1

                mpcost = mp.cost / mp.subclass_specific_data.get('rho', 1) / 10
                if self.agentBudget is not None:
                    self.agentBudget -= mpcost
                return is_valid, visited_states, mpcost
            if self.agentBudget is not None:
                self.agentBudget -= mpcost
            return False,visited_states,None
        if self.agentBudget is not None:
            self.agentBudget -= mpcost
        return False,None, None

    def updatePosOdom(self,pos):
        self.pos_actual = np.array(pos)
        self.pos = np.round(self.pos).astype(int)

    def isValidMP(self,mp):
        is_valid = mp.is_valid
        mp.translate_start_position(self.pos_actual)
        _, sp = mp.get_sampled_position()
        # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
        visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
        #is_valid = is_valid and self.isValidPoses(visited_states)
        final_pos = np.round(mp.end_state[:self.spatial_dim]).astype(int)
        is_valid = is_valid and self.isValidFinalPose(final_pos)
        return is_valid,visited_states

    def isValidPoses(self, poses):
        is_valid = True
        for state in poses.T:
            is_valid = is_valid and self.isValidPos(state)
        return is_valid

    def isValidFinalPose(self, final_pose):
        is_valid = True
        is_valid = is_valid and self.isValidPos(final_pose.T)
        return is_valid

    def isValidPos(self, pos):
        is_valid = (np.array(pos - self.pad) > -1).all()
        is_valid = is_valid and (np.array(pos + self.pad) < self.world_size).all()
        return is_valid

class AgentGP(AgentMP):
    def __init__(self,ID,row,col,map_size,pad,world_size,mp_graph,lookup,spatial_dim,tiles,sensor_params,budget=None):
        super(AgentGP, self).__init__(ID,row,col,map_size,pad,world_size,mp_graph,lookup,spatial_dim,tiles,budget)
        self.beliefMap = np.zeros([self.world_size,self.world_size])
        self.targetMap = np.zeros([self.world_size,self.world_size])
        self.sensor = sensor_setter.set_env(sensor_params)


    def initBeliefMap(self,Map):
        self.beliefMap = Map.copy()


    def updatePos(self,action,beliefThresh = 0.99):
        valid,states,cost = super(AgentGP, self).updatePos(action)
        return valid,states,cost

    def updateInfoTarget(self,visited_states,worldTargetMap,beliefThresh):
        measurement_list = []
        for state in visited_states:
            r = state[0]
            c = state[1]
            range_ = int(self.sensor.sensor_range/2)
            min_x = np.max([r - range_, 0])
            min_y = np.max([c - range_, 0])
            max_x = np.min([r + range_+1, self.beliefMap.shape[0]])
            max_y = np.min([c + range_+1, self.beliefMap.shape[1]])
            measurement = self.sensor.getMeasurement(state,worldTargetMap)
            measurement_list.append(measurement)
            for j in range(min_x,max_x):
                for k in range(min_y,max_y):
                    logodds_b_map = np.log(self.beliefMap[j,k]/(1-self.beliefMap[j,k]))
                    sensor_log_odds = np.log((1-self.sensor.sensor_unc[j-(r-range_),k-(c-range_)])/ \
                                            self.sensor.sensor_unc[j-(r-range_),k-(c-range_)])
                    #print(sensor_log_odds)
                    if measurement[j-(r-range_),k-(c-range_)]==0:
                        logodds_b_map -= sensor_log_odds
                    else:
                        logodds_b_map += sensor_log_odds
                    self.beliefMap[j,k] = 1/(np.exp(-logodds_b_map)+1)

                # Update whether target is found

                    if self.beliefMap[j,k]>=beliefThresh:
                        self.targetMap[j,k]=2

                    self.coverageMap[j,k] = 1.0

        return measurement_list



class AgentSemantic :
    def __init__(self,ID = 0,
                 pos = None,
                 ground_truth = None,
                 mp_object = None,
                 sensor_params = None,
                 budget=None):

        self.ID = ID
        self.pos = np.array(pos)

        self.pad = ground_truth.padding

        self.tiles = mp_object.num_tiles

        self.visited_states = None
        self.mp_graph = mp_object.mp_graph
        self.lookup = mp_object.lookup
        self.index = 0
        self.spatial_dim = mp_object.spatial_dim
        self.agent_budget = budget

        self.prev_action = 0
        self.pos_actual = self.pos.copy()


        self.belief_semantic_map = GPSemanticMap(ground_truth.config)
        self.gt_semantic_map = ground_truth

        self.sensor_params = sensor_params
        self.sensor = sensor_setter.set_env(sensor_params)


    def init_belief_map(self,map):
        self.belief_semantic_map = deepcopy(map)

    def updatemap(self,world_map):
        self.belief_semantic_map = deepcopy(world_map)

    def get_mp(self,action):
        mp = deepcopy(self.mp_graph[self.index, action])
        return mp

    def updatePos(self,action):
        mp = deepcopy(self.mp_graph[self.index, action])
        mpcost = 5000
        if mp is not None:
            #mp.translate_start_position(self.pos)
            #_, sp = mp.get_sampled_position()
            # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
            #visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
            is_valid,visited_states = self.isValidMP(mp)
            mask = [np.all(visited_states[:,j]==self.pos) for j in range(visited_states.shape[-1])]
            visited_states = np.delete(visited_states,np.where(mask),axis= -1)
            self.prev_action = action
            self.current_primitive = deepcopy(mp)
            if is_valid:
                next_index = self.lookup[self.index, action]
                next_index = int(np.floor(next_index / self.tiles))
                self.index = next_index
                self.pos = np.round(mp.end_state[:self.spatial_dim]).astype(int)
                self.pos_actual = mp.end_state[:self.spatial_dim]
                #print("{:d} {:d} {:d} {:d}".format(self.pos[0], self.pos[1], visited_states[0,0], visited_states[1,0]))
                self.visited_states = visited_states
                # for s in visited_states.T:
                #     self.trajectory[s[0],s[1]] = 1

                mpcost = mp.cost / mp.subclass_specific_data.get('rho', 1) / 10
                if self.agent_budget is not None:
                    self.agent_budget -= mpcost
                return is_valid, visited_states, mpcost
            if self.agent_budget is not None:
                self.agent_budget -= mpcost
            return False,visited_states,None
        if self.agent_budget is not None:
            self.agent_budget -= mpcost
        return False,None, None

    def updatePosOdom(self,pos):
        self.pos_actual = np.array(pos)
        self.pos = np.round(self.pos).astype(int)

    def isValidMP(self,mp):
        is_valid = mp.is_valid
        mp.translate_start_position(self.pos_actual)
        _, sp = mp.get_sampled_position()
        # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
        visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
        #is_valid = is_valid and self.isValidPoses(visited_states)
        final_pos = np.round(mp.end_state[:self.spatial_dim]).astype(int)
        is_valid = is_valid and self.isValidFinalPose(final_pos)
        return is_valid,visited_states

    def isValidPoses(self, poses):
        is_valid = True
        for state in poses.T:
            is_valid = is_valid and self.isValidPos(state)
        return is_valid

    def isValidFinalPose(self, final_pose):
        is_valid = True
        is_valid = is_valid and self.isValidPos(final_pose.T)
        return is_valid

    def isValidPos(self, pos):
        is_valid = (np.array(pos - self.pad) > -1).all()
        is_valid = is_valid and (np.array(pos + self.pad) < self.gt_semantic_map.world_map_size).all()
        return is_valid


    def update_semantics(self,visited_states):
        measurement_list = []

        for state in visited_states:
            measurement = self.sensor.get_measurements(state,self.gt_semantic_map)
            measurement_list.append(measurement)
            self.belief_semantic_map.update_semantics(state,measurement,self.sensor_params)

        return measurement_list