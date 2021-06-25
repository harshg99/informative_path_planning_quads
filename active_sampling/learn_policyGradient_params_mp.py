#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
from active_sampling import LearnPolicyGradientParams, Trajectory
from motion_primitives_py import MotionPrimitiveLattice
from copy import deepcopy


class LearnPolicyGradientParamsMP(LearnPolicyGradientParams):

    def __init__(self, mp_graph_file_name):
        super(LearnPolicyGradientParamsMP, self).__init__()
        self.mp_graph_file_name = mp_graph_file_name
        self.load_graph()
        # self.num_actions = self.mp_graph.edges.shape[0]
        self.curr_node_index = 0
        self.spatial_dim = 2
        self.curr_state = np.zeros(self.mp_graph.n)
        self.Tau_horizon = 20
        self.num_iterations = 100
        self.num_trajectories = 10

    def load_graph(self):
        self.mp_graph = MotionPrimitiveLattice.load(self.mp_graph_file_name)
        self.mp_graph.edges = self.mp_graph.edges.T
        self.num_actions = max([len([j for j in i if j!=None]) for i in self.mp_graph.edges])
        self.minimum_action_mp_graph = np.empty((self.mp_graph.edges.shape[0],self.num_actions),dtype=object)
        self.lookup_dictionary = np.ones_like(self.minimum_action_mp_graph)*-1
        for i in range(self.mp_graph.edges.shape[0]):
            k=0
            for j in range(self.mp_graph.edges.shape[1]):
                if self.mp_graph.edges[i,j] is not None:
                    self.minimum_action_mp_graph[i,k] = self.mp_graph.edges[i,j]
                    self.lookup_dictionary[i,k] = j
                    k+=1
        self.num_vertices = len(self.mp_graph.vertices)

    def get_next_state(self, curr_pos, curr_action):
        self.curr_state[:self.spatial_dim] = curr_pos
        reset_map_index = int(np.floor(self.curr_node_index / self.mp_graph.num_tiles))
        mp = deepcopy(self.minimum_action_mp_graph[reset_map_index, curr_action])
        # mp = deepcopy(self.mp_graph.edges[reset_map_index, curr_action])
        # print(self.minimum_action_mp_graph[self.curr_node_index,:])
        if mp is not None:
            mp.translate_start_position(curr_pos)
            worldmap_pos = np.rint(mp.end_state[:self.spatial_dim]).astype(np.int32)
            is_valid = mp.is_valid and self.isValidPos(worldmap_pos)
            if is_valid:
                self.curr_node_index = self.lookup_dictionary[reset_map_index, curr_action]
                self.curr_state = mp.end_state
                return worldmap_pos, is_valid
        return curr_pos, False


if __name__ == '__main__':
    import rospkg

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('motion_primitives')
    pkg_path = f'{pkg_path}/motion_primitives_py/'
    mpl_file = f"{pkg_path}data/lattices/lattice_test2.json"

    import os
    lpgp = LearnPolicyGradientParamsMP(mpl_file)
    script_dir = os.path.dirname(__file__)
    rewardmap = pickle.load(open(f'{script_dir}/trainingData/gaussian_mixture_training_data.pkl', "rb"), encoding='latin1')
    lpgp.run_training(rewardmap)
