#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
from active_sampling import LearnPolicyGradientParams, Trajectory
from motion_primitives_py import MotionPrimitiveLattice
from copy import deepcopy

import active_sampling


class LearnPolicyGradientParamsMP(LearnPolicyGradientParams):

    def __init__(self, mp_graph_file_name):
        super(LearnPolicyGradientParamsMP, self).__init__()
        self.mp_graph_file_name = mp_graph_file_name
        self.load_graph()
        self.spatial_dim = self.mp_graph.num_dims

        self.Tau_horizon = 100
        self.num_iterations = 1000
        self.num_trajectories = 10
        self.Eta = 0.015

    def load_graph(self):
        self.mp_graph = MotionPrimitiveLattice.load(self.mp_graph_file_name)
        self.mp_graph.edges = self.mp_graph.edges.T
        self.num_actions = max([len([j for j in i if j != None]) for i in self.mp_graph.edges])
        print(self.num_actions)
        self.minimum_action_mp_graph = np.empty((self.mp_graph.edges.shape[0], self.num_actions), dtype=object)
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

    def get_next_state(self, absolute_pos, map_indices, action, action_index):
        mp = deepcopy(self.minimum_action_mp_graph[action_index, action])
        if mp is not None:

            mp.translate_start_position(absolute_pos)
            worldmap_pos = self.absolutePosToIndexPos(mp.end_state[:self.spatial_dim])
            is_valid = mp.is_valid and self.isValidPos(worldmap_pos)
            _, sp = mp.get_sampled_position()
            if is_valid:
                visited_map_indices = np.unique(self.absolutePosToIndexPos(sp.T),axis=0)
                next_action_index = int(np.floor(self.lookup_dictionary[action_index, action]/self.mp_graph.num_tiles))
                return mp.end_state[:self.spatial_dim], next_action_index, is_valid, visited_map_indices, mp.cost/mp.subclass_specific_data.get('rho', 1000)

        # else:
        #     print('Warning: invalid MP is being selected')
        return absolute_pos, action_index, False, map_indices.reshape(2, 1), None

    def set_up_training(self):
        self.theta = np.random.rand(self.num_features, self.num_other_states, self.num_actions)*0.1


if __name__ == '__main__':
    import rospkg
    import os

    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('motion_primitives')
    # pkg_path = f'{pkg_path}/motion_primitives_py/'
    # mpl_file = f"{pkg_path}data/lattices/lattice_test.json"

    mpl_file = f'{os.path.dirname(active_sampling.__file__)}/latticeData/10.json'
    script_dir = os.path.dirname(os.path.abspath(__file__))

    lpgp = LearnPolicyGradientParamsMP(mpl_file)
    lpgp.fileNm = "lpgp10"
    # lpgp = pickle.load(open(f'{script_dir}/testingData/{lpgp.fileNm}.pkl', "rb"), encoding='latin1')
    lpgp.load_graph()

    rewardmap = pickle.load(open(f'{script_dir}/trainingData/gaussian_mixture_training_data.pkl', "rb"), encoding='latin1')
    rewardmap = np.load('airport.npy')
    lpgp.xy_resolution = .1
    lpgp.run_training(rewardmap)
