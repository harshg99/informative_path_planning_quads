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
        self.num_iterations = 10
        # self.num_trajectories = 5
        # self.Eta = .3

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
        print(self.num_other_states)
        self.num_actions_per_state = [len([j for j in i if j != None]) for i in self.minimum_action_mp_graph]

    def get_next_state(self, pos, action, index):
        # reset_map_index = int(np.floor(self.curr_state_index / self.mp_graph.num_tiles))
        mp = deepcopy(self.minimum_action_mp_graph[index, action])
        if mp is not None:
            mp.translate_start_position(pos)
            worldmap_pos = np.rint(mp.end_state[:self.spatial_dim]).astype(np.int32)
            is_valid = mp.is_valid and self.isValidPos(worldmap_pos)
            _, sp = mp.get_sampled_position()
            # visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
            visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
            if is_valid:
                next_index = self.lookup_dictionary[index, action]
                next_index = int(np.floor(next_index/self.mp_graph.num_tiles))
                return mp.end_state[:self.spatial_dim], next_index, is_valid, visited_states, mp.cost
        # else:
        #     print('Warning: invalid MP is being selected')
        return pos, index, False, None, None

    def set_up_training(self):
        self.theta = np.random.rand(self.num_features, self.num_other_states, self.num_actions)*0.1
        # for i in range(self.num_other_states):
        #     for j in range(self.num_actions):
        #         if self.minimum_action_mp_graph[i, j] is None:
        #             self.theta[:, i, j] = -1E8


if __name__ == '__main__':
    import rospkg
    import os

    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('motion_primitives')
    # pkg_path = f'{pkg_path}/motion_primitives_py/'
    # mpl_file = f"{pkg_path}data/lattices/lattice_test.json"
    mpl_file = f'{os.path.dirname(active_sampling.__file__)}/latticeData/8_actions.json'

    import os
    lpgp = LearnPolicyGradientParamsMP(mpl_file)
    script_dir = os.path.dirname(active_sampling.__file__)
    print(script_dir)
    rewardmap = pickle.load(open(f'{script_dir}/trainingData/gaussian_mixture_training_data.pkl', "rb"), encoding='latin1')
    lpgp.run_training(rewardmap)
