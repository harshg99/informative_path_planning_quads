#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
from active_sampling import LearnPolicyGradientParams
from motion_primitives_py import MotionPrimitiveLattice
from copy import deepcopy

class LearnPolicyGradientParamsMP(LearnPolicyGradientParams):

    def __init__(self, mp_graph_file_name):
        super(LearnPolicyGradientParamsMP, self).__init__()
        self.mp_graph_file_name = mp_graph_file_name
        self.load_graph()
        self.num_actions = self.mp_graph.edges.shape[0]
        print(self.num_actions)
        self.curr_node_index = 0
        self.spatial_dim = 2
        self.curr_state = np.zeros(self.mp_graph.n)
        self.curr_state[:self.spatial_dim] = self.rand_start_pos
        self.Tau_horizon =10
        self.num_iterations = 100
    
    def load_graph(self):
        self.mp_graph = MotionPrimitiveLattice.load(self.mp_graph_file_name)

    def get_next_state(self, worldmap, curr_pos, curr_action):
        # print(curr_pos)
        reset_map_index = int(np.floor(self.curr_node_index / self.mp_graph.num_tiles))
        mp = deepcopy(self.mp_graph.edges[curr_action, reset_map_index])
        if mp is not None:
            mp.offset_start_position(self.rand_start_pos)
            self.curr_node_index = reset_map_index
            self.curr_state = mp.end_state
            worldmap_pos = np.rint(self.curr_state[:self.spatial_dim]).astype(np.int32)
            is_valid = mp.is_valid and (np.array(worldmap_pos + (self.curr_r_pad)) < worldmap.shape).all() and (np.array(worldmap_pos - self.curr_r_pad) > -1).all()
            if is_valid:
                return worldmap_pos, is_valid
        return curr_pos, False


if __name__ == '__main__':
    import rospkg

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('motion_primitives')
    pkg_path = f'{pkg_path}/motion_primitives_py/'
    mpl_file =  f"{pkg_path}data/lattices/lattice_test.json"

    import os
    lpgp = LearnPolicyGradientParamsMP(mpl_file)
    script_dir = os.path.dirname(__file__)
    rewardmap = pickle.load(open(f'{script_dir}/trainingData/gaussian_mixture_training_data.pkl', "rb"), encoding='latin1')
    lpgp.run_training(rewardmap)
