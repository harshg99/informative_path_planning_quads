from env.searchenvMP import *
from env.env_setter import *
import numpy as np
from copy import deepcopy
from params import *
import os
from Utilities import set_dict
from env.render import *
from multiprocessing import Pool as pool
from env.env_setter import env_setter

class il_wrapper:
    def __init__(self,home_dir):

        import env_params.MotionPrim as parameters
        env_params_dict = set_dict(parameters)
        env_params_dict['home_dir'] = os.getcwd() + home_dir

        import params as args
        args_dict = set_dict(args)
        self.env = SearchEnvMP(env_params_dict,args_dict)

        self.mp_graph = self.env.minimum_action_mp_graph
        self.lookup = self.env.lookup_dictionary
        self.num_tiles = self.env.mp_graph.num_tiles
        self.spatial_dim = self.env.mp_graph.num_dims

    def isValidMP(self, pos, mp, agentID):
        is_valid = mp.is_valid
        mp.translate_start_position(pos)
        _, sp = mp.get_sampled_position()
        # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
        visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
        is_valid = is_valid and self.isValidPoses(visited_states, agentID)
        return is_valid, visited_states

    def isValidPoses(self, poses, agentID):
        is_valid = True
        for state in poses.T:
            is_valid = is_valid and self.isValidPos(state, agentID)
        return is_valid

    def isValidPos(self, pos, agentID):
        is_valid = (np.array(pos - self.env.agents[agentID].pad) > -1).all()
        is_valid = is_valid and (np.array(pos + self.env.agents[agentID].pad) \
                                 < self.env.agents[agentID].world_size).all()
        return is_valid

    def return_action(self, agentID):
        pass


class il_wrapper_semantic:
    def __init__(self,params_dict,home_dir):
        self.gifs = params_dict['GIFS']
        self.gifs_path = params_dict['GIFS_PATH']


        import env_params.Semantic as parameters
        env_params_dict = set_dict(parameters)
        env_params_dict['home_dir'] =home_dir
        self.env_params_dict = env_params_dict
        self.measurement_step = env_params_dict['measurement_step']
        self.sampled_step = env_params_dict['sampled_step_size']

        import params as args

        args_dict = set_dict(args)
        self.args_dict = args_dict

        if self.gifs:
            args_dict['RENDER_TRAINING'] = True
            args_dict['RENDER_TRAINING_WINDOW'] = 1
        self.env = GPSemanticGym(env_params_dict, args_dict)

        self.mp_graph = self.env.minimum_action_mp_graph
        self.lookup = self.env.lookup_dictionary
        self.num_tiles = self.env.mp_graph.num_tiles
        self.spatial_dim = self.env.mp_graph.num_dims

    def isValidMP(self, pos, mp, agentID):
        is_valid = mp.is_valid
        mp.translate_start_position(pos)
        _, sp = mp.get_sampled_position(step_size=self.sampled_step)
        # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
        # visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
        visited_states = np.round(sp).astype(np.int32)
        visited_states_ = []
        dict_ = {}

        for j in range(0, visited_states.shape[-1]):
            val = dict_.get(tuple(visited_states[:, j]), 0)
            if val == 0:
                visited_states_.append(visited_states[:, j])
                dict_[tuple(visited_states[:, j])] = 1.0

        visited_states_ = np.array(visited_states_).T
        indices = [0] + np.arange(self.measurement_step, visited_states_.shape[1] - 1, self.measurement_step).tolist() \
                  + [visited_states_.shape[1] - 1]

        visited_states = np.array(visited_states_)[:, indices]

        # is_valid = is_valid and self.isValidPoses(visited_states)
        is_valid = is_valid and self.isValidFinalPose(visited_states, agentID)
        return is_valid,visited_states


    def isValidPoses(self, poses, agentID):
        is_valid = True
        for state in poses.T:
            is_valid = is_valid and self.isValidPos(state, agentID)
        return is_valid

    def isValidPos(self, pos, agentID):
        is_valid = (np.array(pos - self.env.agents[agentID].pad) > -1).all()
        is_valid = is_valid and (np.array(pos + self.env.agents[agentID].pad) \
                                 < self.env.agents[agentID].belief_semantic_map.world_map_size).all()
        return is_valid

    def isValidFinalPose(self, final_pose,agentID):
        is_valid = True
        is_valid = is_valid and self.isValidPos(final_pose.T,agentID)
        return is_valid

    def return_action(self, agentID):
        pass
