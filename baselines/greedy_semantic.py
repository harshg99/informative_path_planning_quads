import json

from env.env_setter import *
import numpy as np
from copy import deepcopy
from params import *
import os
from Utilities import set_dict
from env.render import *
from multiprocessing import Pool as pool
from baselines.il_wrapper import il_wrapper_semantic
from env.GPSemantic import *
from env.SemanticMap import *

class GreedySemantic(il_wrapper_semantic):
    def __init__(self,params_dict,home_dir="./"):
        super().__init__(params_dict,home_dir)
        self.test_params = params_dict
        self.results_path = params_dict['RESULTS_PATH']
        self.depth = params_dict['depth']
        self.exploration = params_dict['exploration']

        if not os.path.exists(self.gifs_path):
            os.makedirs(self.gifs_path)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)


    def getExpectedEntropy(self, visited_states,world_map):

        world_map_init = deepcopy(world_map)

        for state in visited_states.tolist():

            semantic_obs,_ = world_map.get_observations(state, fov=self.env.sensor_params['sensor_range'],
                                                       scale=None, type='semantic',
                                                       return_distance=False, resolution=None)

            projected_measurement = np.argmax(semantic_obs, axis=-1)

            world_map.update_semantics(state, projected_measurement, self.env.sensor_params)


        init_entropy = world_map_init.get_entropy()
        final_entropy = world_map.get_entropy()
        entropy_reduction = (init_entropy - final_entropy).sum()

        return entropy_reduction/(np.square(self.env.sensor_params['sensor_range'][0])*world_map.resolution**2)
        #return entropy_reduction

    def getMean(self,visited_states,worldMap):
        entropy_reduction = 0
        for state in visited_states.tolist():

            semantic_obs,_ = worldMap.get_observations(state, fov=self.env.sensor_params['sensor_range'],
                                                       scale=None, type='semantic',
                                                       return_distance=False, resolution=None)


            entropy_reduction += semantic_obs.mean(axis=-1).sum()

        return entropy_reduction / (np.square(self.env.sensor_params['sensor_range'][0])*worldMap.resolution**2)
        #return entropy_reduction

    def getmpcost(self,pos,index,action,agentID,worldMap):
        mp = deepcopy(self.mp_graph[index, action])
        reward = 0
        next_pos = pos
        next_index = index
        is_valid = False
        if mp is not None:
            # mp.translate_start_position(self.pos)
            # _, sp = mp.get_sampled_position()
            # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
            # visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
            is_valid, visited_states = self.isValidMP(pos,mp,agentID)
            reward  = 0
            if is_valid:
                next_index = self.lookup[index, action]
                next_index = int(np.floor(next_index / self.num_tiles))
                next_pos =  np.round(mp.end_state[:self.spatial_dim]).astype(int)
                # print("{:d} {:d} {:d} {:d}".format(self.pos[0], self.pos[1], visited_states[0,0], visited_states[1,0]))
                self.visited_states = visited_states.T
                reward = self.exploration * self.getMean(self.visited_states,worldMap)+\
                self.getExpectedEntropy(self.visited_states,worldMap)

                reward -= mp.cost/ mp.subclass_specific_data.get('rho', 1) / 10 / REWARD.MP.value
            elif visited_states is not None:
                # reward += REWARD.COLLISION.value*(visited_states.shape[0]+1)
                reward += REWARD.COLLISION.value
            else:
                reward += REWARD.COLLISION.value * 1.5

        return reward,next_index,next_pos,is_valid

    def plan_action(self,pos,index,agentID,current_depth=0,worldMap=None):
        if current_depth>=self.depth:
            return worldMap.get_entropy().mean()
        else:
            costs = []
            for j in range(self.env.action_size):
                worldMap_ = deepcopy(worldMap)
                cost,next_index,next_pos,is_valid = self.getmpcost(pos,index,j,agentID,worldMap_)
                if is_valid:
                    costs.append(cost + self.plan_action(next_pos,next_index,agentID,current_depth+1,worldMap_))
                else:
                    costs.append(cost + -100000)
            if current_depth==0:
                best_action = np.argmax(np.array(costs))
                return best_action,np.max(np.array(costs))
            else:
                return np.max(np.array(costs))

    '''
    For imitation learning
    '''
    def return_action(self,agentID):
        agent = self.env.agents[agentID]
        action, cost = self.plan_action(deepcopy(agent.pos), deepcopy(agent.index), agentID)
        return action,cost

    def run_test(self,test_map_ID, test_ID):
        '''
        Run a test episode
        @param test_map_ID: ID of the test map (0-24)
        @param test_ID: testing ID for this set of map (0-ENV_PARAMS['TEST_PER_MAP']-1)
        '''

        episode_step = 0.0
        episode_rewards = 0.0
        self.env.reset(episode_num = 0, test_map = test_ID, test_indices = test_map_ID)
        frames = []
        done = False


        # kl_divergence = np.mean(self.env.worldBeliefMap*np.log(
        #     np.clip(self.env.worldBeliefMap,1e-10,1)/np.clip(orig_target_map_dist,1e-10,1)
        # ))
        #kl_divergence = np.mean(np.square(self.env.worldBeliefMap - orig_target_map_dist))

        while ((not self.args_dict['FIXED_BUDGET'] and episode_step < self.env.episode_length) \
               or (self.args_dict['FIXED_BUDGET'])):

            action_dict = {}
            worldMap = deepcopy(self.env.belief_semantic_map)
            for j,agent in enumerate(self.env.agents):
                action_dict[j],cost = self.plan_action(deepcopy(agent.pos),deepcopy(agent.index),j,worldMap=worldMap)
            rewards,done = self.env.step_all(action_dict)
            episode_rewards += np.array(rewards).sum()
            episode_step+=1
            if self.gifs:
                frames += self.env.render(mode='rgb_array')
            if done:
                break

        metrics = self.env.get_final_metrics()
        metrics['episode_reward'] = episode_rewards
        metrics['episode_length'] = episode_step
        metrics['proximity'] = self.env.proximity

        if self.gifs:
            make_gif(np.array(frames),
                 '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(self.gifs_path,
                                                          test_map_ID*self.env_params_dict['TEST_PER_MAP']+test_ID ,
                                                          0,
                                                          episode_rewards))
        return metrics

if __name__=="__main__":
    import baseline_params.GreedySemantic as parameters
    map_index = 20
    planner = GreedySemantic(set_dict(parameters),home_dir='../')
    print(planner.run_test(test_map_ID=0,test_ID=0))