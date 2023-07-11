
from env.env_setter import *
import numpy as np
from copy import deepcopy
from params import *
import os
from Utilities import set_dict
from cmaes import CMA
from multiprocessing import Pool as pool
import functools
from baselines.il_wrapper import il_wrapper_semantic
from env.GPSemantic import *
from env.SemanticMap import *
import cProfile

class Node:
    def __init__(self,incoming,outgoing,state,map_state,current_cost,depth = None,cost_fn=None):
        self.incoming = incoming
        self.outgoing = outgoing
        self.state = state
        self.map_state = map_state
        self.cost_fn =  cost_fn
        self.current_cost = current_cost
        self.depth = None

    def __eq__(self, other):
        if self.incoming == other.incoming and self.outgoing == other.outgoing\
                and np.all(self.state==other.state) and self.depth==other.depth\
                and np.all(self.map_state==other.map_state):
            return True
        return False

    def __lt__(self,other):
        if self.heuristic_cost_fn(self.map_state) + self.current_cost <= \
                self.heuristic_cost_fn(other.map_state)+ other.current_cost:
            return True
        return False

    def heuristic_cost_fn(self):
        return self.cost_fn(self.map_state)


class coverage_planner_semantic(il_wrapper_semantic):
    def __init__(self,params_dict,home_dir="./"):
        super().__init__(params_dict,home_dir)

        self.depth = params_dict['depth']
        self.counter = 0
        self.results_path = params_dict['RESULTS_PATH']

        if not os.path.exists(self.gifs_path):
            os.makedirs(self.gifs_path)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)




    def return_action(self,agentID):
        agent = self.env.agents[agentID]
        action, cost = self.plan_action(deepcopy(agent.pos), deepcopy(agent.index), agentID)
        return action

    def getCoverage(self, visited_states,world_map):

        world_map_init = deepcopy(world_map)

        for state in visited_states.tolist():

            semantic_obs,_ = world_map.get_observations(state, fov=self.env.sensor_params['sensor_range'],
                                                       scale=None, type='semantic',
                                                       return_distance=False, resolution=None)

            projected_measurement = np.argmax(semantic_obs, axis=-1)

            world_map.update_semantics(state, projected_measurement, self.env.sensor_params)


        init_coverage = np.sum(world_map_init.coverage_map - world_map.coverage_map)
        coverage = -init_coverage

        return coverage/(np.square(self.env.sensor_params['sensor_range'][0])*world_map.resolution**2)
        #return entropy_reduction

    def getmpcost(self,pos,index,action,agentID,world_map):
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
                reward = self.getCoverage(self.visited_states,world_map)
                reward -= mp.cost / mp.subclass_specific_data.get('rho', 1) / 10 / self.env.mp_cost_norm
            elif visited_states is not None:
                # reward += REWARD.COLLISION.value*(visited_states.shape[0]+1)
                reward += REWARD.COLLISION.value
            else:
                reward += REWARD.COLLISION.value * 1.5

        return reward,next_index,next_pos,is_valid

    def plan_action(self, pos, index, agentID, current_depth=0, worldMap=None):
        if current_depth >= self.depth:
            return 0
        else:
            costs = []

            for j in range(self.env.action_size):
                coverageMap = deepcopy(worldMap)
                cost, next_index, next_pos,is_valid = self.getmpcost(pos, index, j, agentID, coverageMap)
                if is_valid:
                    costs.append(cost + self.plan_action(next_pos,next_index,agentID,
                                                         current_depth+1,coverageMap))
                else:
                    costs.append(cost + -1000000.0)


            if current_depth == 0:
                best_cost = np.max(np.array(costs))
                best_actions = np.argwhere(costs==best_cost)
                best_action = np.random.choice(best_actions.flatten())
                return best_action, np.array(costs)[best_action]
            else:
                return np.max(np.array(costs))

    def run_test(self,test_map_ID, test_ID):
        '''
        Run a test episode
        @param test_map_ID: ID of the test map (0-24)
        @param test_ID: testing ID for this set of map (0-ENV_PARAMS['TEST_PER_MAP']-1)
        '''

        episode_step = 0.0
        episode_rewards = 0.0
        self.env.reset(episode_num=0, test_map=test_ID, test_indices=test_map_ID)
        frames = []
        done = False

        # kl_divergence = np.mean(self.env.worldBeliefMap*np.log(
        #     np.clip(self.env.worldBeliefMap,1e-10,1)/np.clip(orig_target_map_dist,1e-10,1)
        # ))
        # kl_divergence = np.mean(np.square(self.env.worldBeliefMap - orig_target_map_dist))

        while ((not self.args_dict['FIXED_BUDGET'] and episode_step < self.env.episode_length) \
               or (self.args_dict['FIXED_BUDGET'])):

            action_dict = {}
            worldMap = deepcopy(self.env.belief_semantic_map)
            for j,agent in enumerate(self.env.agents):
                action_dict[j],cost = self.plan_action(deepcopy(agent.pos),deepcopy(agent.index),j,worldMap=worldMap)
            returns = self.env.step_all(action_dict)
            if len(returns) == 3:
                rewards, rewards_dict, done = returns
            else:
                rewards, done = returns
            episode_rewards += np.array(rewards).sum()
            episode_step+=1
            print("Step: {:d}, Reward: {:.2f}, Cost: {:.2f} BudgetRem{:.2f}"\
                  .format(int(episode_step),episode_rewards,cost.item(),self.env.agents[0].agent_budget/self.env.mp_cost_norm))

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
    import baseline_params.CoverageSemantic as parameters
    planner = coverage_planner_semantic(set_dict(parameters),home_dir='../')
    cProfile.run('print(planner.run_test(test_map_ID=0,test_ID=0))')

