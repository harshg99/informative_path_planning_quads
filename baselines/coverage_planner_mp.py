from env.searchenvMP import *
from env.env_setter import *
import numpy as np
from copy import deepcopy
from params import *
import os
from Utilities import set_dict
from cmaes import CMA
from multiprocessing import Pool as pool
import functools
from baselines.il_wrapper import il_wrapper

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


class coverage_planner_mp(il_wrapper):
    def __init__(self,params_dict,home_dir="/"):
        super().__init__(home_dir)

        self.depth = params_dict['depth']
        self.counter = 0
        self.gifs = params_dict['GIFS']
        self.gifs_path = params_dict['GIFS_PATH']
        self.results_path = params_dict['RESULTS_PATH']

        if not os.path.isdir(self.gifs_path):
            os.makedirs(self.gifs_path)
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        import env_params.GPPrim as parameters
        env_params_dict = set_dict(parameters)
        env_params_dict['home_dir'] = os.getcwd() + home_dir

        import params as args
        self.args_dict = set_dict(args)
        self.env = GPEnvMP(env_params_dict, self.args_dict)

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

    def return_action(self,agentID):
        agent = self.env.agents[agentID]
        action, cost = self.plan_action(deepcopy(agent.pos), deepcopy(agent.index), agentID)
        return action

    def getmpcost(self,pos,index,action,agentID,coverageMap):
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
                init_coverage_map = deepcopy(coverageMap)
                for state in self.visited_states.tolist():
                    r = state[0]
                    c = state[1]
                    range_ = int(self.env.sensor_params['sensor_range'] / 2)
                    min_x = np.max([r - range_, 0])
                    min_y = np.max([c - range_, 0])
                    max_x = np.min([r + range_ + 1, self.env.worldBeliefMap.shape[0]])
                    max_y = np.min([c + range_ + 1, self.env.worldBeliefMap.shape[1]])
                    init_coverage_map[min_x:max_x,min_y:max_y] = 0.0

                reward = coverageMap.sum()-init_coverage_map.sum()

            elif visited_states is not None:
                # reward += REWARD.COLLISION.value*(visited_states.shape[0]+1)
                reward += REWARD.COLLISION.value
            else:
                reward += REWARD.COLLISION.value * 1.5

        return reward,next_index,next_pos,is_valid

    def plan_action(self, pos, index, agentID, current_depth=0, coverageMap=None):
        if current_depth >= self.depth:
            return 0
        else:
            costs = []

            for j in range(self.env.action_size):
                coverageMap = deepcopy(coverageMap)
                cost, next_index, next_pos,is_valid = self.getmpcost(pos, index, j, agentID, coverageMap)
                if is_valid:
                    costs.append(cost + self.plan_action(next_pos,next_index,agentID,
                                                         current_depth+1,coverageMap))
                else:
                    costs.append(cost + -100000)


            if current_depth == 0:
                best_action = np.argmax(np.array(costs))
                return best_action, np.max(np.array(costs))
            else:
                return np.max(np.array(costs))

    def run_test(self, rewardmap, ID=0, targetMap=None,orig_target_map_dist=None):
        episode_step = 0.0
        episode_rewards = 0.0
        np.random.seed(seed=ID)
        self.env.reset(rewardmap, targetMap,orig_target_map_dist)
        frames = []
        done = False
        coverageMap = np.ones(self.env.worldMap.shape)

        kl_divergence = np.mean(self.env.worldBeliefMap*np.log(
            np.clip(self.env.worldBeliefMap,1e-10,1)/np.clip(orig_target_map_dist,1e-10,1)
        ))
        #kl_divergence = np.mean(np.square(self.env.worldBeliefMap - orig_target_map_dist))

        while ((not self.args_dict['FIXED_BUDGET'] and episode_step < self.env.episode_length) \
               or (self.args_dict['FIXED_BUDGET'])):
            if self.gifs:
                frames.append(self.env.render(mode='rgb_array'))
            action_dict = {}
            for j, agent in enumerate(self.env.agents):
                action_dict[j], cost = self.plan_action(deepcopy(agent.pos),
                                                        deepcopy(agent.index), j,
                                                        coverageMap=coverageMap)

            rewards, done = self.env.step_all(action_dict)
            for j, agent in enumerate(self.env.agents):
                for state in agent.visited_states.T.tolist():
                    r = state[0]
                    c = state[1]
                    range_ = int(self.env.sensor_params['sensor_range'] / 2)
                    min_x = np.max([r - range_, 0])
                    min_y = np.max([c - range_, 0])
                    max_x = np.min([r + range_ + 1, self.env.worldBeliefMap.shape[0]])
                    max_y = np.min([c + range_ + 1, self.env.worldBeliefMap.shape[1]])
                    coverageMap[min_x:max_x, min_y:max_y] = 0.0

            episode_rewards += np.array(rewards).sum()
            episode_step += 1
            if done:
                break

        metrics = self.env.get_final_metrics()
        metrics['episode_reward'] = episode_rewards
        metrics['episode_length'] = episode_step
        metrics['divergence'] = kl_divergence

        if self.gifs:
            make_gif(np.array(frames),
                     '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(self.gifs_path, ID, 0, episode_rewards))
        return metrics

if __name__=="__main__":
    import baseline_params.CoverageGPParams as parameters
    map_index = 50
    dir_name = os.getcwd() + "/../" + MAP_TEST_DIR + '/' + TEST_TYPE.format(30) +'/'
    file_name = dir_name + "tests{}env.npy".format(map_index)
    rewardmap = np.load(file_name)
    file_name = dir_name + "tests{}target.npy".format(map_index)
    targetmap = np.load(file_name)
    file_name = dir_name + "tests{}target_orig_dist.npy".format(map_index)
    orig_target_map = np.load(file_name)
    planner = coverage_planner_mp(set_dict(parameters),home_dir='/../')
    print(planner.run_test(rewardmap,map_index,orig_target_map_dist=orig_target_map))

