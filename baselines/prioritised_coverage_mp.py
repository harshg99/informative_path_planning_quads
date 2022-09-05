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
from baselines.coverage_planner_mp import coverage_planner_mp

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


class prioritised_coverage_mp(coverage_planner_mp):
    def __init__(self,params_dict,home_dir="/"):
        super().__init__(params_dict,home_dir)

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

    def run_test(self, rewardmap, ID=0, targetMap=None,orig_target_map_dist=None):
        episode_step = 0.0
        episode_rewards = 0.0
        np.random.seed(seed=ID)
        self.env.reset(rewardmap, targetMap,orig_target_map_dist)
        frames = []
        done = False
        #coverageMap = np.ones(self.env.worldMap.shape)
        coverageMap = deepcopy(self.env.worldMap)
        # kl_divergence = np.mean(self.env.worldBeliefMap*np.log(
        #     np.clip(self.env.worldBeliefMap,1e-10,1)/np.clip(orig_target_map_dist,1e-10,1)
        # ))
        kl_divergence = np.mean(np.square(self.env.worldBeliefMap - orig_target_map_dist))

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
            coverageMap = deepcopy(self.env.worldMap)

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
    map_index = 48
    dir_name = os.getcwd() + "/../" + MAP_TEST_DIR + '/' + TEST_TYPE.format(30) +'/'
    file_name = dir_name + "tests{}env.npy".format(map_index)
    rewardmap = np.load(file_name)
    file_name = dir_name + "tests{}target.npy".format(map_index)
    targetmap = np.load(file_name)
    file_name = dir_name + "tests{}target_orig_dist.npy".format(map_index)
    orig_target_map = np.load(file_name)
    planner = coverage_planner_mp(set_dict(parameters),home_dir='/../')
    print(planner.run_test(rewardmap,map_index,targetMap=targetmap,orig_target_map_dist=orig_target_map))

