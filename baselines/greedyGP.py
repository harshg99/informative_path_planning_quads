import json

from env.GPEnvMP import *
from env.env_setter import *
import numpy as np
from copy import deepcopy
from params import *
import os
from Utilities import set_dict
from env.render import *
from multiprocessing import Pool as pool
from baselines.il_wrapper import il_wrapper

class GreedyGP(il_wrapper):
    def __init__(self,params_dict,home_dir="/"):
        super().__init__(home_dir)
        self.gifs = params_dict['GIFS']
        self.gifs_path = params_dict['GIFS_PATH']
        self.results_path = params_dict['RESULTS_PATH']
        self.depth = params_dict['depth']
        self.exploration = params_dict['exploration']

        if not os.path.isdir(self.gifs_path):
            os.makedirs(self.gifs_path)
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        import env_params.GPPrim as parameters
        env_params_dict = set_dict(parameters)
        env_params_dict['home_dir'] = os.getcwd() + home_dir

        import params as args

        args_dict = set_dict(args)
        self.args_dict = args_dict
        self.env = GPEnvMP(env_params_dict,args_dict)

    def isValidMP(self,pos,mp,agentID):
        is_valid = mp.is_valid
        mp.translate_start_position(pos)
        _, sp = mp.get_sampled_position()
        # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
        visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
        final_pos = np.round(mp.end_state[:self.spatial_dim]).astype(int)
        #is_valid = is_valid and self.isValidPoses(visited_states, agentID)
        is_valid = is_valid and self.isValidFinalPose(final_pos.T,agentID)
        return is_valid,visited_states

    def isValidFinalPose(self, final_pose,agentID):
        is_valid = True
        is_valid = is_valid and self.isValidPos(final_pose.T,agentID)
        return is_valid


    def isValidPoses(self, poses,agentID):
        is_valid = True
        for state in poses.T:
            is_valid = is_valid and self.isValidPos(state,agentID)
        return is_valid

    def isValidPos(self, pos,agentID):
        is_valid = (np.array(pos - self.env.agents[agentID].pad) > -1).all()
        is_valid = is_valid and (np.array(pos + self.env.agents[agentID].pad) \
                                 < self.env.agents[agentID].world_size).all()
        return is_valid


    def getExpectedEntropy(self, visited_states,worldMap):

        beliefMap = worldMap.copy()
        entropy_reduction = 0

        for state in visited_states.tolist():
            r = state[0]
            c = state[1]
            range_ = int(self.env.sensor_params['sensor_range'] / 2)
            min_x = np.max([r - range_, 0])
            min_y = np.max([c - range_, 0])
            max_x = np.min([r + range_ + 1, self.env.worldBeliefMap.shape[0]])
            max_y = np.min([c + range_ + 1, self.env.worldBeliefMap.shape[1]])
            # beliefMap_ = worldMap.copy()

            logodds_b_map = np.log(np.clip(worldMap[min_x:max_x, min_y:max_y] /
                                           (1 - worldMap[min_x:max_x, min_y:max_y]), 1e-7, 1e7))
            sensor = self.env.sensor_params['sensor_unc'][min_x - (r - range_):max_x - (r - range_),
                     min_y - (c - range_):max_y - (c - range_)]
            sensor_log_odds = np.log(np.clip((1 - sensor) / sensor, 1e-7, 1e7))
            map_free = 1 / (np.exp(-logodds_b_map + sensor_log_odds) + 1)
            map_occ = 1 / (np.exp(-logodds_b_map - sensor_log_odds) + 1)
            # entropy_free = self.env.getEntropy(map_free)
            # entropy_occ = self.env.getEntropy(map_occ)
            entropy_init = self.env.getEntropy(worldMap[min_x:max_x, min_y:max_y]).sum()

            # entropy_redfree = (1-sensor)*entropy_free + sensor*entropy_occ - entropy
            # entropy_redocc = sensor*entropy_free + (1-sensor)*entropy_occ - entropy

            # entropy_reduction = entropy_redfree[worldMap[min_x:max_x,min_y:max_y]<0.5].sum()\
            #                     + entropy_redocc[worldMap[min_x:max_x,min_y:max_y]>=0.5].sum()
            worldMap[min_x:max_x, min_y:max_y][worldMap[min_x:max_x, min_y:max_y] < 0.5] = \
                map_free[worldMap[min_x:max_x, min_y:max_y] < 0.5]
            worldMap[min_x:max_x, min_y:max_y][worldMap[min_x:max_x, min_y:max_y] >= 0.5] = \
                map_occ[worldMap[min_x:max_x, min_y:max_y] >= 0.5]
            entropy_final = self.env.getEntropy(worldMap[min_x:max_x, min_y:max_y]).sum()
            entropy_reduction = entropy_final - entropy_init


        return entropy_reduction/np.square(self.env.sensor_params['sensor_range'])
        #return entropy_reduction

    def getMean(self,visited_states,worldMap):
        beliefMap = worldMap.copy()
        entropy_reduction = 0

        for state in visited_states.tolist():
            r = state[0]
            c = state[1]
            range_ = int(self.env.sensor_params['sensor_range'] / 2)
            min_x = np.max([r - range_, 0])
            min_y = np.max([c - range_, 0])
            max_x = np.min([r + range_ + 1, self.env.worldBeliefMap.shape[0]])
            max_y = np.min([c + range_ + 1, self.env.worldBeliefMap.shape[1]])
            # beliefMap_ = worldMap.copy()


            entropy_reduction += worldMap[min_x:max_x, min_y:max_y].sum()


        return entropy_reduction / np.square(self.env.sensor_params['sensor_range'])
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
                reward -= mp.cost/ mp.subclass_specific
                next_index = self.lookup[index, action]
                next_index = int(np.floor(next_index / self.num_tiles))
                next_pos =  np.round(mp.end_state[:self.spatial_dim]).astype(int)
                # print("{:d} {:d} {:d} {:d}".format(self.pos[0], self.pos[1], visited_states[0,0], visited_states[1,0]))
                self.visited_states = visited_states.T
                reward = self.exploration * self.getMean(self.visited_states,worldMap)+\
                self.getExpectedEntropy(self.visited_states,worldMap)

                reward -= mp.cost/ mp.subclass_specific_data.get('rho', 1) / 10 / self.env.mp_cost_norm
            elif visited_states is not None:
                # reward += REWARD.COLLISION.value*(visited_states.shape[0]+1)
                reward += REWARD.COLLISION.value
            else:
                reward += REWARD.COLLISION.value * 1.5

        return reward,next_index,next_pos,is_valid

    def get_action_cost(self,args,action):
        pos = args[0]
        index = args[1]
        agentID = args[2]
        worldMap = args[3]
        current_depth = args[4]

    def plan_action(self,pos,index,agentID,current_depth=0,worldMap=None):
        if current_depth>=self.depth:
            return 0
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

    def run_test(self,rewardmap,ID=0,targetMap=None,orig_target_map_dist=None):
        episode_step = 0.0
        episode_rewards = 0.0
        np.random.seed(seed=ID)
        self.env.reset(rewardmap,targetMap,orig_target_map_dist)
        frames = []
        done = False
        beleif1 = self.env.worldBeliefMap / self.env.worldBeliefMap.sum()
        belief2 = self.env.orig_target_distribution_map / self.env.orig_target_distribution_map.sum()
        div = beleif1 * np.log(
            np.clip(beleif1, 1e-10, 1) / np.clip(belief2, 1e-10, 1)) + belief2 * np.log(
            np.clip(belief2, 1e-10, 1) / np.clip(beleif1, 1e-10, 1))

        kl_divergence = div.sum()
        # kl_divergence = np.mean(self.env.worldBeliefMap*np.log(
        #     np.clip(self.env.worldBeliefMap,1e-10,1)/np.clip(orig_target_map_dist,1e-10,1)
        # ))
        #kl_divergence = np.mean(np.square(self.env.worldBeliefMap - orig_target_map_dist))

        while ((not self.args_dict['FIXED_BUDGET'] and episode_step < self.env.episode_length) \
               or (self.args_dict['FIXED_BUDGET'])):
            if self.gifs:
                frames += self.env.render(mode='rgb_array')
            action_dict = {}
            worldMap = deepcopy(self.env.worldMap)
            worldMap[self.env.worldTargetMap == 2] = 0
            for j,agent in enumerate(self.env.agents):
                action_dict[j],cost = self.plan_action(deepcopy(agent.pos),deepcopy(agent.index),j,worldMap=worldMap)
            returns = self.env.step_all(action_dict)
            if len(returns) == 3:
                rewards, rewards_dict, done = returns
            else:
                rewards, done = returns
            episode_rewards += np.array(rewards).sum()
            episode_rewards += np.array(rewards).sum()
            episode_step+=1
            if done:
                break

        metrics = self.env.get_final_metrics()
        metrics['episode_reward'] = episode_rewards
        metrics['episode_length'] = episode_step
        metrics['divergence'] = kl_divergence

        if self.gifs:
            make_gif(np.array(frames),
                 '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(self.gifs_path,ID , 0, episode_rewards))
        return metrics

if __name__=="__main__":
    import baseline_params.GreedyGPparams as parameters
    map_index = 20
    dir_name = os.getcwd() + "/../" + MAP_TEST_DIR + '/' + TEST_TYPE.format(30) +'/'
    file_name = dir_name + "tests{}env.npy".format(map_index)
    rewardmap = np.load(file_name)
    file_name = dir_name + "tests{}target.npy".format(map_index)
    targetmap = np.load(file_name)
    file_name = dir_name + "tests{}target_orig_dist.npy".format(map_index)
    orig_target_map = np.load(file_name)
    planner = GreedyGP(set_dict(parameters),home_dir='/../')
    print(planner.run_test(rewardmap,map_index,targetMap=targetmap,orig_target_map_dist=orig_target_map))