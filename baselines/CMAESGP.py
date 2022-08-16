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

class CMAESGP(il_wrapper):
    def __init__(self,params_dict,home_dir="/"):
        super().__init__(home_dir)
        self.population_size = params_dict['population_size']
        self.depth = params_dict['depth']
        self.threads = params_dict['threads']
        self.min_iterations = params_dict['min_iterations']
        self.iterations = params_dict['iterations']
        self.thresh = params_dict['thresh']
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

    def objective(self,args,vector):
        pos = np.stack(args[0])
        index = args[1]
        agentID = args[2]
        worldMap = deepcopy(np.stack(args[3]))
        # pos = 0
        # index = 0
        # agentID = 0
        # worldMap = 0
        # Remap between -1 and 1
        bins = 2*np.arange(self.env.action_size+1)/self.env.action_size - 1.0
        action_list = np.digitize(vector,bins,right=False) - 1
        action_list[action_list==self.env.action_size] -=1

        next_pos = pos
        next_index = index
        objective = 0
        for action in action_list:
            cost, next_index, next_pos = self.getmpcost(next_pos, next_index, int(action), agentID, worldMap)
            objective += cost

        objective += self.env.getEntropy(worldMap.copy()).mean()

        return objective

    def isValidMP(self, pos, mp, agentID):
        is_valid = mp.is_valid
        mp.translate_start_position(pos)
        _, sp = mp.get_sampled_position()
        # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
        visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
        final_pos = np.round(mp.end_state[:self.spatial_dim]).astype(int)
        #is_valid = is_valid and self.isValidPoses(visited_states, agentID)
        is_valid = is_valid and self.isValidFinalPose(final_pos.T,agentID)
        return is_valid, visited_states

    def isValidPoses(self, poses, agentID):
        is_valid = True
        for state in poses.T:
            is_valid = is_valid and self.isValidPos(state, agentID)
        return is_valid

    def isValidFinalPose(self, final_pose,agentID):
        is_valid = True
        is_valid = is_valid and self.isValidPos(final_pose.T,agentID)
        return is_valid

    def isValidPos(self, pos, agentID):
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
            #beliefMap_ = worldMap.copy()

            logodds_b_map = np.log(np.clip(worldMap[min_x:max_x,min_y:max_y]/
                                           (1- worldMap[min_x:max_x,min_y:max_y]),1e-7,1e7))
            sensor = self.env.sensor_params['sensor_unc'][min_x-(r-range_):max_x-(r-range_),
                                                          min_y-(c - range_):max_y-(c-range_)]
            sensor_log_odds = np.log(np.clip((1 - sensor) / sensor, 1e-7,1e7))
            map_free = 1 / (np.exp(-logodds_b_map + sensor_log_odds) + 1)
            map_occ = 1 / (np.exp(-logodds_b_map - sensor_log_odds) + 1)
            entropy_free = self.env.getEntropy(map_free)
            entropy_occ = self.env.getEntropy(map_occ)
            entropy = self.env.getEntropy(worldMap[min_x:max_x, min_y:max_y])

            entropy_redfree = (1-sensor)*entropy_free + sensor*entropy_occ - entropy
            entropy_redocc = sensor*entropy_free + (1-sensor)*entropy_occ - entropy

            entropy_reduction = entropy_redfree[worldMap[min_x:max_x,min_y:max_y]<0.5].sum()\
                                + entropy_redocc[worldMap[min_x:max_x,min_y:max_y]>=0.5].sum()
            worldMap[min_x:max_x,min_y:max_y][worldMap[min_x:max_x,min_y:max_y]<0.5] = \
                map_free[worldMap[min_x:max_x,min_y:max_y]<0.5]
            worldMap[min_x:max_x, min_y:max_y][worldMap[min_x:max_x, min_y:max_y] >= 0.5] =\
                map_occ[worldMap[min_x:max_x, min_y:max_y] >= 0.5]

            # for j in range(min_x, max_x):
            #     for k in range(min_y, max_y):
            #         logodds_b_map = np.log(worldMap[j, k] / (1 - worldMap[j, k]))
            #         sensor_log_odds = np.log(
            #             (1 - self.env.sensor_params['sensor_unc'][j - (r - range_), k - (c - range_)]) / \
            #             self.env.sensor_params['sensor_unc'][j - (r - range_), k - (c - range_)])
            #         # print(sensor_log_odds)
            #         map_free = 1 / (np.exp(-logodds_b_map + sensor_log_odds)+1)
            #         map_occ = 1/(np.exp(-logodds_b_map - sensor_log_odds)+1)
            #         entropy_free = self.env.getEntropy(map_free)
            #         entropy_occ = self.env.getEntropy(map_occ)
            #         entropy = self.env.getEntropy(worldMap[ j,k])
            #
            #         if worldMap[j,k]<0.5:
            #             entropy_reduction += (1-self.env.sensor_params['sensor_unc'][j - (r - range_), k - (c - range_)])* entropy_free + \
            #                                  self.env.sensor_params['sensor_unc'][j - (r - range_), k - (c - range_)] * entropy_occ - \
            #                                  entropy
            #             worldMap[j,k] = map_free
            #         else:
            #             entropy_reduction += self.env.sensor_params['sensor_unc'][j - (r - range_), k - (c - range_)] * entropy_free + \
            #                                  (1 - self.env.sensor_params['sensor_unc'][j - (r - range_), k - (c - range_)]) * entropy_occ - \
            #                                  entropy
            #             worldMap[j,k] = map_occ

        return entropy_reduction

    def getmpcost(self,pos,index,action,agentID,worldMap):
        mp = deepcopy(self.mp_graph[index, action])
        reward = 0
        next_pos = pos
        next_index = index
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
                self.visited_states = visited_states
                for state in visited_states.T:
                    reward += worldMap[state[0], state[1]]
                    worldMap[state[0], state[1]] = 0
                reward -= mp.cost/ mp.subclass_specific_data.get('rho', 1) / 10 / 10000
            else:
                reward += REWARD.COLLISION.value * 2
            reward += worldMap[int(next_pos[0]), int(next_pos[1])]
            worldMap[next_pos[0], next_pos[1]] = 0
        else:
            reward += REWARD.COLLISION.value * 3
        return reward,next_index,next_pos


    def plan_action(self,pos,index,agentID,worldMap=None):
        args = [deepcopy(pos.tolist()),index,agentID,deepcopy(worldMap.tolist())]
        bounds = np.zeros((self.depth,2))
        bounds[:,0] = -1.0
        bounds[:,1] = 1.0
        optimiser = CMA(mean=np.zeros(self.depth),sigma = 1.0,bounds=bounds,population_size=self.population_size)
        #partial_objective = functools.partial(self.objective,args)
        #with pool(self.threads) as p:
        std_list = []
        best_costs = []
        flag = False
        for gen in range(self.iterations):
            vectors = []
            costs = []
            for _ in range(optimiser.population_size):
                vectors.append(deepcopy(optimiser.ask().tolist()))
                costs.append(self.objective(args,vectors[-1]))
            #p.starmap(partial_objective,vectors)

            solutions = [(vectors[j],-costs[j]) for j in range(optimiser.population_size)]
            optimiser.tell(solutions)
            std_list.append(np.std(np.array(costs)))
            best_costs.append(np.max(np.array(costs)))
            #print(std_list[-1])
            if gen>=self.min_iterations:
                if np.mean(np.array(std_list)[-self.min_iterations:])<self.thresh:
                    flag = True
                if np.all(np.abs(costs-best_costs[-1])<self.thresh):
                    flag = True

            if flag:
                break

        best_vector = optimiser.ask()
        cost = self.objective(args,best_vector)

        bins = 2*np.arange(self.env.action_size+1)/self.env.action_size - 1.0
        action_list = np.digitize(best_vector,bins,right=False) - 1
        action_list[action_list==self.env.action_size] -=1

        return action_list,cost

    def return_action(self,agentID):
        agent  = self.env.agents[agentID]
        if self.counter>=self.depth:
            self.counter = 0
            self.action_list, self.costs = self.plan_action(deepcopy(agent.pos), deepcopy(agent.index), agentID,
                                              deepcopy(self.env.worldMap.copy()))
        self.counter+=1
        return self.action_list[self.counter-1],self.costs[self.counter-1]

    def run_test(self,rewardMap,ID=0,targetMap=None):
        episode_step = 0.0
        episode_rewards = 0

        np.random.seed(seed=ID)
        self.env.reset(rewardMap,targetMap)

        frames = []
        metrics = dict()
        done = False
        while ((not self.args_dict['FIXED_BUDGET'] and episode_step < self.env.episode_length)\
               or (self.args_dict['FIXED_BUDGET'])):
            action_dicts = [{} for j in range(self.depth)]
            for j,agent in enumerate(self.env.agents):
                action_list,costs  = self.plan_action(deepcopy(agent.pos),deepcopy(agent.index),\
                                                      j,deepcopy(self.env.worldMap.copy()))
                for k,action in enumerate(action_list):
                    action_dicts[k][j] = action

            for j in range(self.depth):
                if self.gifs:
                    frames.append(self.env.render(mode='rgb_array'))
                rewards,done = self.env.step_all(action_dicts[j])
                episode_rewards += np.array(rewards).sum()
                episode_step+=1
                if done:
                    break
            if done:
                break

        metrics = self.env.get_final_metrics()
        metrics['episode_reward'] = episode_rewards
        metrics['episode_length'] = episode_step

        if self.gifs:
            make_gif(np.array(frames),
                     '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(self.gifs_path, ID, 0, episode_rewards))
        return metrics

if __name__=="__main__":
    import baseline_params.CMAESGPparams as parameters
    map_index = 20
    dir_name = os.getcwd() + "/../" + MAP_TEST_DIR + '/' + TEST_TYPE.format(30) +'/'
    file_name = dir_name + "tests{}env.npy".format(map_index)
    rewardmap = np.load(file_name)
    file_name = dir_name + "tests{}target.npy".format(map_index)
    targetmap = np.load(file_name)
    planner = CMAESGP(set_dict(parameters),home_dir='/../')
    print(planner.run_test(rewardmap))