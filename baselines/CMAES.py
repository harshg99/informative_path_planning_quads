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
class CMAES:
    def __init__(self,params_dict):
        self.population_size = params_dict['population_size']
        self.depth = params_dict['depth']
        self.threads = params_dict['threads']
        self.min_iterations = params_dict['min_iterations']
        self.iterations = params_dict['iterations']
        self.thresh = params_dict['thresh']
        import env_params.MotionPrim as parameters
        env_params_dict = set_dict(parameters)
        env_params_dict['home_dir'] = os.getcwd()+"/.."
        self.env = SearchEnvMP(env_params_dict)

        self.mp_graph = self.env.minimum_action_mp_graph
        self.lookup = self.env.lookup_dictionary
        self.num_tiles = self.env.mp_graph.num_tiles
        self.spatial_dim = self.env.mp_graph.num_dims

    def isValidMP(self,pos,mp,agentID):
        is_valid = mp.is_valid
        mp.translate_start_position(pos)
        _, sp = mp.get_sampled_position()
        # visited_states = np.round(mp.end_state[:mp.num_dims]).astype(np.int32).reshape(mp.num_dims,1)
        visited_states = np.unique(np.round(sp).astype(np.int32), axis=1)
        is_valid = is_valid and self.isValidPoses(visited_states,agentID)
        return is_valid,visited_states

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
            objective +=cost

        return objective


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
        with pool(self.threads) as p:
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

    def run_test(self,rewardMap):
        episode_step = 0
        episode_rewards = 0

        self.env.reset(rewardMap)

        while(episode_step <self.env.episode_length):
            action_dicts = [{} for j in range(self.depth)]
            for j,agent in enumerate(self.env.agents):
                action_list,costs  = self.plan_action(deepcopy(agent.pos),deepcopy(agent.index),j,deepcopy(self.env.worldMap.copy()))
                for k,action in enumerate(action_list):
                    action_dicts[k][j] = action

            for j in range(self.depth):
                rewards,done = self.env.step_all(action_dicts[j])
                episode_rewards += np.array(rewards).sum()
                episode_step+=1

        return episode_rewards

if __name__=="__main__":
    import baseline_params.CMAESparams as parameters
    map_index = 20
    dir_name = os.getcwd() + "/../" + MAP_TEST_DIR
    file_name = dir_name + "tests{}.npy".format(map_index)
    rewardmap = np.load(file_name)
    planner = CMAES(set_dict(parameters))
    print(planner.run_test(rewardmap))