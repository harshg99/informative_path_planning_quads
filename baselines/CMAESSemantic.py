from env.GPSemantic import *
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

class CMAESSemantic(il_wrapper_semantic):
    def __init__(self,params_dict,home_dir="/"):
        super().__init__(home_dir)
        self.population_size = params_dict['population_size']
        self.depth = params_dict['depth']
        self.threads = params_dict['threads']
        self.min_iterations = params_dict['min_iterations']
        self.iterations = params_dict['iterations']
        self.thresh = params_dict['thresh']
        self.counter = 0
        self.results_path = params_dict['RESULTS_PATH']
        self.exploration = params_dict['exploration']

        if not os.path.exists(self.gifs_path):
            os.makedirs(self.gifs_path)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)



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

        objective += worldMap.get_entropy().sum()

        return objective


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

    def run_test(self,rewardMap,ID=0,targetMap=None,orig_target_map_dist=None):
        episode_step = 0.0
        episode_rewards = 0

        np.random.seed(seed=ID)
        self.env.reset(rewardMap,targetMap,orig_target_map_dist)

        frames = []
        metrics = dict()
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
        #kl_divergence = np.mean(np.square(self.env.worldBeliefMap-orig_target_map_dist))

        while ((not self.args_dict['FIXED_BUDGET'] and episode_step < self.env.episode_length)\
               or (self.args_dict['FIXED_BUDGET'])):
            action_dicts = [{} for j in range(self.depth)]
            for j,agent in enumerate(self.env.agents):
                worldMap = deepcopy(self.env.worldMap)
                worldMap[self.env.worldTargetMap==2] = 0
                action_list,costs  = self.plan_action(
                    deepcopy(agent.pos),
                    deepcopy(agent.index),
                    j,
                    worldMap
                )
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
        metrics['divergence'] = kl_divergence

        if self.gifs:
            make_gif(np.array(frames),
                     '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(
                         self.gifs_path,
                         ID,
                         0,
                         episode_rewards))
        return metrics

if __name__=="__main__":
    import baseline_params.CMAESSemantic as parameters
    map_index = 20
    planner = CMAESSemantic(set_dict(parameters),home_dir='../')
    print(planner.run_test(test_map_ID=0,test_ID=0))