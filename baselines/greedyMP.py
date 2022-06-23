from env.searchenvMP import *
from env.env_setter import *
import numpy as np
from copy import deepcopy
from params import *
import os
from Utilities import set_dict
from env.render import *
from multiprocessing import Pool as pool
from baselines.il_wrapper import il_wrapper

class GreedyMP(il_wrapper):
    def __init__(self,params_dict,home_dir="/"):
        super(GreedyMP).__init__(home_dir)
        self.gifs = params_dict['GIFS']
        self.gifs_path = params_dict['GIFS_PATH']
        self.depth = params_dict['depth']


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
                if current_depth == 0:
                    worldMap = deepcopy(self.env.worldMap.copy())
                else:
                    worldMap = deepcopy(worldMap.copy())
                cost,next_index,next_pos = self.getmpcost(pos,index,j,agentID,worldMap)
                costs.append(cost + self.plan_action(next_pos,next_index,agentID,current_depth+1,worldMap))
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

    def run_test(self,rewardmap,ID=0):
        episode_step = 0
        episode_rewards = 0

        self.env.reset(rewardmap)
        frames = []
        while(episode_step <self.env.episode_length):
            if self.gifs:
                frames.append(self.env.render(mode='rgb_array'))
            action_dict = {}
            for j,agent in enumerate(self.env.agents):
                action_dict[j],cost = self.plan_action(deepcopy(agent.pos),deepcopy(agent.index),j)
            rewards,done = self.env.step_all(action_dict)
            episode_rewards += np.array(rewards).sum()
            episode_step+=1
        if self.gifs:
            make_gif(np.array(frames),
                 '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(self.gifs_path,ID , 0, episode_rewards))
        return episode_rewards

if __name__=="__main__":
    import baseline_params.GreedyMPparams as parameters
    map_index = 20
    dir_name = os.getcwd() + "/../" + MAP_TEST_DIR
    file_name = dir_name + "tests{}.npy".format(map_index)
    rewardmap = np.load(file_name)
    planner = GreedyMP(set_dict(parameters),home_dir='/../')
    print(planner.run_test(rewardmap,map_index))