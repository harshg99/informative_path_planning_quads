import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing as mp

import numpy as np

from baselines import baseline_setter
import os
import sys,getopt
from params import *
import json
from pdb import set_trace as T
import Utilities
from models.alg_setter import alg_setter
from env.env_setter import env_setter
from copy import  deepcopy
import argparse
from env.searchenv import REWARD

class TrajectoryVisualisation:
    def __init__(self,args_dict:dict,map_index,map_size,model_path: str,gifs: bool):
        self.args_dict = args_dict
        self.args_dict['GPU'] = False  # CPU testing
        self.gifs = args_dict['TEST_GIFS']
        self.env = env_setter.set_env(args_dict)
        self.model = alg_setter.set_model(self.env, args_dict)
        self.gifs_path = self.args_dict['TEST_GIFS_PATH'].format(model_path)

        print('Loading Model')
        model_path = 'data/models/' + model_path
        checkpoint = torch.load(model_path + "/checkpoint.pkl", map_location=self.args_dict['DEVICE'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_episode = checkpoint['epoch']
        print('Model results at Episode: {}'.format(curr_episode))

        dir_name = os.getcwd() + "/" + MAP_TEST_DIR + '/' + TEST_TYPE.format(map_size) + '/'
        file_name = dir_name + "tests{}env.npy".format(map_index)
        self.rewardmap = np.load(file_name)
        file_name = dir_name + "tests{}target.npy".format(map_index)
        self.targetmap = np.load(file_name)
        file_name = dir_name + "tests{}target_orig_dist.npy".format(map_index)
        orig_target_map = np.load(file_name)
        self.args_dict = args_dict
        self.gifs = gifs
        self.env.reset(deepcopy(self.rewardmap), deepcopy(self.targetmap),deepcopy(orig_target_map))

        plt.imshow(np.transpose(self.env.orig_worldMap), cmap='viridis',
                   interpolation=None,
                   extent=[0, self.env.orig_worldMap.shape[0],
                           0, self.env.orig_worldMap.shape[1]],
                   origin="lower")
        plt.colorbar(label='Probability of Target Occupancy')
        plt.title('Agent Prior')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        plt.figure()
        plt.imshow(np.transpose(
            self.env.orig_target_distribution_map/self.env.orig_target_distribution_map.max()),
            cmap='viridis',
            interpolation='spline36',
            extent=[0, self.env.orig_worldMap.shape[0],
                    0, self.env.orig_worldMap.shape[1]],
            origin="lower"
        )

        plt.colorbar(label='Probability of Target Occupancy')
        plt.title('Ground Truth Distribution')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        plt.figure()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.imshow(np.transpose(self.env.orig_worldTargetMap), cmap='viridis',
                   interpolation=None,
                   extent=[0, self.env.orig_worldMap.shape[0],
                           0, self.env.orig_worldMap.shape[1]],
                   origin="lower")
        plt.colorbar(label='Probability of Target Occupancy')
        plt.title('Target Map')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

    def plan_action(self, observation,hidden_in=None):

        # print(observation)
        hidden_in = hidden_in
        if self.args_dict['LSTM']:
            policy, value,hidden_out = self.model.forward_step(observation,hidden_in)
            hidden_in = hidden_out.cpu().detach().numpy()
        else:
            policy, value = self.model.forward_step(observation)
        policy = policy.cpu().detach().numpy()
        action_dict = Utilities.best_actions(policy)

        return action_dict,hidden_in

    # Choosing which test to run
    def trajectory(self):
        observation = self.env.get_obs_all()
        episode_step = 0

        agent_poses = [self.env.agents[0].pos_actual]
        primitives = []
        frames = []
        hidden_in = None
        plt.figure(figsize=(7, 6))


        while ((not self.args_dict['FIXED_BUDGET'] and
                episode_step < self.env.episode_length) \
               or (self.args_dict['FIXED_BUDGET'])):
            if self.gifs:
                frames.append(self.env.render(mode='rgb_array'))

            if self.args_dict['LSTM']:
                action_dict, hidden_in = self.plan_action(observation, hidden_in)
            else:
                action_dict, _ = self.plan_action(observation, hidden_in)
            rewards, done = self.env.step_all(action_dict)

            if self.env.agents[0].current_primitive is not None:
                primitives.append(self.env.agents[0].current_primitive)

            agent_poses.append(self.env.agents[0].pos_actual)
            #episode_rewards += np.array(rewards).sum()
            observation = self.env.get_obs_all()

            episode_step += 1
            if done:
                break

            frac =  self.env.agents[0].agentBudget/(REWARD.MP.value*self.args_dict['BUDGET'])
            if (frac<=0.5 and frac>=0.48) or \
                    (frac<= 0.25 and frac>=0.23) or \
                    (frac<= 0.01 and frac>=-0.01) or\
                    (frac<=0.75 and frac>=0.73):
                plt.figure(figsize=(7, 6))
                plt.imshow(np.transpose(self.env.orig_worldMap), cmap='viridis',
                           interpolation=None,
                           extent=[0, self.env.orig_worldMap.shape[0],
                                   0, self.env.orig_worldMap.shape[1]],
                           origin="lower")
                plt.colorbar(label='Probability of Target Occupancy')
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')

                plt.plot(agent_poses[0][0], agent_poses[0][1], 'go', markersize=12)
                plt.plot(agent_poses[-1][0], agent_poses[-1][1], 'ro', markersize=12)

                for pose,mp in zip(agent_poses,primitives):
                    if mp is not None:
                        mp.translate_start_position(pose)
                        mp.plot(position_only=True, step_size=.01)
                        plt.plot(pose[0], pose[1], 'y.')



                plt.figure(figsize=(7, 6))
                plt.imshow(np.transpose(self.env.worldMap), cmap='viridis',
                           interpolation=None,
                           extent=[0, self.env.worldMap.shape[0],
                                   0, self.env.worldMap.shape[1]],
                           origin="lower")
                plt.colorbar(label='Probability of Target Occupancy')
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')

                plt.plot(agent_poses[0][0], agent_poses[0][1], 'go', markersize=12)
                plt.plot(agent_poses[-1][0], agent_poses[-1][1], 'ro', markersize=12)

                for pose, mp in zip(agent_poses, primitives):
                    if mp is not None:
                        mp.translate_start_position(pose)
                        mp.plot(position_only=True, step_size=.01)
                        plt.plot(pose[0], pose[1], 'y.')


        #pass
        plt.show()

class MapVisualisation:
    def __init__(self,args_dict,map_index,map_size):
        dir_name = os.getcwd() + "/" + MAP_TEST_DIR + '/' + TEST_TYPE.format(map_size) + '/'
        file_name = dir_name + "tests{}env.npy".format(map_index)
        self.rewardmap = np.load(file_name)
        file_name = dir_name + "tests{}target.npy".format(map_index)
        self.targetmap = np.load(file_name)
        self.args_dict = args_dict
        

    def visualise(self):
        plt.figure(figsize=(8, 6))
        x = np.linspace(0, 29, 30)
        y = np.linspace(0, 29, 30)
        X, Y = np.meshgrid(x, y)
        levels = np.linspace(0, 1, 100)
        plt.contourf(X, Y, np.transpose(self.rewardmap)*100/2+0.1, levels, cmap='viridis')
        plt.colorbar()
        plt.xlabel('Distance (m)', fontsize=24, color='red')
        matplotlib.rcParams.update({'font.size': 16})
        #plt.show()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--index", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--size", type=int, default=30,
                        help="Size of the map")
    parser.add_argument('--path',type=str,default="GPAC_Model6",
                        help="path to the model")
    parser.add_argument('--gifs', type=bool, default=True,
                        help="outputting gifs")

    args = parser.parse_args()
    # fmt: on
    return args

if __name__=="__main__":
    import params as args
    from Utilities import set_dict
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = Utilities.set_dict(args)
    input_args = parse_args()

    # map_vis = MapVisualisation(
    #     args_dict=args,
    #     map_index = input_args.index,
    #     map_size = input_args.size
    # )
    #
    # map_vis.visualise()

    trajectory_vis = TrajectoryVisualisation(
        args_dict=args,
        map_index = input_args.index,
        map_size = input_args.size,
        model_path=input_args.path,
        gifs=input_args.gifs
    )
    trajectory_vis.trajectory()

