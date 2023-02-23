import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing as mp

import numpy as np

from baselines import baseline_setter
import os
import sys, getopt
from params import *
import json
from pdb import set_trace as T
import Utilities
from models.alg_setter import alg_setter
from env.env_setter import env_setter
from copy import deepcopy
import argparse
from env.GPSemantic import REWARD
from env.GPSemantic import GPSemanticGym
import seaborn


class TrajectoryVisualisation:
    def __init__(self, args_dict: dict, map_index:int,map_ID:int, model_path: str):
        self.args_dict = args_dict
        self.args_dict['GPU'] = False  # CPU testing
        self.gifs = args_dict['TEST_GIFS']

        self.args_dict = args_dict
        import env_params.Semantic as parameters
        env_params_dict = set_dict(parameters)
        env_params_dict['home_dir'] = "./"
        self.env_params_dict = env_params_dict

        self.gifs = self.args_dict['TEST_GIFS']

        self.gifs_path = self.args_dict['TEST_GIFS_PATH'].format(model_path)
        if not os.path.exists(self.gifs_path):
            os.makedirs(self.gifs_path)

        if self.gifs:
            self.args_dict['RENDER_TRAINING'] = True
            self.args_dict['RENDER_TRAINING_WINDOW'] = 1

        self.env = GPSemanticGym(env_params_dict, self.args_dict)
        self.model = alg_setter.set_model(self.env, self.args_dict)

        self.gifs_path = self.args_dict['TEST_GIFS_PATH'].format(model_path)

        print('Loading Model')
        model_path = 'data/models/' + model_path
        checkpoint = torch.load(model_path + "/checkpoint.pkl", map_location=self.args_dict['DEVICE'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_episode = checkpoint['epoch']
        print('Model results at Episode: {}'.format(curr_episode))


        self.env.reset(episode_num=0, test_map=map_ID, test_indices=map_index)

    def plan_action(self, observation, hidden_in=None):

        # print(observation)
        hidden_in = hidden_in
        if self.args_dict['LSTM']:
            policy, value, hidden_out = self.model.forward_step(observation, hidden_in)
            hidden_in = hidden_out.cpu().detach().numpy()
        else:
            policy, value = self.model.forward_step(observation)
        policy = policy.cpu().detach().numpy()
        action_dict = Utilities.best_actions(policy)

        return action_dict, hidden_in

    # Choosing which test to run

    def agent_obs_visualisation(self, frac):

        if (frac <= 0.72 and frac >= 0.70):
            observation = self.env.get_obs_all()
            observation_agent = observation['obs'][0]
            fig, axs = plt.subplots(observation_agent.shape[-1], 1)

            for j in range(observation_agent.shape[-1]):
                img = np.transpose(observation_agent[:, :, j])
                cax_00 = axs[j].imshow(img, cmap='YlGn', origin='lower')
                axs[j].xaxis.set_major_formatter(plt.NullFormatter())  # kill xlabels
                axs[j].yaxis.set_major_formatter(plt.NullFormatter())  # kill ylabels

        plt.show()
        return

    def plot_trajectory(self,frame,primitives,agent_poses):
        fig,axs = plt.subplots(1,3)

        # we know each frame has 3 images
        semantic_image_frame, entropy_image_frame, gt_image_frame = np.split(frame, 3, axis=1)

        axs[0].imshow(semantic_image_frame,
                   interpolation=None,
                   extent=[0, self.env.belief_semantic_map.semantic_map.shape[0],
                           0, self.env.belief_semantic_map.semantic_map.shape[1]],
                   origin="upper")
        axs[0].set_title('Detected Semantic Map')
        axs[0].set_xlabel('x (m)')
        axs[0].set_ylabel('y (m)')
        axs[0].plot(agent_poses[0][1]*self.env.resolution,
                    (self.env.world_map_size - agent_poses[0][0])*self.env.resolution, 'bo', markersize=4)
        axs[0].plot(agent_poses[-1][1]*self.env.resolution,
                    (self.env.world_map_size - agent_poses[-1][0])*self.env.resolution, 'ro', markersize=4)



        # for pose,mp in zip(agent_poses,primitives):
        #     if mp is not None:
        #         mp.translate_start_position(pose)
        #         mp.plot(position_only=True, step_size=.01,color='black')
        #         plt.plot(pose[0], pose[1], 'y.')

        axs[1].imshow(entropy_image_frame,
                   interpolation=None,
                      extent=[0, self.env.belief_semantic_map.semantic_map.shape[0],
                              0, self.env.belief_semantic_map.semantic_map.shape[1]],
                   origin="upper")
        axs[1].set_xlabel('x (m)')
        axs[1].set_ylabel('y (m)')
        axs[1].set_title('Prior on the semantic map')

        axs[1].plot(agent_poses[0][1]*self.env.resolution,
                    (self.env.world_map_size - agent_poses[0][0])*self.env.resolution, 'bo', markersize=4)
        axs[1].plot(agent_poses[-1][1]*self.env.resolution,
                    (self.env.world_map_size - agent_poses[-1][0])*self.env.resolution, 'ro', markersize=4)

        axs[2].imshow(gt_image_frame,
                   interpolation=None,
                      extent=[0, self.env.belief_semantic_map.semantic_map.shape[0],
                              0, self.env.belief_semantic_map.semantic_map.shape[1]],
                   origin="upper")
        axs[2].set_xlabel('x (m)')
        axs[2].set_ylabel('y (m)')
        axs[2].set_title('Ground truth image map')
        # plt.imshow(np.transpose(self.env.worldTargetMap), cmap='OrRd',
        #            interpolation=None,
        #            extent=[0, self.env.worldMap.shape[0],
        #                    0, self.env.worldMap.shape[1]],
        #            origin="lower",
        #            alpha=0.4)

        axs[2].plot(agent_poses[0][1]*self.env.resolution,
                    (self.env.world_map_size - agent_poses[0][0])*self.env.resolution, 'bo', markersize=4)
        axs[2].plot(agent_poses[-1][1]*self.env.resolution,
                    (self.env.world_map_size - agent_poses[-1][0])*self.env.resolution, 'ro', markersize=4)

        for pose, mp in zip(agent_poses, primitives):
            if mp is not None:
                mp.translate_start_position([pose[0], pose[1]])
                state_sampling = mp.get_sampled_states(step_size=0.01)
                xs = state_sampling[2,:]
                ys = self.env.world_map_size - state_sampling[1,:]
                for i in range(3):
                    axs[i].plot(xs * self.env.belief_semantic_map.resolution,
                                ys* self.env.belief_semantic_map.resolution, '.',
                                color='black',alpha=0.5,markersize=1)

                    axs[i].plot(pose[1]*self.env.belief_semantic_map.resolution,
                                (self.env.world_map_size - pose[0])*self.env.belief_semantic_map.resolution, 'y.')

        #plt.show()


    def trajectory(self):
        observation = self.env.get_obs_all()
        episode_step = 0

        agent_poses = []
        primitives = []
        frames = []
        hidden_in = None

        while ((not self.args_dict['FIXED_BUDGET'] and
                episode_step < self.env.episode_length) \
               or (self.args_dict['FIXED_BUDGET'])):

            # Flip the poses to visulaize the trajectory
            agent_poses.append(self.env.agents[0].pos_actual)
            if self.args_dict['LSTM']:
                action_dict, hidden_in = self.plan_action(observation, hidden_in)
            else:
                action_dict, _ = self.plan_action(observation, hidden_in)
            rewards, done = self.env.step_all(action_dict)


            frame = self.env.render(mode='rgb_array')
            frames += frame

            if self.env.agents[0].current_primitive is not None:
                primitives.append(self.env.agents[0].current_primitive)


            # episode_rewards += np.array(rewards).sum()
            observation = self.env.get_obs_all()

            episode_step += 1
            if done:
                break

            frac = self.env.agents[0].agent_budget / (REWARD.MP.value * self.args_dict['BUDGET'])
            # agent channel visualisations
            #self.agent_obs_visualisation(frac)

            if (frac <= 0.32 and frac >= 0.30) or (frac <= 0.72 and frac >= 0.70) or\
                (frac <= 0.02 and frac >= 0.00) or  (frac <= 1.00 and frac >= 0.98) :
                self.plot_trajectory(frame[0], primitives, agent_poses)

        # pass
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--index", type=int, default=0,
                        help="seed of the experiment")
    parser.add_argument('--path', type=str, default="SemanticAC_ModelLSTM_Range6_v3",
                        help="path to the model")
    parser.add_argument('--gifs', type=bool, default=True,
                        help="outputting gifs")
    parser.add_argument('--id', type=int, default=0)

    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
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
        map_index=input_args.index,
        model_path=input_args.path,
        map_ID=input_args.id
    )
    trajectory_vis.trajectory()

