import numpy as np
import pickle
import os
from active_sampling import LearnPolicyGradientParamsMP
from matplotlib import pyplot as plt
from mav_traj_gen import *
from planning_ros_msgs.msg import SplineTrajectory, Spline, Polynomial
from copy import deepcopy
import rospy
import actionlib
from action_trackers.msg import RunTrajectoryAction, RunTrajectoryGoal
from nav_msgs.msg import Odometry
import Utilities
import torch
from models.alg_setter import alg_setter
from env.env_setter import env_setter
import argparse

class DRLROS:
    def __init__(self,odom_topic: str,
                 traj_pub_topic: str,
                 traj_client_topic: str,
                 args_dict: dict,
                 model_path: str,
                 map_path: str ):
        self.odom_topic = odom_topic
        self.traj_pub_topic = traj_pub_topic
        self.traj_client = actionlib.SimpleActionClient(traj_client_topic, RunTrajectoryAction)

        # listening for goals.
        rospy.loginfo("waiting for action server")
        self.traj_client.wait_for_server()
        rospy.loginfo("found action server")
        self.traj_client.cancel_all_goals()
        rospy.Subscriber(self.odom_topic,Odometry, self.odom_cb, queue_size=1)

        self.traj_opt_pub = rospy.Publisher("/quadrotor/spline_traj", SplineTrajectory, queue_size=10)

        self.args_dict = args_dict
        self.args_dict['GPU'] = False  # CPU testing
        self.gifs = args_dict['TEST_GIFS']

        self.env = env_setter.set_env(args_dict)
        self.model = alg_setter.set_model(self.env, args_dict)
        self.gifs_path = self.args_dict['TEST_GIFS_PATH'].format(model_path,
                                                                 map_size,
                                                                 args_dict['BUDGET']
                                                                 )

        print('Loading Model')
        model_path = 'data/models/' + model_path
        checkpoint = torch.load(model_path + "/checkpoint.pkl", map_location=self.args_dict['DEVICE'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_episode = checkpoint['epoch']
        print('Model results at Episode: {}'.format(curr_episode))

        #Plotting stuff
        self.hidden_in = None
        self.primitives = []
        self.agent_poses = []

        # Set the map
        file_name = map_path + "tests39env.npy"
        rewardmap = np.load(file_name)
        file_name = map_path + "tests39target.npy"
        targetmap = np.load(file_name)
        file_name = map_path + "tests39target_orig_dist.npy"
        orig_target_map = np.load(file_name)
        self.env.reset(rewardmap, targetmap, orig_target_map)



    def odom_cb(self, msg):
        self.odom = msg

    def run_action(self):

        UAV_pose = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y])
        self.env.agents[0].updatePosOdom(UAV_pose)
        observation = self.env.get_obs_all()

        if self.args_dict['LSTM']:
            action_dict, hidden_in = self.plan_action(observation, self.hidden_in)
        else:
            action_dict, _ = self.plan_action(observation, self.hidden_in)

        rewards, done = self.env.step_all(action_dict)
        episode_rewards = np.array(rewards).sum()

        goal = RunTrajectoryGoal()
        mp_list = self.get_mp(action_dict)
        spline_traj = self.mp_list_to_spline_traj(mp_list)
        goal.traj =  self.mp_list_to_spline_traj(mp_list)
        self.traj_opt_pub.publish(spline_traj)

        self.traj_client.send_goal(goal)

        result = self.traj_client.wait_for_result()
        return episode_rewards,result,done

    def run_simulation(self):
        observation = self.env.get_obs_all()
        episode_step = 0

        self.agent_poses = [self.env.agents[0].pos_actual]
        self.primitives = []
        frames = []
        while ((not self.args_dict['FIXED_BUDGET'] and
                episode_step < self.env.episode_length) \
               or (self.args_dict['FIXED_BUDGET'])):
            if self.gifs:
                frames.append(self.env.render(mode='rgb_array'))

            # Plans and runs the action from the DL model
            reward,result,done = self.run_action()

            if self.env.agents[0].current_primitive is not None:
                self.primitives.append(self.env.agents[0].current_primitive)

            self.agent_poses.append(self.env.agents[0].pos_actual)
            #episode_rewards += np.array(rewards).sum()
            observation = self.env.get_obs_all()

            episode_step += 1
            if done:
                break

    def plot_data(self):
        plt.figure(figsize=(7, 6))
        plt.imshow(np.transpose(self.env.orig_worldMap), cmap='viridis',
                   interpolation='spline36',
                   extent=[0, self.env.orig_worldMap.shape[0],
                           0, self.env.orig_worldMap.shape[1]],
                   origin="lower")
        plt.colorbar()

        plt.plot(self.agent_poses[0][0], self.agent_poses[0][1], 'go', markersize=12)
        plt.plot(self.agent_poses[-1][0], self.agent_poses[-1][1], 'ro', markersize=12)

        for pose, mp in zip(self.agent_poses, self.primitives):
            if mp is not None:
                mp.translate_start_position(pose)
                mp.plot(position_only=True, step_size=.01)
                plt.plot(pose[0], pose[1], 'y.')

        plt.figure(figsize=(7, 6))
        plt.imshow(np.transpose(self.env.worldMap), cmap='viridis',
                   interpolation='spline36',
                   extent=[0, self.env.worldMap.shape[0],
                           0, self.env.worldMap.shape[1]],
                   origin="lower")
        plt.colorbar()

        plt.plot(self.agent_poses[0][0], self.agent_poses[0][1], 'go', markersize=12)
        plt.plot(self.agent_poses[-1][0], self.agent_poses[-1][1], 'ro', markersize=12)

        for pose, mp in zip(self.agent_poses, self.primitives):
            if mp is not None:
                mp.translate_start_position(pose)
                mp.plot(position_only=True, step_size=.01)
                plt.plot(pose[0], pose[1], 'y.')

        plt.show()

    def get_mp(self,action_dict):
        action = action_dict['0']
        mp = self.env.agents[0].get_mp(action)
        # return default motion primitive is current primitve is None
        if mp is not None:
            return [mp]
        else:
            return [self.env.agents[0].get_mp(0)]

    def plan_action(self, observation,hidden_in=None):

        # print(observation)
        hidden_in = hidden_in
        if self.args_dict['LSTM']:
            policy, value,hidden_out = self.model.forward_step(observation,hidden_in)
            self.hidden_in = hidden_out.cpu().detach().numpy()
        else:
            policy, value = self.model.forward_step(observation)
        policy = policy.cpu().detach().numpy()
        action_dict = Utilities.best_actions(policy)

        return action_dict,hidden_in

    def mp_list_to_spline_traj(self, mp_list):
        spline_traj = SplineTrajectory()
        for dimension in range(2):
            spline = Spline()
            for mp in mp_list:
                poly = Polynomial()
                poly.degree = mp.poly_coeffs.shape[1] - 1
                poly.basis = 0
                poly.dt = mp.traj_time
                # need to reverse the order due to different conventions in spline polynomial and mp.polys
                poly.coeffs = np.flip(mp.poly_coeffs[dimension, :])
                # convert between Mike's parametrization and mine
                for degree in range(len(poly.coeffs)):
                    poly.coeffs[degree] *= poly.dt ** degree
                spline.segs.append(poly)
                spline.segments += 1
                spline.t_total += mp.traj_time
            spline_traj.data.append(spline)

        # add z-dim
        spline = Spline()
        for mp in mp_list:
            poly = Polynomial()
            poly.degree = mp.poly_coeffs.shape[1] - 1
            poly.basis = 0
            poly.dt = mp.traj_time
            # set poly with 0 coeffs for z direction
            poly.coeffs = np.zeros(mp.poly_coeffs.shape[1])
            # arbitrary z height 1
            poly.coeffs[0] = self.odom.pose.pose.position.z
            spline.segs.append(poly)
        spline_traj.data.append(spline)

        spline.segments = len(mp_list)
        spline.t_total = np.sum([mp.traj_time for mp in mp_list])
        spline_traj.header.frame_id = "quadrotor/map"
        spline_traj.header.stamp = rospy.Time.now()
        spline_traj.dimensions = 3

        return spline_traj

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--map_path", type=str, default="tests/GPPrim30/",
                        help="seed of the experiment")
    parser.add_argument('--model_path',type=str,default="GPAC_Model6_RANGE8_SENS5_ENV2_FINAL",
                        help="path to the model")
    parser.add_argument('--gifs', type=bool, default=True,
                        help="outputting gifs")

    args = parser.parse_args()
    # fmt: on
    return args

if __name__ == '__main__':
    rospy.init_node('lpgp_policy_publisher')
    import params as args
    args_dict = Utilities.set_dict(args)
    input_args = parse_args()

    obj = DRLROS(
        odom_topic="/quadrotor/odom",
        traj_pub_topic="/quadrotor/spline_traj",
        traj_client_topic='quadrotor/trackers_manager/execute_trajectory',
        model_path=input_args.model_path,
        map_path=input_args.map_path,
        args_dict=args_dict
    )


    obj.run_simulation()
    obj.plot_data()
