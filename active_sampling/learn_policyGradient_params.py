#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import time
import os

class LearnPolicyGradientParams:
    def __init__(self):
        self.reward_map_size = 30
        self.pad_size = self.reward_map_size-1  # TODO clean up these
        self.world_map_size = self.reward_map_size + 2*(self.pad_size)
        self.curr_r_map_size = self.reward_map_size + self.pad_size
        self.curr_r_pad = (self.curr_r_map_size-1)/2
        self.num_actions = 4  # Actions: 0 - LEFT, 1 - UP, 2 - RIGHT, and 3 - DOWN
        self.num_features = 24

        self.num_iterations = 50
        self.num_trajectories = 20
        self.Tau_horizon = 400
        self.rand_start = True
        self.rand_start_pos = np.random.choice(range(self.reward_map_size), 2) + np.array([self.curr_r_pad,self.curr_r_pad]).astype(np.int32)
        self.plot = False
        self.fileNm = "lpgp"
        if(len(sys.argv) > 1):
            self.fileNm = sys.argv[1]

        # Discount factor gamma
        self.gamma = 0.5
        # Learning rate
        self.Eta = 0.015
        if(len(sys.argv) > 2):
            self.Eta = float(sys.argv[2])
        
    def get_phi_prime(self, worldmap, curr_pos):
        def phi_from_map_coords(r, c):
            map_section = worldmap[r[0]:r[1], c[0]:c[1]]
            size = np.size(map_section)
            if size == 0:  # TODO ask about this
                size = 1
            return np.sum(map_section)/size

        r = curr_pos[1]
        c = curr_pos[0]
        phi_prime = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not (i == 0 and j == 0):
                    phi_prime.append(worldmap[r+i, c+j])
                    phi_prime.append(phi_from_map_coords((r-1+3*i, r-1+3*(i+1)), (c-1+3*j, c-1+3*(j+1))))

        for i in ((0, r-4), (r-4, r+5), (r+5, self.world_map_size)):
            for j in ((0, c-4), (c-4, c+5), (c+5, self.world_map_size)):
                if not (i == (r-4, r+5) and j == (c-4, c+5)):
                    phi_prime.append(phi_from_map_coords(i, j))

        phi_prime = np.squeeze(np.array([phi_prime]))
        return phi_prime

    def get_phi(self, worldmap, curr_pos, curr_action):
        phi_prime = self.get_phi_prime(worldmap, curr_pos)
        phi = np.zeros((self.num_features, self.num_actions))
        phi[:, curr_action] = phi_prime
        return phi

    def compute_softmax(self, worldmap, pos, theta):
        phi_prime = self.get_phi_prime(worldmap, pos)
        theta_dot_phi = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            theta_dot_phi[i] = theta[:, i] @ phi_prime
        theta_dot_phi -= np.max(theta_dot_phi)
        exp_theta_dot_phi = np.exp(theta_dot_phi)
        prob = exp_theta_dot_phi/np.sum(exp_theta_dot_phi)
        return prob

    def get_pi(self, worldmap, pos, act, theta):
        prob = self.compute_softmax(worldmap, pos, theta)
        pi = prob[act]
        return pi

    def sample_action(self, worldmap, pos, theta, maxPolicy=False):
        prob = self.compute_softmax(worldmap, pos, theta)

        if maxPolicy:
            next_action = np.argmax(prob)
        else:
            next_action = np.random.choice(self.num_actions, size=1, p=prob)[0]
        return next_action

    def get_next_state(self, worldmap, curr_pos, curr_action):
        """
        Given the current state and action, return the next state
        Ensures that next_pos is still in the reward map area
        """
        def isValidPos(pos,action):
            is_valid = (np.array(pos-self.curr_r_pad) > -1).all()
            is_valid = is_valid and (np.array(pos + self.curr_r_pad) < worldmap.shape).all()
            return is_valid
        actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        next_pos = curr_pos + actions[curr_action]
        is_action_valid = isValidPos(next_pos, curr_action)
        if is_action_valid:
            return next_pos, is_action_valid
        else:
            return curr_pos, is_action_valid

    def generate_trajectories(self, num_trajectories, curr_pos, theta, maxPolicy=False, rand_start=True):
        # Array of trajectories starting from current position.
        # Generate multiple trajectories (<action, state> pairs) using the current Theta.
        Tau = np.ndarray(shape=(num_trajectories, self.Tau_horizon), dtype=object)
        for i in range(num_trajectories): #TODO multiprocessing
            if rand_start:
                curr_pos = self.rand_start_pos
            local_worldmap = np.copy(self.orig_worldmap)
            for j in range(self.Tau_horizon):
                curr_action = self.sample_action(local_worldmap, curr_pos, theta, maxPolicy)
                next_pos, is_action_valid = self.get_next_state(local_worldmap, curr_pos, curr_action)
                if is_action_valid:
                    curr_reward = local_worldmap[next_pos[1], next_pos[0]]
                    local_worldmap[curr_pos[1], curr_pos[0]] = 0
                else:
                    curr_reward = -2
                Tau[i][j] = Trajectory(curr_pos, curr_action, curr_reward)
                curr_pos = next_pos
        return Tau

    def get_derivative(self, Tau, worldmap, theta):
        pos = Tau.curr_pos
        act = Tau.curr_act
        phi = self.get_phi(worldmap, pos, act)
        sum_b = 0
        for b in range(self.num_actions):
            sum_b = sum_b + (self.get_pi(worldmap, pos, b, theta) * self.get_phi(worldmap, pos, b))
        delta = phi - sum_b
        return delta

    def get_maximum_path_reward(self, curr_pos, theta):
        max_path_reward = 0
        discount_reward = 0
        Tau = self.generate_trajectories(1, curr_pos, theta, maxPolicy=True, rand_start=False)
        for j in range(self.Tau_horizon):
            max_path_reward = max_path_reward + Tau[0][j].curr_reward
            discount_reward = discount_reward + (self.gamma**j)*Tau[0][j].curr_reward
        max_path_reward = max_path_reward 
        discount_reward = discount_reward
        print(f'The Maximum reward with trajectory = {max_path_reward}')
        print(f'The discounted reward = {discount_reward}')
        self.path_max_reward_list.append(max_path_reward)
        self.discount_reward_list.append(discount_reward)

    def run_training(self, rewardmap):
        self.maximum_reward = sum(-np.sort(-np.reshape(rewardmap, (1, self.reward_map_size**2))[0])[0:self.Tau_horizon])
        worldmap = np.zeros((self.world_map_size, self.world_map_size))
        worldmap[self.pad_size:self.pad_size+self.reward_map_size, self.pad_size:self.pad_size+self.reward_map_size] = rewardmap
        self.orig_worldmap = np.copy(worldmap)
        curr_pos = np.array([self.reward_map_size, self.reward_map_size])

        theta = np.random.rand(self.num_features, self.num_actions)*0.1
        plt.ion()
        self.traj_reward_list = list()
        self.path_max_reward_list = list()
        self.discount_reward_list = list()
        self.tot_time = 0

        #*******************************************************************#
        # num_iterations --> the number of learning iterations
        # num_trajectories --> No. of trajectries used for policy estimation
        # Tau_horizon --> Finite horizon of each trajectory
        #*******************************************************************#

        for iterations in range(self.num_iterations):
            start = time.time()
            # Generate multiple trajectories (<action, state> pairs) using the current Theta.
            Tau = self.generate_trajectories(self.num_trajectories, curr_pos, theta, rand_start=self.rand_start)
            g_T = 0
            tot_reward = 0
            sum_R_t = np.zeros(self.Tau_horizon)
            for i in range(self.num_trajectories):
                g_Tau = 0
                traj_reward = 0
                worldmap = np.copy(self.orig_worldmap)
                for j in range(self.Tau_horizon):
                    # Rolling out each of the trajectories
                    R_t = 0
                    # Total reward in a trajectory
                    tot_reward += Tau[i][j].curr_reward
                    traj_reward += Tau[i][j].curr_reward
                    # Discounted future rewards
                    for t in range(j, self.Tau_horizon):
                        R_t = R_t + self.gamma**(t-j)*Tau[i][t].curr_reward
                    sum_R_t[j] = sum_R_t[j] + R_t
                    A_t = R_t - (sum_R_t[j]/(i+1))
                    worldmap[Tau[i][j].curr_pos[1], Tau[i][j].curr_pos[0]] = 0
                    g_t = self.get_derivative(Tau[i][j], worldmap, theta) * A_t
                    g_Tau = g_Tau + g_t
                g_T = g_T + g_Tau
            g_T = g_T / self.num_trajectories
            g_T = g_T / self.num_features
            tot_reward = tot_reward / self.num_trajectories
            theta = theta + self.Eta*g_T

            print(f"Iteration {iterations+1}/{self.num_iterations}")
            print(f"total accumulated reward = {tot_reward}")

            if self.plot == True:
                self.traj_reward_list.append(tot_reward)
                curr_pos1 = np.array([self.reward_map_size, self.reward_map_size])
                self.get_maximum_path_reward(curr_pos1, theta)
                self.makeFig(iterations)

            end = time.time()
            self.tot_time += (end-start)
            print(f'Computation Time: {end-start}')

        # print theta
        curr_pos = np.array([self.reward_map_size, self.reward_map_size])
        Tau = self.generate_trajectories(1,curr_pos, theta, maxPolicy=True, rand_start=False)
        for j in range(self.Tau_horizon):
            print(Tau[0][j])

        # Saving the trained data
        self.rewardmap = rewardmap
        self.theta = theta
        self.Tau = Tau
        self.mp_graph = None
        script_dir = os.path.dirname(__file__)
        pickle.dump(self, open(f'{script_dir}/testingData/{self.fileNm}.pkl', "wb"))

    def makeFig(self,iterations):
        plt.plot(np.arange(iterations+1), self.traj_reward_list)
        plt.plot(np.arange(iterations+1), [self.maximum_reward]*(iterations+1))
        plt.plot(np.arange(iterations+1), self.path_max_reward_list)
        plt.draw()
        plt.pause(0.00001)

class Trajectory:
    def __init__(self, curr_pos, curr_act, curr_reward):
        self.curr_pos = curr_pos
        self.curr_act = curr_act
        self.curr_reward = curr_reward
        
    def __str__(self):
        return f"{self.curr_pos} {self.curr_act} {self.curr_reward}"


if __name__ == '__main__':

    lpgp = LearnPolicyGradientParams()
    
    script_dir = os.path.dirname(__file__)
    rewardmap = pickle.load(open(f'{script_dir}/trainingData/gaussian_mixture_training_data.pkl', "rb"), encoding='latin1')
    lpgp.run_training(rewardmap)
