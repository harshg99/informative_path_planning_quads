#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import time
import os
from copy import deepcopy


class LearnPolicyGradientParams:
    def __init__(self):
        self.reward_map_size = 30
        self.pad_size = self.reward_map_size-1  # TODO clean up these
        self.world_map_size = self.reward_map_size + 2*(self.pad_size)
        self.curr_r_map_size = self.reward_map_size + self.pad_size
        self.curr_r_pad = (self.curr_r_map_size-1)/2
        self.num_actions = 4  # Actions: 0 - LEFT, 1 - UP, 2 - RIGHT, and 3 - DOWN
        self.num_features = 24
        self.num_other_states = 1

        self.num_iterations = 10
        self.num_trajectories = 5
        self.Tau_horizon = 400
        self.plot = False
        self.fileNm = "lpgp"
        if(len(sys.argv) > 2):
            self.fileNm = sys.argv[2]

        # Discount factor gamma
        self.gamma = 0.5
        # Learning rate
        self.Eta = 0.015
        if(len(sys.argv) > 1):
            self.Eta = float(sys.argv[1])
    
    def get_phi_prime(self, worldmap, pos):
        def phi_from_map_coords(r, c):
            map_section = worldmap[r[0]:r[1], c[0]:c[1]]
            size = (r[1]-r[0])*(c[1]-c[0])
            return np.sum(map_section)/size

        r = pos[1]
        c = pos[0]
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

    def get_phi(self, worldmap, pos, action, index):
        phi_prime = self.get_phi_prime(worldmap, pos)
        phi = np.zeros((self.num_features, self.num_other_states, self.num_actions))
        phi[:, index, action] = phi_prime
        return phi

    def compute_softmax(self, worldmap, pos, index):
        phi_prime = self.get_phi_prime(worldmap, pos)
        theta_dot_phi = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            theta_dot_phi[i] = self.theta[:,index, i] @ phi_prime
        theta_dot_phi -= np.max(theta_dot_phi)
        exp_theta_dot_phi = np.exp(theta_dot_phi)
        prob = exp_theta_dot_phi/np.sum(exp_theta_dot_phi)
        return prob

    def get_pi(self, worldmap, pos, act, index):
        prob = self.compute_softmax(worldmap, pos, index)
        pi = prob[act]
        return pi

    def sample_action(self, worldmap, pos, index, maxPolicy=False):
        prob = self.compute_softmax(worldmap, pos, index)
        prob[self.num_actions_per_state[index]:] =0
        prob = prob/sum(prob)
        if maxPolicy:
            next_action = np.argmax(prob)
        else:
            next_action = np.random.choice(self.num_actions, size=1, p=prob)[0]
        return next_action

    def isValidPos(self, pos):
        is_valid = (np.array(pos-self.curr_r_pad) > -1).all()
        is_valid = is_valid and (np.array(pos + self.curr_r_pad) < self.orig_worldmap.shape).all()
        return is_valid

    def get_next_state(self, pos, action, index):
        """
        Given the current state and action, return the next state
        Ensures that next_pos is still in the reward map area
        """
        actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        next_pos = pos + actions[action]
        is_action_valid = self.isValidPos(next_pos)
        if is_action_valid:
            return next_pos, 0, is_action_valid, next_pos.reshape(2,1), 0
        else:
            return pos, 0, is_action_valid, None, None

    def generate_trajectories(self, num_trajectories, maxPolicy=False, rand_start=True):
        # Array of trajectories starting from current position.
        # Generate multiple trajectories (<action, state> pairs) using the current Theta.
        Tau = np.ndarray(shape=(num_trajectories, self.Tau_horizon), dtype=object)
        for i in range(num_trajectories):
            if rand_start:
                pos = np.random.choice(range(self.reward_map_size), 2) + \
                    np.array([self.curr_r_pad, self.curr_r_pad]).astype(np.int32)
            else:
                pos = np.array([self.reward_map_size, self.reward_map_size])
            index = 0
            local_worldmap = np.copy(self.orig_worldmap)
            for j in range(self.Tau_horizon):
                worldmap_pos = np.rint(pos).astype(np.int32)
                action = self.sample_action(local_worldmap, worldmap_pos, index, maxPolicy)
                next_pos, next_index, is_action_valid, visited_states, traj_cost = self.get_next_state(pos, action, index)
                # worldmap_next_pos =  np.rint(next_pos).astype(np.int32)
                curr_reward = 0
                if is_action_valid:
                    for state in visited_states.T:
                        curr_reward += local_worldmap[state[1], state[0]]
                        curr_reward -= traj_cost
                        local_worldmap[state[1], state[0]] = 0
                else:
                    curr_reward = -20
                Tau[i][j] = Trajectory(worldmap_pos, pos, action, curr_reward, index)
                pos = next_pos
                index = next_index
        return Tau

    def get_derivative(self, Tau, worldmap):
        pos = Tau.pos
        act = Tau.action
        index = Tau.index
        phi = self.get_phi(worldmap, pos, act, index)
        sum_b = 0
        for b in range(self.num_actions):
            sum_b = sum_b + (self.get_pi(worldmap, pos, b, index) * self.get_phi(worldmap, pos, b, index))
        delta = phi - sum_b
        return delta

    def get_maximum_path_reward(self):
        max_path_reward = 0
        discount_reward = 0
        Tau = self.generate_trajectories(1, maxPolicy=True, rand_start=False)
        for j in range(self.Tau_horizon):
            max_path_reward = max_path_reward + Tau[0][j].reward
            discount_reward = discount_reward + (self.gamma**j)*Tau[0][j].reward
        print(f'The Maximum reward with trajectory = {max_path_reward}')
        print(f'The discounted reward = {discount_reward}')
        self.path_max_reward_list.append(max_path_reward)
        self.discount_reward_list.append(discount_reward)

    def set_up_training(self):
        self.theta = np.random.rand(self.num_features, self.num_other_states, self.num_actions)*0.1

    def run_training(self, rewardmap):
        self.set_up_training()
        self.maximum_reward = sum(-np.sort(-np.reshape(rewardmap, (1, self.reward_map_size**2))[0])[0:self.Tau_horizon])
        worldmap = np.zeros((self.world_map_size, self.world_map_size))
        worldmap[self.pad_size:self.pad_size+self.reward_map_size, self.pad_size:self.pad_size+self.reward_map_size] = rewardmap
        self.orig_worldmap = np.copy(worldmap)

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
            Tau = self.generate_trajectories(self.num_trajectories, rand_start=True)
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
                    tot_reward += Tau[i][j].reward
                    traj_reward += Tau[i][j].reward
                    # Discounted future rewards
                    for t in range(j, self.Tau_horizon):
                        R_t = R_t + self.gamma**(t-j)*Tau[i][t].reward
                    sum_R_t[j] = sum_R_t[j] + R_t
                    A_t = R_t - (sum_R_t[j]/(i+1))
                    worldmap[Tau[i][j].pos[1], Tau[i][j].pos[0]] = 0
                    g_t = self.get_derivative(Tau[i][j], worldmap) * A_t
                    g_Tau = g_Tau + g_t
                g_T = g_T + g_Tau
            g_T = g_T / self.num_trajectories
            g_T = g_T / (self.num_features*self.num_other_states)
            tot_reward = tot_reward / self.num_trajectories
            self.theta = self.theta + self.Eta*g_T

            print(f"Iteration {iterations+1}/{self.num_iterations}")
            print(f"total accumulated reward = {tot_reward:.2f} / {self.maximum_reward:.2f}")

            if self.plot == True:
                self.traj_reward_list.append(tot_reward)
                self.get_maximum_path_reward()
                self.makeFig(iterations)

            end = time.time()
            self.tot_time += (end-start)
            print(f'Computation Time: {(end-start):.2f}')
            
            x = deepcopy(self)
            script_dir = os.path.dirname(__file__)
            x.mp_graph = None
            x.minimum_action_mp_graph = None
            pickle.dump(x, open(f'{script_dir}/testingData/{self.fileNm}.pkl', "wb"))

        # print theta
        pos = np.array([self.reward_map_size, self.reward_map_size])
        Tau = self.generate_trajectories(1, maxPolicy=True, rand_start=False)
        for j in range(self.Tau_horizon):
            print(Tau[0][j])

        # Saving the trained data
        self.Tau = Tau
        self.mp_graph = None
        script_dir = os.path.dirname(__file__)
        self.mp_graph = None
        self.minimum_action_mp_graph = None
        pickle.dump(self, open(f'{script_dir}/testingData/{self.fileNm}.pkl', "wb"))

    def makeFig(self, iterations):
        plt.plot(np.arange(iterations+1), self.traj_reward_list)
        plt.plot(np.arange(iterations+1), [self.maximum_reward]*(iterations+1))
        plt.plot(np.arange(iterations+1), self.path_max_reward_list)
        plt.draw()
        plt.pause(0.00001)


class Trajectory:
    def __init__(self, pos, exact_pos, action, reward, index):
        self.pos = pos
        self.exact_pos = exact_pos
        self.action = action
        self.reward = reward
        self.index = index

    def __str__(self):
        return f"{self.pos} {self.exact_pos} {self.index} {self.action} {self.reward}"


if __name__ == '__main__':

    lpgp = LearnPolicyGradientParams()

    script_dir = os.path.dirname(__file__)
    rewardmap = pickle.load(open(f'{script_dir}/trainingData/gaussian_mixture_training_data.pkl', "rb"), encoding='latin1')
    lpgp.run_training(rewardmap)
