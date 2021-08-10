#!/usr/bin/env python3

# Code to test run from the learnt parameters
# The output of this code is the path for the robot to cover high rewarding regions
# Author : Sandeep Manjanna (McGill University)

from active_sampling import LearnPolicyGradientParams, LearnPolicyGradientParamsMP, Trajectory
import numpy as np
import pickle
from matplotlib import pyplot as plt
import os

# pickle file with the learnt parameters (Theta):
script_dir = os.path.dirname(__file__)
pg = pickle.load(open(f'{script_dir}/testingData/lpgp.pkl', "rb"), encoding='latin1')
if type(pg) is LearnPolicyGradientParamsMP:
    pg.load_graph()

pg.rewardmap = pickle.load(open(f'{script_dir}/testingData/gaussian_mixture_test2.pkl', "rb"), encoding='latin1')

Tau = pg.generate_trajectories(1, maxPolicy=True, rand_start=True)
tot = 0
dis_tot = 0

px = []
py = []

for j in range(pg.Tau_horizon):
    px.append(Tau[0][j].pos[0]-(pg.reward_map_size-1))
    py.append(Tau[0][j].pos[1]-(pg.reward_map_size-1))
    tot += Tau[0][j].reward
    dis_tot += (pg.gamma**j)*Tau[0][j].reward

for i in range(len(Tau[0])):
    print(Tau[0][i])
    # traj = Tau[0][i]
    # print(pg.minimum_action_mp_graph[traj.index,traj.action].start_state)
print(f"Maximum reward possible = {pg.maximum_reward}")
print(f"Trajectory Reward= {tot}")
print(f"Discounted Trajectory Reward= {dis_tot}")

plt.figure(figsize=(7, 6))
plt.imshow(pg.rewardmap, cmap='viridis')
plt.plot(px[0], py[0], 'go', markersize=12)
plt.plot(px[-1], py[-1], 'ro', markersize=12)
plt.colorbar()
if isinstance(pg,LearnPolicyGradientParamsMP):
    for pt in Tau[0]:
        mp = pg.minimum_action_mp_graph[pt.index, pt.action]
        mp.translate_start_position(pt.exact_pos - [pg.reward_map_size-1]*pg.spatial_dim )
        mp.plot(position_only=True, step_size = .01)
        plt.plot(pt.exact_pos[0]-(pg.reward_map_size-1),pt.exact_pos[1]-(pg.reward_map_size-1), 'w.')
        # print(mp.start_state[pg.spatial_dim:pg.spatial_dim*2])
else:
    plt.plot(px, py, 'w.')
    plt.plot(px, py, 'k-', linewidth=3)
if not pg.traj_reward_list == []:
    pg.makeFig(pg.num_iterations)
plt.show()
