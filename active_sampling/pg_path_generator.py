#!/usr/bin/env python3

# Code to test run from the learnt parameters
# The output of this code is the path for the robot to cover high rewarding regions
# Author : Sandeep Manjanna (McGill University)

import active_sampling
from active_sampling import LearnPolicyGradientParams, LearnPolicyGradientParamsMP, Trajectory
import numpy as np
import pickle
from matplotlib import pyplot as plt
import os

# pickle file with the learnt parameters (Theta):
script_dir = os.path.dirname(os.path.abspath(__file__))
pg = pickle.load(open(f'{script_dir}/testingData/lpgp10.pkl', "rb"), encoding='latin1')
# pg.mp_graph_file_name = f'{os.path.dirname(active_sampling.__file__)}/latticeData/10.json'

# pg = pickle.load(open(f'{script_dir}/testingData/lpgp15.pkl', "rb"), encoding='latin1')
if type(pg) is LearnPolicyGradientParamsMP:
    pg.load_graph()

# pg.rewardmap = pickle.load(open(f'{script_dir}/testingData/gaussian_mixture_test2.pkl', "rb"), encoding='latin1') * 1000
pg.rewardmap = (pickle.load(open(f'{script_dir}/testingData/cShaped_test3.pkl', "rb"), encoding='latin1')) * 1000
# pg.rewardmap = np.load('airport.npy')*1000
pg.reward_map_size = pg.rewardmap.shape[0]
pg.pad_size = pg.reward_map_size-1  # TODO clean up these
pg.world_map_size = pg.reward_map_size + 2*(pg.pad_size)
pg.orig_worldmap = np.zeros((pg.world_map_size, pg.world_map_size))
pg.orig_worldmap[pg.pad_size:pg.pad_size+pg.reward_map_size, pg.pad_size:pg.pad_size+pg.reward_map_size] = pg.rewardmap
pg.curr_r_map_size = pg.reward_map_size + pg.pad_size
pg.curr_r_pad = (pg.curr_r_map_size-1)/2
pg.xy_resolution = 1

pg.Tau_horizon = 50
Tau = pg.generate_trajectories(1, maxPolicy=True, rand_start=True)
tot = 0
dis_tot = 0

px = []
py = []

for j in range(pg.Tau_horizon):
    print(Tau[0][j])

    px.append(Tau[0][j].exact_pos[0])
    py.append(Tau[0][j].exact_pos[1])
    tot += Tau[0][j].reward
    dis_tot += (pg.gamma**j)*Tau[0][j].reward

print(f"Maximum reward possible = {pg.maximum_reward}")
print(f"Trajectory Reward= {tot}")
print(f"Discounted Trajectory Reward= {dis_tot}")

plt.figure(figsize=(7, 6))
plt.imshow(pg.rewardmap, cmap='viridis', interpolation='spline36', extent=[
           0, pg.xy_resolution*pg.reward_map_size, 0, pg.xy_resolution*pg.reward_map_size])
plt.colorbar()

plt.plot(px[0], py[0], 'go', markersize=12)
plt.plot(px[-1], py[-1], 'ro', markersize=12)
if isinstance(pg, LearnPolicyGradientParamsMP):
    for pt in Tau[0]:
        mp = pg.minimum_action_mp_graph[pt.index, pt.action]
        if mp is not None:
            mp.translate_start_position(pt.exact_pos)
            mp.plot(position_only=True, step_size = .01)
            plt.plot(pt.exact_pos[0],pt.exact_pos[1], 'y.')
else:
    plt.plot(px, py, 'w.')
    plt.plot(px, py, 'k-', linewidth=3)
if not pg.traj_reward_list == []:
    pg.makeFig(pg.num_iterations)
plt.show()
