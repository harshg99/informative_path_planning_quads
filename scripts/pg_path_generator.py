#!/usr/bin/env python3

# Code to test run from the learnt parameters
# The output of this code is the path for the robot to cover high rewarding regions
# Author : Sandeep Manjanna (McGill University)

from learn_policyGradient_params import LearnPolicyGradientParams, Trajectory
import numpy as np
import pickle
from matplotlib import pyplot as plt

# pickle file with the learnt parameters (Theta):
pg = pickle.load(open('totreward_5x5_200_base.pkl', "rb"), encoding='latin1')

print(f"the rewardmap size = {pg.reward_map_size}")

curr_pos = np.array([pg.reward_map_size, pg.reward_map_size])
Tau = pg.generate_trajectories(1, curr_pos, pg.theta, maxPolicy=True, rand_start=False)
tot = 0
dis_tot = 0

px = []
py = []

for j in range(pg.Tau_horizon):
    px.append(Tau[0][j].curr_pos[0]-(pg.reward_map_size-1))
    py.append(Tau[0][j].curr_pos[1]-(pg.reward_map_size-1))
    tot += Tau[0][j].curr_reward
    dis_tot += (pg.gamma**j)*Tau[0][j].curr_reward

print(f"Maximum reward possible = {pg.maximum_reward}")
print(f"Trajectory Reward= {tot}")
print(f"Discounted Trajectory Reward= {dis_tot}")

plt.figure(figsize=(7, 6))
plt.imshow(pg.rewardmap, cmap='viridis')
plt.plot(px, py, 'k-', linewidth=3)
plt.plot(px[0], py[0], 'ro', markersize=12)
plt.plot(px[-1], py[-1], 'go', markersize=12)
plt.colorbar()
plt.show()
