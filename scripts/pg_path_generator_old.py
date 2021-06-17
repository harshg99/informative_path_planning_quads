# Code to test run from the learnt parameters
# The output of this code is the path for the robot to cover high rewarding regions
# Author : Sandeep Manjanna (McGill University)

from learn_policyGradient_params import LearnPolicyGradientParams, Trajectory
import numpy as np
import pickle
from matplotlib import pyplot as plt

# pickle file with the learnt parameters (Theta):
data = pickle.load(open('../testingData/gaussian_model_100iter_trial1.pkl', "rb"), encoding='latin1')

# pickle files with test datasets:
#rewardmap = pickle.load(open('../testingData/gaussian_mixture_test2.pkl',"r"))
rewardmap = pickle.load(open('../testingData/cShaped_test3.pkl', "rb"), encoding='latin1')
#rewardmap = pickle.load(open('../testingData/gaussian_mixture_test1.pkl',"r"))

pg = LearnPolicyGradientParams()
pg.reward_map_size = rewardmap.shape[0]
pg.pad_size = pg.reward_map_size-1
pg.world_map_size = pg.reward_map_size + 2*(pg.pad_size)
pg.curr_r_map_size = pg.reward_map_size + pg.pad_size
pg.curr_r_pad = (pg.curr_r_map_size-1)/2
pg.num_features = int(data[6].shape[0]/4)
pg.Tau_horizon = 250
pg.gamma = data[1]

print(f"the rewardmap size = {pg.reward_map_size}")

rewardmap = rewardmap/float(np.sum(rewardmap))

maximum_reward = sum(-np.sort(-np.reshape(rewardmap, (1, pg.reward_map_size**2))[0])[0:pg.Tau_horizon])

worldmap = np.zeros((pg.world_map_size, pg.world_map_size))-0.0001
worldmap[pg.pad_size:pg.pad_size+pg.reward_map_size, pg.pad_size:pg.pad_size+pg.reward_map_size] = rewardmap

curr_pos = np.array([pg.reward_map_size, pg.reward_map_size])
pg.num_trajectories = 1
theta = data[6]
if theta.shape[1] == 1:
    theta = data[6].reshape(pg.num_actions, pg.num_features).T
Tau = pg.generate_trajectories(worldmap, curr_pos, theta, maxPolicy=True, rand_start=False)
tot = 0
dis_tot = 0

px = []
py = []

for i in range(pg.num_trajectories):
    for j in range(pg.Tau_horizon):
        px.append(Tau[i][j].curr_pos[0]-(pg.reward_map_size-1))
        py.append(Tau[i][j].curr_pos[1]-(pg.reward_map_size-1))
        tot += Tau[i][j].curr_reward
        dis_tot += (pg.gamma**j)*Tau[i][j].curr_reward

print(f"Maximum reward possible = {maximum_reward}")
print(tot)
print(dis_tot)

plt.figure(figsize=(7, 6))
plt.imshow(rewardmap, cmap='viridis')
plt.plot(px, py, 'k-', linewidth=3)
plt.plot(px[0], py[0], 'ro', markersize=12)
plt.plot(px[-1], py[-1], 'go', markersize=12)
plt.colorbar()
plt.show()
