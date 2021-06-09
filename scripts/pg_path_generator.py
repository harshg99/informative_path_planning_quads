#Code to test run from the learnt parameters
#The output of this code is the path for the robot to cover high rewarding regions
# Author : Sandeep Manjanna (McGill University)

import sys
import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.interpolate import spline
import matplotlib
import matplotlib.collections as mcoll
import matplotlib.path as mpath

class Trajectory:
    def __init__(self,curr_pos,curr_act,curr_reward):
        self.curr_pos=list(curr_pos)
        self.curr_act=curr_act
        self.curr_reward=curr_reward
    def get_curr_pos(self):
        return self.curr_pos
    def get_curr_act(self):
        return self.curr_act
    def get_curr_reward(self):
        return self.curr_reward
    def __str__(self):
        return str(self.curr_pos)+" "+str(self.curr_act)+" "+str(self.curr_reward)
    
#sys.path.insert(0, '../scripts')
import pg_sampling_struct as pg

#pickle file with the learnt parameters (Theta):
data = pickle.load(open('../testingData/gaussian_model_100iter_trial1.pkl',"r"))

#pickle files with test datasets:
#rewardmap = pickle.load(open('../testingData/gaussian_mixture_test2.pkl',"r"))
rewardmap = pickle.load(open('../testingData/cShaped_test3.pkl',"r"))
#rewardmap = pickle.load(open('../testingData/gaussian_mixture_test1.pkl',"r"))

pg.reward_map_size = rewardmap.shape[0]
pg.pad_size = pg.reward_map_size-1
pg.world_map_size = pg.reward_map_size + 2*(pg.pad_size)
pg.curr_r_map_size = pg.reward_map_size + pg.pad_size
pg.curr_r_pad = (pg.curr_r_map_size-1)/2
pg.num_features = data[6].shape[0]/4
pg.Tau_horizon = 250
pg.gamma = data[1]

print "the rewardmap size = "+str(pg.reward_map_size)

total_reward = []
discounted_reward = []
horizon = []

for H in [1]:

    #pickle files with test datasets:
    #rewardmap = pickle.load(open('../testingData/gaussian_mixture_test2.pkl',"r"))
    rewardmap = pickle.load(open('../testingData/cShaped_test3.pkl',"r"))
    #rewardmap = pickle.load(open('../testingData/gaussian_mixture_test1.pkl',"r"))
    
    rewardmap = rewardmap/float(np.sum(rewardmap))
    
    maximum_reward = sum(-np.sort(-np.reshape(rewardmap,(1,pg.reward_map_size**2))[0])[0:pg.Tau_horizon])

    worldmap = np.zeros((pg.world_map_size,pg.world_map_size))-0.0001
    worldmap[pg.pad_size:pg.pad_size+pg.reward_map_size,pg.pad_size:pg.pad_size+pg.reward_map_size] = rewardmap
    
    curr_pos = np.array([pg.reward_map_size,pg.reward_map_size])
    num_traj = 1
    theta = data[6]
    Tau = pg.generate_trajectories(num_traj,worldmap,curr_pos,theta,isPrint=False,maxPolicy=True,rand_start=False)
    tot = 0
    dis_tot = 0

    px=[]
    py=[]

    for i in range(num_traj):
        for j in range(pg.Tau_horizon):
            px.append(Tau[i][j].curr_pos[0]-(pg.reward_map_size-1))
            py.append(Tau[i][j].curr_pos[1]-(pg.reward_map_size-1))
            tot += Tau[i][j].curr_reward
            dis_tot += (pg.gamma**j)*Tau[i][j].curr_reward

    total_reward.append(tot)
    discounted_reward.append(dis_tot)
    horizon.append(H)
    
x = np.array([np.linspace(0, pg.reward_map_size-1, pg.reward_map_size),
              np.linspace(0, pg.reward_map_size-1, pg.reward_map_size)]).T
x1Mesh, x2Mesh = np.meshgrid(x[:,0:1], x[:,1:2])
levels = np.linspace(np.min(rewardmap),np.max(rewardmap),100)

print "Maximum reward possible = "+str(maximum_reward)
print tot
print dis_tot


plt.figure(figsize=(7,6))
plt.imshow(rewardmap,cmap='viridis')
plt.plot(px,py,'k-',linewidth=3)
plt.plot(px[0],py[0],'ro',markersize=12)
plt.plot(px[-1],py[-1],'go',markersize=12)
plt.colorbar()
plt.show()

rmap = rewardmap

plt.show()
plt.figure(figsize=(7,6))
plt.imshow(rmap,cmap='viridis')
plt.plot(px,py,'k-',linewidth=3)
plt.plot(px[0],py[0],'ro',markersize=12)
plt.plot(px[-1],py[-1],'go',markersize=12)
plt.colorbar()
plt.show()