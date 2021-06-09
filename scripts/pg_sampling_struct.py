#Structure imported in the pg_path_generator.py
# Author : Sandeep Manjanna (McGill University)

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import pickle
import time

reward_map_size = 50
pad_size = reward_map_size-1
world_map_size = reward_map_size + 2*(pad_size)
curr_r_map_size = reward_map_size + pad_size
curr_r_pad = (curr_r_map_size-1)/2
num_actions = 4
num_features = 0
param_len = num_features*num_actions

num_iterations = 500
num_trajectories = 20
Tau_horizon = 400
is_action_valid = True
rand_start_pos = True
plot = True
fileNm = "totreward_5x5_200_base"

#Discount factor gamma
gamma = 0.95
#Learning rate
Eta = 0

#phi_prime for 24-feature aggregation --> High resolution to low resolution
def get_phi_prime(worldmap,curr_pos):
    wmap = np.copy(worldmap)
    r=curr_pos[1]
    c=curr_pos[0]
    phi_prime = []
    phi_prime.append(wmap[r,c-1])
    phi_prime.append(wmap[r-1,c-1])
    phi_prime.append(wmap[r-1,c])
    phi_prime.append(wmap[r-1,c+1])
    phi_prime.append(wmap[r,c+1])
    phi_prime.append(wmap[r+1,c+1])
    phi_prime.append(wmap[r+1,c])
    phi_prime.append(wmap[r+1,c-1])
    
    temp = np.copy(wmap[r-1:r-1+3,c-4:c-4+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4:r-4+3,c-4:c-4+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4:r-4+3,c-1:c-1+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4:r-4+3,c+2:c+2+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-1:r-1+3,c+2:c+2+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r+2:r+2+3,c+2:c+2+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r+2:r+2+3,c-1:c-1+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r+2:r+2+3,c-4:c-4+3])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    
    temp = np.copy(wmap[r-4:r-4+9,0:c-4])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[0:r-4,0:c-4])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[0:r-4,c-4:c-4+9])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[0:r-4,c-4+9:])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4:r-4+9,c-4+9:])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4+9:,c-4+9:])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4+9:,c-4:c-4+9])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    temp = np.copy(wmap[r-4+9:,0:c-4])
    avg = np.size(temp)
    if(avg==0):avg=1
    phi_prime.append(np.sum(temp)/avg)
    
    phi_prime = np.array([phi_prime])
    return phi_prime

def get_phi(worldmap,curr_pos,curr_action):
    #Actions: 0 - LEFT, 1 - UP, 2 - RIGHT, and 3 - DOWN
    identity_left = np.zeros(num_features)
    identity_up = np.zeros(num_features)
    identity_right = np.zeros(num_features)
    identity_down = np.zeros(num_features)

    if curr_action == 0:
        identity_left = np.ones(num_features)
    elif curr_action == 1:
        identity_up = np.ones(num_features)
    elif curr_action == 2:
        identity_right = np.ones(num_features)
    else:
        identity_down = np.ones(num_features)

    identity = np.concatenate((identity_left,identity_up,identity_right,identity_down))
    phi_prime = get_phi_prime(worldmap,curr_pos)
    phi = np.tile(phi_prime,num_actions)
    phi = np.multiply(phi,identity)
    phi = np.transpose(phi)
    return phi

def sample_action(worldmap,pos,prev_action,theta,isprint=False,maxPolicy=False):
    
    phi = []
    dot = []

    for i in range(num_actions):
        phi.append(get_phi(worldmap,pos,i))
        dot.append(np.dot(np.transpose(theta),phi[i])[0,0])

    #Making sure the range for numpy.exp
    dot_max = np.max(dot)
    exp = []

    for i in range(num_actions):
        dot[i] = dot[i] - dot_max
        exp.append(np.exp(dot[i]))

    exp_sum = np.sum(exp)
    prob = []
    for i in range(num_actions):
        prob.append(exp[i] / exp_sum)

    p = prob

    if maxPolicy:
        next_action = np.argmax(p)
    else:
        next_action = np.random.choice(num_actions,1,p)[0]
    return next_action

def get_next_state(worldmap,curr_pos,curr_action):
    global is_action_valid
    next_pos = list(curr_pos)
    if(curr_action == 0):
        if(curr_pos[0]-(curr_r_pad+1) > -1):
            next_pos[0] = curr_pos[0]-1
        else:
            is_action_valid = False
    elif(curr_action == 1):
        if(curr_pos[1]-(curr_r_pad+1) > -1):
            next_pos[1] = curr_pos[1]-1
        else:
            is_action_valid = False
    elif(curr_action == 2):
        if(curr_pos[0]+(curr_r_pad+1) < worldmap.shape[1]):
            next_pos[0] = curr_pos[0]+1
        else:
            is_action_valid = False
    elif(curr_action == 3):
        if(curr_pos[1]+(curr_r_pad+1) < worldmap.shape[0]):
            next_pos[1] = curr_pos[1]+1
        else:
            is_action_valid = False

    return next_pos

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

def generate_trajectories(num_trajectories,worldmap,curr_pos,theta,isPrint=False,maxPolicy=False,rand_start=True):
    #Array of trajectories starting from current position.
    print "In trajectory generation"
    global is_action_valid
    copy_worldmap = np.copy(worldmap)
    Tau = np.ndarray(shape=(num_trajectories,Tau_horizon), dtype=object)
    start_pos = list(curr_pos)
    for i in range(num_trajectories):
        p=[]
        for prob in range(reward_map_size):
            p.append(1.0/reward_map_size)
        pos1 = np.random.choice(range(reward_map_size),1,p)[0]+pad_size
        pos2 = np.random.choice(range(reward_map_size),1,p)[0]+pad_size
        curr_pos = np.array([pos1,pos2])
        if not rand_start:
            curr_pos = []
            curr_pos = list(start_pos)
        worldmap = None
        worldmap = np.copy(copy_worldmap)
        curr_action = -1
        for j in range(Tau_horizon):
            prev_action = curr_action
            curr_action = sample_action(worldmap,curr_pos,prev_action,theta,isPrint,maxPolicy)
            next_pos = get_next_state(worldmap,curr_pos,curr_action)
            if is_action_valid:
                curr_reward = worldmap[next_pos[1],next_pos[0]]
                worldmap[curr_pos[1],curr_pos[0]] = -0.0001
            else:
                curr_reward = -2
                is_action_valid = True
            t = Trajectory(curr_pos,curr_action,curr_reward)
            Tau[i][j] = t
            curr_pos = next_pos
    return Tau

