#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace as T

'''
   Adds a Gaussian
'''

min_var = 3.0
max_var = 6.0
def Gaussian(mean, cov,map_size):
    x = np.array([np.linspace(0, map_size - 1, map_size),
                  np.linspace(0, map_size - 1, map_size)]).T

    x1Mesh, x2Mesh = np.meshgrid(x[:, 0:1], x[:, 1:2])
    # Making the gaussians circular

    cov1 = np.copy(cov)
    cov[0][0] = np.clip(cov[0][0], min_var, max_var)
    cov[1][1] = np.clip(cov[1][1], min_var, max_var)
    cov[0][1] = np.clip(cov[0][1], 0, min_var / 2)
    cov[1][0] = cov[0][1]

    if np.linalg.det(cov) < 0:
        cov[0][0] = np.clip(cov1[0][1], min_var, max_var)
        cov[1][1] = np.clip(cov1[1][0], min_var, max_var)
        cov[0][1] = np.clip(cov1[1][1], min_var / 2, max_var / 2)
        cov[1][0] = cov[0][1]
    elif np.linalg.det(cov) == 0:
        cov[0][0] = np.clip(cov1[0][1] + np.random.rand(1), min_var, max_var)
        cov[1][1] = np.clip(cov1[1][0] + np.random.rand(1), min_var, max_var)

    xPred = np.array([np.reshape(x1Mesh, (map_size * map_size,)) \
                         , np.reshape(x2Mesh, (map_size * map_size,))])
    gaussian = np.diag(1 / np.sqrt(2 * np.pi * np.abs(np.linalg.det(cov))) * \
                       np.exp(-(xPred - mean.reshape((mean.shape[0], 1))).T @ np.linalg.inv(cov) @ \
                              (xPred - mean.reshape((mean.shape[0], 1)))))
    gaussian = gaussian.reshape((map_size, map_size))
    return gaussian

x_bounds = [-15,15]
y_bounds = [-15,15]

resolution = 0.5
#T()
num_bins = (int((x_bounds[1]-x_bounds[0])/ resolution) +1, int((y_bounds[1]-y_bounds[0]) / resolution) +1)

interest_array = np.zeros(num_bins)

origin = [0,0]

# X axis is columns , Y axis is rows

map_origin = [int((origin[1]-y_bounds[0])/resolution)+1,int((origin[1]- x_bounds[0])/resolution) + 1]

box_pop_centers = [[0,3],[5,9],[-5,9]]
box_pop_steps = [[3,3,0],[3,3,0],[3,3,0]]
box_pop_nums = [[2,3],[3,2],[3,2]]
#T()

for i,centre in enumerate(box_pop_centers):
    for j in range(box_pop_nums[i][1]):
        for k in range(box_pop_nums[i][0]):
            r_pos = centre[0] + float((j-float(box_pop_nums[i][1]/2.0)))*box_pop_steps[i][1]
            c_pos = centre[1] + float((k-float(box_pop_nums[i][0]/2.0)))*box_pop_steps[i][0]
            r_pos_map = int((c_pos - y_bounds[0])/resolution) + 1
            c_pos_map = int((r_pos - x_bounds[0])/resolution) + 1
            interest_array += Gaussian(np.array([c_pos_map,r_pos_map]),\
                                       np.array([[6.0,0.0],[0.0,6.0]]),num_bins[0])
            #interest_array[r_pos_map,c_pos_map] = 1

#Normalising the gaussian map
interest_array/=interest_array.sum()
plt.figure()
plt.imshow(interest_array, origin='lower', extent=[0, resolution*num_bins[0], 0, resolution*num_bins[1]])

np.save("interest_Array.npy",interest_array)
plt.show()