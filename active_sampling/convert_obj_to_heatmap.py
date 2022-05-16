#!/usr/bin/python3
import pywavefront
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

obj = pywavefront.Wavefront(
    '/home/laura/Desktop/SanCarlosAirport_Buildings.obj',
    strict=True,
)

vert = np.array(obj.vertices)
vert = vert * .01
vert = vert[vert[:,0] > 0]
vert = vert[vert[:,1] > 0]
vert[:,2] = vert[:,2] - min(vert[:,2])


plt.scatter(vert[:,0],vert[:,1],1,vert[:,2])
plt.colorbar()

num_bins = 100
interest_array = np.zeros((num_bins+1,num_bins+1))
x_max = max(vert[:,0])
y_max = max(vert[:,1])

for v in vert:
    if v[2] > 0:
        coord = [0,0]
        coord[0] = int(v[0]/x_max * num_bins)
        coord[1] = int(v[1]/y_max * num_bins)
        interest_array[coord[1],coord[0]] = 1

plt.figure()
np.save("airport.npy",interest_array)
plt.imshow(interest_array, origin='lower')

plt.show()
