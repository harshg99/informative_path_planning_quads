#!/usr/bin/python3
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
x = pickle.load(open(f'{script_dir}/testingData/gaussian_mixture_test2.pkl', "rb"), encoding='latin1')
plt.imshow(x, cmap='Greys', interpolation='spline36', origin='lower')


plt.show()
plt.imsave('foo.png',x, cmap='Greys',origin='lower')
