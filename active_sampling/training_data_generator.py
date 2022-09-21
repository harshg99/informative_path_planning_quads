# Code to generate the input Gaussian data for learning / training
# Author : Sandeep Manjanna (McGill University)

from matplotlib import pyplot as plt
import math
import GPy
import numpy as np
import pickle
import os

testSamples = 5
X = np.random.uniform(-3., 3., (testSamples, 2))
np.shape(X)
Y = np.sin(X[:, 0:1]) * np.sin(X[:, 1:2])+np.random.randn(testSamples, 1)*0.05
np.shape(Y)

# Sub-test to check the gaussian with very sparse data
X = np.array([[6, 6], [24, 24], [9, 9], [10, 20]])
Y = np.array([[15], [7.5], [11.25], [13.5]])

#m = GPy.models.GPRegression(X,Y)
m = GPy.models.SparseGPRegression(X, Y, num_inducing=10)
m.rbf.variance = 1
m.rbf.lengthscale = 3
print(m.rbf)
# m.optimize()


x = np.array([np.linspace(0, 29, 30), np.linspace(0, 29, 30)]).T  # np.random.uniform(-3.,3.,(200,2))

x1Mesh, x2Mesh = np.meshgrid(x[:, 0:1], x[:, 1:2])
xPred = np.array([np.reshape(x1Mesh, (900,)), np.reshape(x2Mesh, (900,))]).T

yPred, Var = m.predict(xPred)
x1len = math.floor(np.max(x[:, 0:1]) - np.min(x[:, 0:1]))+1
x2len = math.floor(np.max(x[:, 1:2]) - np.min(x[:, 1:2]))+1

yMesh = np.reshape(yPred, (np.size(x, 0), np.size(x, 0))).T
print(yMesh.shape)
levels = np.linspace(np.min(yMesh), np.max(yMesh), 1000)
levels1 = np.linspace(np.min(yMesh), np.max(yMesh), 10)
# yMesh[:] = 0.5
plt.contourf(x1Mesh, x2Mesh, yMesh, levels, cmap='viridis')
#plt.contour(x1Mesh, x2Mesh, yMesh, levels1, colors='k')
plt.colorbar()
plt.show()

script_dir = os.path.dirname(__file__)
#pickle.dump(yMesh, open(f'{script_dir}/trainingData/gaussian_mixture_training_data.pkl', "wb"))
