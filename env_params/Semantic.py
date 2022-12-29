import numpy as np

graph_file_name = 'latticeData/faster20.json'
numAgents=1
seed=45
episode_length = 100
pad_size = 5

rewardMapSizeList = [30,45,60,90]
randomMapSize = False
defaultMapChoice = 0
scale = [1,2,4]

# Sets the belief for randomising target locations
TARGET_RANDOM_SCALE = 1.0 # scale for setting random locations
RANDOM_CENTRES = 5 # number of random centres
MAXDISP = 10*(defaultMapChoice+1) # max displacement of prior mean

num_centers=[3*(defaultMapChoice+1),5*(defaultMapChoice+1)]
max_var = 40.0
min_var = 30.0

noise_max_var = 50.0*(float(defaultMapChoice)/2.0+1)
noise_min_var = 40.0*(float(defaultMapChoice)/2.0+1)

sensor_range = 5
sensor_unc = np.zeros([sensor_range,sensor_range])
ceoff = 0.1

num_targets = [10,30]
defaultBelief = 0.05
targetBeliefThresh = 0.95

for j in range(sensor_range):
    for k in range(sensor_range):
        dist = max(np.abs(j-int((sensor_range/2))),np.abs(k-int((sensor_range/2))))
        sensor_unc[j][k] = 0.01 + ceoff*dist

sensor_params = {}
sensor_params['type'] = 'FieldSensor' # FieldSensor
sensor_params['sensor_unc'] = sensor_unc
sensor_params['sensor_range'] = sensor_range