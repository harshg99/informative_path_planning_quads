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
sensor_max_acc = 0.95
sensor_decay_coeff = 0.1

num_targets = [10,30]
defaultBelief = 0.05
targetBeliefThresh = 0.95

sensor_params = {}
sensor_params['type'] = 'FieldSensor' # FieldSensor
sensor_params['sensor_unc'] = sensor_unc
sensor_params['sensor_range'] = sensor_range