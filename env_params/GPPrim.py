import numpy as np

graph_file_name = 'latticeData/faster20.json'
numAgents=1
num_centers=[5, 10]
max_var= 20.0
min_var= 3.0
seed=45
episode_length = 100
pad_size = 5

rewardMapSizeList = [30,45,60,90]
randomMapSize = False
defaultMapChoice = 0
scale = [1,2,4]

sensor_range = 5
sensor_unc = np.zeros([sensor_range,sensor_range])
ceoff = 0.1

num_targets = [10,20]
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

