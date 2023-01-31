import numpy as np

graph_file_name = 'latticeData/faster20.json'
numAgents=1
seed=45
episode_length = 100
pad_size = 5

rewardMapSizeList = 40 # m m size of the environment
defaultMapChoice = 0
resolution = 0.2 # m m to pixel conversion (map will be 200 by 200)
num_semantics = 5  # include background semantic

# Sets the belief for randomising target locations
TARGET_NOISE_SCALE = 1.0 # scale for setting random locations
RANDOM_CENTRES = 30
CENTRE_SIZE = 40

sensor_range = [5,5]
sensor_max_acc = 0.95
sensor_decay_coeff = 0.1

defaultBelief = 0.50
TargetBeliefThresh = 0.95

sensor_params = {}
sensor_params['type'] = 'FieldSensor' # FieldSensor
sensor_params['sensor_unc'] = sensor_max_acc
sensor_params['sensor_range'] = sensor_range