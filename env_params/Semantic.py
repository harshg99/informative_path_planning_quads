import numpy as np

graph_file_name = 'latticeData/faster20.json'
numAgents=1
seed=45
episode_length = 100
pad_size = 5

rewardMapSize = 32 # m m size of the environment
defaultMapChoice = 0
resolution = 8 # m m to pixel conversion (map will be 200 by 200)
num_semantics = 4  # include background semantic



sensor_range = [5,5]
sensor_max_acc = 0.95
sensor_decay_coeff = 0.1

defaultBelief = 0.50
TargetBeliefThresh = 0.95

sensor_params = {}
sensor_params['type'] = 'SemanticSensor' # FieldSensor
sensor_params['sensor_unc'] = sensor_max_acc
sensor_params['sensor_range'] = sensor_range

#Loading the maps
assets_folder = 'env/assets/map_images/'
TRAIN_PROP = 0.8
TOTAL_MAPS = 25
# Sets the belief for randomising target locations

TARGET_NOISE_SCALE = 1.0 # scale for setting random locations
RANDOM_CENTRES = 64
CENTRE_SIZE = 25 # In pixels
MAX_CLIP = 0.9


import os

'''finds the number of files in a directory'''
def get_num_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
