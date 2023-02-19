import numpy as np

graph_file_name = 'latticeData/faster20.json'
numAgents=1
seed=45
episode_length = 100
pad_size = 4

rewardMapSize = 32 # m m size of the environment
defaultMapChoice = 0
resolution = 8 # m m to pixel conversion (map will be 200 by 200)
num_semantics = 4  # include background semantic



sensor_range = [3,3]
sensor_max_acc = 0.95
sensor_decay_coeff = 0.05

defaultBelief = 0.50
TargetBeliefThresh = 0.95

sensor_params = {}
sensor_params['type'] = 'SemanticSensor' # FieldSensor
sensor_params['sensor_max_acc'] = sensor_max_acc
sensor_params['sensor_range'] = sensor_range
sensor_params['sensor_decay_coeff'] = sensor_decay_coeff
#Loading the maps
assets_folder = 'env/assets/map_images/'
TRAIN_PROP = 0.8
TOTAL_MAPS = 25
# Sets the belief for randomising target locations

TARGET_NOISE_SCALE = 1.0 # scale for setting random locations
RANDOM_CENTRES = 64
CENTRE_SIZE = 25 # In pixels
MAX_CLIP = 0.9

# For tests
TEST_PER_MAP = 2


