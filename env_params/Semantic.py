import numpy as np

graph_file_name = 'latticeData/faster20.json' # Short motion primitives that can be scaled

#graph_file_name = 'latticeData/70.json' # longer motion primitives
graph_file_name = 'latticeData/20.json' # longer motion primitives with fewer vertices
numAgents=1
seed=45
episode_length = 100


if graph_file_name ==  'latticeData/faster20.json':
    rewardMapSize = 32 # m m size of the environment
    defaultMapChoice = 0

    measurement_step = 1 # how many measurements to discretize ber step
    resolution = 8 # m m to pixel conversion (map will be 200 by 200)
    obs_resolution = 1 # m m to pixel conversion (map will be 200 by 200)
    sensor_range = [3, 3]
    pad_size = 4
    sensor_decay_coeff = 0.08
    sampled_step_size = 0.1

elif graph_file_name ==  'latticeData/70.json' or graph_file_name ==  'latticeData/20.json':
    rewardMapSize = 256  # m m size of the environment
    defaultMapChoice = 0
    measurement_step = 4  # how many measurements to discretize ber step
    obs_resolution = 8  # m m to pixel conversion (map will be 200 by 200)
    resolution = 1   # m m to pixel conversion (map will be 200 by 200)
    sensor_range = [24, 24]
    pad_size = 32
    sensor_decay_coeff = 0.08/obs_resolution
    sampled_step_size = 0.4


num_semantics = 4  # include background semantic
sensor_max_acc = 0.85


defaultBelief = 0.50
TargetBeliefThresh = 0.95

sensor_params = {}
sensor_params['type'] = 'SemanticSensor' # FieldSensor
sensor_params['sensor_max_acc'] = sensor_max_acc
sensor_params['sensor_range'] = sensor_range
sensor_params['sensor_decay_coeff'] = sensor_decay_coeff
sensor_params['measurement_step'] = measurement_step
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


