# Training Parameters
GPU=False
NUM_META_AGENTS = 4
EPISODE_LENGTH = 256
MAX_EPISODES = 20000
DISCOUNT = 0.9
LR = 3e-5
DECAY = 1e-5

class JOB_TYPES:
    getExperience      = 1
    getGradient        = 2

class TRAINING_OPTIONS:
    multiThreaded      = 1
    singleThreaded        = 2

JOB_TYPE = JOB_TYPES.getGradient
TRAINING_TYPE = TRAINING_OPTIONS.singleThreaded

DEVICE = 'cuda:0'

RENDER_TRAINING = False
RAY_RESET_EPS = 5000

#Environment Parameters
NUM_AGENTS = 1
SPAWN_RANDOM_AGENTS = True

#Network Params
HIDDEN_SIZES = [64,128,128,32,16]
value_weight = 0.05
policy_weight = 1.0
entropy_weight = 0.01


#Logging Params
MODEL_NAME = 'VanillaAC_v1'
DESCRIPTION = 'ACV1'
TRAIN_PATH = 'data/train/'+MODEL_NAME+'_'+DESCRIPTION
MODEL_PATH = 'data/models/'+MODEL_NAME+'_'+DESCRIPTION
LOAD_MODEL = False
SUMMARY_WINDOW = 10

NEPTUNE = True
NEPTUNE_RUN = None
neptune_project        = "harshg99/SearchKR" # USER-ID
NEPTUNE_API_TOKEN      = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOTI4NjE0Yi00ZjNmLTQ5NjktOTdhNy04YTk3ZGQyZTg1MDIifQ=="

# Observation Type
OBSERVER = 'FULL' # TILED(Original),FULL(whole reward map)
