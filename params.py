# Training Parameters
GPU=False
NUM_META_AGENTS = 3
EPISODE_LENGTH = 400
MAX_EPISODES = 20000
DISCOUNT = 0.9
LR = 1e-3
DECAY = 1/MAX_EPISODES

class JOB_TYPES:
    getExperience      = 1
    getGradient        = 2

class TRAINING_OPTIONS:
    multiThreaded      = 1
    singleThreaded        = 2

class GRADIENT_OPTIONS:
    episodic = 0
    batches = 4

JOB_TYPE = JOB_TYPES.getGradient
TRAINING_TYPE = TRAINING_OPTIONS.singleThreaded

GRADIENT_TYPE = GRADIENT_OPTIONS.batches

DEVICE = 'cuda:0'

RENDER_TRAINING = True
RENDER_TRAINING_WINDOW = 300
RAY_RESET_EPS = 1000
SET_SEED = False # Sets seed to ensure similar form of training
SAME_MAP = True  # Parameter that doesnt update seed in env, map would be spawned deterministically

#Environment Parameters
NUM_AGENTS = 1
SPAWN_RANDOM_AGENTS = True

#MODEL TYPE
MODEL_TYPE = 'ActorCritic2'

#Logging Params
MODEL_NAME = 'VanillaAC_v17'
DESCRIPTION = 'RangewObs'
TRAIN_PATH = 'data/train/'+MODEL_NAME+'_'+DESCRIPTION
MODEL_PATH = 'data/models/'+MODEL_NAME+'_'+DESCRIPTION
GIFS_PATH = 'data/gifs/'+MODEL_NAME+'_'+DESCRIPTION
LOAD_MODEL = False
SUMMARY_WINDOW = 10

NEPTUNE = True
NEPTUNE_RUN = None
neptune_project        = "harshg99/SearchKR" # USER-ID
NEPTUNE_API_TOKEN      = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOTI4NjE0Yi00ZjNmLTQ5NjktOTdhNy04YTk3ZGQyZTg1MDIifQ=="

# Observation Type
OBSERVER = 'RANGEwOBS' # TILED(Original),RANGE(portion of reward map),TILEDwOBS,RANGEwOBS
RANGE = 15