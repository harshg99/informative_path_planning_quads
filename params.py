# Training Parameters
import torch
GPU = False and torch.cuda.is_available()
DEVICE = 'cuda:0' if GPU else 'cpu'
NUM_DEVICES = torch.cuda.device_count()

NUM_META_AGENTS = 1
MAX_EPISODES = 50000
DISCOUNT = 0.9
LR = 1e-3
DECAY = 1/MAX_EPISODES
LAMBDA_RET = True
LAMBDA = 0.8

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

RENDER_TRAINING = True
RENDER_TRAINING_WINDOW = 300
RAY_RESET_EPS = 1000



#Environment Parameters
SPAWN_RANDOM_AGENTS = True
SET_SEED = False # Sets seed to ensure similar form of training
SAME_MAP = True  # Parameter that doesnt update seed in env, map would be spawned deterministically

ENV_TYPE = 'GPPrim' # MotionPrim or Discrete or GPPrim

#Episode Parameters
FIXED_BUDGET = True
BUDGET = 10.0

# Observation Type
RANGE = 8

#MODEL TYPE
CLASS_CONFIG = 'LSTM'  #Linear OR Transformer OR RSSM
if CLASS_CONFIG =='Linear':
    MODEL_TYPE = 'Model6'#Model1 Model2 etc
    ALG_TYPE = 'AC' #AC or PPO or SAC
    QVALUE = True
    COMPUTE_VALIDS = True # Soft Loss on Valid actions (not supported for ActorCritic 1 or 2
    # Observation Type
    OBSERVER = 'RANGEwOBSwMULTIwCOV'  # TILED(Original),RANGE(portion of reward map),TILEDwOBS,RANGEwOBS,RANGEwOBSwPENC,RANGEwOBSwMULTI
    LSTM = False
    BUDGET_LAYER = False

elif CLASS_CONFIG == 'Transformer':
    MODEL_TYPE = 'ModelTrans2'  # Model1 Model2 etc
    ALG_TYPE = 'AC'  # AC or PPO or SAC
    QVALUE = True
    COMPUTE_VALIDS = True  # Soft Loss on Valid actions (not supported for ActorCritic 1 or 2
    # Observation Type
    OBSERVER = 'RANGEwOBSwMULTI'
    LSTM = False
    BUDGET_LAYER = False

elif CLASS_CONFIG == 'RSSM':
    MODEL_TYPE = 'ModeL1'  # Model1 Model2 etc
    ALG_TYPE = 'AC'  # AC or PPO or SAC
    QVALUE = True
    COMPUTE_VALIDS = True  # Soft Loss on Valid actions (not supported for ActorCritic 1 or 2
    # Observation Type
    OBSERVER = 'RANGEwOBSwMULTI'
    LSTM = False
    BUDGET_LAYER = False

elif CLASS_CONFIG == 'LSTM':
    MODEL_TYPE = 'ModelMF4'  # Model1 Model2 etc
    ALG_TYPE = 'ACLSTM'  # ACLSTM or PPOLSTM or SACLSTM
    QVALUE = True
    COMPUTE_VALIDS = True  # Soft Loss on Valid actions (not supported for ActorCritic 1 or 2
    # Observation Type
    OBSERVER = 'RANGEwOBSwMULTIwCOV'
    LSTM = True
    BUDGET_LAYER = False

if ALG_TYPE=='PPO':
    JOB_TYPE = JOB_TYPES.getExperience
    TRAINING_TYPE = TRAINING_OPTIONS.singleThreaded

if ALG_TYPE=='SAC':
    JOB_TYPE = JOB_TYPES.getExperience
    TRAINING_TYPE = TRAINING_OPTIONS.singleThreaded
    CAPACITY = 12000
    NUM_GPUS = 1
    SAC_GRAD_ITERATIONS = 5
    BATCH_SIZE = 32
    MIN_CAPACITY = 200

#Logging Params
MODEL_NAME = 'GPAC_Model6'
DESCRIPTION = 'Final_Version_Range16'
TRAIN_PATH = 'data/train/'+MODEL_NAME+'_'+DESCRIPTION
MODEL_PATH = 'data/models/'+MODEL_NAME+'_'+DESCRIPTION
GIFS_PATH = 'data/gifs/'+MODEL_NAME+'_'+DESCRIPTION
LOAD_MODEL = True
LOAD_BEST_MODEL = 0 # None for older versions, if LOAD Model gives an error
# integer between 0 to 5 to load particular models
NUM_SAVE_MODEL = 5

SUMMARY_WINDOW = 10

TEST_GIFS = False
TEST_GIFS_PATH = "data/test/{}/GIFS/"
TEST_RESULTS_PATH= "data/test/{}/RESULTS/"

NEPTUNE = False
NEPTUNE_RUN = None

neptune_project        = "harshg99/SearchKR2" # USER-ID
NEPTUNE_API_TOKEN      = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOTI4NjE0Yi00ZjNmLTQ5NjktOTdhNy04YTk3ZGQyZTg1MDIifQ=="



# Test directory
MAP_TEST_DIR = 'tests/maps/'
TEST_TYPE = 'GPPrim{}'
