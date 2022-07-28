import gym
import time
import numpy as np
from gym.envs.classic_control import rendering
from enum import Enum
from env.render import *
from env.searchenv import *
from env.searchenvMP import *
import math
import matplotlib.pyplot as plt
import GPy
from params import *
from motion_primitives_py import MotionPrimitiveLattice
from copy import deepcopy
from env.agents import AgentGPObs
import os
from env.Metrics import Metrics

# Obstacle environment

class GPObsEnvMP(GPEnvMP):
    def __init__(self,params_dict,args_dict):
        super().__init__(params_dict,args_dict)
        self.defaultBelief = params_dict['defaultBelief']
        self.sensor_params = params_dict['sensor_params']
        self.numrand_targets = params_dict['num_targets']
        self.targetBeliefThresh = params_dict['targetBeliefThresh']
        self.metrics = Metrics()