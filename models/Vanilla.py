import torch
import torch.nn as nn
from params import *
import numpy as np
import Utilities


class Vanilla(nn.Module):
    def __init__(self,input_size,action_size,params_dict,PPO=False):
        super(Vanilla,self).__init__()

    # Implement these functions according to various model requirements

    def forward(self):
        pass

    def backward(self):
        pass

    def reset(self):
        pass

    def buffer_keys_required(self):
        keysList = ['obs','next_obs','actions','rewards','dones']
        return keysList