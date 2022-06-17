import torch
from models.subnets import *
import params as arguments

class subnet_setter:

    @staticmethod
    def set_model(type,hidden_size,input_size,device='cpu'):
        if type == 'MLP':
            input_dims = 1
            for size in input_size:
                input_dims *= size
            return MLPLayer(hidden_size,input_dims,device)
        elif type == 'Conv':
            return ConvLayer(hidden_size,input_size,device)