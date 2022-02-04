import torch
import torch.nn as nn

class Vanilla(torch.model):
    def __init__(self,size):
        self.hidden_sizes = []
        self.policy_net = nn.Sequential()
        self.value_net = nn.Sequential()

    def forward(self,input):
        pass

    def loss(self,input):
        pass

    def backward(self,train_buffer):
        pass
