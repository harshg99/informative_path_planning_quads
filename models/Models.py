import torch
import torch.nn as nn
from params import *
import numpy as np
import Utilities
from models.Vanilla import Vanilla
from torch.optim.lr_scheduler import ExponentialLR
from models.subnets import *
from models.subnet_setter import subnet_setter
import torch.nn.functional as F

class Model1(Vanilla):
    '''
        2 separate networks for actor and critic
    '''
    def __init__(self,input_size,action_size,params_dict,args_dict):
        super(Vanilla,self).__init__()
        self.hidden_sizes = params_dict['hidden_sizes']
        self.input_size = input_size
        self.action_size = action_size
        self.params_dict = params_dict
        self.args_dict = args_dict

        self.layers1= subnet_setter.set_model(params_dict['obs_model'],\
                                              self.hidden_sizes,self.input_size)

        self.policy_layers = [nn.Linear(self.hidden_sizes[-1],self.action_size)]
        self.policy_net = nn.Sequential(*self.policy_layers)

        self.layers2= subnet_setter.set_model(params_dict['obs_model'],\
                                              self.hidden_sizes,self.input_size)

        self.value_layers = [nn.Linear(self.hidden_sizes[-1], 1)]
        self.value_net = nn.Sequential(*self.value_layers)


        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        #print(input.shape)
        inputpol  = self.layers1(input)
        inputval = self.layers2(input)

        policy = self.policy_net(inputpol)
        value = self.value_net(inputval)
        return self.softmax(policy),value

    def forward_step(self,input):
        input = input['obs']
        input = torch.tensor([input], dtype=torch.float32)
        return self.forward(input)

    def compute_valids(self,input):
        # print(input.shape)
        inputpol = self.layers1(input)
        policy = self.policy_net(inputpol)
        return self.sigmoid(policy)

    def forward_buffer(self, obs_buffer):
        obs = []
        valids = []

        for j in obs_buffer:
            obs.append(j['obs'])
            valids.append(j['valids'])

        obs = torch.tensor(np.array(obs), dtype=torch.float32)


        policy, value = self.forward(obs)
        valids_net = self.compute_valids(obs)
        valids = torch.tensor(valids, dtype=torch.float32)
        return policy.squeeze(),value.squeeze(), \
               valids.squeeze(), valids_net.squeeze()

class Model2(Model1):
    '''
    Actor and Critic with shared parameters
    '''
    def __init__(self,input_size,action_size,params_dict,args_dict):
        super(Vanilla,self).__init__()
        self.hidden_sizes = params_dict['hidden_sizes']
        self.input_size = input_size
        self.action_size = action_size
        self.params_dict = params_dict
        self.args_dict = args_dict

        self.layers = subnet_setter.set_model(params_dict['obs_model'],\
                                              self.hidden_sizes,self.input_size)

        self.policy_layers = mlp_block(self.hidden_sizes[-1], self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU)
        for j in range(2):
            self.policy_layers.extend(mlp_block(self.hidden_sizes[-1], self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU))

        self.policy_layers.extend([nn.Linear(self.hidden_sizes[-1],self.action_size)])
        self.policy_net = nn.Sequential(*self.policy_layers)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.value_layers = mlp_block(self.hidden_sizes[-1], self.hidden_sizes[-1],dropout=False,activation=nn.LeakyReLU)
        for j in range(2):
            self.value_layers.extend(mlp_block(self.hidden_sizes[-1], self.hidden_sizes[-1],dropout=False,activation=nn.LeakyReLU))
        self.value_layers.extend([nn.Linear(self.hidden_sizes[-1], 1)])
        self.value_net = nn.Sequential(*self.value_layers)

    def forward(self, input):
        # print(input.shape)
        input = self.layers(input)

        policy = self.policy_net(input)
        value = self.value_net(input)
        return self.softmax(policy), value

    def compute_valids(self,input):
        # print(input.shape)
        input = self.layers(input)

        policy = self.policy_net(input)
        return self.sigmoid(policy)

    def forward_step(self, input):
        input = input['obs']
        input = torch.tensor([input], dtype=torch.float32)
        return self.forward(input)

    def forward_buffer(self, obs_buffer):
        obs = []
        valids = []

        for j in obs_buffer:
            obs.append(j['obs'])
            valids.append(j['valids'])


        obs = torch.tensor(np.array(obs), dtype=torch.float32)

        policy, value = self.forward(obs)
        valids_net = self.compute_valids(obs)
        valids = torch.tensor(valids, dtype=torch.float32)
        return policy.squeeze(),value.squeeze(), \
               valids.squeeze(), valids_net.squeeze()

class Model3(Model2):
    '''
    With agent position information
    '''
    def __init__(self,input_size,action_size,params_dict,args_dict):
        super(Model3,self).__init__(input_size,action_size,params_dict,args_dict)
        self.hidden_sizes = params_dict['hidden_sizes']
        self.hidden_sizes1 = params_dict['hidden_sizes1']
        self.pos_layer_size = params_dict['pos_layer_size']
        self.input_size = input_size
        self.action_size = action_size
        self.params_dict = params_dict
        self.args_dict = args_dict

        self.position_layer = mlp_block(2,self.pos_layer_size[0],dropout=False,activation=nn.LeakyReLU)
        for j in range(len(self.position_layer)-1):
            self.position_layer.extend(mlp_block(self.pos_layer_size[j],\
                                                      self.pos_layer_size[j+1],dropout=False,activation=nn.LeakyReLU))
        self.position_layer = nn.Sequential(*self.position_layer)
        self.policy_layers = mlp_block(self.hidden_sizes[-1] + self.pos_layer_size[-1], \
                                                 self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU)
        for j in range(2):
            self.policy_layers.extend(mlp_block(self.hidden_sizes[-1],\
                                                     self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU))

        self.policy_layers.extend([nn.Linear(self.hidden_sizes[-1],self.action_size)])
        self.policy_net = nn.Sequential(*self.policy_layers)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.value_layers = mlp_block(self.hidden_sizes[-1] + self.pos_layer_size[-1], \
                                                self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU)
        for j in range(2):
            self.value_layers.extend(mlp_block(self.hidden_sizes[-1],\
                                                    self.hidden_sizes[-1],dropout=False,activation=nn.LeakyReLU))
        self.value_layers.extend([nn.Linear(self.hidden_sizes[-1], 1)])
        self.value_net = nn.Sequential(*self.value_layers)

    def forward_step(self, input):
        obs = input['obs']
        pos = input['position']
        obs = torch.tensor([obs], dtype=torch.float32)
        pos = torch.tensor([pos],dtype=torch.float32)
        return self.forward(obs,pos)

    def compute_valids(self,input,pos):
        pos = self.position_layer(pos)
        input = self.layers(input)
        input = torch.concat([input,pos],dim=-1)
        policy = self.policy_net(input)
        return self.sigmoid(policy)

    def forward(self,input,pos):
        pos = self.position_layer(pos)
        input = self.layers(input)
        input = torch.concat([input,pos],dim=-1)
        policy = self.policy_net(input)
        return self.softmax(policy),self.value_net(input)

    def forward_buffer(self, obs_buffer):
        obs = []
        valids = []
        pos = []
        prev_a = []
        for j in obs_buffer:
            obs.append(j['obs'])
            valids.append(j['valids'])
            pos.append(j['position'])

        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        pos = torch.tensor(np.array(pos), dtype=torch.float32)

        policy, value = self.forward(obs, pos)
        valids_net = self.compute_valids(obs, pos)
        valids = torch.tensor(valids, dtype=torch.float32)
        return policy.squeeze(),value.squeeze(), \
               valids.squeeze(), valids_net.squeeze()


'''
With positonal encoder and prior motion primitives
'''
class Model4(Model3):
    def __init__(self,input_size,action_size,params_dict,args_dict):
        super(Model3,self).__init__(input_size,action_size,params_dict,args_dict)
        self.hidden_sizes = params_dict['hidden_sizes']
        self.hidden_sizes1 = params_dict['hidden_sizes1']
        self.pos_layer_size = params_dict['pos_layer_size']
        self.input_size = input_size
        self.action_size = action_size
        self.params_dict = params_dict
        self.args_dict = args_dict

        self.position_layer = mlp_block(2, self.pos_layer_size[0], dropout=False, activation=nn.LeakyReLU)
        for j in range(len(self.position_layer)-1):
            self.position_layer.extend(mlp_block(self.pos_layer_size[j], \
                                                 self.pos_layer_size[j + 1], dropout=False, activation=nn.LeakyReLU))
        self.position_layer = nn.Sequential(*self.position_layer)

        self.policy_layers = mlp_block(self.hidden_sizes[-1] + self.pos_layer_size[-1] +self.action_size, \
                                                 self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU)
        for j in range(2):
            self.policy_layers.extend(mlp_block(self.hidden_sizes[-1],\
                                                     self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU))

        self.policy_layers.extend([nn.Linear(self.hidden_sizes[-1],self.action_size)])
        self.policy_net = nn.Sequential(*self.policy_layers)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.value_layers = mlp_block(self.hidden_sizes[-1] + self.pos_layer_size[-1]+self.action_size, \
                                                self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU)
        for j in range(2):
            self.value_layers.extend(mlp_block(self.hidden_sizes[-1],\
                                                    self.hidden_sizes[-1],dropout=False,activation=nn.LeakyReLU))
        self.value_layers.extend([nn.Linear(self.hidden_sizes[-1], 1)])
        self.value_net = nn.Sequential(*self.value_layers)


    def forward_step(self, input):
        obs = input['obs']
        pos = input['position']
        previous_actions = input['previous_actions']
        obs = torch.tensor([obs], dtype=torch.float32)
        pos = torch.tensor([pos],dtype=torch.float32)
        prev_a = F.one_hot(torch.tensor([previous_actions]),num_classes = self.action_size)
        return self.forward(obs,pos,prev_a)

    def compute_valids(self,input,pos,prev_a):
        pos = self.position_layer(pos)
        input = self.layers(input)
        input = torch.concat([input,pos,prev_a],dim=-1)
        policy = self.policy_net(input)
        return self.sigmoid(policy)

    def forward(self,input,pos,prev_a):
        #print(input.shape)
        pos = self.position_layer(pos)
        input = self.layers(input)
        input = torch.concat([input,pos,prev_a],dim=-1)
        policy = self.policy_net(input)
        return self.softmax(policy),self.value_net(input)

    def forward_buffer(self, obs_buffer):
        obs = []
        valids = []
        pos = []
        prev_a = []
        for j in obs_buffer:
            obs.append(j['obs'])
            valids.append(j['valids'])
            pos.append(j['position'])
            prev_a.append(j['previous_actions'])

        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        pos = torch.tensor(np.array(pos),dtype=torch.float32)
        prev_a = F.one_hot(torch.tensor(prev_a),num_classes = self.action_size)

        policy, value = self.forward(obs, pos,prev_a)
        valids_net = self.compute_valids(obs, pos,prev_a)
        valids = torch.tensor(valids,dtype = torch.float32)
        return policy.squeeze(),value.squeeze(), \
               valids.squeeze(), valids_net.squeeze()