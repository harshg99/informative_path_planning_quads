import torch.nn as nn
import torch
import numpy as np
from models.Models import *
from models.Vanilla import *
from models.subnets import *
import torch.nn.functional as F
from models.subnet_setter import subnet_enc_setter
from copy import deepcopy

class ModelMF1(Vanilla):
    def __init__(self,env,params_dict,args_dict):
        super(Vanilla,self).__init__()
        self.conv_sizes = params_dict['hidden_sizes']
        self.hidden_sizes = self.conv_sizes
        self.pos_layer_size = params_dict['pos_layer_size']
        self.graph_layer_size = params_dict['graph_node_layer_size']
        self.env = env
        self.input_size = env.input_size
        self.action_size = env.action_size
        self.num_graph_nodes = env.num_graph_nodes
        self.args_dict = args_dict
        self.params_dict = params_dict


        self.hidden_state = None
        self._invert_hidden_sizes = np.array(self.conv_sizes)[::-1].tolist()

        #self.encoder_layer = subnet_setter.set_model(params_dict['obs_model'], \
        #                                       self.hidden_sizes, self.input_size)
        self.init_backbone()



        self.multi_head_input_size = int(params_dict['token_length']*self.conv_sizes[-1])
        self.attention_model = nn.MultiheadAttention(embed_dim=self.multi_head_input_size,
                                                     num_heads=params_dict['num_heads'],
                                                     batch_first=True)
        #TODO: define GRU
        if self.args_dict['FIXED_BUDGET']:
            self.lstminsize = self.multi_head_input_size + self.pos_layer_size[-1] + \
                                               self.action_size + self.graph_layer_size[-1] + \
                                               params_dict['budget_layer_size'][-1]

        else:
            self.lstminsize = self.multi_head_input_size + self.pos_layer_size[-1] + \
                                               self.action_size + self.graph_layer_size[-1]
        self.LSTM = nn.GRU(input_size=self.lstminsize,
                           hidden_size=params_dict['embed_size'],
                           num_layers=params_dict['gru_layers'],
                           batch_first=True
                           )
        # Position layer
        self.position_layer = mlp_block(2, self.pos_layer_size[0], dropout=False, activation=nn.LeakyReLU)
        for j in range(len(self.position_layer) - 1):
            self.position_layer.extend(mlp_block(self.pos_layer_size[j], \
                                                 self.pos_layer_size[j + 1], dropout=False, activation=nn.LeakyReLU))
        self.position_layer = nn.Sequential(*self.position_layer)
        self.position_layer.to(self.args_dict['DEVICE'])

        # Graph encoding layer
        self.graph_node_layer = mlp_block(self.num_graph_nodes, self.graph_layer_size[0], \
                                          dropout=False, activation=nn.LeakyReLU)
        for j in range(len(self.graph_layer_size) - 1):
            self.graph_node_layer.extend(mlp_block(self.graph_layer_size[j], \
                                                   self.graph_layer_size[j + 1], \
                                                   dropout=False, activation=nn.LeakyReLU))



        self.graph_node_layer = nn.Sequential(*self.graph_node_layer)

        self.graph_node_layer.to(self.args_dict['DEVICE'])

        if self.args_dict['FIXED_BUDGET']:
            self.budget_layers = mlp_block(1,params_dict['budget_layer_size'][0],dropout=False, activation=nn.LeakyReLU)
            for j in range(len(params_dict['budget_layer_size']) - 1):
                self.budget_layers.extend(mlp_block(params_dict['budget_layer_size'][j], \
                                                       params_dict['budget_layer_size'][j+1], \
                                                       dropout=False, activation=nn.LeakyReLU))
            self.budget_layers = nn.Sequential(*self.budget_layers)
            self.budget_layers.to(self.args_dict['DEVICE'])


        self.policy_layers = mlp_block(params_dict['embed_size']+self.lstminsize, \
                                       params_dict['embed_size'], dropout=False, activation=nn.LeakyReLU)

            # for j in range(2):
            #     self.policy_layers.extend(mlp_block(self.hidden_sizes[-1],\
            #                                              self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU))

        self.policy_layers.extend([nn.Linear(params_dict['embed_size'], self.action_size)])
        self.policy_net = nn.Sequential(*self.policy_layers)
        self.policy_net.to(self.args_dict['DEVICE'])
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()


        self.value_layers = mlp_block(params_dict['embed_size']+self.lstminsize, \
                                      params_dict['embed_size'], dropout=False, activation=nn.LeakyReLU)

        # for j in range(2):
        #     self.value_layers.extend(mlp_block(self.hidden_sizes[-1],\
        #                                             self.hidden_sizes[-1],dropout=False,activation=nn.LeakyReLU))
        self.value_layers.extend([nn.Linear(params_dict['embed_size'], self.action_size)])
        self.value_net = nn.Sequential(*self.value_layers)
        self.value_net.to(self.args_dict['DEVICE'])


    def init_backbone(self):
        # Different conv layers for different scales
        size = deepcopy(self.input_size)
        size[-1] = int(size[-1]/len(self.env.scale))
        self.conv_blocks = [subnet_enc_setter.set_model(self.params_dict['obs_model'],self.conv_sizes,size\
                                      ,self.args_dict['DEVICE'],self.params_dict) for _ in self.env.scale]

    def get_conv_embeddings(self,input_obs):
        conv_embeddings = []
        B,N,H,W,C = input_obs.shape
        step = int(C/len(self.env.scale))
        for j,layers in enumerate(self.conv_blocks):
            embeddings = layers(input_obs[:,:,:,:,j*step:(j+1)*step])
            if self.params_dict['obs_model']=='Conv':
                conv_embeddings.append(embeddings.permute([0,1,3,4,2]).reshape((B,N,-1)))
            else:
                conv_embeddings.append(embeddings.reshape((B,N,-1)))

        return conv_embeddings

    def encoder_forward(self,input,hidden_in=None):
        encoded_gru,hidden_state = self.LSTM(input,hidden_in.permute([1,0,2]))
        return encoded_gru,hidden_state.permute([1,0,2])

    def reset_hidden_state(self):
        self.hidden_state = None

    def set_hidden_state(self,hidden_in):
        self.hidden_state = hidden_in

    def forward(self,input,pos,previous_actions,graph_node,budget=None,hidden_in=None):

        conv_embeddings = torch.stack(self.get_conv_embeddings(input),dim=-1)
        conv_embeddings = conv_embeddings.permute([0,1,3,2])
        B,N,L,D = conv_embeddings.shape
        conv_embeddings = conv_embeddings.reshape(B*N,L,D)
        # get the first attention value
        #attention_values,attention_weights = self.attention_model(conv_embeddings,conv_embeddings,conv_embeddings)
        #attention_values = attention_values.reshape((B,N,L,D)).sum(dim=-2)
        attention_values = torch.max(conv_embeddings.reshape((B,N,L,D)),dim=-2)
        pos = self.position_layer(pos)
        graph_node = self.graph_node_layer(graph_node.to(torch.float32))
        if self.args_dict['FIXED_BUDGET']:
            budget = self.budget_layers(budget)

        # For attention block
        if self.args_dict['FIXED_BUDGET']:
            tokens = torch.cat([pos, attention_values, graph_node, previous_actions,budget], dim=-1)
        else:
            tokens = torch.cat([pos, attention_values, graph_node, previous_actions], dim=-1)

        encoded_gru, hidden_state = self.encoder_forward(tokens,hidden_in)
        encoded_gru = torch.cat([encoded_gru,tokens],dim=-1)
        values = self.value_net(encoded_gru)
        policy = self.policy_net(encoded_gru)

        return self.softmax(policy),values,self.sigmoid(policy),hidden_state


    def forward_step(self, input,hidden_in=None):
        observation = input['obs']
        pos = input['position']
        prev_a = input['previous_actions']
        graph_nodes = input['node']
        budget = []
        if self.args_dict['FIXED_BUDGET']:
            budget = input['budget']

        if hidden_in is None:
            hidden_in = torch.zeros((1,len(self.env.agents),self.params_dict['embed_size']))\
                .to(torch.float32)

        observation = torch.tensor([observation], dtype=torch.float32).to(self.args_dict['DEVICE'])
        pos = torch.tensor([pos], dtype=torch.float32).to(self.args_dict['DEVICE'])
        prev_a = F.one_hot(torch.tensor([prev_a]),
                           num_classes = self.action_size).to(self.args_dict['DEVICE'])
        graph_nodes = F.one_hot(torch.tensor(np.array([graph_nodes])),
                                num_classes = self.num_graph_nodes).to(self.args_dict['DEVICE'])
        hidden_in = torch.tensor(hidden_in).to(torch.float32).to(self.args_dict['DEVICE'])
        if self.args_dict['FIXED_BUDGET']:
            budget = torch.tensor(np.array([budget]), dtype=torch.float32).to(self.args_dict['DEVICE'])
        policy,values,valids,hidden_state = self.forward(observation,pos,prev_a,graph_nodes,budget,hidden_in)

        #Update hidden state, make sure hidden state is reset after every episode
        self.hidden_state = hidden_state
        return policy,values,hidden_state


    def compute_valids(self,input):
        # print(input.shape)
        input = self.layers(input)

        policy = self.policy_net(input)
        return self.sigmoid(policy)

    def forward_buffer(self, obs_buffer,hidden_in_buffer):
        obs = []
        valids = []
        pos = []
        prev_a = []
        graph_nodes = []
        budget = []
        hidden_in = []

        for j,hidden in zip(obs_buffer,hidden_in_buffer):
            obs.append(j['obs'])
            valids.append(j['valids'])
            pos.append(j['position'])
            prev_a.append(j['previous_actions'])
            graph_nodes.append(j['node'])
            if hidden is not None:
                hidden_in.append(hidden[0])
            else:
                hidden_in.append(np.zeros((len(self.env.agents),
                                                       self.params_dict['embed_size'])))

            if self.args_dict['FIXED_BUDGET']:
                budget.append(j['budget'])
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.args_dict['DEVICE'])
        pos = torch.tensor(np.array(pos), dtype=torch.float32).to(self.args_dict['DEVICE'])

        hidden_in = torch.tensor(np.array(hidden_in),dtype=torch.float32).to(self.args_dict['DEVICE'])
        prev_a = F.one_hot(torch.tensor(prev_a), \
                           num_classes=self.action_size).to(self.args_dict['DEVICE'])
        graph_nodes = F.one_hot(torch.tensor(np.array(graph_nodes)),
                                num_classes=self.num_graph_nodes).to(self.args_dict['DEVICE'])
        if self.args_dict['FIXED_BUDGET']:
            budget = torch.tensor(np.array(budget), dtype=torch.float32).to(self.args_dict['DEVICE'])

        policy, value,valids_net,hout = self.forward(obs, pos, prev_a, graph_nodes,budget,hidden_in)
        valids = torch.tensor(valids, dtype=torch.float32).to(self.args_dict['DEVICE'])
        return policy.squeeze(), value.squeeze(), \
               valids.squeeze(), valids_net.squeeze()