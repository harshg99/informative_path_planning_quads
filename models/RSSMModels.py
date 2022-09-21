import torch.nn as nn
import torch
import numpy as np
from models.Models import *
from models.Vanilla import *
from models.subnets import *
import torch.nn.functional as F
from subnet_setter import subnet_enc_setter

class Model(Vanila):
    def __init__(self,env,params_dict,args_dict):
        super(Vanilla,self).__init__()
        self.hidden_sizes = params_dict['hidden_sizes']
        self.env = env
        self.input_size = env.input_size
        self.action_size = env.action_size
        self.params_dict = params_dict
        self.args_dict = args_dict

        self.hidden_state = None
        self._invert_hidden_sizes = np.array(self.hidden_sizes)[::-1].tolist()
        self.encoder_layer = subnet_setter.set_model(params_dict['obs_model'], \
                                               self.hidden_sizes, self.input_size)

        self.decoder_layer = subnet_setter.set_model(params_dict['obs_model'],self._invert_hidden_sizes,
                                                     self.hidden_sizes,self.input_size)


        #TODO: define GRU
        self.LSTM = nn.GRU(input_size=self.hidden_sizes[-1],
                           hidden_size=2*self.hidden_sizes[-1],
                           num_layers=self.grulayers,
                           batch_first=True
                           )

        self.attention_model = nn.MultiheadAttention(embed_dim=1,num_heads=params_dict['num_heads'])



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

        self.policy_layers = mlp_block(self.hidden_sizes[-1], self.hidden_sizes[-1], dropout=False,
                                      activation=nn.LeakyReLU)
        # for j in range(2):
        #     self.policy_layers.extend(mlp_ block(self.hidden_sizes[-1], self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU))

        self.policy_layers.extend([nn.Linear(self.hidden_sizes[-1], self.action_size)])
        self.policy_net = nn.Sequential(*self.policy_layers)
        self.policy_net.to(self.args_dict['DEVICE'])
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.value_layers = mlp_block(self.hidden_sizes[-1], self.hidden_sizes[-1], dropout=False,
                                      activation=nn.LeakyReLU)
        # for j in range(2):
        #     self.value_layers.extend(mlp_block(self.hidden_sizes[-1], self.hidden_sizes[-1],dropout=False,activation=nn.LeakyReLU))
        self.value_layers.extend([nn.Linear(self.hidden_sizes[-1], 1)])
        self.value_net = nn.Sequential(*self.value_layers)
        self.value_net.to(self.args_dict['DEVICE'])
        self.optim = torch.optim.Adam(self._model.parameters(), lr=params_dict['LR'], betas=(0.9, 0.99))

    def encoder_forward(self,input):
        encoded_values = self.encoder_layer(input)
        encoded_gru,hidden_state = self.LSTM(encoded_values,self.hidden_state)
        return encoded_values,encoded_gru,hidden_state


    def forward(self,input,pos,previous_actions,graph_node):

        encoded_values, encoded_gru, hidden_state, decoded_values = self.encoder_forward(input)
        pos = self.position_layer(pos)
        input = self.layers(input)
        graph_node = self.graph_node_layer(graph_node.to(torch.float32))

        # For attention block
        tokens = torch.cat([pos,encoded_gru.detach(),graph_node,previous_actions],dim=0)

        values = self.value_net(tokens)

        tokens = tokens.unsqueeze(dim=-1)
        attention_tokens = self.attention_model(tokens,tokens,tokens)

        attention_tokens = attention_tokens.squeeze(dim=-1)
        policy = self.policy_layers(attention_tokens)

        return self.softmax(policy),values,self.sigmoid(policy),hidden_state


    def forward_step(self, input):
        observation = input['obs']
        pos = input['position']
        previous_actions = input['previous_actions']
        graph_node = input['node']

        observation = torch.tensor([observation], dtype=torch.float32).to(self.args_dict['DEVICE'])
        pos = torch.tensor([pos], dtype=torch.float32).to(self.args_dict['DEVICE'])
        previous_actions = torch.tensor([previous_actions], dtype=torch.float32).to(self.args_dict['DEVICE'])
        graph_node = torch.tensor([graph_node], dtype=torch.float32).to(self.args_dict['DEVICE'])

        policy,values,valids,hidden_state = self.forward(observation,pos,previous_actions,graph_node)
        #Update hidden state, make sure hidden state is reset after every episode
        self.hidden_state = hidden_state
        return policy,values


    def reconstrunction_backward(self,input,targets):
        encoded_values, encoded_gru, hidden_state, decoded_targets = self.encoder_forward(input)
        targets = torch.tensor(targets,dtype=torch.float32).view(targets.shape[0],targets.shape[1],-1)

        encoded_gru_mean,encoded_gru_var = torch.split(encoded_gru,2,dim=-1)

        sampled_vectors_shape = encoded_gru_mean.shape.cpu().detach().numpy().tolist()
        sampled_vectors_shape.append(self.params_dict['samples'])
        sampled_vectors = torch.normal(0,std=torch.ones(sampled_vectors_shape))

        kl_loss = (1 + encoded_gru_var - torch.square(encoded_gru_mean) - torch.exp(encoded_gru_var)).sum()

        encoded_gru_var_reshaped = torch.repeat_interleave(encoded_gru_var.unsqueeze(dim=-1),\
                                                           repeats=self.params_dict['samples'],
                                                           dim=-1)
        encoded_gru_mean_reshaped = torch.repeat_interleave(encoded_gru_mean.unsqueeze(dim=-1),\
                                                           repeats=self.params_dict['samples'],
                                                           dim=-1)
        # Shape ( B,TS,n_repeats_dim)
        targets = torch.repeat_interleave(encoded_gru_mean.unsqueeze(dim=-2),\
                                                           repeats=self.params_dict['samples'],
                                                           dim=-2)
        encoded_gru_samples = encoded_gru_mean_reshaped + sampled_vectors*torch.exp(0.5*encoded_gru_var_reshaped)

        encoded_gru_samples = torch.permute(encoded_gru_samples,[0,1,3,2])
        decoded_gru_samples = self.decoder_layer(encoded_gru_samples)
        reconstruction_error = torch.square(targets - decoded_gru_samples).sum()

        total_loss = kl_loss + reconstruction_error
        total_loss.backward()

        gradient = []
        for params in self.parameters():
            gradient.append(params.grad)

        norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), 50)

        train_metrics = {'Reconstruction Loss': reconstruction_error.sum().cpu().detach().numpy().item(),
                         'KL Loss': kl_loss.sum().cpu().detach().numpy().item() ,
                         'Grad Norm': norm.detach().cpu().numpy().item()}
        return train_metrics,gradient

    def compute_valids(self,input):
        # print(input.shape)
        input = self.layers(input)

        policy = self.policy_net(input)
        return self.sigmoid(policy)

    # For supervised learning of any form
    def supervised_forward_buffer(self,obs_buffer):
        obs = []
        targets = []
        for j in obs_buffer:
            obs.append(j['obs'])
            targets.append(j['targets'])


        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.args_dict['DEVICE'])
        targets = torch.tensor(np.array(targets), dtype=torch.float32).to(self.args_dict['DEVICE'])
        return self.reconstrunction_backward(obs,targets)

    def forward_buffer(self, obs_buffer):
        obs = []
        valids = []
        pos = []
        prev_a = []
        graph_nodes = []
        for j in obs_buffer:
            obs.append(j['obs'])
            valids.append(j['valids'])
            pos.append(j['position'])
            prev_a.append(j['previous_actions'])
            graph_nodes.append(j['node'])
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.args_dict['DEVICE'])
        pos = torch.tensor(np.array(pos), dtype=torch.float32).to(self.args_dict['DEVICE'])
        prev_a = F.one_hot(torch.tensor(prev_a), \
                           num_classes=self.action_size).to(self.args_dict['DEVICE'])
        graph_nodes = F.one_hot(torch.tensor(np.array(graph_nodes)),
                                num_classes=self.num_graph_nodes).to(self.args_dict['DEVICE'])
        policy, value = self.forward(obs, pos, prev_a, graph_nodes)
        valids_net = self.compute_valids(obs, pos, prev_a, graph_nodes)
        valids = torch.tensor(valids, dtype=torch.float32).to(self.args_dict['DEVICE'])
        return policy.squeeze(), value.squeeze(), \
               valids.squeeze(), valids_net.squeeze()
