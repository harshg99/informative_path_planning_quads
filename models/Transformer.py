import torch
import torch.nn as nn
from models.ActorCritic import ActorCritic3
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import Utilities
from params import *

class MHA(nn.Module):
    def __init__(self,config):
        # Number of attention blocks
        super(MHA, self).__init__()
        self.num_heads = config['num_heads']
        self.hidden = config['hidden']
        self.config = config
        # Concat and normalisation
        self.attnblocks = [ScaledDotProductAttention(config) for _ in range(self.num_heads)]
        # Feedforward to compressed state
        self.outlayer = nn.Linear(in_features=self.num_heads,out_features=1)

    def forward(self,keys_in,query_in=None):
        for j,blocks in enumerate(self.attnblocks):
            if j==0:
                concat_output = blocks(keys_in,query_in).unsqueeze(-1)
            else:
                concat_output = torch.concat([concat_output,blocks(keys_in,query_in).unsqueeze(-1)],dim=-1)
        mhaout = self.outlayer(concat_output).squeeze(-1)
        return mhaout

# scaled dot product attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self,config):
        super(ScaledDotProductAttention, self).__init__()
        self.key_length = config['action_size']
        #self.key_size = config['key_size']
        self.hidden = config['hidden']
        self.embed_size = config['embed']
        self.key_network = nn.Linear(self.embed_size,self.hidden)
        self.value_network = nn.Linear(self.embed_size,self.hidden)
        self.query_network = nn.Linear(self.embed_size,self.hidden)

    def forward(self,keys_in,query_in=None):
        if query_in is None:
            query_in = keys_in
        keys = self.key_network(keys_in)
        query = self.query_network(query_in)
        value = self.value_network(keys_in)
        factor = np.sqrt(self.hidden)
        kT = keys.transpose(-2,-1)
        QK = torch.matmul(query,kT)
        QK = torch.softmax(QK/factor,dim=-1)
        QKV = torch.matmul(QK,value)
        return QKV

#TODO Transformer Actor Critic with Keys as Actions and State Decoder

class TransformerAC(ActorCritic3):
    def __init__(self,input_size,action_size,params_dict):
        super(ActorCritic3, self).__init__(input_size,action_size,params_dict)
        self.hidden_sizes = params_dict['hidden_sizes']
        self.input_size = input_size
        self.action_size = action_size
        self.params_dict = params_dict
        params_dict['action_size'] = self.action_size

        self.layers = self.mlp_block(self.input_size, self.hidden_sizes[0])
        for j in range(len(self.hidden_sizes) - 1):
            self.layers.extend(
                self.mlp_block(self.hidden_sizes[j], self.hidden_sizes[j + 1], dropout=False, activation=nn.LeakyReLU))
        self.layers.extend(
            self.mlp_block(self.hidden_sizes[-1], self.params_dict['embed']*self.action_size, dropout=False, activation=nn.LeakyReLU))
        self.layers_net = nn.Sequential(*self.layers)
        self.attention_layer = MHA(params_dict)
        self.action_layer = nn.Linear(self.params_dict['action_depth'],self.params_dict['embed'])
        self.policy_layers = []
        self.policy_layers.extend([nn.Linear(params_dict['embed'], 1)])
        self.policy_net = nn.Sequential(*self.policy_layers)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.value_layers = []

        self.value_layers.extend([nn.Linear(params_dict['embed'], 1)])
        self.value_net = nn.Sequential(*self.value_layers)
        self.optim = torch.optim.Adam(self.parameters(), lr=params_dict['LR'], betas=(0.9, 0.99))
        self.scheduler = ExponentialLR(self.optim, gamma=params_dict['DECAY'])

    def forward(self,input,actions):
        '''
        @param: input: tuple of observation and action
        '''
        processed = self.layers_net(input)
        #reshaping here
        processed = processed.view((processed.shape[0],processed.shape[1],\
                                    self.action_size,self.params_dict['embed']))
        processed_actions = self.action_layer(actions)
        processed_attn = self.attention_layer(processed, processed_actions)
        value = self.value_net(processed_attn).squeeze(-1)
        policy = self.softmax(self.policy_net(processed_attn).squeeze(-1))
        return policy,value

    def forward_step(self,input):
        obs = input['obs']
        actions = input['mps']
        obs = torch.tensor([obs], dtype=torch.float32)
        actions = torch.tensor([actions],dtype=torch.float32)
        return self.forward(obs,actions)

    def compute_valids(self,input,actions):
        processed = self.layers_net(input)
        # reshaping here
        processed = processed.view((processed.shape[0], processed.shape[1],\
                                    self.action_size, self.params_dict['embed']))
        processed_actions = self.action_layer(actions)
        processed_attn = self.attention_layer(processed, processed_actions)
        valids = self.sigmoid(self.policy_net(processed_attn).squeeze(-1))
        return valids

    def get_advantages(self,train_buffer):
        rewards_plus = np.copy(train_buffer['rewards']).tolist()
        rewards_plus.append(train_buffer['bootstrap_value'][:,0].tolist())
        rewards_plus = np.array(rewards_plus).squeeze()
        discount_rewards = Utilities.discount(rewards_plus,DISCOUNT)[:-1]

        values_plus = train_buffer['values']
        policy = train_buffer['policy']
        values_plus = np.stack(values_plus)
        policy = np.stack(policy).squeeze(axis=1)
        values = np.sum(np.stack(values_plus*policy), axis=-1).squeeze(axis=-1)
        values_plus = values.tolist()+[np.max(train_buffer['bootstrap_value']).squeeze()]
        values_plus = np.array(values_plus).squeeze()
        advantages = np.array(train_buffer['rewards']).squeeze() + DISCOUNT*values_plus[1:] - values_plus[:-1]
        #advantages = Utilities.discount(advantages,DISCOUNT)
        train_buffer['advantages'] = advantages.copy()
        train_buffer['discounted_rewards'] = np.copy(discount_rewards)

        return train_buffer

    def compute_loss(self, train_buffer):
        advantages = train_buffer['advantages']
        target_v = train_buffer['discounted_rewards']
        a_batch = np.array(train_buffer['actions'])

        obs = []
        valids = []
        mps = []
        for j in train_buffer['obs']:
            obs.append(j['obs'])
            mps.append(j['mps'])
            valids.append(j['valids'])

        valids = torch.tensor(np.array(valids),dtype=torch.float32)

        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        mps = torch.tensor(np.array(mps),dtype=torch.float32)
        policy, value = self.forward(obs,mps)
        valids_net = self.compute_valids(obs,mps)
        policy = policy.squeeze()
        value = value.squeeze()
        target_v = torch.tensor(target_v, dtype=torch.float32)
        a_batch = torch.tensor(a_batch, dtype=torch.int64)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        responsible_outputs = policy.gather(-1, a_batch)
        v_l = self.params_dict['value_weight'] * torch.square(value.gather(-1,a_batch).squeeze()-target_v)
        e_l = -self.params_dict['entropy_weight'] * (policy * torch.log(torch.clamp(policy, min=1e-10, max=1.0)))
        p_l = -self.params_dict['policy_weight'] * torch.log(
            torch.clamp(responsible_outputs.squeeze(), min=1e-15, max=1.0)) * advantages.squeeze()
        #valid_l = self.params_dict['valids_weight']* (valids*torch.log(torch.clip(valids_net,1e-7,1))+ (1-valids)*torch.log(torch.clip(1 - valids_net,1e-7,1)))
        valid_l = self.params_dict['valids_weight'] * ((1 - valids) * torch.log(torch.clip(1 - valids_net, 1e-7, 1)) + \
                                                       valids * torch.log(torch.clip(valids_net, 1e-7, 1)))
        return v_l, p_l, e_l,valid_l

    def backward(self, train_buffer):
        self.optim.zero_grad()

        v_l, p_l, e_l,valid_l = self.compute_loss(train_buffer)

        loss = v_l.sum() + p_l.sum() - e_l.sum() + valid_l.sum()
        self.optim.zero_grad()
        loss.sum().backward()
        # self.optimizer.step()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 50)
        v_n = torch.linalg.norm(
            torch.stack(
                [torch.linalg.norm(p.detach()).to("cpu") for p in self.parameters()])).detach().numpy().item()

        gradient = []
        for local_param in self.parameters():
            gradient.append(local_param.grad)
        g_n = norm.detach().cpu().numpy().item()
        episode_length = train_buffer['episode_length']
        train_metrics = {'Value Loss': v_l.sum().cpu().detach().numpy().item() / episode_length,
                         'Policy Loss': p_l.sum().cpu().detach().numpy().item() / episode_length,
                         'Entropy Loss': e_l.sum().cpu().detach().numpy().item() / episode_length,
                         'Valid Loss': valid_l.sum().cpu().detach().numpy().item()/episode_length,
                         'Grad Norm': g_n, 'Var Norm': v_n}
        return train_metrics, gradient

