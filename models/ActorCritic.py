import torch
import torch.nn as nn
from params import *
import numpy as np
import Utilities
from models.Vanilla import Vanilla
from torch.optim.lr_scheduler import ExponentialLR


class ActorCritic(Vanilla):
    def __init__(self,input_size,action_size,params_dict):
        super(Vanilla,self).__init__()
        self.hidden_sizes = params_dict['hidden_sizes']
        self.input_size = input_size
        self.action_size = action_size
        self.params_dict = params_dict

        self.policy_layers = self.mlp_block(self.input_size,self.hidden_sizes[0],dropout=False)
        for j in range(len(self.hidden_sizes)-1):
            self.policy_layers.extend(self.mlp_block(self.hidden_sizes[j],self.hidden_sizes[j+1],dropout=False,activation=nn.LeakyReLU))
        self.policy_layers.extend([nn.Linear(self.hidden_sizes[-1],self.action_size)])
        self.policy_net = nn.Sequential(*self.policy_layers)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.value_layers = self.mlp_block(self.input_size, self.hidden_sizes[0], dropout=False)
        for j in range(len(self.hidden_sizes) - 1):
            self.value_layers.extend(self.mlp_block(self.hidden_sizes[j], self.hidden_sizes[j + 1],dropout=False,activation=nn.LeakyReLU))
        self.value_layers.extend([nn.Linear(self.hidden_sizes[-1], 1)])
        self.value_net = nn.Sequential(*self.value_layers)

        self.optim = torch.optim.Adam(self.parameters(),lr=params_dict['LR'],betas=(0.9,0.99))
        self.scheduler = ExponentialLR(self.optim,gamma=params_dict['DECAY'])

    def mlp_block(self,hidden_size,out_size,dropout=True,droputProb = 0.5,activation = None,batchNorm = False):
        if activation is None:
            layers = [nn.Linear(hidden_size,out_size)]
        else:
            layers = [nn.Linear(hidden_size,out_size),activation()]
        if batchNorm:
            layers.append(nn.BatchNorm1d(out_size))
        if dropout:
            layers.append(nn.Dropout(p=droputProb))
        return layers

    def forward(self,input):
        #print(input.shape)
        policy = self.policy_net(input)
        return self.softmax(policy),self.value_net(input)

    def forward_step(self,input):
        input = input['obs']
        input = torch.tensor([input], dtype=torch.float32)
        return self.forward(input)

    def reset(self,episodeNum):
        self.episodeNum = episodeNum

    def get_advantages(self,train_buffer):
        rewards_plus = np.copy(train_buffer['rewards']).tolist()
        rewards_plus.append(train_buffer['bootstrap_value'][:,0].tolist())
        rewards_plus = np.array(rewards_plus).squeeze()
        discount_rewards = Utilities.discount(rewards_plus,DISCOUNT)[:-1]

        values_plus = train_buffer['values']
        values_plus.append(train_buffer['bootstrap_value'])
        values_plus = np.array(values_plus).squeeze()
        advantages = np.array(train_buffer['rewards']).squeeze() + DISCOUNT*values_plus[1:] - values_plus[:-1]

        train_buffer['advantages'] = advantages
        train_buffer['discounted_rewards'] = np.copy(discount_rewards)

        return train_buffer

    def compute_loss(self,train_buffer):
        advantages = train_buffer['advantages']
        target_v = train_buffer['discounted_rewards']
        a_batch = np.array(train_buffer['actions'])

        obs = []
        for j in train_buffer['obs']:
            obs.append(j['obs'])

        obs = torch.tensor(np.array(obs).squeeze(axis=1), dtype=torch.float32)
        policy, value = self.forward(obs)
        target_v = torch.tensor(target_v, dtype=torch.float32)
        a_batch = torch.tensor(a_batch, dtype=torch.int64)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        responsible_outputs = policy.gather(-1, a_batch)
        v_l = self.params_dict['value_weight'] * torch.square(value.squeeze() - target_v)
        e_l = -self.params_dict['entropy_weight'] * (policy * torch.log(torch.clamp(policy, min=1e-10, max=1.0)))
        p_l = -self.params_dict['policy_weight'] * torch.log(
        torch.clamp(responsible_outputs.squeeze(), min=1e-15, max=1.0)) * advantages.squeeze()
        return v_l,p_l,e_l

    def backward(self,train_buffer):
        self.optim.zero_grad()

        v_l,p_l,e_l = self.compute_loss(train_buffer)

        loss = v_l.sum() + p_l.sum() - e_l.sum()
        self.optim.zero_grad()
        loss.sum().backward()
        # self.optimizer.step()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 50)
        v_n = torch.linalg.norm(
            torch.stack([torch.linalg.norm(p.detach()).to("cpu") for p in self.parameters()])).detach().numpy().item()

        gradient = []
        for local_param in self.parameters():
            gradient.append(local_param.grad)
        g_n = norm.detach().cpu().numpy().item()
        episode_length = train_buffer['episode_length']
        train_metrics = {'Value Loss': v_l.sum().cpu().detach().numpy().item()/episode_length,
                         'Policy Loss': p_l.sum().cpu().detach().numpy().item()/episode_length,
                         'Entropy Loss': e_l.sum().cpu().detach().numpy().item()/episode_length,
                         'Grad Norm': g_n, 'Var Norm': v_n}
        return train_metrics, gradient

class ActorCritic2(ActorCritic):
    def __init__(self,input_size,action_size,params_dict):
        super(Vanilla,self).__init__()
        self.hidden_sizes = params_dict['hidden_sizes']
        self.input_size = input_size
        self.action_size = action_size
        self.params_dict = params_dict

        self.layers = self.mlp_block(self.input_size,self.hidden_sizes[0],dropout=False)
        for j in range(len(self.hidden_sizes)-1):
            self.layers.extend(self.mlp_block(self.hidden_sizes[j],self.hidden_sizes[j+1],dropout=False,activation=nn.LeakyReLU))

        self.policy_layers = self.layers.copy()
        for j in range(2):
            self.policy_layers.extend(self.mlp_block(self.hidden_sizes[-1], self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU))

        self.policy_layers.extend([nn.Linear(self.hidden_sizes[-1],self.action_size)])
        self.policy_net = nn.Sequential(*self.policy_layers)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.value_layers = self.layers.copy()
        for j in range(2):
            self.value_layers.extend(self.mlp_block(self.hidden_sizes[-1], self.hidden_sizes[-1],dropout=False,activation=nn.LeakyReLU))
        self.value_layers.extend([nn.Linear(self.hidden_sizes[-1], 1)])
        self.value_net = nn.Sequential(*self.value_layers)

        self.optim = torch.optim.Adam(self.parameters(),lr=params_dict['LR'],betas=(0.9,0.99))
        self.scheduler = ExponentialLR(self.optim,gamma=params_dict['DECAY'])

# With valid losses
class ActorCritic3(ActorCritic2):
    def __init__(self,input_size,action_size,params_dict):
        super(ActorCritic2,self).__init__(input_size,action_size,params_dict)

    def compute_loss(self, train_buffer):
        advantages = train_buffer['advantages']
        target_v = train_buffer['discounted_rewards']
        a_batch = np.array(train_buffer['actions'])

        obs = []
        valids = []
        for j in train_buffer['obs']:
            obs.append(j['obs'])
            valids.append(j['valids'])

        valids = torch.tensor(np.array(valids),dtype=torch.float32)

        obs = torch.tensor(np.array(obs).squeeze(axis=1), dtype=torch.float32)
        policy, value = self.forward(obs)
        valids_net = self.compute_valids(obs)

        target_v = torch.tensor(target_v, dtype=torch.float32)
        a_batch = torch.tensor(a_batch, dtype=torch.int64)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        responsible_outputs = policy.gather(-1, a_batch)
        v_l = self.params_dict['value_weight'] * torch.square(value.squeeze() - target_v)
        e_l = -self.params_dict['entropy_weight'] * (policy * torch.log(torch.clamp(policy, min=1e-10, max=1.0)))
        p_l = -self.params_dict['policy_weight'] * torch.log(
            torch.clamp(responsible_outputs.squeeze(), min=1e-15, max=1.0)) * advantages.squeeze()
        #valid_l = self.params_dict['valids_weight']* (valids*torch.log(torch.clip(valids_net,1e-7,1))+ (1-valids)*torch.log(torch.clip(1 - valids_net,1e-7,1)))
        valid_l = self.params_dict['valids_weight'] * ((1 - valids) * torch.log(torch.clip(1 - valids_net, 1e-7, 1)))
        return v_l, p_l, e_l,valid_l

    def compute_valids(self,input):
        policy = self.policy_net(input)
        return self.sigmoid(policy)

    def backward(self, train_buffer):
        self.optim.zero_grad()

        v_l, p_l, e_l,valid_l = self.compute_loss(train_buffer)

        loss = v_l.sum() + p_l.sum() - e_l.sum()
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






